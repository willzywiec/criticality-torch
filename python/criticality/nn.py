"""nn.py

Train an ensemble of deep neural networks to predict keff values.

Python/torch port of ``R/nn.R``.

Low-hanging-fruit cleanup: the R code trained each model, then *retrained* it
for ``epochs/10`` while writing one ``.h5`` checkpoint per epoch, then later
re-read every checkpoint to find the epoch with the lowest train+val MAE. That
whole dance is just "keep the best epoch's weights". Here a single training
loop records the full per-epoch history and snapshots the best weights
(by train MAE + val MAE) in memory, then restores them at the end.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .ensemble import test_ensemble
from .model import build_model
from .plot import plot_history
from .scale import Dataset


@dataclass
class History:
    """Per-epoch training history (mirrors the R Keras history data frame)."""

    epoch: list[int]
    loss: list[float]
    mae: list[float]
    val_loss: list[float]
    val_mae: list[float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": self.epoch,
                "val.loss": self.val_loss,
                "val.mae": self.val_mae,
                "loss": self.loss,
                "mae": self.mae,
            }
        )


def _split_tensors(
    dataset: Dataset, val_split: float, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Build train/validation tensors, holding out the last ``val_split``.

    Keras' ``validation_split`` uses the final fraction of the data without
    shuffling, so we replicate that to keep results comparable.
    """
    x = torch.tensor(dataset.training_df.to_numpy(dtype=np.float32), device=device)
    y = torch.tensor(
        dataset.training_data["keff"].to_numpy(dtype=np.float32), device=device
    )
    n_val = int(round(len(x) * val_split))
    if n_val == 0:
        return x, y, x[:0], y[:0]
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def fit(
    dataset: Dataset,
    model,
    loss_fn,
    optimizer,
    batch_size: int,
    epochs: int,
    val_split: float,
    device: torch.device,
    verbose: bool = False,
) -> tuple[History, dict]:
    """Train one model, returning its history and the best-epoch state dict."""
    x_tr, y_tr, x_val, y_val = _split_tensors(dataset, val_split, device)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    hist = History([], [], [], [], [])
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        # Evaluate on the full train and validation partitions.
        model.eval()
        with torch.no_grad():
            tr_pred = model(x_tr)
            tr_loss = loss_fn(tr_pred, y_tr).item()
            tr_mae = torch.mean(torch.abs(tr_pred - y_tr)).item()
            if len(x_val) > 0:
                val_pred = model(x_val)
                val_loss = loss_fn(val_pred, y_val).item()
                val_mae = torch.mean(torch.abs(val_pred - y_val)).item()
            else:
                val_loss = val_mae = float("nan")

        hist.epoch.append(epoch)
        hist.loss.append(tr_loss)
        hist.mae.append(tr_mae)
        hist.val_loss.append(val_loss)
        hist.val_mae.append(val_mae)

        # Snapshot the weights that minimize train MAE + val MAE.
        score = tr_mae + (0.0 if np.isnan(val_mae) else val_mae)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"loss={tr_loss:.4f}  mae={tr_mae:.4f}  val_mae={val_mae:.4f}"
            )

    return hist, best_state


def _settings_text(**kwargs) -> str:
    lines = ["model settings"] + [f"{k}: {v}" for k, v in kwargs.items()]
    return "\n".join(lines) + "\n"


def train_nn(
    dataset: Dataset,
    *,
    batch_size: int = 8192,
    code: str = "mcnp",
    ensemble_size: int = 5,
    epochs: int = 1500,
    layers: str = "8192-256-256-256-256-16",
    loss: str = "sse",
    opt_alg: str = "adamax",
    learning_rate: float = 0.00075,
    val_split: float = 0.2,
    overwrite: bool = False,
    replot: bool = True,
    reweight: bool = False,
    verbose: bool = False,
    ext_dir: str | Path,
    training_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Train (or load) an ensemble of metamodels and compute ensemble weights.

    Returns a dict ``{"models": [...], "weights": np.ndarray}`` analogous to the
    ``list(metamodel, wt)`` returned by the R ``NN`` function.
    """
    device = torch.device(device)
    ext_dir = Path(ext_dir)
    training_dir = Path(training_dir) if training_dir else ext_dir / "training"
    model_dir = training_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    settings = _settings_text(
        batch_size=batch_size,
        code=code,
        ensemble_size=ensemble_size,
        epochs=epochs,
        layers=layers,
        loss=loss,
        optimizer=opt_alg,
        learning_rate=learning_rate,
        validation_split=val_split,
    )
    settings_path = training_dir / "model-settings.txt"
    settings_changed = (
        not settings_path.exists() or settings_path.read_text() != settings
    )
    if settings_changed and settings_path.exists() and not overwrite:
        raise RuntimeError("model settings changed; rerun with overwrite=True")
    if settings_changed and overwrite:
        for ckpt in model_dir.glob("*.pt"):
            ckpt.unlink()
    settings_path.write_text(settings)

    input_dim = dataset.training_df.shape[1]
    models = []

    existing = sorted(model_dir.glob("*.pt"))
    can_load = (
        not settings_changed
        and not reweight
        and len(existing) >= ensemble_size
    )

    if can_load:
        for i in range(ensemble_size):
            model, _, _ = build_model(
                input_dim, layers, loss, opt_alg, learning_rate, device
            )
            model.load_state_dict(torch.load(model_dir / f"{i + 1}.pt"))
            models.append(model)
            if replot and (model_dir / f"{i + 1}.csv").exists():
                plot_history(i + 1, plot_dir=model_dir)
    else:
        for i in range(ensemble_size):
            model, loss_fn, optimizer = build_model(
                input_dim, layers, loss, opt_alg, learning_rate, device
            )
            if verbose:
                print(f"training model {i + 1}/{ensemble_size}")
            hist, best_state = fit(
                dataset, model, loss_fn, optimizer,
                batch_size, epochs, val_split, device, verbose,
            )
            model.load_state_dict(best_state)
            torch.save(model.state_dict(), model_dir / f"{i + 1}.pt")
            hist.to_frame().to_csv(model_dir / f"{i + 1}.csv", index=False)
            plot_history(i + 1, history=hist.to_frame(), plot_dir=model_dir)
            models.append(model)

    weights = test_ensemble(
        dataset, models, ext_dir=ext_dir, training_dir=training_dir, device=device
    )
    return {"models": models, "weights": weights}
