"""model.py

Build the deep-neural-network metamodel architecture and loss/optimizer.

Python/torch port of ``R/model.R`` (originally a Keras sequential model).

Low-hanging-fruit cleanup: the R version repeated nine near-identical
``if (length(layers) >= n)`` blocks to add hidden layers. Here the hidden
layers are built in a single loop of arbitrary depth.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def sse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Sum of squared errors, matching the R ``SSE`` custom loss."""
    return torch.sum((y_true - y_pred) ** 2)


# Loss functions keyed by the string names used in the R API.
LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "sse": sse_loss,
    "mse": nn.functional.mse_loss,
    "mae": nn.functional.l1_loss,
}

# Optimizers keyed by the ``opt.alg`` names used in the R API.
OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamax": torch.optim.Adamax,
    "nadam": torch.optim.NAdam,
    "rmsprop": torch.optim.RMSprop,
}


class Metamodel(nn.Module):
    """Fully connected ReLU network with a single linear output (keff)."""

    def __init__(self, input_dim: int, layers: list[int]):
        super().__init__()
        modules: list[nn.Module] = []
        prev = input_dim
        for units in layers:
            modules.append(nn.Linear(prev, units))
            modules.append(nn.ReLU())
            prev = units
        modules.append(nn.Linear(prev, 1))  # linear output activation
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_layers(layers: str | list[int]) -> list[int]:
    """Parse a ``"8192-256-256"`` architecture string into a list of widths."""
    if isinstance(layers, str):
        return [int(tok) for tok in layers.split("-")]
    return list(layers)


def build_model(
    input_dim: int,
    layers: str | list[int] = "8192-256-256-256-256-16",
    loss: str = "sse",
    opt_alg: str = "adamax",
    learning_rate: float = 0.00075,
    device: str | torch.device = "cpu",
) -> tuple[Metamodel, Callable, torch.optim.Optimizer]:
    """Build the metamodel, loss function, and optimizer.

    Args:
        input_dim: Number of input features (columns of ``training.df``).
        layers: Architecture string (e.g. ``"256-256-16"``) or list of widths.
        loss: Loss key (``"sse"``, ``"mse"``, ``"mae"``).
        opt_alg: Optimizer key (e.g. ``"adamax"``, ``"adam"``, ``"rmsprop"``).
        learning_rate: Optimizer learning rate.
        device: Torch device.

    Returns:
        ``(model, loss_fn, optimizer)``.
    """
    widths = parse_layers(layers)
    model = Metamodel(input_dim, widths).to(device)

    if loss not in LOSSES:
        raise ValueError(f"unknown loss '{loss}', choose from {sorted(LOSSES)}")
    if opt_alg not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{opt_alg}', choose from {sorted(OPTIMIZERS)}")

    loss_fn = LOSSES[loss]
    optimizer = OPTIMIZERS[opt_alg](model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer
