"""predict.py

Sample the Bayesian network and predict keff values with the DNN metamodel.

Python/torch port of ``R/predict.R``.

Fix applied during the port: the R code called ``Scale`` without passing the
trained dataset, which (via R's ``exists('dataset')`` check) silently fell back
to the *training* code path and recomputed the mean/sd from the BN samples
themselves. That is almost certainly unintended -- new data must be scaled with
the training statistics. Here we pass the trained ``dataset`` explicitly so the
encoding and standardization are reused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bn import BayesNet
from .scale import Dataset, scale_data
from .simulate import FISS_DENSITY


def _predict(model, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def predict_keff(
    bn: BayesNet,
    metamodel: dict,
    dataset: Dataset,
    *,
    keff_cutoff: float = 0.9,
    mass_cutoff: float = 100,
    rad_cutoff: float = 7,
    sample_size: int = 1_000_000,
    ext_dir: str | Path,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample the BN and append predicted keff values.

    Args:
        bn: Fitted :class:`~criticality.bn.BayesNet`.
        metamodel: ``{"models": [...], "weights": np.ndarray}`` from ``train_nn``.
        dataset: Trained :class:`~criticality.scale.Dataset` (for scaling).
        keff_cutoff: Keep only samples with predicted keff at or above this.
        mass_cutoff: Minimum mass (g) to retain.
        rad_cutoff: Minimum radius (cm) to retain.
        sample_size: Number of BN samples to draw.
        ext_dir: External directory.
        device: Torch device.
        rng: Optional numpy RNG.
        verbose: Whether to print progress.

    Returns:
        The surviving BN samples with a ``keff`` column.
    """
    device = torch.device(device)
    rng = rng or np.random.default_rng()
    models = metamodel["models"]
    weights = np.asarray(metamodel["weights"], dtype=float)

    bn_dist = bn.sample(sample_size, rng).dropna()
    if verbose:
        print("BN samples generated")

    bn_dist = bn_dist[
        (bn_dist["mass"].astype(float) > mass_cutoff)
        & (bn_dist["rad"].astype(float) > rad_cutoff)
    ].reset_index(drop=True)
    if verbose:
        print(f"BN filtering complete ({len(bn_dist)})")

    if len(bn_dist) == 0:
        bn_dist["keff"] = []
        return bn_dist

    mass = bn_dist["mass"].astype(float).to_numpy()
    rad = bn_dist["rad"].astype(float).to_numpy()
    thk = bn_dist["thk"].astype(float).to_numpy().copy()
    form = bn_dist["form"].astype(str).to_numpy()
    mod = bn_dist["mod"].astype(str).to_numpy().astype(object)
    ref = bn_dist["ref"].astype(str).to_numpy().astype(object)

    density = np.array([FISS_DENSITY[f] for f in form])
    vol = 4 / 3 * np.pi * rad**3

    # When the sphere can't physically hold the mass, drop moderation and
    # shrink the volume to the bare-metal volume, then recompute the radius.
    too_small = vol <= mass / density
    mod[too_small] = "none"
    vol[too_small] = mass[too_small] / density[too_small]
    rad = (3 / 4 * vol / np.pi) ** (1 / 3)

    # Reflector / thickness consistency.
    ref[thk == 0] = "none"
    thk[ref == "none"] = 0

    conc = np.where(vol == 0, 0.0, mass / vol)

    processed = pd.DataFrame(
        {
            "mass": mass,
            "form": form,
            "mod": mod,
            "rad": rad,
            "ref": ref,
            "thk": thk,
            "vol": vol,
            "conc": conc,
        }
    )
    if verbose:
        print("BN processing complete")

    bn_df = scale_data(processed, ext_dir=ext_dir, dataset=dataset)
    x = torch.tensor(bn_df.to_numpy(dtype=np.float32), device=device)

    # First pass: cheap single-model cutoff before the full ensemble.
    if keff_cutoff > 0 and len(bn_df) > 1:
        old_n = len(bn_df)
        keff0 = _predict(models[0], x)
        keep = keff0 >= keff_cutoff
        bn_df = bn_df[keep].reset_index(drop=True)
        processed = processed[keep].reset_index(drop=True)
        x = x[torch.tensor(keep)]
        if verbose:
            print(f"Initial predictions complete ({old_n} --> {len(bn_df)})")

    if len(bn_df) >= 1:
        preds = np.column_stack([_predict(m, x) for m in models])
        processed["keff"] = preds @ weights

    return processed
