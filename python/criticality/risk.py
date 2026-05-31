"""risk.py

Estimate process criticality accident risk.

Python/torch port of ``R/risk.R``. Repeatedly samples the Bayesian network and
runs the metamodel, then reports the fraction of samples whose predicted keff
meets or exceeds the upper subcritical limit (USL).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bn import BayesNet
from .predict import predict_keff
from .scale import Dataset


def estimate_risk(
    bn: BayesNet,
    metamodel: dict,
    dataset: Dataset,
    *,
    facility: str = "facility",
    dist: str = "gamma",
    keff_cutoff: float = 0.9,
    mass_cutoff: float = 100,
    rad_cutoff: float = 7,
    risk_pool: int = 100,
    sample_size: int = 1_000_000,
    usl: float = 0.95,
    ext_dir: str | Path,
    training_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate process criticality accident risk.

    Returns ``(risk, bn_dist)`` where ``risk`` is the per-pool risk array and
    ``bn_dist`` is the concatenated set of supercritical-leaning samples.
    """
    ext_dir = Path(ext_dir)
    training_dir = Path(training_dir) if training_dir else ext_dir / "training"
    rng = rng or np.random.default_rng()

    stamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    risk_dir = ext_dir / "risk" / f"{facility}-{stamp}"
    risk_dir.mkdir(parents=True, exist_ok=True)

    settings = [
        "risk settings",
        f"distribution: {dist}",
        f"facility: {facility}",
        f"keff cutoff: {keff_cutoff}",
        f"mass cutoff (g): {mass_cutoff}",
        f"rad cutoff (cm): {rad_cutoff}",
        f"risk pool: {risk_pool}",
        f"sample size: {sample_size}",
        f"upper subcritical limit: {usl}",
        f"external directory: {ext_dir}",
        f"training directory: {training_dir}",
        f"risk directory: {risk_dir}",
    ]
    (risk_dir / "risk-settings.txt").write_text("\n".join(settings) + "\n")

    # Cap memory: very large sample sizes are split into more pools.
    if sample_size > 5e8:
        risk_pool = int(risk_pool * sample_size / 5e8)
        sample_size = int(5e8)

    risk = np.zeros(risk_pool)
    pools = []
    for i in range(risk_pool):
        samples = predict_keff(
            bn,
            metamodel,
            dataset,
            keff_cutoff=keff_cutoff,
            mass_cutoff=mass_cutoff,
            rad_cutoff=rad_cutoff,
            sample_size=sample_size,
            ext_dir=ext_dir,
            device=device,
            rng=rng,
            verbose=False,
        )
        if "keff" in samples and len(samples):
            risk[i] = np.sum(samples["keff"] >= usl) / sample_size
        pools.append(samples)
        if verbose:
            print(f"risk pool {i + 1}/{risk_pool}: {risk[i]:.3e}")

    pd.DataFrame({"risk": risk}).to_csv(risk_dir / "risk.csv", index=False)
    bn_dist = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    bn_dist.to_pickle(risk_dir / "bn-risk.pkl")

    if risk.mean() != 0:
        print(f"Risk = {risk.mean():.3e}")
        print("SD = NA" if risk_pool == 1 else f"SD = {risk.std(ddof=1):.3e}")
    else:
        print(f"Risk < {risk_pool * sample_size:.0e}")

    return risk, bn_dist
