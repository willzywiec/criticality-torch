"""tabulate.py

Load or build the training/test dataset.

Python port of ``R/tabulate.R``. If a cached dataset pickle exists it is
loaded; otherwise the raw Monte Carlo output CSV is read, derived columns
(volume, concentration) are computed, and the data is scaled/encoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .scale import Dataset, scale_data


def tabulate(code: str = "mcnp", ext_dir: str | Path = ".", *, seed: int | None = None) -> Dataset:
    """Load training and test data, building it from CSV if no cache exists.

    Args:
        code: Monte Carlo code label (used to locate ``<code>-dataset.pkl`` and
            the raw ``<code>*.csv`` output file).
        ext_dir: External directory containing the data.
        seed: Optional RNG seed for the train/test split.

    Returns:
        A :class:`~criticality.scale.Dataset`.
    """
    code = code.lower()
    ext_dir = Path(ext_dir)

    cache = ext_dir / f"{code}-dataset.pkl"
    if cache.exists():
        return Dataset.load(cache)

    # Prefer the exact raw-output file, then fall back to any <code>*.csv.
    exact = ext_dir / f"{code}.csv"
    if exact.exists():
        csv_path = exact
    else:
        matches = sorted(ext_dir.glob(f"*{code}*.csv")) + sorted(ext_dir.glob(f"*{code.upper()}*.csv"))
        csv_files = list(dict.fromkeys(matches))
        if not csv_files:
            raise FileNotFoundError(
                f"could not find data: no {code}-dataset.pkl or {code}*.csv in {ext_dir}"
            )
        csv_path = csv_files[0]

    output = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()
    output = output.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Derived geometry columns.
    vol = 4 / 3 * np.pi * output["rad"] ** 3
    conc = output["mass"] / vol

    cols = ["mass", "form", "mod", "rad", "ref", "thk", "shape"]
    processed = pd.DataFrame({c: output[c] for c in cols if c in output.columns})
    processed["vol"] = vol
    processed["conc"] = conc
    processed["keff"] = output["keff"]
    processed["sd"] = output["sd"]

    return scale_data(processed, ext_dir=ext_dir, code=code, seed=seed)
