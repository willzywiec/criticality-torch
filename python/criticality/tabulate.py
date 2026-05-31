"""tabulate.py

Load (or generate with OpenMC) the training/test dataset.

Replaces the R ``Tabulate`` function, which loaded a precomputed MCNP
``.RData``/CSV. Here the data source is OpenMC: if a cached dataset or an
``openmc.csv`` output table exists it is loaded, otherwise it can be generated
on the fly by running OpenMC over sampled configurations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .scale import Dataset, scale_data
from .simulate import SimSettings, sample_configs, simulate_dataset


def _derive_geometry(output: pd.DataFrame) -> pd.DataFrame:
    """Add volume/concentration columns and keep the expected schema."""
    vol = 4 / 3 * np.pi * output["rad"] ** 3
    cols = ["mass", "form", "mod", "rad", "ref", "thk", "shape"]
    processed = pd.DataFrame({c: output[c] for c in cols if c in output.columns})
    processed["vol"] = vol
    processed["conc"] = output["mass"] / vol
    processed["keff"] = output["keff"]
    processed["sd"] = output["sd"]
    return processed


def tabulate(
    code: str = "openmc",
    ext_dir: str | Path = ".",
    *,
    seed: int | None = None,
) -> Dataset:
    """Load training/test data, reading the cached dataset or ``<code>.csv``.

    Args:
        code: Monte Carlo code label (OpenMC). Used to locate
            ``<code>-dataset.pkl`` and the raw ``<code>.csv`` output file.
        ext_dir: External directory containing the data.
        seed: Optional RNG seed for the train/test split.

    Returns:
        A :class:`~criticality.scale.Dataset`.

    Raises:
        FileNotFoundError: If no cached dataset or output CSV is found. Use
            :func:`generate_dataset` to produce one with OpenMC.
    """
    code = code.lower()
    ext_dir = Path(ext_dir)

    cache = ext_dir / f"{code}-dataset.pkl"
    if cache.exists():
        return Dataset.load(cache)

    csv_path = ext_dir / f"{code}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"no {code}-dataset.pkl or {code}.csv in {ext_dir}; "
            f"run criticality.generate_dataset(...) to simulate data with OpenMC"
        )

    output = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()
    output = output.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return scale_data(_derive_geometry(output), ext_dir=ext_dir, code=code, seed=seed)


def generate_dataset(
    ext_dir: str | Path,
    *,
    code: str = "openmc",
    n: int | None = None,
    configs: pd.DataFrame | None = None,
    settings: SimSettings | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> Dataset:
    """Generate a dataset by running OpenMC, cache it, and return it.

    Provide either ``n`` (number of random configurations to sample) or an
    explicit ``configs`` DataFrame with columns ``mass, form, mod, rad, ref,
    thk``. The raw OpenMC output is written to ``<code>.csv`` and the scaled
    dataset to ``<code>-dataset.pkl``.
    """
    ext_dir = Path(ext_dir)
    ext_dir.mkdir(parents=True, exist_ok=True)
    code = code.lower()

    if configs is None:
        if n is None:
            raise ValueError("provide either 'n' or an explicit 'configs' frame")
        configs = sample_configs(n, seed=seed)

    output = simulate_dataset(configs, settings, verbose=verbose)
    output.to_csv(ext_dir / f"{code}.csv", index=False)
    return scale_data(_derive_geometry(output), ext_dir=ext_dir, code=code, seed=seed)
