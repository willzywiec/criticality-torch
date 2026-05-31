"""scale.py

Center, scale, and one-hot encode model inputs.

Python/torch port of ``R/scale.R``. The original used ``caret::dummyVars``
for one-hot encoding and ``scale()`` for standardization; here we use pandas
for encoding and a stored mean/standard-deviation vector for standardization
so the exact same transform can be re-applied to new data at prediction time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Continuous variables that get centered and scaled. The R code notes that
# 'ht' and 'hd' are intentionally omitted from the current facility model.
SCALE_LABELS = ["mass", "rad", "thk", "vol", "conc"]

# Columns carried alongside the features but never fed to the network.
TARGET_COLS = ["keff", "sd"]


@dataclass
class Dataset:
    """Container mirroring the named list returned by the R ``Scale`` function."""

    output: pd.DataFrame
    training_data: pd.DataFrame
    training_mean: pd.Series
    training_sd: pd.Series
    training_df: pd.DataFrame
    test_data: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None
    # Ordered feature columns; new data is reindexed onto these so the encoding
    # is stable across train/test/prediction.
    feature_cols: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "Dataset":
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _nullify(df: pd.DataFrame) -> pd.DataFrame:
    """Drop single-factor columns (the R ``Nullify`` helper)."""
    return df.loc[:, df.nunique(dropna=False) > 1]


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns in place, preserving column order.

    Replicates ``caret::dummyVars(sep = '')`` which keeps every factor level
    (no reference level is dropped) and names columns ``<var><level>``.
    """
    pieces = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            pieces.append(df[[col]].astype(float))
        else:
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="").astype(float)
            pieces.append(dummies)
    return pd.concat(pieces, axis=1)


def scale_data(
    output: pd.DataFrame,
    ext_dir: str | Path,
    code: str = "mcnp",
    dataset: Dataset | None = None,
    *,
    save: bool = True,
    seed: int | None = None,
) -> Dataset | pd.DataFrame:
    """Center, scale, and one-hot encode variables.

    Args:
        output: Processed Monte Carlo output. Expected columns include
            ``mass, form, mod, rad, ref, thk, vol, conc`` and, for training
            data, ``keff`` and ``sd``.
        ext_dir: External directory used for caching the dataset.
        code: Monte Carlo code label, used in the cache file name.
        dataset: If provided, ``output`` is transformed using the encoding and
            mean/sd already stored in this dataset (prediction path). Otherwise
            a new train/test split is created (training path).
        save: Whether to persist the dataset cache to ``ext_dir``.
        seed: Optional RNG seed for the train/test split.

    Returns:
        A :class:`Dataset` on the training path, or a feature ``DataFrame``
        when ``dataset`` is supplied (transform-only path).
    """
    code = code.lower()
    output = output.copy()

    # The R code drops 'shape' before encoding (single, dataset-wide constant).
    output = output.drop(columns=[c for c in ["shape"] if c in output.columns])

    # --- Transform-only path: reuse an existing dataset's encoding ---------
    if dataset is not None:
        feature_frame = _one_hot(output)
        # Align to the trained feature columns (missing dummies -> 0).
        feature_frame = feature_frame.reindex(
            columns=dataset.feature_cols, fill_value=0.0
        )
        for label in SCALE_LABELS:
            if label in feature_frame.columns:
                feature_frame[label] = (
                    feature_frame[label] - dataset.training_mean[label]
                ) / dataset.training_sd[label]
        return feature_frame

    # --- Training path: build a fresh train/test split --------------------
    encoded = _one_hot(_nullify(output))

    # Keep only well-converged simulations (low Monte Carlo uncertainty).
    if "sd" in encoded.columns:
        encoded = encoded[encoded["sd"] < 0.001].reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Test set: oversized-mass cases, 20% of the data sampled at random.
    candidate = encoded[encoded["mass"] > 200]
    n_test = round(len(encoded) * 0.2)
    n_test = min(n_test, len(candidate))
    test_idx = rng.choice(candidate.index.to_numpy(), size=n_test, replace=False)
    test_data = encoded.loc[test_idx].reset_index(drop=True)
    training_data = encoded.drop(index=test_idx).reset_index(drop=True)

    # Standardization statistics come from the training partition only.
    training_mean = training_data[SCALE_LABELS].mean()
    training_sd = training_data[SCALE_LABELS].std(ddof=1)

    feature_cols = [c for c in training_data.columns if c not in TARGET_COLS]
    training_df = training_data[feature_cols].copy()
    test_df = test_data[feature_cols].copy()

    for label in SCALE_LABELS:
        training_df[label] = (training_df[label] - training_mean[label]) / training_sd[label]
        test_df[label] = (test_df[label] - training_mean[label]) / training_sd[label]

    result = Dataset(
        output=output,
        training_data=training_data,
        training_mean=training_mean,
        training_sd=training_sd,
        training_df=training_df,
        test_data=test_data,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    if save:
        ext_dir = Path(ext_dir)
        ext_dir.mkdir(parents=True, exist_ok=True)
        result.save(ext_dir / f"{code}-dataset.pkl")

    return result
