"""bn.py

Build a Bayesian network from pre-formatted nuclear-facility data and sample
from it.

Python port of ``R/bn.R`` (originally ``bnlearn`` + ``fitdistrplus`` + ``evd``).

The network's DAG is fixed and shallow: ``op`` is the root, ``ctrl`` depends on
``op``, and every remaining node (``mass, form, mod, rad, ref, thk``) depends on
both ``op`` and ``ctrl``. That structure means a full bnlearn dependency isn't
needed -- ancestral (forward) sampling is exact and is implemented directly
here with numpy, while continuous parameters are fit with ``scipy.stats``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Discrete support grids (centimeters for rad/thk; grams for mass).
MASS_GRID = np.arange(0, 4001, 1, dtype=float)
RAD_GRID = np.arange(0, 18.0 + 1e-9, 0.25) * 2.54
THK_GRID = np.arange(0, 11.0 + 1e-9, 0.25) * 2.54

CATEGORICAL = ["form", "mod", "ref"]
CONTINUOUS = {"mass": MASS_GRID, "rad": RAD_GRID, "thk": THK_GRID}


def _fit_continuous(values: np.ndarray, grid: np.ndarray, dist: str) -> np.ndarray:
    """Fit a truncated distribution and return a normalized pmf over ``grid``."""
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        pmf = np.zeros_like(grid)
        pmf[0] = 1.0
        return pmf

    # Avoid a degenerate fit when every observation is identical.
    if np.unique(values).size == 1:
        values = values.copy()
        values[-1] = np.ceil(values[-1]) if values[-1] == values[0] else values[-1]

    if dist == "gamma":
        a, loc, scale = stats.gamma.fit(values, floc=0)
        density = stats.gamma.pdf(grid, a, loc=loc, scale=scale)
    elif dist == "normal":
        loc, scale = stats.norm.fit(values)
        density = stats.norm.pdf(grid, loc=loc, scale=scale)
    elif dist == "log-normal":
        s, loc, scale = stats.lognorm.fit(values, floc=0)
        density = stats.lognorm.pdf(grid, s, loc=loc, scale=scale)
    elif dist == "weibull":
        c, loc, scale = stats.weibull_min.fit(values, floc=0)
        density = stats.weibull_min.pdf(grid, c, loc=loc, scale=scale)
    elif dist == "gev":
        c, loc, scale = stats.genextreme.fit(values)
        density = stats.genextreme.pdf(grid, c, loc=loc, scale=scale)
    else:
        raise ValueError(f"unknown distribution '{dist}'")

    total = density.sum()
    if total <= 0 or not np.isfinite(total):
        pmf = np.zeros_like(grid)
        pmf[0] = 1.0
        return pmf
    return density / total


def _fit_categorical(values: pd.Series, levels: list[str]) -> np.ndarray:
    """Empirical frequencies over ``levels``; defaults to the first level."""
    if len(values) == 0:
        pmf = np.zeros(len(levels))
        pmf[0] = 1.0
        return pmf
    counts = values.value_counts()
    pmf = np.array([counts.get(lvl, 0) for lvl in levels], dtype=float)
    return pmf / pmf.sum()


@dataclass
class BayesNet:
    """Fitted Bayesian network with conditional probability tables."""

    levels: dict[str, list]          # categorical level sets (op, ctrl, form, ...)
    grids: dict[str, np.ndarray]     # continuous support grids
    op_cpt: np.ndarray               # P(op), shape (n_op,)
    ctrl_cpt: np.ndarray             # P(ctrl | op), shape (n_op, n_ctrl)
    child_cpt: dict[str, np.ndarray] # node -> P(value | op, ctrl), (n_op, n_ctrl, n_val)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Forward-sample ``n`` rows from the network."""
        rng = rng or np.random.default_rng()
        op_levels = self.levels["op"]
        ctrl_levels = self.levels["ctrl"]
        n_op, n_ctrl = len(op_levels), len(ctrl_levels)

        op_idx = rng.choice(n_op, size=n, p=self.op_cpt)
        ctrl_idx = np.empty(n, dtype=int)
        for o in range(n_op):
            mask = op_idx == o
            if mask.any():
                ctrl_idx[mask] = rng.choice(n_ctrl, size=mask.sum(), p=self.ctrl_cpt[o])

        columns: dict[str, np.ndarray] = {
            "op": np.array(op_levels, dtype=object)[op_idx],
            "ctrl": np.array(ctrl_levels, dtype=object)[ctrl_idx],
        }

        # Each child depends only on (op, ctrl); sample cell by cell.
        for node, cpt in self.child_cpt.items():
            if node in CONTINUOUS:
                support = self.grids[node]
            else:
                support = np.array(self.levels[node], dtype=object)
            out = np.empty(n, dtype=support.dtype)
            for o in range(n_op):
                for c in range(n_ctrl):
                    mask = (op_idx == o) & (ctrl_idx == c)
                    m = int(mask.sum())
                    if m:
                        choice = rng.choice(len(support), size=m, p=cpt[o, c])
                        out[mask] = support[choice]
            columns[node] = out

        return pd.DataFrame(columns)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "BayesNet":
        with open(path, "rb") as handle:
            return pickle.load(handle)


def build_bn(
    facility_data: str,
    ext_dir: str | Path,
    dist: str = "gamma",
    *,
    save: bool = True,
) -> BayesNet:
    """Create a Bayesian network from a facility ``.csv`` file.

    Args:
        facility_data: CSV file name (within ``ext_dir``) with columns
            ``op, ctrl, mass, form, mod, rad, ref, thk``.
        ext_dir: External directory holding the CSV and receiving the output.
        dist: Distribution for continuous parameters
            (``gamma``, ``gev``, ``normal``, ``log-normal``, ``weibull``).
        save: Whether to pickle the fitted network to ``ext_dir``.
    """
    ext_dir = Path(ext_dir)
    facility = facility_data.replace(".csv", "")
    df = pd.read_csv(ext_dir / facility_data)

    # Categorical level sets, alphabetically sorted (matching R's table()).
    levels = {
        "op": sorted(df["op"].unique()),
        "ctrl": sorted(df["ctrl"].unique()),
        "form": sorted(df["form"].unique()),
        "mod": sorted(df["mod"].unique()),
        "ref": sorted(df["ref"].unique()),
    }
    grids = dict(CONTINUOUS)

    n_op, n_ctrl = len(levels["op"]), len(levels["ctrl"])

    # P(op) and P(ctrl | op).
    op_cpt = np.array([np.mean(df["op"] == o) for o in levels["op"]])
    ctrl_cpt = np.zeros((n_op, n_ctrl))
    for i, o in enumerate(levels["op"]):
        sub = df[df["op"] == o]
        for j, c in enumerate(levels["ctrl"]):
            ctrl_cpt[i, j] = np.mean(sub["ctrl"] == c) if len(sub) else 0.0

    # P(child | op, ctrl) for every remaining node.
    child_cpt: dict[str, np.ndarray] = {}
    for node in ["mass", "form", "mod", "rad", "ref", "thk"]:
        if node in CONTINUOUS:
            grid = grids[node]
            cpt = np.zeros((n_op, n_ctrl, len(grid)))
        else:
            cpt = np.zeros((n_op, n_ctrl, len(levels[node])))
        for i, o in enumerate(levels["op"]):
            for j, c in enumerate(levels["ctrl"]):
                cell = df[(df["op"] == o) & (df["ctrl"] == c)]
                if node in CONTINUOUS:
                    cpt[i, j] = _fit_continuous(cell[node].to_numpy(), grids[node], dist)
                else:
                    cpt[i, j] = _fit_categorical(cell[node], levels[node])
        child_cpt[node] = cpt

    bn = BayesNet(levels=levels, grids=grids, op_cpt=op_cpt,
                  ctrl_cpt=ctrl_cpt, child_cpt=child_cpt)

    if save:
        bn.save(ext_dir / f"{facility}-{dist}.pkl")
    return bn
