"""plot.py

Generate and save training-history plots.

Python/matplotlib port of ``R/plot.R`` (originally ggplot2). Plots training and
cross-validation MAE on a log scale and marks the training minimum.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for batch/CI use
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(
    i: int,
    history: pd.DataFrame | None = None,
    *,
    plot_dir: str | Path,
) -> None:
    """Plot and save the MAE history for ensemble member ``i``.

    Args:
        i: Model number (used for the file name).
        history: Training-history data frame with ``epoch``, ``mae``, and
            ``val.mae`` columns. If ``None``, it is read from ``<plot_dir>/<i>.csv``.
        plot_dir: Directory to read/write history files and the ``.png``.
    """
    plot_dir = Path(plot_dir)
    if history is None:
        history = pd.read_csv(plot_dir / f"{i}.csv")
    else:
        history.to_csv(plot_dir / f"{i}.csv", index=False)

    min_idx = int(history["mae"].idxmin())
    min_epoch = history["epoch"].iloc[min_idx]
    min_mae = history["mae"].iloc[min_idx]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(history["epoch"], history["val.mae"], color="#a9a9a9", label="cross-validation data")
    ax.plot(history["epoch"], history["mae"], color="black", label="training data")
    ax.scatter([min_epoch], [min_mae], color="red", zorder=5, label="training minimum")
    ax.annotate(
        f"{min_mae:.3e}",
        (min_epoch, min_mae),
        textcoords="offset points",
        xytext=(0, -14),
        color="red",
        ha="center",
        fontsize=9,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean absolute error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(plot_dir / f"{i}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
