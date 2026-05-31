"""ensemble.py

Compute deep-neural-network metamodel weights and generate keff predictions
for all training and test data.

Python/torch port of ``R/test.R`` (the ``Test`` function). Renamed to
``test_ensemble`` so it is not mistaken for a pytest module.

The R code searched for the best ensemble weights at every ensemble size using
three optimizers (Nelder-Mead, BFGS, and simulated annealing) plus a simple
average, then kept whichever gave the lowest test MAE. We keep that behavior
but vectorize the predictions and use ``scipy.optimize``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import dual_annealing, minimize

from .scale import Dataset


def _predict(model, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def test_ensemble(
    dataset: Dataset,
    models: list,
    *,
    ext_dir: str | Path,
    training_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Find ensemble weights minimizing test-set MAE.

    Returns a weight vector of length ``len(models)`` (unused members get a
    weight of 0). The dot product of model predictions with this vector is the
    ensemble keff prediction.
    """
    device = torch.device(device)
    ext_dir = Path(ext_dir)
    training_dir = Path(training_dir) if training_dir else ext_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    n = len(models)
    keff_test = dataset.test_data["keff"].to_numpy()
    x_test = torch.tensor(dataset.test_df.to_numpy(dtype=np.float32), device=device)

    # Predictions for every model, shape (n_samples, n_models).
    test_pred = np.column_stack([_predict(m, x_test) for m in models])
    test_mae = [np.mean(np.abs(keff_test - test_pred[:, i])) for i in range(n)]

    def objective(weights: np.ndarray, k: int) -> float:
        return float(np.mean(np.abs(keff_test - test_pred[:, :k] @ weights)))

    avg, nm, bfgs, sa = [], [], [], []
    nm_wt, bfgs_wt, sa_wt = [], [], []

    for k in range(1, n + 1):
        x0 = np.full(k, 1.0 / k)
        avg.append(float(np.mean(np.abs(keff_test - test_pred[:, :k].mean(axis=1)))))

        r_nm = minimize(objective, x0, args=(k,), method="Nelder-Mead")
        r_bfgs = minimize(objective, x0, args=(k,), method="BFGS")
        r_sa = dual_annealing(
            objective, bounds=[(-2.0, 2.0)] * k, args=(k,), x0=x0, maxiter=200
        )

        nm.append(float(r_nm.fun))
        bfgs.append(float(r_bfgs.fun))
        sa.append(float(r_sa.fun))
        nm_wt.append(r_nm.x)
        bfgs_wt.append(r_bfgs.x)
        sa_wt.append(r_sa.x)

    pd.DataFrame({"avg": avg, "nm": nm, "bfgs": bfgs, "sa": sa}).to_csv(
        training_dir / "test-mae.csv", index=False
    )

    print(f"Mean Test MAE = {np.mean(test_mae):.6f}")
    print(f"Ensemble Test MAE (avg) = {avg[-1]:.6f}")

    # Pick the single best configuration over all ensemble sizes and methods.
    methods = {"avg": avg, "nm": nm, "bfgs": bfgs, "sa": sa}
    best_method, best_k, best_val = "avg", n, avg[-1]
    for name, vals in methods.items():
        k = int(np.argmin(vals)) + 1
        if vals[k - 1] < best_val:
            best_method, best_k, best_val = name, k, vals[k - 1]

    weights = np.zeros(n)
    if best_method == "avg":
        weights[:best_k] = 1.0 / best_k
    elif best_method == "nm":
        weights[:best_k] = nm_wt[best_k - 1]
    elif best_method == "bfgs":
        weights[:best_k] = bfgs_wt[best_k - 1]
    else:
        weights[:best_k] = sa_wt[best_k - 1]

    print(f"Selected: {best_method} with {best_k} model(s), test MAE = {best_val:.6f}")

    # Write out predictions for the training and test partitions.
    x_train = torch.tensor(
        dataset.training_df.to_numpy(dtype=np.float32), device=device
    )
    train_pred = np.column_stack([_predict(m, x_train) for m in models])

    training_data = dataset.training_data.copy()
    test_data = dataset.test_data.copy()
    training_data["keff_pred"] = train_pred @ weights
    test_data["keff_pred"] = test_pred @ weights
    training_data.to_csv(training_dir / "training-data.csv", index=False)
    test_data.to_csv(training_dir / "test-data.csv", index=False)
    np.save(training_dir / "weights.npy", weights)

    return weights
