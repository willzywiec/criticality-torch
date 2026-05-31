# criticality (Python / torch)

A Python + [PyTorch](https://pytorch.org/) port of the R
[`criticality`](https://github.com/willzywiec/criticality) package — a
collection of functions for modeling fissile material operations in nuclear
facilities, based on Zywiec et al. (2021)
[doi:10.1016/j.ress.2020.107322](https://doi.org/10.1016/j.ress.2020.107322).

The original used Keras/TensorFlow (via `reticulate`) for the deep-neural-network
metamodel and `bnlearn` for the Bayesian network. This port replaces Keras with
torch and implements the (shallow, fixed-structure) Bayesian network directly,
so the only runtime dependencies are torch / numpy / pandas / scipy / matplotlib.

## Install

```bash
cd python
pip install -e .          # or: pip install -e ".[dev]" for tests
```

## Pipeline at a glance

```python
import criticality as cx

# 1. Load training/test data (rebuilds split + scaling from <code>.csv,
#    or loads a cached <code>-dataset.pkl).
dataset = cx.tabulate(code="mcnp", ext_dir="extdata")

# 2. Train an ensemble of DNN metamodels and compute ensemble weights.
metamodel = cx.train_nn(
    dataset,
    ensemble_size=5,
    epochs=1500,
    layers="8192-256-256-256-256-16",
    loss="sse",
    opt_alg="adamax",
    learning_rate=0.00075,
    ext_dir="extdata",
    device="cuda",          # or "cpu"
)

# 3. Build the Bayesian network from facility walkthrough data.
bn = cx.build_bn("facility.csv", ext_dir="extdata", dist="gamma")

# 4. Estimate process criticality accident risk.
risk, samples = cx.estimate_risk(
    bn, metamodel, dataset,
    keff_cutoff=0.9, mass_cutoff=100, rad_cutoff=7,
    risk_pool=100, sample_size=1_000_000, usl=0.95,
    ext_dir="extdata",
)
```

## R → Python mapping

| R function (`pkg/criticality/R`) | Python (`criticality`)        |
| -------------------------------- | ----------------------------- |
| `BN()`        (`bn.R`)           | `bn.build_bn()`               |
| `Tabulate()`  (`tabulate.R`)     | `tabulate.tabulate()`         |
| `Scale()`     (`scale.R`)        | `scale.scale_data()`          |
| `Model()`     (`model.R`)        | `model.build_model()`         |
| `NN()`        (`nn.R`)           | `nn.train_nn()`               |
| `Test()`      (`test.R`)         | `ensemble.test_ensemble()`    |
| `Plot()`      (`plot.R`)         | `plot.plot_history()`         |
| `Predict()`   (`predict.R`)      | `predict.predict_keff()`      |
| `Risk()`      (`risk.R`)         | `risk.estimate_risk()`        |

## Getting the data

`facility.csv` (the Bayesian-network input) ports directly. The MCNP training
data ships only as an R binary (`mcnp-dataset.RData`). With R available, export
it to CSV once:

```bash
Rscript convert_rdata.R /path/to/extdata mcnp
```

This writes `mcnp.csv` (raw output), which `cx.tabulate()` consumes to rebuild
the train/test split and scaling.

## Changes vs. the R original (optimizations / fixes)

- **`model.R`** — the nine repeated `if (length(layers) >= n)` blocks that added
  hidden layers are collapsed into a single loop, so any network depth works.
- **`nn.R`** — the R code trained each model, *retrained* it for `epochs/10`
  while writing one `.h5` checkpoint per epoch, then re-read every checkpoint to
  find the lowest-MAE epoch. That is just "keep the best epoch's weights", so the
  port uses one training loop that records full history and snapshots the best
  weights in memory (by train MAE + val MAE).
- **`predict.R`** — the R code called `Scale()` without passing the trained
  dataset; via R's `exists('dataset')` check this silently re-fit the scaling
  statistics on the prediction data instead of reusing the training mean/sd. The
  port passes the trained `dataset` explicitly so new data is scaled correctly.
- **`bn.R`** — `bnlearn` + `fitdistrplus` + `evd` are replaced with direct
  ancestral sampling (exact for this DAG: `op` → root, `ctrl | op`, all other
  nodes `| op, ctrl`) and `scipy.stats` distribution fits. Sampling is
  vectorized over the ≤42 `(op, ctrl)` cells.
- Predictions in `ensemble`/`predict` are vectorized into a single
  `(n_samples, n_models)` matrix multiply against the ensemble weights.

## Tests

```bash
cd python && pytest -q
```

The smoke test exercises every module end-to-end on synthetic data (no MCNP
dataset required).
```
