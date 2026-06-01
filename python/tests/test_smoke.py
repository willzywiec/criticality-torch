"""End-to-end smoke test of the torch criticality pipeline on synthetic data.

This exercises every module (scale -> model -> nn/train -> ensemble ->
bn -> predict -> risk) without needing OpenMC, which generates the training
data but is not pip-installable. The OpenMC data-generation wiring is tested
separately with the simulation step mocked.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from criticality import (
    SimSettings,
    build_bn,
    build_model,
    estimate_risk,
    generate_dataset,
    predict_keff,
    sample_configs,
    scale_data,
    simulate_output,
    tabulate,
    train_nn,
)
import criticality as _cr
import sys
import criticality.tabulate  # noqa: F401  (ensure submodule is imported)

tabulate_mod = sys.modules["criticality.tabulate"]


@pytest.fixture
def synthetic_output(tmp_path):
    """A small MCNP-like output table with a learnable keff signal."""
    rng = np.random.default_rng(0)
    n = 400
    mass = rng.uniform(50, 400, n)
    rad = rng.uniform(5, 20, n)
    form = rng.choice(["alpha", "heu"], n)
    mod = rng.choice(["h2o", "none"], n)
    ref = rng.choice(["h2o", "none"], n)
    thk = rng.uniform(0, 10, n)
    vol = 4 / 3 * np.pi * rad**3
    conc = mass / vol
    # keff increases with mass/conc -- something the net can fit.
    keff = 0.5 + 0.0015 * mass + 2.0 * conc + rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "mass": mass,
            "form": form,
            "mod": mod,
            "rad": rad,
            "ref": ref,
            "thk": thk,
            "vol": vol,
            "conc": conc,
            "keff": keff,
            "sd": rng.uniform(0, 0.0009, n),  # all below the 0.001 filter
        }
    )


def test_scale_split(synthetic_output, tmp_path):
    ds = scale_data(synthetic_output, ext_dir=tmp_path, seed=1)
    assert ds.training_df.shape[1] == ds.test_df.shape[1]
    assert "keff" not in ds.feature_cols and "sd" not in ds.feature_cols
    assert len(ds.training_data) > len(ds.test_data) > 0
    # Cache written.
    assert (tmp_path / "openmc-dataset.pkl").exists()


def test_build_model_shapes():
    model, loss_fn, opt = build_model(input_dim=8, layers="16-8", loss="sse")
    import torch

    x = torch.randn(5, 8)
    assert model(x).shape == (5,)
    assert loss_fn(model(x), torch.zeros(5)).ndim == 0


def test_train_and_predict(synthetic_output, tmp_path):
    ds = scale_data(synthetic_output, ext_dir=tmp_path, seed=2)
    meta = train_nn(
        ds,
        batch_size=64,
        ensemble_size=2,
        epochs=40,
        layers="32-16",
        loss="sse",
        learning_rate=0.005,
        val_split=0.2,
        ext_dir=tmp_path,
    )
    assert len(meta["models"]) == 2
    assert meta["weights"].shape == (2,)
    # Plots and prediction CSVs were written.
    assert (tmp_path / "training" / "model" / "1.png").exists()
    assert (tmp_path / "training" / "test-data.csv").exists()

    # Model should track the target with low error on training data.
    import torch

    x = torch.tensor(ds.training_df.to_numpy(dtype=np.float32))
    with torch.no_grad():
        pred = np.column_stack([m(x).numpy() for m in meta["models"]]) @ meta["weights"]
    mae = np.mean(np.abs(pred - ds.training_data["keff"].to_numpy()))
    assert mae < 0.5  # loosely fit, but clearly learning


def _write_facility_csv(path):
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame(
        {
            "op": rng.choice(["op1", "op2"], n),
            "ctrl": rng.choice(["A", "B"], n),
            "mass": rng.uniform(50, 400, n),
            "form": rng.choice(["alpha", "heu"], n),
            "mod": rng.choice(["h2o", "none"], n),
            "rad": rng.uniform(5, 20, n),
            "ref": rng.choice(["h2o", "none"], n),
            "thk": rng.uniform(0, 10, n),
        }
    )
    df.to_csv(path / "facility.csv", index=False)


def test_bn_and_risk(synthetic_output, tmp_path):
    ds = scale_data(synthetic_output, ext_dir=tmp_path, seed=4)
    meta = train_nn(
        ds,
        batch_size=64,
        ensemble_size=2,
        epochs=20,
        layers="16-8",
        learning_rate=0.005,
        ext_dir=tmp_path,
    )
    _write_facility_csv(tmp_path)
    bn = build_bn("facility.csv", ext_dir=tmp_path, dist="gamma")
    samples = bn.sample(2000, np.random.default_rng(5))
    assert set(samples.columns) == {"op", "ctrl", "mass", "form", "mod", "rad", "ref", "thk"}

    preds = predict_keff(
        bn, meta, ds,
        keff_cutoff=0.0, mass_cutoff=60, rad_cutoff=6,
        sample_size=2000, ext_dir=tmp_path, rng=np.random.default_rng(6),
        verbose=False,
    )
    assert "keff" in preds.columns

    risk, bn_dist = estimate_risk(
        bn, meta, ds,
        keff_cutoff=0.0, mass_cutoff=60, rad_cutoff=6,
        risk_pool=2, sample_size=1500, usl=0.9,
        ext_dir=tmp_path, rng=np.random.default_rng(7), verbose=False,
    )
    assert risk.shape == (2,)


# --- OpenMC data-generation layer (simulation mocked) ----------------------

def test_sample_configs():
    cfg = sample_configs(50, seed=0)
    assert len(cfg) == 50
    assert set(cfg.columns) == {"mass", "form", "mod", "rad", "ref", "thk"}
    assert cfg["form"].isin(["alpha", "delta", "puo2", "heu", "uo2"]).all()


def _fake_simulate_dataset(configs, settings=None, *, max_workers=None,
                           output_csv=None, verbose=True):
    """Cheap synthetic stand-in for the OpenMC batch (no cross sections needed)."""
    out = configs.copy()
    vol = 4 / 3 * np.pi * out["rad"] ** 3
    out["vol"] = vol
    out["conc"] = out["mass"] / vol
    out["keff"] = 0.5 + 0.0015 * out["mass"] + 2.0 * out["conc"]
    out["sd"] = 0.0005
    if output_csv is not None:
        out.to_csv(output_csv, index=False)
    return out


def test_generate_dataset_mocked(tmp_path, monkeypatch):
    """generate_dataset() should wire sampling -> simulation -> scaling."""
    monkeypatch.setattr(tabulate_mod, "simulate_dataset", _fake_simulate_dataset)

    ds = generate_dataset(tmp_path, n=300, settings=SimSettings(particles=10), seed=1)
    assert (tmp_path / "openmc.csv").exists()
    assert (tmp_path / "openmc-dataset.pkl").exists()
    assert ds.training_df.shape[1] == ds.test_df.shape[1]
    assert len(ds.training_data) > 0


def test_simulate_output_then_tabulate(tmp_path, monkeypatch):
    """Generation and training stages are separable (generate-only path)."""
    monkeypatch.setattr(tabulate_mod, "simulate_dataset", _fake_simulate_dataset)

    # Stage 1: generation only -- writes the raw CSV, no scaled dataset yet.
    out_path = simulate_output(tmp_path, n=300, seed=1)
    assert out_path == tmp_path / "openmc.csv"
    assert out_path.exists()
    assert not (tmp_path / "openmc-dataset.pkl").exists()

    # Stage 2: the training pipeline builds the dataset from that output.
    ds = tabulate(code="openmc", ext_dir=tmp_path, seed=1)
    assert (tmp_path / "openmc-dataset.pkl").exists()
    assert len(ds.training_data) > 0


def test_simulate_output_invalidates_stale_cache(tmp_path, monkeypatch):
    """A fresh generation run drops a stale dataset cache so training rebuilds."""
    monkeypatch.setattr(tabulate_mod, "simulate_dataset", _fake_simulate_dataset)
    cache = tmp_path / "openmc-dataset.pkl"
    cache.write_bytes(b"stale")
    simulate_output(tmp_path, n=200, seed=2)
    assert not cache.exists()


def test_simulate_dataset_parallel(monkeypatch):
    """simulate_dataset runs configs through the pool and preserves input order."""
    # Patch the per-run keff so no OpenMC/cross sections are needed. The worker
    # imports run_keff from the module, so patch it there; fork carries it.
    monkeypatch.setattr(
        _cr.simulate, "run_keff",
        lambda mass, *a, **k: (0.5 + 0.001 * mass, 0.0001),
    )
    cfg = sample_configs(12, seed=7)
    out = _cr.simulate.simulate_dataset(cfg, max_workers=4, verbose=False)
    assert len(out) == len(cfg)
    assert list(out.columns) == _cr.simulate.OUTPUT_COLUMNS
    # Order matches the input configs (results are re-sorted by index).
    assert list(out["mass"]) == [float(m) for m in cfg["mass"]]
    expected = [0.5 + 0.001 * float(m) for m in cfg["mass"]]
    assert np.allclose(out["keff"].to_numpy(), expected)


def test_simulate_dataset_records_failures(monkeypatch):
    """A failing run is recorded with NaN keff instead of aborting the batch."""
    def flaky(mass, *a, **k):
        if mass > 1000:
            raise RuntimeError("boom")
        return 0.8, 0.0001

    monkeypatch.setattr(_cr.simulate, "run_keff", flaky)
    cfg = pd.DataFrame({
        "mass": [100.0, 2000.0, 300.0], "form": ["heu"] * 3,
        "mod": ["none"] * 3, "rad": [8.0] * 3,
        "ref": ["none"] * 3, "thk": [0.0] * 3,
    })
    # max_workers=1 keeps the raising call inline so the monkeypatch applies
    # without relying on fork semantics.
    out = _cr.simulate.simulate_dataset(cfg, max_workers=1, verbose=False)
    assert np.isnan(out["keff"].iloc[1])
    assert not np.isnan(out["keff"].iloc[0])
    assert not np.isnan(out["keff"].iloc[2])


def test_run_keff_requires_openmc():
    """Without OpenMC installed, simulation calls raise a clear error."""
    pytest.importorskip  # noqa
    from criticality import run_keff

    try:
        import openmc  # noqa: F401
        pytest.skip("OpenMC is installed; skipping the missing-dependency check")
    except ImportError:
        with pytest.raises(ImportError, match="OpenMC is required"):
            run_keff(100, "heu", "none", 8.0, "none", 0.0)
