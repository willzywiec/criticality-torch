"""Command-line entry point: run the full criticality pipeline end to end.

    python -m criticality --ext-dir extdata [options]

Stitches the library modules together in the same order as the R workflow:
    tabulate -> train_nn -> build_bn -> estimate_risk

Use --skip-risk to stop after training the metamodel (e.g. when no facility
walkthrough data is available).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import (
    SimSettings,
    build_bn,
    estimate_risk,
    generate_dataset,
    tabulate,
    train_nn,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m criticality",
        description="Train a DNN metamodel and estimate criticality accident risk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data / environment")
    g.add_argument("--ext-dir", required=True, help="External data directory (full path).")
    g.add_argument("--training-dir", default=None, help="Training directory (default: <ext-dir>/training).")
    g.add_argument("--code", default="openmc", help="Monte Carlo code label.")
    g.add_argument("--device", default="cpu", help='Torch device, e.g. "cpu" or "cuda".')
    g.add_argument("--seed", type=int, default=None, help="RNG seed for the train/test split and sampling.")

    g = p.add_argument_group("data generation (OpenMC)")
    g.add_argument("--simulate", type=int, default=0, metavar="N",
                   help="Generate the dataset by running OpenMC over N sampled configurations.")
    g.add_argument("--sim-particles", type=int, default=5000, help="OpenMC particles per batch.")
    g.add_argument("--sim-batches", type=int, default=130, help="OpenMC total batches.")
    g.add_argument("--sim-inactive", type=int, default=30, help="OpenMC inactive batches.")

    g = p.add_argument_group("metamodel training (NN)")
    g.add_argument("--batch-size", type=int, default=8192)
    g.add_argument("--ensemble-size", type=int, default=5)
    g.add_argument("--epochs", type=int, default=1500)
    g.add_argument("--layers", default="8192-256-256-256-256-16", help='Architecture string, e.g. "256-256-16".')
    g.add_argument("--loss", default="sse", choices=["sse", "mse", "mae"])
    g.add_argument("--opt-alg", default="adamax",
                   choices=["adadelta", "adagrad", "adam", "adamax", "nadam", "rmsprop"])
    g.add_argument("--learning-rate", type=float, default=0.00075)
    g.add_argument("--val-split", type=float, default=0.2)
    g.add_argument("--overwrite", action="store_true", help="Overwrite cached models if settings changed.")
    g.add_argument("--no-replot", dest="replot", action="store_false", help="Skip regenerating history plots.")
    g.add_argument("--reweight", action="store_true", help="Recompute ensemble weights even if cached.")
    g.add_argument("--verbose", action="store_true")

    g = p.add_argument_group("Bayesian network / risk")
    g.add_argument("--facility-data", default="facility.csv", help="Facility CSV file name within --ext-dir.")
    g.add_argument("--dist", default="gamma",
                   choices=["gamma", "gev", "normal", "log-normal", "weibull"])
    g.add_argument("--keff-cutoff", type=float, default=0.9)
    g.add_argument("--mass-cutoff", type=float, default=100)
    g.add_argument("--rad-cutoff", type=float, default=7)
    g.add_argument("--risk-pool", type=int, default=100)
    g.add_argument("--sample-size", type=int, default=1_000_000)
    g.add_argument("--usl", type=float, default=0.95)
    g.add_argument("--skip-risk", action="store_true", help="Stop after training the metamodel.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ext_dir = Path(args.ext_dir)

    if args.simulate > 0:
        print(f"== Generating {args.simulate} configurations with OpenMC ==")
        dataset = generate_dataset(
            ext_dir,
            code=args.code,
            n=args.simulate,
            settings=SimSettings(
                particles=args.sim_particles,
                batches=args.sim_batches,
                inactive=args.sim_inactive,
                seed=args.seed,
            ),
            seed=args.seed,
            verbose=args.verbose,
        )
    else:
        print("== Loading dataset ==")
        dataset = tabulate(code=args.code, ext_dir=ext_dir, seed=args.seed)
    print(f"   training rows: {len(dataset.training_data)}, "
          f"test rows: {len(dataset.test_data)}, features: {dataset.training_df.shape[1]}")

    print("== Training metamodel ensemble ==")
    metamodel = train_nn(
        dataset,
        batch_size=args.batch_size,
        code=args.code,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        layers=args.layers,
        loss=args.loss,
        opt_alg=args.opt_alg,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        overwrite=args.overwrite,
        replot=args.replot,
        reweight=args.reweight,
        verbose=args.verbose,
        ext_dir=ext_dir,
        training_dir=args.training_dir,
        device=args.device,
    )

    if args.skip_risk:
        print("== Done (skipping risk) ==")
        return 0

    facility = args.facility_data.replace(".csv", "")
    print("== Building Bayesian network ==")
    bn = build_bn(args.facility_data, ext_dir=ext_dir, dist=args.dist)

    print("== Estimating risk ==")
    risk, _ = estimate_risk(
        bn, metamodel, dataset,
        facility=facility,
        dist=args.dist,
        keff_cutoff=args.keff_cutoff,
        mass_cutoff=args.mass_cutoff,
        rad_cutoff=args.rad_cutoff,
        risk_pool=args.risk_pool,
        sample_size=args.sample_size,
        usl=args.usl,
        ext_dir=ext_dir,
        training_dir=args.training_dir,
        device=args.device,
        rng=np.random.default_rng(args.seed),
    )
    print(f"== Done. Mean risk = {risk.mean():.3e} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
