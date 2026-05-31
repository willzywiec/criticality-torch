"""simulate.py

OpenMC Monte Carlo transport backend.

This replaces the precomputed MCNP/COG datasets the package used to depend on.
Given a fissile-material configuration (mass, form, mod, rad, ref, thk), it
builds a spherical OpenMC model and runs a k-eigenvalue calculation to obtain
keff and its standard deviation -- the labels the DNN metamodel is trained on.

OpenMC is imported lazily so the rest of the package (scaling, the torch
metamodel, the Bayesian network, risk) imports and runs without it; you only
need OpenMC (and a cross-section library, via the ``OPENMC_CROSS_SECTIONS``
environment variable) to *generate* data.

Geometry model (matching the R parameterization):
    * A homogeneous sphere of radius ``rad`` holds ``mass`` grams of fissile
      material at concentration ``conc = mass / vol``; when moderated, water
      fills the remaining volume (a solution/slurry). When ``mod == 'none'`` the
      sphere is bare metal at full density.
    * An optional reflector is a spherical shell of thickness ``thk`` around the
      core (``ref == 'h2o'`` -> water; ``ref == 'none'`` -> bare).
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Fissile material density (g/cc) keyed by material form. Single source of truth
# shared with predict.py.
FISS_DENSITY = {
    "alpha": 19.86,   # alpha-phase Pu metal
    "delta": 15.92,   # delta-phase Pu metal
    "puo2": 11.5,     # plutonium dioxide
    "heu": 18.85,     # highly enriched uranium metal
    "uo2": 10.97,     # uranium dioxide
}

# Default isotopics (weight fractions). Weapons-grade Pu and 93%-enriched U.
PU_VECTOR = {"Pu239": 0.94, "Pu240": 0.06}
U_VECTOR = {"U235": 0.93, "U238": 0.07}

# Weight fraction of the heavy metal in the oxide (Pu/U ~ 88%, O ~ 12%).
_PU_IN_PUO2 = 239.0 / (239.0 + 2 * 16.0)
_U_IN_UO2 = 238.0 / (238.0 + 2 * 16.0)

WATER_DENSITY = 0.998  # g/cc at room temperature


def _require_openmc():
    try:
        import openmc  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "OpenMC is required to run simulations. Install it (conda-forge or "
            "from source) and set OPENMC_CROSS_SECTIONS to a cross-section "
            "library. See https://docs.openmc.org/."
        ) from exc
    return openmc


def _fissile_material(form: str, density: float):
    """Build a pure fissile-material OpenMC material at ``density`` (g/cc)."""
    openmc = _require_openmc()
    mat = openmc.Material(name=form)
    if form in ("alpha", "delta"):  # Pu metal
        for nuc, wo in PU_VECTOR.items():
            mat.add_nuclide(nuc, wo, "wo")
    elif form == "heu":  # U metal
        for nuc, wo in U_VECTOR.items():
            mat.add_nuclide(nuc, wo, "wo")
    elif form == "puo2":
        for nuc, wo in PU_VECTOR.items():
            mat.add_nuclide(nuc, _PU_IN_PUO2 * wo, "wo")
        mat.add_nuclide("O16", 1.0 - _PU_IN_PUO2, "wo")
    elif form == "uo2":
        for nuc, wo in U_VECTOR.items():
            mat.add_nuclide(nuc, _U_IN_UO2 * wo, "wo")
        mat.add_nuclide("O16", 1.0 - _U_IN_UO2, "wo")
    else:
        raise ValueError(f"unknown fissile form '{form}'")
    mat.set_density("g/cm3", density)
    return mat


def _water(thermal: bool = True):
    openmc = _require_openmc()
    water = openmc.Material(name="water")
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    water.set_density("g/cm3", WATER_DENSITY)
    # Thermal scattering improves moderation accuracy but needs the c_H_in_H2O
    # S(alpha,beta) data; disable it for a minimal (free-gas) cross-section set.
    if thermal:
        water.add_s_alpha_beta("c_H_in_H2O")
    return water


def _core_material(form: str, conc: float, thermal: bool = True):
    """Homogeneous core: fissile material at ``conc`` g/cc, water filling rest."""
    openmc = _require_openmc()
    rho_fiss = FISS_DENSITY[form]
    fiss = _fissile_material(form, rho_fiss)
    vf_fiss = min(max(conc / rho_fiss, 0.0), 1.0)
    if vf_fiss >= 1.0:  # bare metal -- no moderator
        return fiss
    return openmc.Material.mix_materials(
        [fiss, _water(thermal)], [vf_fiss, 1.0 - vf_fiss], "vo"
    )


@dataclass
class SimSettings:
    """OpenMC k-eigenvalue run controls."""

    particles: int = 5000
    batches: int = 130
    inactive: int = 30
    seed: int | None = None
    thermal_scattering: bool = True  # set False for a free-gas-only data library
    threads: int = 1  # OpenMC threads per run; keep 1 when fanning out many runs


def build_model(
    mass: float,
    form: str,
    mod: str,
    rad: float,
    ref: str,
    thk: float,
    settings: SimSettings | None = None,
):
    """Build an :class:`openmc.Model` for one fissile-material configuration."""
    openmc = _require_openmc()
    settings = settings or SimSettings()

    vol = 4.0 / 3.0 * math.pi * rad**3
    conc = mass / vol if vol > 0 else 0.0
    # When unmoderated, the core is bare metal regardless of nominal concentration.
    if mod == "none":
        conc = FISS_DENSITY[form]

    core = _core_material(form, conc, settings.thermal_scattering)
    materials = [core]

    has_reflector = ref != "none" and thk > 0
    if has_reflector:
        reflector = _water(settings.thermal_scattering)
        materials.append(reflector)

    core_sphere = openmc.Sphere(r=rad)
    if has_reflector:
        outer = openmc.Sphere(r=rad + thk, boundary_type="vacuum")
        core_cell = openmc.Cell(fill=core, region=-core_sphere)
        refl_cell = openmc.Cell(fill=reflector, region=+core_sphere & -outer)
        cells = [core_cell, refl_cell]
    else:
        core_sphere.boundary_type = "vacuum"
        cells = [openmc.Cell(fill=core, region=-core_sphere)]

    geometry = openmc.Geometry(openmc.Universe(cells=cells))

    run = openmc.Settings()
    run.run_mode = "eigenvalue"
    run.particles = settings.particles
    run.batches = settings.batches
    run.inactive = settings.inactive
    if settings.seed is not None:
        run.seed = settings.seed
    run.source = openmc.IndependentSource(space=openmc.stats.Point((0.0, 0.0, 0.0)))

    return openmc.Model(geometry, openmc.Materials(materials), run)


def run_keff(
    mass: float,
    form: str,
    mod: str,
    rad: float,
    ref: str,
    thk: float,
    settings: SimSettings | None = None,
) -> tuple[float, float]:
    """Run one OpenMC k-eigenvalue calculation; return ``(keff, sd)``."""
    openmc = _require_openmc()
    settings = settings or SimSettings()
    model = build_model(mass, form, mod, rad, ref, thk, settings)
    with tempfile.TemporaryDirectory() as run_dir:
        # One thread per run by default: when many runs are dispatched across a
        # process pool, per-run threading would oversubscribe the CPU.
        sp_path = model.run(cwd=run_dir, output=False, threads=settings.threads)
        with openmc.StatePoint(sp_path) as sp:
            keff = sp.keff
    return float(keff.nominal_value), float(keff.std_dev)


OUTPUT_COLUMNS = [
    "mass", "form", "mod", "rad", "ref", "thk", "vol", "conc", "keff", "sd",
]


def _simulate_one(task: tuple) -> tuple[int, dict, str | None]:
    """Run a single configuration. Top-level so it is picklable by the pool.

    Returns ``(index, row, error)`` where ``error`` is ``None`` on success. A
    failed run is reported with NaN ``keff``/``sd`` rather than raised so one
    bad configuration never aborts a large batch.
    """
    idx, mass, form, mod, rad, ref, thk, settings = task
    vol = 4.0 / 3.0 * math.pi * rad**3
    row = {
        "mass": mass, "form": form, "mod": mod, "rad": rad,
        "ref": ref, "thk": thk, "vol": vol,
        "conc": mass / vol if vol > 0 else 0.0,
        "keff": float("nan"), "sd": float("nan"),
    }
    try:
        keff, sd = run_keff(mass, form, mod, rad, ref, thk, settings)
        row["keff"], row["sd"] = keff, sd
        return idx, row, None
    except Exception as exc:  # noqa: BLE001 - keep the batch going
        return idx, row, str(exc)


class _CsvAppender:
    """Stream completed rows to disk so a long batch survives interruption."""

    def __init__(self, path: Path):
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=OUTPUT_COLUMNS)
        self._writer.writeheader()
        self._fh.flush()

    def append(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def simulate_dataset(
    configs: pd.DataFrame,
    settings: SimSettings | None = None,
    *,
    max_workers: int | None = None,
    output_csv: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a large series of OpenMC calculations, in parallel, over ``configs``.

    Each row is an independent k-eigenvalue calculation dispatched to a worker
    process (one OpenMC run per task, single-threaded by default), making this
    embarrassingly parallel across CPU cores.

    Args:
        configs: DataFrame with columns ``mass, form, mod, rad, ref, thk``.
        settings: OpenMC run controls (``settings.threads`` sets threads/run).
        max_workers: Number of worker processes (default: all CPU cores). Use 1
            to run inline in the current process (no pool).
        output_csv: If given, completed rows are streamed here as they finish,
            so partial progress survives a crash or interruption.
        verbose: Print per-simulation progress.

    Returns:
        A DataFrame with the inputs plus derived ``vol``/``conc`` and the
        simulated ``keff``/``sd``, ordered to match ``configs``. Failed runs
        carry NaN ``keff``/``sd``.
    """
    settings = settings or SimSettings()
    n = len(configs)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, n)) if n else 1

    tasks = [
        (
            i, float(r.mass), str(r.form), str(r.mod),
            float(r.rad), str(r.ref), float(r.thk), settings,
        )
        for i, r in enumerate(configs.itertuples(index=False))
    ]

    writer = _CsvAppender(Path(output_csv)) if output_csv is not None else None
    results: dict[int, dict] = {}
    failures = 0

    def _record(done: int, idx: int, row: dict, err: str | None) -> None:
        nonlocal failures
        results[idx] = row
        if writer is not None:
            writer.append(row)
        if err is not None:
            failures += 1
        if verbose:
            tag = f"[{done}/{n}]"
            head = f"{row['form']} mass={row['mass']:.0f}g rad={row['rad']:.2f}cm"
            if err is None:
                print(f"  {tag} {head} -> keff={row['keff']:.5f} ± {row['sd']:.5f}")
            else:
                print(f"  {tag} {head} -> FAILED: {err}")

    try:
        if max_workers == 1:
            # Inline path: no process pool (simpler, and easy to test/mock).
            for done, task in enumerate(tasks, start=1):
                idx, row, err = _simulate_one(task)
                _record(done, idx, row, err)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_simulate_one, t) for t in tasks]
                for done, fut in enumerate(as_completed(futures), start=1):
                    idx, row, err = fut.result()
                    _record(done, idx, row, err)
    finally:
        if writer is not None:
            writer.close()

    if failures and verbose:
        print(f"  {failures}/{n} configurations failed (recorded with NaN keff).")

    ordered = [results[i] for i in range(n)] if n else []
    return pd.DataFrame(ordered, columns=OUTPUT_COLUMNS)


def sample_configs(
    n: int,
    *,
    mass_range: tuple[float, float] = (10.0, 4000.0),
    rad_range: tuple[float, float] = (1.0, 45.0),
    thk_range: tuple[float, float] = (0.0, 28.0),
    forms: tuple[str, ...] = ("alpha", "delta", "puo2", "heu", "uo2"),
    mods: tuple[str, ...] = ("h2o", "none"),
    refs: tuple[str, ...] = ("h2o", "none"),
    seed: int | None = None,
) -> pd.DataFrame:
    """Latin-hypercube-ish random sampling of the input space for data generation."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mass": rng.uniform(*mass_range, n),
            "form": rng.choice(forms, n),
            "mod": rng.choice(mods, n),
            "rad": rng.uniform(*rad_range, n),
            "ref": rng.choice(refs, n),
            "thk": rng.uniform(*thk_range, n),
        }
    )
