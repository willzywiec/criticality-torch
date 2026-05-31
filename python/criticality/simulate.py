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

import math
import tempfile
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
    model = build_model(mass, form, mod, rad, ref, thk, settings)
    with tempfile.TemporaryDirectory() as run_dir:
        sp_path = model.run(cwd=run_dir, output=False)
        with openmc.StatePoint(sp_path) as sp:
            keff = sp.keff
    return float(keff.nominal_value), float(keff.std_dev)


def simulate_dataset(
    configs: pd.DataFrame,
    settings: SimSettings | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run OpenMC for each row of ``configs`` and return a labeled output table.

    Args:
        configs: DataFrame with columns ``mass, form, mod, rad, ref, thk``.
        settings: OpenMC run controls.
        verbose: Print per-simulation progress.

    Returns:
        A DataFrame with the inputs plus derived ``vol``/``conc`` and the
        simulated ``keff``/``sd`` -- the same schema Tabulate/Scale expect.
    """
    rows = []
    n = len(configs)
    for idx, row in enumerate(configs.itertuples(index=False), start=1):
        mass, form, mod = float(row.mass), str(row.form), str(row.mod)
        rad, ref, thk = float(row.rad), str(row.ref), float(row.thk)
        keff, sd = run_keff(mass, form, mod, rad, ref, thk, settings)
        vol = 4.0 / 3.0 * math.pi * rad**3
        rows.append(
            {
                "mass": mass, "form": form, "mod": mod, "rad": rad,
                "ref": ref, "thk": thk, "vol": vol,
                "conc": mass / vol if vol > 0 else 0.0,
                "keff": keff, "sd": sd,
            }
        )
        if verbose:
            print(f"  [{idx}/{n}] {form} mass={mass:.0f}g rad={rad:.2f}cm -> keff={keff:.5f} ± {sd:.5f}")
    return pd.DataFrame(rows)


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
