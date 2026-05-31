"""Verify that the full native stack (OpenMC + cross sections) runs.

Runs a single small k-eigenvalue calculation for a bare HEU sphere and a
water-reflected case, printing keff. If this works, `generate_dataset(...)`
will too. Intended to be run after scripts/setup_wsl.sh on WSL.
"""

import os
import sys

from criticality import SimSettings, run_keff


def main() -> int:
    if not os.environ.get("OPENMC_CROSS_SECTIONS"):
        print("OPENMC_CROSS_SECTIONS is not set; run scripts/download_data.sh first.",
              file=sys.stderr)
        return 1

    # Small, fast settings -- this is a smoke test, not a converged result.
    settings = SimSettings(particles=2000, batches=60, inactive=15, seed=1)

    cases = [
        dict(mass=50_000, form="heu", mod="none", rad=8.7, ref="none", thk=0.0),
        dict(mass=50_000, form="heu", mod="none", rad=8.7, ref="h2o", thk=5.0),
    ]
    for case in cases:
        keff, sd = run_keff(settings=settings, **case)
        label = "reflected" if case["ref"] != "none" else "bare     "
        print(f"  {label}  keff = {keff:.5f} ± {sd:.5f}")

    print("OpenMC native verification OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
