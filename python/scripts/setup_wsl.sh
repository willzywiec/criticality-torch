#!/usr/bin/env bash
#
# setup_wsl.sh -- one-shot native setup for the criticality pipeline on WSL.
#
# Installs Miniforge (if conda is missing), creates the `criticality` conda
# environment (OpenMC + torch + scientific stack), installs this package in
# editable mode, downloads a cross-section library, and verifies that a real
# OpenMC k-eigenvalue calculation runs end to end.
#
# Usage:
#   bash scripts/setup_wsl.sh                 # full setup + minimal data + verify
#   SKIP_DATA=1 bash scripts/setup_wsl.sh     # env only, skip cross-section data
#
# Everything runs natively in WSL -- no Docker.

set -euo pipefail

ENV_NAME="criticality"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the python/ dir
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

echo "==> criticality WSL setup"
echo "    package dir: $HERE"

# --- 1. Ensure conda (Miniforge) -------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    if [ ! -d "$MINIFORGE_DIR" ]; then
        echo "==> Installing Miniforge to $MINIFORGE_DIR"
        arch="$(uname -m)"
        url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
        tmp="$(mktemp --suffix=.sh)"
        curl -fsSL "$url" -o "$tmp"
        bash "$tmp" -b -p "$MINIFORGE_DIR"
        rm -f "$tmp"
    fi
    # shellcheck disable=SC1091
    source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# --- 2. Create or update the environment -----------------------------------
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "==> Updating conda env '$ENV_NAME'"
    conda env update -n "$ENV_NAME" -f "$HERE/environment.yml" --prune
else
    echo "==> Creating conda env '$ENV_NAME'"
    conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

# --- 3. Install the package (editable) -------------------------------------
echo "==> Installing criticality (editable)"
conda run -n "$ENV_NAME" pip install -e "$HERE"

# --- 4. Cross-section data --------------------------------------------------
if [ "${SKIP_DATA:-0}" != "1" ]; then
    echo "==> Downloading cross-section data"
    conda run -n "$ENV_NAME" bash "$HERE/scripts/download_data.sh"
else
    echo "==> Skipping cross-section data (SKIP_DATA=1)"
fi

# --- 5. Verify ---------------------------------------------------------------
echo "==> Running test suite"
conda run -n "$ENV_NAME" pytest -q "$HERE/tests"

if [ "${SKIP_DATA:-0}" != "1" ]; then
    echo "==> Verifying a live OpenMC k-eigenvalue run"
    conda run -n "$ENV_NAME" python "$HERE/scripts/verify_openmc.py"
fi

echo
echo "==> Done. Activate the environment with:"
echo "      conda activate $ENV_NAME"
echo "    Then run the pipeline, e.g.:"
echo "      python -m criticality --ext-dir extdata --simulate 2000"
