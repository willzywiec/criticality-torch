#!/usr/bin/env bash
#
# download_data.sh -- fetch an OpenMC cross-section library and persist
# OPENMC_CROSS_SECTIONS for the conda environment.
#
# Default: use `openmc-data-downloader` to grab only the nuclides this package
# uses (H, O, U-235/238, Pu-239/240) plus the c_H_in_H2O thermal scattering
# data -- a small, fast download sufficient to run the pipeline.
#
# Alternative (production accuracy): download the full official ENDF/B-VIII.0
# HDF5 library by setting OPENMC_DATA_URL to the tarball from
# https://openmc.org/official-data-libraries/ -- e.g.
#   OPENMC_DATA_URL=https://.../endfb-viii.0-hdf5.tar.xz bash download_data.sh
#
# Override the install location with DATA_DIR (default: ~/openmc-data).

set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/openmc-data}"
LIBRARY="${OPENMC_DATA_LIBRARY:-ENDFB-8.0-NNDC}"
mkdir -p "$DATA_DIR"

if [ -n "${OPENMC_DATA_URL:-}" ]; then
    echo "==> Downloading full library from OPENMC_DATA_URL"
    tarball="$DATA_DIR/library.tar.xz"
    curl -fsSL "$OPENMC_DATA_URL" -o "$tarball"
    tar -xf "$tarball" -C "$DATA_DIR"
    rm -f "$tarball"
    XS_XML="$(find "$DATA_DIR" -name cross_sections.xml | head -n1)"
else
    echo "==> Downloading minimal nuclide set with openmc-data-downloader ($LIBRARY)"
    # See https://github.com/openmc-data-storage/openmc_data_downloader
    openmc-data-downloader \
        -l "$LIBRARY" \
        -d "$DATA_DIR" \
        -i H1 H2 O16 O17 U235 U238 Pu239 Pu240 \
        -s c_H_in_H2O
    XS_XML="$DATA_DIR/cross_sections.xml"
fi

if [ -z "${XS_XML:-}" ] || [ ! -f "$XS_XML" ]; then
    echo "ERROR: could not locate cross_sections.xml under $DATA_DIR" >&2
    exit 1
fi

echo "==> Cross sections: $XS_XML"

# Persist OPENMC_CROSS_SECTIONS so it is set on every `conda activate`.
if [ -n "${CONDA_PREFIX:-}" ]; then
    ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    mkdir -p "$ACT_DIR"
    echo "export OPENMC_CROSS_SECTIONS=\"$XS_XML\"" > "$ACT_DIR/openmc_xs.sh"
    echo "==> Wrote $ACT_DIR/openmc_xs.sh (sets OPENMC_CROSS_SECTIONS on activate)"
fi

# Also export for the current shell/session.
export OPENMC_CROSS_SECTIONS="$XS_XML"
echo "==> OPENMC_CROSS_SECTIONS=$OPENMC_CROSS_SECTIONS"
