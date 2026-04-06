#!/usr/bin/env bash
# One-time environment setup on the CSL cluster.
# Run this once from a development node (cslE##) or login shell — NOT via sbatch.
#
# Workflow:
#   ssh ivan.romero.moreno@csl.calcula.tsc.upc.edu
#   tmux new -s setup                    # persistent session in case SSH drops
#   cd ~/agrixel-fair-dockerV2
#   bash hpc/setup_env.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Repo root: $REPO_DIR"

# ── 1. Load conda module ──────────────────────────────────────────────────
module load conda/24.3

# ── 2. Init conda for bash (only needed the very first time) ─────────────
if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
    echo "Running conda init (one-time) ..."
    conda init bash
fi

# Source conda.sh directly — this activates conda in the current shell
# without needing to close/reopen it (avoids the "conda activate" not found error).
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
# shellcheck source=/dev/null
source "$CONDA_SH"
echo "NOTE: conda init also modified ~/.bashrc. Future interactive shells will"
echo "      load conda automatically. For this session we sourced conda.sh directly."

# ── 3. Create environment ─────────────────────────────────────────────────
ENV_NAME="agrixel-fair"

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '$ENV_NAME' already exists — skipping create."
else
    echo "Creating environment from environment.yml (this takes a few minutes) ..."
    conda env create -f "$REPO_DIR/environment.yml"
fi

conda activate agrixel-fair

# Quick import check
python -c "import rasterio, shapely, yaml, zarr; print('  OK: core imports work')"

# ── 4. Credential file ────────────────────────────────────────────────────
CREDS_FILE="$REPO_DIR/hpc/.env"
if [[ ! -f "$CREDS_FILE" ]]; then
    cat > "$CREDS_FILE" << 'EOF'
# API credentials — fill in and save.
# Planetary Computer: no key needed for S1/S2/S3 public STAC.

# ERA5 / Copernicus CDS — get key at https://cds.climate.copernicus.eu
ERA5_APIKEY=

# NASA Earthdata (for MODIS) — register at https://urs.earthdata.nasa.gov
MODIS_USERNAME=
MODIS_PASSWORD=
EOF
    echo "Created $CREDS_FILE — edit it and fill in your keys if using ERA5/MODIS."
else
    echo "Credentials file already exists: $CREDS_FILE"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Reload your shell so 'conda activate' works in future sessions:"
echo "       source ~/.bashrc"
echo "  2. Activate the environment:"
echo "       conda activate agrixel-fair"
echo "  3. Edit hpc/.env if you need ERA5 or MODIS credentials."
echo "  4. Dry-run to confirm everything works:"
echo "       python hpc/run_parcel.py --index 1 --dry-run"
echo "  5. Submit all jobs:"
echo "       bash hpc/submit.sh"
