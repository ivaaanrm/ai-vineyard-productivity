#!/usr/bin/env bash
# Submit a SLURM array job for the dataset pipeline.
#
# Each task processes one parcel from aoi_table.csv.  After all tasks finish,
# run with --merge to combine partial CSVs into the final dataset.csv.
#
# Usage:
#   bash hpc/submit_dataset.sh                    # submit all parcels
#   bash hpc/submit_dataset.sh --parcels-at-once 8
#   bash hpc/submit_dataset.sh --dry-run          # preview without submitting
#   bash hpc/submit_dataset.sh --merge            # merge after jobs finish

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_DIR="${HOME}/ai-vineyard-productivity"
ZARR_DIR="${HOME}/agrixel-fair-dockerV2/data/output/files"
CONFIG_FILE="${REPO_DIR}/src/config/dataset.yml"
OUTPUT_DIR="${REPO_DIR}/experiments/data"
AOI_TABLE="${HOME}/agrixel-fair-dockerV2/data/input/aoi_table.csv"
PARCELS_AT_ONCE=10
DRY_RUN=false
MERGE=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parcels-at-once) PARCELS_AT_ONCE="$2"; shift 2 ;;
        --zarr-dir)        ZARR_DIR="$2";        shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift   ;;
        --merge)           MERGE=true;            shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Merge mode ────────────────────────────────────────────────────────────────
if $MERGE; then
    conda activate ai-vineyard 2>/dev/null || true
    python "${REPO_DIR}/src/dataset/run_calcula.py" \
        --merge \
        --config     "${CONFIG_FILE}" \
        --output-dir "${OUTPUT_DIR}"
    exit 0
fi

# ── Count parcels ─────────────────────────────────────────────────────────────
if [[ ! -f "${AOI_TABLE}" ]]; then
    echo "ERROR: aoi_table not found: ${AOI_TABLE}" >&2
    exit 1
fi

N_PARCELS=$(( $(wc -l < "${AOI_TABLE}") - 1 ))  # subtract header row

printf "%-20s: %s\n" "Parcels"         "${N_PARCELS}"
printf "%-20s: %s\n" "Parcels at once" "${PARCELS_AT_ONCE}"
printf "%-20s: %s\n" "zarr-dir"        "${ZARR_DIR}"
printf "%-20s: %s\n" "output-dir"      "${OUTPUT_DIR}"

# ── Write runtime config ──────────────────────────────────────────────────────
CONFIG_OUT="${REPO_DIR}/hpc/.job_dataset_config"
cat > "${CONFIG_OUT}" <<EOF
REPO_DIR=${REPO_DIR}
ZARR_DIR=${ZARR_DIR}
CONFIG_FILE=${CONFIG_FILE}
OUTPUT_DIR=${OUTPUT_DIR}
EOF
echo ""
echo "Written: ${CONFIG_OUT}"

# ── Patch array range ─────────────────────────────────────────────────────────
JOB_SCRIPT="${REPO_DIR}/hpc/job_dataset.sh"
sed -i "s|^#SBATCH --array=.*|#SBATCH --array=1-${N_PARCELS}%${PARCELS_AT_ONCE}|" "${JOB_SCRIPT}"
echo "Updated: ${JOB_SCRIPT} → --array=1-${N_PARCELS}%${PARCELS_AT_ONCE}"

mkdir -p "${REPO_DIR}/hpc/logs"

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Would run: sbatch ${JOB_SCRIPT}"
    exit 0
fi

# ── Submit ────────────────────────────────────────────────────────────────────
echo ""
CMD="sbatch ${JOB_SCRIPT}"
echo "Command: ${CMD}"
eval "${CMD}"

echo ""
echo "After all tasks finish, merge results with:"
echo "  bash hpc/submit_dataset.sh --merge"
