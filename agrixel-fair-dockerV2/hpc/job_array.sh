#!/usr/bin/env bash
# SLURM array job for CSL cluster (csl.calcula.tsc.upc.edu).
#
# Each task = 1 parcel × 1 sensor.
# submit.sh sets the array range and writes hpc/.job_config before submitting.
#
# Layout:
#   task 1  → parcel 1, sensor 0  (e.g. S2)
#   task 2  → parcel 1, sensor 1  (e.g. S1)
#   task 3  → parcel 1, sensor 2  (e.g. S3)
#   task 4  → parcel 2, sensor 0
#   ...
#
# Submit: bash hpc/submit.sh [--parcels-at-once 5]

#SBATCH --job-name=agrixel
#SBATCH -p csl
#SBATCH -A csl
#SBATCH --array=1-2889%15          # patched by submit.sh: N_PARCELS×N_SENSORS % (parcels_at_once×N_SENSORS)
#SBATCH --time=04:00:00            # 4 h per sensor per parcel (~36 windows)
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=hpc/logs/agrixel_%A_%a.out
#SBATCH --error=hpc/logs/agrixel_%A_%a.err
#SBATCH --mail-type=FAIL

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/agrixel-fair-dockerV2"
DATA_DIR="$REPO_DIR/data"

# ── Job config (written by submit.sh) ─────────────────────────────────────
# Provides: N_SENSORS, SENSORS (array), N_PARCELS
JOB_CONFIG="$REPO_DIR/hpc/.job_config"
if [[ ! -f "$JOB_CONFIG" ]]; then
    echo "ERROR: $JOB_CONFIG not found. Run bash hpc/submit.sh first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$JOB_CONFIG"

# ── Map task ID → (parcel index, sensor) ──────────────────────────────────
TASK_0=$(( SLURM_ARRAY_TASK_ID - 1 ))
PARCEL_INDEX=$(( TASK_0 / N_SENSORS + 1 ))    # 1-based row in aoi_table.csv
SENSOR_IDX=$(( TASK_0 % N_SENSORS ))
SENSOR_KEY="${SENSORS[$SENSOR_IDX]}"

echo "Task $SLURM_ARRAY_TASK_ID: parcel $PARCEL_INDEX / $N_PARCELS  sensor $SENSOR_KEY  host $(hostname)  $(date)"

# ── Conda env ──────────────────────────────────────────────────────────────
# Hardcode the conda path — module load is unreliable in non-interactive batch shells.
source /opt/conda-24.3/etc/profile.d/conda.sh
conda activate agrixel-fair

if ! python -c "import yaml, rasterio, shapely" &>/dev/null; then
    echo "ERROR: conda env 'agrixel-fair' not ready on $(hostname)" >&2
    exit 1
fi

mkdir -p "$REPO_DIR/hpc/logs"

# ── Run ────────────────────────────────────────────────────────────────────
cd "$REPO_DIR"

python hpc/run_parcel.py \
    --index    "$PARCEL_INDEX" \
    --sensor   "$SENSOR_KEY" \
    --config   cli_app/run_config.yml \
    --data-dir "$DATA_DIR" \
    --app-dir  "$REPO_DIR/app" \
    --env-file "$REPO_DIR/hpc/.env"

EXIT_CODE=$?
echo "Finished task $SLURM_ARRAY_TASK_ID (parcel $PARCEL_INDEX / $SENSOR_KEY) — exit $EXIT_CODE  $(date)"
exit $EXIT_CODE
