#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --partition=csl
#SBATCH --account=csl
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=1-1%1
#SBATCH --output=hpc/logs/dataset_%A_%a.out
#SBATCH --error=hpc/logs/dataset_%A_%a.err

# Written by submit_dataset.sh — do not edit manually.
# Each task processes one parcel from aoi_table.csv.

set -euo pipefail

source hpc/.job_dataset_config

mkdir -p hpc/logs

echo "=== Task ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX} ==="
echo "  Host     : $(hostname)"
echo "  zarr-dir : ${ZARR_DIR}"
echo "  output   : ${OUTPUT_DIR}"
echo ""

module load conda/24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-vineyard

cd "${REPO_DIR}"

python src/dataset/run_calcula.py \
    --parcel-index "${SLURM_ARRAY_TASK_ID}" \
    --zarr-dir     "${ZARR_DIR}" \
    --config       "${CONFIG_FILE}" \
    --output-dir   "${OUTPUT_DIR}"
