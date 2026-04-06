# HPC Download Jobs — CSL Cluster (UPC)

Runs the satellite image download pipeline (S1/S2/S3) on the
[CSL CommSensLab cluster](https://tsc.upc.edu/en/it-services/computing-services)
without Docker, using SLURM array jobs.

Cluster address: `csl.calcula.tsc.upc.edu`

---

## Architecture

Each SLURM task processes **one parcel × one sensor**. All sensors for the same
parcel run in parallel on separate nodes. After all windows finish, the output
directory is cleaned automatically — only `cube.zarr` is kept.

```
parcel L2403 ──┬── task 1 → S2  (24 windows, 1 CPU, 1 GB)  → cleanup → cube.zarr
               ├── task 2 → S1  (24 windows, 1 CPU, 1 GB)  → cleanup → cube.zarr
               └── task 3 → S3  (24 windows, 1 CPU, 1 GB)  → cleanup → cube.zarr

parcel L2411 ──┬── task 4 → S2  ...
               ├── task 5 → S1  ...
               └── task 6 → S3  ...
...
```

With `--parcels-at-once 5` and 3 sensors: **15 tasks run simultaneously**
(5 parcels × 3 sensors). Total tasks: 963 × 3 = 2,889.

---

## Files in this directory

| File | Purpose |
|---|---|
| `setup_env.sh` | One-time conda environment creation on the cluster |
| `job_array.sh` | SLURM array job script (1 task = 1 parcel × 1 sensor) |
| `run_parcel.py` | Per-task runner — loops windows, calls `app/runner.py`, cleans output |
| `submit.sh` | Reads config, computes task count, writes `.job_config`, calls `sbatch` |
| `status.sh` | Progress report from log files |
| `.env` | API credentials (created by `setup_env.sh`, never committed) |
| `.job_config` | Runtime config written by `submit.sh`, sourced by `job_array.sh` |
| `logs/` | SLURM stdout/stderr per task (created on first submit) |

---

## Cluster layout

```
Login / development nodes:  cslE##   ← you land here after SSH
Compute nodes:              cslC##   ← SLURM sends tasks here automatically
Scheduler:                  SLURM
Partition / account:        -p csl -A csl
Conda module:               conda/24.3
```

---

## Step 1 — Copy the repo to the cluster

Run from your local machine:

```bash
rsync -av --exclude='data/output' --exclude='__pycache__' --exclude='.git' \
  /Users/ivanr/Developer/ai-vineyard-productivity/agrixel-fair-dockerV2/ \
  ivan.romero.moreno@csl.calcula.tsc.upc.edu:~/agrixel-fair-dockerV2/
```

Only needs repeating when code or config changes.

---

## Step 2 — SSH in and start a tmux session

```bash
ssh ivan.romero.moreno@csl.calcula.tsc.upc.edu
tmux new -s agrixel
```

If your SSH connection drops, reattach with:
```bash
tmux attach -t agrixel
```

Useful tmux keys: `Ctrl+b d` detach · `Ctrl+b c` new window · `Ctrl+b 0/1` switch

---

## Step 3 — One-time environment setup

```bash
cd ~/agrixel-fair-dockerV2
bash hpc/setup_env.sh
```

This loads `conda/24.3`, creates the `agrixel-fair` environment from
`environment.yml`, and writes a credentials template at `hpc/.env`.

After the script finishes, **reload your shell** so `conda activate` works:

```bash
source ~/.bashrc
conda activate agrixel-fair
# prompt shows: (agrixel-fair) ivan.romero.moreno@cslE##:~$
```

> Only needed once per login session. Future SSH logins load conda automatically.

### Credentials

Edit `hpc/.env` if you use ERA5 or MODIS (S1/S2/S3 need no API keys):

```bash
nano hpc/.env
```

```
ERA5_APIKEY=your-cds-uid:your-cds-key
MODIS_USERNAME=your-earthdata-username
MODIS_PASSWORD=your-earthdata-password
```

---

## Step 4 — Test before submitting everything

### Dry-run one parcel + one sensor (no downloads)

```bash
conda activate agrixel-fair
python hpc/run_parcel.py --index 1 --sensor S2 --dry-run
```

Expected output:
```
============================================================
  PARCEL 1/963: L2403  sensor(s): ['S2']
============================================================
  [L2403/S2] 24 windows  years=[2022, 2023]
    window 1/24: 2022-01-01 → 2022-01-31
  [DRY RUN] PARAMS_FILE=hpc_S2_xxxx.json python app/runner.py
  ...

============================================================
  DONE: L2403
    S2: completed (24/24 windows OK)
============================================================
```

### Real test on a compute node (one parcel, one sensor)

```bash
srun -p csl -A csl -c1 --mem=1G --time=0:30:00 --pty /bin/bash

# Now on a cslC## node:
module load conda/24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agrixel-fair
cd ~/agrixel-fair-dockerV2

python hpc/run_parcel.py --index 1 --sensor S2

exit  # back to login node
```

Check output:
```bash
ls data/output/files/SENTINEL-2/L2403/
# should only contain: cube.zarr/
```

---

## Step 5 — Submit all jobs

```bash
conda activate agrixel-fair
bash hpc/submit.sh
```

`submit.sh` will:
1. Read enabled sensors from `cli_app/run_config.yml`
2. Count parcels in `data/input/aoi_table.csv`
3. Compute total tasks = N_PARCELS × N_SENSORS
4. Write `hpc/.job_config` (sourced by each SLURM task at runtime)
5. Patch `--array=1-N%MAX_CONCURRENT` in `job_array.sh`
6. Call `sbatch`

Example output:
```
Parcels        : 963
Sensors        : S2 S1 S3  (3)
Total tasks    : 2889  (963 × 3)
Parcels at once: 5  → 15 concurrent tasks
Written: hpc/.job_config
Updated: job_array.sh → --array=1-2889%15
Command: sbatch hpc/job_array.sh
Submitted batch job 48271
```

### Options

```bash
# More parcels at once (raises API load — use with care):
bash hpc/submit.sh --parcels-at-once 8

# Preview without submitting:
bash hpc/submit.sh --dry-run
```

---

## Step 6 — Monitor progress

### Live queue

```bash
squeue -j 48271
```

```
  JOBID        PARTITION  NAME     USER  ST   TIME  NODES  NODELIST
48271_1        csl        agrixel  ivan  R    0:42      1  cslc03
48271_2        csl        agrixel  ivan  R    0:38      1  cslc07
48271_3        csl        agrixel  ivan  R    0:35      1  cslc09
48271_4        csl        agrixel  ivan  PD   0:00      1  (Priority)
```

States: `R` running · `PD` pending (waiting for a slot) · `CG` completing

### Summary from logs

```bash
bash hpc/status.sh 48271
```

```
=== Log summary (from hpc/logs/) ===
  Total log files : 450
  Finished        : 441
  Failed          :   3
  Still running   :   9
```

### Tail a specific task

Each task has its own log. Task ID maps to `(parcel, sensor)`:

```
task_id  = (parcel_index - 1) × N_SENSORS + sensor_index + 1

Example (3 sensors: S2 S1 S3):
  task 1  → parcel 1 / S2
  task 2  → parcel 1 / S1
  task 3  → parcel 1 / S3
  task 4  → parcel 2 / S2
  ...
```

```bash
tail -f hpc/logs/agrixel_48271_4.out
```

### Cancel all tasks

```bash
scancel 48271
```

---

## Step 7 — Sync results back to your Mac

```bash
rsync -av \
  ivan.romero.moreno@csl.calcula.tsc.upc.edu:~/agrixel-fair-dockerV2/data/output/ \
  /Users/ivanr/Developer/ai-vineyard-productivity/agrixel-fair-dockerV2/data/output/
```

Output structure (after cleanup):

```
data/output/
├── xml/
│   ├── SENTINEL-1/<parcel_id>/  → ISO-19139 XML metadata
│   ├── SENTINEL-2/<parcel_id>/
│   └── SENTINEL-3/<parcel_id>/
└── files/
    ├── SENTINEL-1/<parcel_id>/cube.zarr   ← only this is kept
    ├── SENTINEL-2/<parcel_id>/cube.zarr
    └── SENTINEL-3/<parcel_id>/cube.zarr
```

---

## What happens inside a single task

```
SLURM assigns task 4 to node cslc07
  └── job_array.sh
       ├── source hpc/.job_config  →  N_SENSORS=3, SENSORS=(S2 S1 S3)
       ├── task 4 → PARCEL_INDEX=2, SENSOR_KEY=S2
       ├── conda activate agrixel-fair
       └── python hpc/run_parcel.py --index 2 --sensor S2
            ├── reads parcel #2 from aoi_table.csv  →  L2411, years=[2021..2024]
            ├── 48 monthly windows (Jan–Dec × 4 years)
            ├── for each window:
            │    write params.json → python app/runner.py → delete params.json
            └── cleanup: remove all files except cube.zarr
```

---

## Re-running failed tasks

```bash
bash hpc/status.sh 48271   # find failed task IDs

# Resubmit specific tasks (comma-separated or range):
sbatch --array=4,7,91 hpc/job_array.sh
```

---

## Configuration reference

**Parcels at once** (controls API load):
```bash
bash hpc/submit.sh --parcels-at-once 5   # default — safe
bash hpc/submit.sh --parcels-at-once 8   # faster, higher API load
```

**Sensors / bands / cloud cover** — edit on the cluster then resubmit:
```bash
nano ~/agrixel-fair-dockerV2/cli_app/run_config.yml
bash hpc/submit.sh   # recalculates task count automatically
```

**Walltime** — if 4 h is not enough for parcels with 4 years (48 windows):
```
#SBATCH --time=06:00:00   # in job_array.sh
```

**Check remaining allocation**:
```bash
sinfo-usage csl
```
