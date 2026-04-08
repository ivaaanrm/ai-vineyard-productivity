#!/usr/bin/env python3
"""Dataset pipeline runner for HPC (Calcula cluster at UPC).

Mirrors hpc/run_parcel.py from the agrixel downloader, but for the
post-processing step: reads zarr cubes produced by agrixel and computes
per-parcel temporal stats.

Three modes
-----------
  --parcel-index N  Process one parcel (1-based row in aoi_table.csv) and
                    write a partial CSV to <output-dir>/partials/.
                    Use with SLURM array jobs (set N = $SLURM_ARRAY_TASK_ID).

  --merge           Concatenate all partial CSVs in <output-dir>/partials/
                    into the final dataset.csv.  Run after all array tasks finish.

  (neither)         Process all parcels sequentially in one shot.
                    Useful for small runs and local testing.

Usage
-----
  # Single parcel (SLURM array task):
  python src/dataset/run_calcula.py --parcel-index $SLURM_ARRAY_TASK_ID \\
      --zarr-dir ~/agrixel-fair-dockerV2/data/output/files \\
      --output-dir ~/experiments/data

  # Merge after all tasks finish:
  python src/dataset/run_calcula.py --merge \\
      --output-dir ~/experiments/data

  # All parcels in one shot (local / sequential fallback):
  python src/dataset/run_calcula.py \\
      --zarr-dir ~/agrixel-fair-dockerV2/data/output/files \\
      --output-dir ~/experiments/data

  # Dry run (shows which parcel would be processed, no I/O):
  python src/dataset/run_calcula.py --parcel-index 1 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Allow running from anywhere: add repo root so `from src.xxx import ...` works.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402 — must come after sys.path patch

from src.dataset.config import DatasetConfig  # noqa: E402
from src.dataset.pipeline import make_pipeline_from_config  # noqa: E402
from src.utils.export import export  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = ROOT / "src/config/dataset.yml"
DEFAULT_ZARR_DIR = ROOT / "agrixel-fair-dockerV2/data/output/files"
DEFAULT_OUTPUT_DIR = ROOT / "experiments/data"

PARTIALS_SUBDIR = "partials"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(config_path: Path, aoi_table: Path | None) -> DatasetConfig:
    cfg = DatasetConfig.from_yaml(config_path)
    if aoi_table:
        cfg.paths["aoi_table"] = str(aoi_table)
    return cfg


def _load_aoi_table(cfg: DatasetConfig) -> pd.DataFrame:
    path = cfg.paths.get("aoi_table")
    if not path:
        raise ValueError("No 'aoi_table' key found in config paths.")
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"aoi_table not found: {p}")
    return pd.read_csv(p)


def _parcel_csv_name(parcel_id: str) -> str:
    return f"dataset_{parcel_id}.csv"


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_one(
    parcel_index: int,
    zarr_dir: Path,
    config_path: Path,
    aoi_table: Path | None,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Process a single parcel (1-based index) and write a partial CSV."""
    cfg = _load_config(config_path, aoi_table)
    df_parcels = _load_aoi_table(cfg)
    n = len(df_parcels)
    idx = parcel_index - 1

    if idx < 0 or idx >= n:
        print(f"ERROR: --parcel-index {parcel_index} out of range (1–{n})", file=sys.stderr)
        sys.exit(1)

    row = df_parcels.iloc[idx]
    parcel_id = row["parcel_id"]
    geometry = row.get("parcel_geometry")

    print(f"\n{'='*60}")
    print(f"  PARCEL {parcel_index}/{n}: {parcel_id}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] would process {parcel_id} and write to {output_dir}/partials/")
        return

    pipeline = make_pipeline_from_config(zarr_dir, cfg)

    sensors = list(cfg.sensors.keys())
    geometries = {parcel_id: geometry} if geometry is not None else None

    result = pipeline.execute(
        parcel_keys=[parcel_id],
        sensors=sensors,
        geometries=geometries,
    )

    if result.empty:
        print(f"  WARNING: no data produced for {parcel_id} — skipping write")
        return

    partials_dir = output_dir / PARTIALS_SUBDIR
    partials_dir.mkdir(parents=True, exist_ok=True)
    out_path = partials_dir / _parcel_csv_name(parcel_id)
    result.to_csv(out_path, index=False)
    print(f"  Written: {out_path}  ({len(result)} rows)")


def run_all(
    zarr_dir: Path,
    config_path: Path,
    aoi_table: Path | None,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Process all parcels sequentially and write a single dataset.csv."""
    cfg = _load_config(config_path, aoi_table)
    df_parcels = _load_aoi_table(cfg)
    n = len(df_parcels)

    print(f"\n{'='*60}")
    print(f"  Running all {n} parcels sequentially")
    print(f"  zarr-dir  : {zarr_dir}")
    print(f"  output-dir: {output_dir}")
    print(f"{'='*60}\n")

    if dry_run:
        print(f"  [DRY RUN] would process {n} parcels and write dataset.csv")
        return

    parcel_ids = df_parcels["parcel_id"].tolist()
    geometries = (
        dict(zip(df_parcels["parcel_id"], df_parcels["parcel_geometry"]))
        if "parcel_geometry" in df_parcels.columns
        else None
    )

    pipeline = make_pipeline_from_config(zarr_dir, cfg)
    sensors = list(cfg.sensors.keys())

    result = pipeline.execute(
        parcel_keys=parcel_ids,
        sensors=sensors,
        geometries=geometries,
    )

    if result.empty:
        print("WARNING: pipeline produced no data — nothing to export", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    export(round(result, 4), output_dir=output_dir, name="dataset", config=cfg)
    print(f"\nDataset written to {output_dir}/dataset.csv  ({len(result)} rows)")


def run_merge(output_dir: Path, config_path: Path) -> None:
    """Merge all partial CSVs in <output_dir>/partials/ into dataset.csv."""
    partials_dir = output_dir / PARTIALS_SUBDIR
    if not partials_dir.exists():
        print(f"ERROR: partials directory not found: {partials_dir}", file=sys.stderr)
        sys.exit(1)

    partial_files = sorted(partials_dir.glob("dataset_*.csv"))
    if not partial_files:
        print(f"ERROR: no partial CSVs found in {partials_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(partial_files)} partial CSVs from {partials_dir}")

    frames = [pd.read_csv(p) for p in partial_files]
    merged = pd.concat(frames, ignore_index=True)

    cfg = DatasetConfig.from_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    export(round(merged, 4), output_dir=output_dir, name="dataset", config=cfg)
    print(f"Merged dataset written to {output_dir}/dataset.csv  ({len(merged)} rows)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset pipeline runner for Calcula HPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--parcel-index", type=int, default=None, metavar="N",
        help="1-based parcel index (use $SLURM_ARRAY_TASK_ID for array jobs).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all partial CSVs in <output-dir>/partials/ into dataset.csv.",
    )
    parser.add_argument(
        "--zarr-dir", type=Path, default=DEFAULT_ZARR_DIR,
        help=f"Base directory containing sensor/parcel/cube.zarr trees. "
             f"Default: {DEFAULT_ZARR_DIR}",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to dataset.yml config file. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--aoi-table", type=Path, default=None,
        help="Override the aoi_table path from the config YAML.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output CSV(s). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without reading/writing any data.",
    )
    args = parser.parse_args()

    if args.merge and args.parcel_index is not None:
        parser.error("--merge and --parcel-index are mutually exclusive.")

    if args.merge:
        run_merge(args.output_dir, args.config)
    elif args.parcel_index is not None:
        run_one(
            parcel_index=args.parcel_index,
            zarr_dir=args.zarr_dir,
            config_path=args.config,
            aoi_table=args.aoi_table,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    else:
        run_all(
            zarr_dir=args.zarr_dir,
            config_path=args.config,
            aoi_table=args.aoi_table,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
