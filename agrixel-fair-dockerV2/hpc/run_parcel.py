#!/usr/bin/env python3
"""Per-parcel / per-sensor runner for HPC (no Docker).

Each SLURM task calls this script for exactly one parcel + one sensor.
It loops over all monthly windows for that sensor, calls app/runner.py
for each window, then cleans the output directory (keep only cube.zarr).

Usage (called by job_array.sh):
    python hpc/run_parcel.py --index 42 --sensor S2 \\
        --config cli_app/run_config.yml --data-dir /path/to/data

--index is 1-based and maps to a row in aoi_table.csv.
--sensor must match a key enabled in run_config.yml (S1, S2, S3, ERA5, MODIS).
Omit --sensor to run all sensors sequentially (useful for local testing).
"""

from __future__ import annotations

import argparse
import ast
import calendar
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────

SENSOR_FOLDER_NAMES = {
    "S1": "SENTINEL-1",
    "S2": "SENTINEL-2",
    "S3": "SENTINEL-3",
    "MODIS": "MODIS",
    "ERA5": "ERA5",
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_years(raw: str) -> list[int]:
    return sorted(int(y) for y in ast.literal_eval(raw))


def _monthly_windows(years: list[int]) -> list[tuple[date, date]]:
    """One window per month per year (equidistant=True)."""
    windows = []
    for year in sorted(years):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            windows.append((date(year, month, 1), date(year, month, last_day)))
    return windows


def _yearly_windows(years: list[int]) -> list[tuple[date, date]]:
    """One full-year window per year (equidistant=False)."""
    return [(date(y, 1, 1), date(y, 12, 31)) for y in sorted(years)]


# ── Config loading ────────────────────────────────────────────────────────

def _load_run_config(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        raise RuntimeError("PyYAML not available — activate the agrixel-fair conda env first.")


def _enabled_sensors(cfg: dict) -> dict[str, dict]:
    sensors_raw = cfg.get("sensors", {}) or {}
    return {k: v for k, v in sensors_raw.items() if v is not None}


# ── Params builder ────────────────────────────────────────────────────────

def _build_params(
    sensor_key: str,
    sensor_cfg: dict,
    date_from: str,
    date_to: str,
    run_cfg: dict,
    max_items: int | None,
    parcel_csv_path: str,
    output_base: str,
) -> dict:
    params: dict = {
        "sensor": sensor_key,
        "date_from": date_from,
        "date_to": date_to,
        "buffer_m": run_cfg.get("buffer_m", 100),
        "max_items": max_items,
        "target_res_m": run_cfg.get("target_res_m"),
        "token_ttl_days": 7,
        "enable_crop_links": True,
        "out_dir": str(Path(output_base) / "xml"),
        "download_dir": str(Path(output_base) / "files"),
        "aoi_table": {
            "path": parcel_csv_path,
            "id_column": "parcel_id",
            "geometry_column": "geometry",
            "geometry_format": "wkt",
            "geometry_epsg": 3857,
        },
    }

    key = sensor_key.upper()
    if key == "S2":
        params["select_products_s2"] = sensor_cfg.get("bands", ["B04", "B08", "NDVI"])
        params["cloud_cover_lte"] = sensor_cfg.get("cloud_cover", 20)
        params["save_ndvi"] = "NDVI" in params["select_products_s2"]
    elif key == "S1":
        params["select_products_s1"] = sensor_cfg.get("polarizations", ["VV", "VH"])
    elif key == "S3":
        params["select_products_s3"] = sensor_cfg.get("products", ["lst-in"])
    elif key == "MODIS":
        params["select_products_modis"] = sensor_cfg.get("products", ["ET_500m", "PET_500m"])
    elif key == "ERA5":
        params["variable"] = sensor_cfg.get("variable", "tp")
        params["daily_agg"] = sensor_cfg.get("daily_agg", "sum")
        params["data_format"] = "netcdf"

    return params


# ── Cleanup ───────────────────────────────────────────────────────────────

def _cleanup_parcel_sensor(data_dir: Path, sensor_key: str, parcel_id: str) -> dict:
    """Keep only cube.zarr inside the parcel's sensor output directory."""
    sensor_folder = SENSOR_FOLDER_NAMES.get(sensor_key, sensor_key)
    parcel_dir = data_dir / "output" / "files" / sensor_folder / parcel_id

    if not parcel_dir.is_dir():
        return {"files_removed": 0, "dirs_removed": 0, "zarr_kept": 0}

    zarr_kept = 1 if (parcel_dir / "cube.zarr").is_dir() else 0
    files_removed = dirs_removed = 0

    for item in sorted(parcel_dir.iterdir()):
        if item.name == "cube.zarr":
            continue
        if item.is_dir():
            shutil.rmtree(item)
            dirs_removed += 1
        else:
            item.unlink()
            files_removed += 1

    return {"files_removed": files_removed, "dirs_removed": dirs_removed, "zarr_kept": zarr_kept}


# ── Window runner ─────────────────────────────────────────────────────────

def _run_window(
    params: dict,
    input_dir: Path,
    app_dir: Path,
    env_vars: dict[str, str],
    dry_run: bool,
) -> int:
    with tempfile.NamedTemporaryFile(
        dir=input_dir,
        prefix=f"hpc_{params['sensor']}_",
        suffix=".json",
        mode="w",
        delete=False,
    ) as tmp:
        json.dump(params, tmp, indent=2)
        params_path = Path(tmp.name)

    if dry_run:
        print(f"  [DRY RUN] PARAMS_FILE={params_path.name} python app/runner.py")
        params_path.unlink(missing_ok=True)
        return 0

    run_env = {**os.environ, **env_vars, "PARAMS_FILE": str(params_path)}
    result = subprocess.run(
        [sys.executable, str(app_dir / "runner.py")],
        env=run_env,
        cwd=str(app_dir),
    )
    params_path.unlink(missing_ok=True)
    return result.returncode


def _write_parcel_csv(parcel_row: dict, input_dir: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=input_dir,
        prefix="parcel_",
        suffix=".csv",
        mode="w",
        newline="",
        delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=parcel_row.keys())
        writer.writeheader()
        writer.writerow(parcel_row)
        return Path(tmp.name)


# ── Main sensor runner ────────────────────────────────────────────────────

def run_sensor(
    parcel_row: dict,
    sensor_key: str,
    sensor_cfg: dict,
    run_cfg: dict,
    data_dir: Path,
    app_dir: Path,
    env_vars: dict[str, str],
    dry_run: bool,
) -> dict:
    parcel_id = parcel_row["parcel_id"]
    years = _parse_years(parcel_row["years"])
    equidistant = run_cfg.get("equidistant", True)
    output_base = str(data_dir / "output")
    input_dir = data_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if equidistant:
        windows = _monthly_windows(years)
        max_items = 1
    else:
        windows = _yearly_windows(years)
        max_items = None

    print(f"  [{parcel_id}/{sensor_key}] {len(windows)} windows  years={years}")

    parcel_csv = _write_parcel_csv(parcel_row, input_dir)
    succeeded = failed = 0

    try:
        for i, (win_start, win_end) in enumerate(windows):
            print(f"    window {i+1}/{len(windows)}: {win_start} → {win_end}")
            params = _build_params(
                sensor_key, sensor_cfg,
                str(win_start), str(win_end),
                run_cfg, max_items,
                str(parcel_csv),
                output_base,
            )
            rc = _run_window(params, input_dir, app_dir, env_vars, dry_run)
            if rc == 0:
                succeeded += 1
            else:
                failed += 1
                print(f"    WARNING: exit code {rc}")
    finally:
        parcel_csv.unlink(missing_ok=True)

    # Cleanup: keep only cube.zarr
    if not dry_run:
        stats = _cleanup_parcel_sensor(data_dir, sensor_key, parcel_id)
        print(f"  [{parcel_id}/{sensor_key}] Cleanup: {stats}")

    status = "completed" if failed == 0 else ("partial" if succeeded > 0 else "failed")
    return {
        "status": status,
        "windows_total": len(windows),
        "windows_succeeded": succeeded,
        "windows_failed": failed,
    }


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HPC per-parcel/sensor runner (no Docker).")
    parser.add_argument("--index", type=int, required=True,
                        help="1-based parcel index (maps to $SLURM_ARRAY_TASK_ID row in aoi_table.csv).")
    parser.add_argument("--sensor", type=str, default=None,
                        help="Sensor key to run (S1, S2, S3, ERA5, MODIS). "
                             "Omit to run all enabled sensors sequentially.")
    parser.add_argument("--config", type=Path, default=Path("cli_app/run_config.yml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--app-dir", type=Path, default=Path("app"))
    parser.add_argument("--env-file", type=Path, default=Path("hpc/.env"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load credentials
    env_vars: dict[str, str] = {}
    if args.env_file.is_file():
        for line in args.env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if v:
                    env_vars[k.strip()] = v.strip()

    run_cfg = _load_run_config(args.config)
    enabled = _enabled_sensors(run_cfg)

    # Read parcel by 1-based index
    csv_path = args.data_dir / "input" / "aoi_table.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        parcels = list(csv.DictReader(f))

    n = len(parcels)
    idx = args.index - 1
    if idx < 0 or idx >= n:
        print(f"ERROR: index {args.index} out of range (1–{n})", file=sys.stderr)
        sys.exit(1)

    parcel_row = parcels[idx]
    parcel_id = parcel_row["parcel_id"]

    # Filter to the requested sensor (or run all if not specified)
    if args.sensor:
        sensor_key = args.sensor.upper()
        if sensor_key not in enabled:
            print(f"ERROR: sensor '{sensor_key}' not enabled in config. "
                  f"Enabled: {list(enabled.keys())}", file=sys.stderr)
            sys.exit(1)
        sensors_to_run = {sensor_key: enabled[sensor_key]}
    else:
        sensors_to_run = enabled

    print(f"\n{'='*60}")
    print(f"  PARCEL {args.index}/{n}: {parcel_id}  sensor(s): {list(sensors_to_run.keys())}")
    print(f"{'='*60}")

    all_results: dict[str, dict] = {}
    for sensor_key, sensor_cfg in sensors_to_run.items():
        all_results[sensor_key] = run_sensor(
            parcel_row=parcel_row,
            sensor_key=sensor_key,
            sensor_cfg=sensor_cfg,
            run_cfg=run_cfg,
            data_dir=args.data_dir.resolve(),
            app_dir=args.app_dir.resolve(),
            env_vars=env_vars,
            dry_run=args.dry_run,
        )

    print(f"\n{'='*60}")
    print(f"  DONE: {parcel_id}")
    for sk, res in all_results.items():
        print(f"    {sk}: {res['status']} ({res['windows_succeeded']}/{res['windows_total']} windows OK)")
    print(f"{'='*60}\n")

    any_failed = any(r["status"] == "failed" for r in all_results.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
