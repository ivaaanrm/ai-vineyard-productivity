"""Docker execution and parallel orchestration for CLI runs.

Execution order: parcel → sensors (parallel) → time windows → cleanup.
"""

from __future__ import annotations

import ast
import calendar
import csv
import json
import random
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from .cleanup import cleanup_parcel_dir
from .config import (
    ERA5Config,
    MODISConfig,
    RunConfig,
    S1Config,
    S2Config,
    S3Config,
)

IMAGE_NAME = "agrixel-fair:latest"

# Global shutdown event — set to cancel all running work
_shutdown = threading.Event()

# Track running Docker containers so we can kill them on shutdown
_active_containers: list[subprocess.Popen] = []
_containers_lock = threading.Lock()

# AOI table column conventions (match the main pipeline)
AOI_ID_COLUMN = "parcel_id"

ZARR_CLEANUP_SENSORS = {"S1", "S2", "S3", "MODIS", "ERA5"}

SENSOR_FOLDER_NAMES = {
    "S1": "SENTINEL-1",
    "S2": "SENTINEL-2",
    "S3": "SENTINEL-3",
    "MODIS": "MODIS",
    "ERA5": "ERA5",
}


def _parse_years(raw: str) -> list[int]:
    """Parse a years column value like '[2021, 2022, 2023]' into a list of ints."""
    return sorted(int(y) for y in ast.literal_eval(raw))


def generate_yearly_windows(years: list[int]) -> list[tuple[date, date]]:
    """Generate one window per month (1st-end) for each year.

    Returns at most 12 windows per year, sorted chronologically.
    """
    windows: list[tuple[date, date]] = []
    for year in sorted(years):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            windows.append((date(year, month, 1), date(year, month, last_day)))
    return windows


# ── Parcels ──────────────────────────────────────────────────────────────


def _read_parcels(data_dir: Path) -> list[dict[str, str]]:
    """Read all parcel rows from aoi_table.csv."""
    csv_path = data_dir / "input" / "aoi_table.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_single_parcel_csv(parcel_row: dict[str, str], input_dir: Path) -> Path:
    """Write a temp CSV with a single parcel row for Docker to consume."""
    input_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=input_dir,
        prefix="parcel_",
        suffix=".csv",
        mode="w",
        newline="",
        delete=False,
    )
    writer = csv.DictWriter(tmp, fieldnames=parcel_row.keys())
    writer.writeheader()
    writer.writerow(parcel_row)
    tmp.close()
    return Path(tmp.name)


# ── Params & Docker ─────────────────────────────────────────────────────


def _build_params(
    sensor_key: str,
    sensor_cfg: Any,
    date_from: str,
    date_to: str,
    run_cfg: RunConfig,
    max_items: int | None,
    parcel_csv_name: str,
) -> dict:
    """Build a params dict for the Docker runner."""
    params: dict = {
        "sensor": sensor_key,
        "date_from": date_from,
        "date_to": date_to,
        "buffer_m": run_cfg.buffer_m,
        "max_items": max_items,
        "target_res_m": run_cfg.target_res_m,
        "token_ttl_days": 7,
        "enable_crop_links": True,
        "out_dir": "/workspace/data/output/xml",
        "download_dir": "/workspace/data/output/files",
        "aoi_table": {
            "path": f"/workspace/data/input/{parcel_csv_name}",
            "id_column": "parcel_id",
            "geometry_column": "geometry",
            "geometry_format": "wkt",
            "geometry_epsg": 3857,
        },
    }

    if isinstance(sensor_cfg, S2Config):
        params["select_products_s2"] = sensor_cfg.bands
        params["cloud_cover_lte"] = sensor_cfg.cloud_cover
        params["save_ndvi"] = "NDVI" in sensor_cfg.bands
    elif isinstance(sensor_cfg, S1Config):
        params["select_products_s1"] = sensor_cfg.polarizations
    elif isinstance(sensor_cfg, S3Config):
        params["select_products_s3"] = sensor_cfg.products
    elif isinstance(sensor_cfg, MODISConfig):
        params["select_products_modis"] = sensor_cfg.products
    elif isinstance(sensor_cfg, ERA5Config):
        params["variable"] = sensor_cfg.variable
        params["daily_agg"] = sensor_cfg.daily_agg
        params["data_format"] = "netcdf"

    return params


def _write_params_file(params: dict, input_dir: Path) -> Path:
    """Write params dict to a temporary JSON file."""
    input_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=input_dir,
        prefix=f"params_cli_{params["sensor"]}",
        suffix=".json",
        mode="w",
        delete=False,
    )
    json.dump(params, tmp, indent=2)
    tmp.close()
    return Path(tmp.name)


def _env_flags_for_sensor(sensor_key: str, env: dict[str, str]) -> list[str]:
    """Return Docker -e flags for sensor-specific credentials."""
    flags: list[str] = []
    if sensor_key == "ERA5":
        flags += ["-e", "CDSAPI_URL=https://cds.climate.copernicus.eu/api"]
        if "ERA5_APIKEY" in env:
            flags += ["-e", f"CDSAPI_KEY={env['ERA5_APIKEY']}"]
    elif sensor_key == "MODIS":
        if "MODIS_USERNAME" in env:
            flags += ["-e", f"EARTHDATA_USERNAME={env['MODIS_USERNAME']}"]
        if "MODIS_PASSWORD" in env:
            flags += ["-e", f"EARTHDATA_PASSWORD={env['MODIS_PASSWORD']}"]
    return flags


def _run_docker(
    params_file: Path, data_dir: Path, sensor_key: str, env: dict[str, str],
) -> tuple[int, list[str]]:
    """Launch a Docker container and return (exit_code, log_lines).

    Supports graceful cancellation via the module-level _shutdown event.
    Reads stdout and stderr concurrently to avoid pipe-buffer deadlocks.
    """
    if _shutdown.is_set():
        return -1, ["Cancelled before start"]

    container_params = "/workspace/data/input/" + params_file.name
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{data_dir}:/workspace/data",
        "-e", f"PARAMS_FILE={container_params}",
        *_env_flags_for_sensor(sensor_key, env),
        IMAGE_NAME,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    with _containers_lock:
        _active_containers.append(proc)

    log_lines: list[str] = []
    log_lock = threading.Lock()

    def _read_stream(stream, prefix: str = "") -> None:
        for line in stream:
            stripped = line.rstrip("\n")
            with log_lock:
                log_lines.append(f"{prefix}{stripped}")
            print(f"{prefix}{stripped}")

    # Read stdout and stderr in parallel threads to avoid deadlock
    stderr_thread = threading.Thread(
        target=_read_stream, args=(proc.stderr, "STDERR: "), daemon=True,
    )
    stderr_thread.start()
    _read_stream(proc.stdout)  # stdout in current thread
    stderr_thread.join(timeout=10)

    proc.wait()

    with _containers_lock:
        if proc in _active_containers:
            _active_containers.remove(proc)

    return proc.returncode, log_lines


# ── Per-sensor execution (all windows for one parcel) ───────────────────


def _run_sensor(
    sensor_key: str,
    sensor_cfg: Any,
    run_cfg: RunConfig,
    data_dir: Path,
    dry_run: bool,
    env: dict[str, str],
    parcel_csv_name: str,
    parcel_id: str,
    years: list[int],
) -> dict:
    """Run all date windows for a single sensor on a single parcel."""
    print(f"    [{parcel_id}] Starting {sensor_key}")

    if run_cfg.equidistant:
        windows = generate_yearly_windows(years)
        max_items = 1
    else:
        # One full-year window per year
        windows = [(date(y, 1, 1), date(y, 12, 31)) for y in sorted(years)]
        max_items = None

    input_dir = data_dir / "input"
    window_results = []

    for i, (win_start, win_end) in enumerate(windows):
        if _shutdown.is_set():
            print(f"    [{parcel_id}/{sensor_key}] Shutdown — skipping remaining windows.")
            break

        label = f"[{parcel_id}/{sensor_key}] window {i+1}/{len(windows)}: {win_start} -> {win_end}"
        print(f"    {label}")

        params = _build_params(
            sensor_key, sensor_cfg,
            str(win_start), str(win_end),
            run_cfg, max_items, parcel_csv_name,
        )

        if dry_run:
            print(f"      [DRY RUN] {sensor_key} {win_start}->{win_end}")
            window_results.append({
                "date_from": str(win_start),
                "date_to": str(win_end),
                "exit_code": 0,
                "dry_run": True,
            })
            continue

        params_file = _write_params_file(params, input_dir)
        exit_code, log_lines = _run_docker(params_file, data_dir, sensor_key, env)
        params_file.unlink(missing_ok=True)

        window_results.append({
            "date_from": str(win_start),
            "date_to": str(win_end),
            "exit_code": exit_code,
        })

        if exit_code != 0:
            print(f"      WARNING: Docker exited with code {exit_code}")

    # Per-parcel cleanup: keep only cube.zarr
    cleanup_stats = None
    if sensor_key in ZARR_CLEANUP_SENSORS and not dry_run:
        sensor_folder = SENSOR_FOLDER_NAMES[sensor_key]
        parcel_dir = data_dir / "output" / "files" / sensor_folder / parcel_id
        if parcel_dir.is_dir():
            cleanup_stats = cleanup_parcel_dir(parcel_dir)
            print(f"    [{parcel_id}/{sensor_key}] Cleanup: {cleanup_stats}")

    succeeded = sum(1 for w in window_results if w.get("exit_code") == 0)
    failed = len(window_results) - succeeded

    result = {
        "status": "completed" if failed == 0 else ("partial" if succeeded > 0 else "failed"),
        "windows_total": len(window_results),
        "windows_succeeded": succeeded,
        "windows_failed": failed,
    }
    if cleanup_stats is not None:
        result["cleanup"] = cleanup_stats

    return result


# ── Shutdown ─────────────────────────────────────────────────────────────


def request_shutdown() -> None:
    """Signal all running work to stop and kill active Docker containers."""
    _shutdown.set()
    with _containers_lock:
        for proc in _active_containers:
            try:
                proc.terminate()
            except OSError:
                pass
    print("\nShutdown requested — stopping all containers...")


# ── Main orchestrator ────────────────────────────────────────────────────


def _slice_parcels(
    parcels: list[dict[str, str]], batch_index: int, batch_total: int,
) -> list[dict[str, str]]:
    """Return the slice of parcels for batch batch_index out of batch_total."""
    n = len(parcels)
    chunk_size = n // batch_total
    remainder = n % batch_total
    # Distribute remainder across the first `remainder` batches
    start = chunk_size * (batch_index - 1) + min(batch_index - 1, remainder)
    end = start + chunk_size + (1 if batch_index <= remainder else 0)
    return parcels[start:end]


def run_all(
    run_cfg: RunConfig,
    data_dir: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    batch_index: int | None = None,
    batch_total: int | None = None,
) -> dict[str, dict[str, dict]]:
    """Run all enabled sensors for each parcel.

    Args:
        batch_index: 1-based batch number (e.g. 1 of 3).
        batch_total: total number of batches.

    Returns: {parcel_id: {sensor_key: result_dict}}
    """
    _shutdown.clear()

    if env is None:
        env = {}
    enabled = run_cfg.enabled_sensors()
    if not enabled:
        print("No sensors enabled in config.")
        return {}

    parcels = _read_parcels(data_dir)
    random.shuffle(parcels)
    total_parcels = len(parcels)

    if batch_index is not None and batch_total is not None:
        parcels = _slice_parcels(parcels, batch_index, batch_total)
        print(f"Batch {batch_index}/{batch_total}: {len(parcels)} parcels (of {total_parcels} total)")
    else:
        print(f"Parcels: {total_parcels}")
    print(f"Sensors: {list(enabled.keys())}")
    print(f"Max parallel: {run_cfg.max_parallel}")
    print(f"Equidistant: {run_cfg.equidistant} (1 sample/month)")

    all_results: dict[str, dict[str, dict]] = {}

    for idx, parcel_row in enumerate(parcels):
        if _shutdown.is_set():
            print("\nShutdown — skipping remaining parcels.")
            break

        parcel_id = parcel_row[AOI_ID_COLUMN]
        years = _parse_years(parcel_row["years"])
        print(f"\n{'='*60}")
        print(f"  PARCEL {idx+1}/{len(parcels)}: {parcel_id}  years={years}")
        print(f"{'='*60}")

        # Write a single-row CSV so Docker processes only this parcel
        parcel_csv = _write_single_parcel_csv(parcel_row, data_dir / "input")

        sensor_results: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=min(run_cfg.max_parallel, len(enabled))) as pool:
            futures = {
                pool.submit(
                    _run_sensor, sensor_key, sensor_cfg, run_cfg, data_dir,
                    dry_run, env, parcel_csv.name, parcel_id, years,
                ): sensor_key
                for sensor_key, sensor_cfg in enabled.items()
            }

            try:
                for future in as_completed(futures):
                    sensor_key = futures[future]
                    try:
                        sensor_results[sensor_key] = future.result()
                    except Exception as exc:
                        print(f"    ERROR: {sensor_key} failed: {exc}")
                        sensor_results[sensor_key] = {
                            "status": "error",
                            "error": str(exc),
                        }
            except KeyboardInterrupt:
                request_shutdown()
                for future in futures:
                    future.cancel()
                for future, sk in futures.items():
                    if sk not in sensor_results:
                        sensor_results[sk] = {"status": "cancelled"}

        parcel_csv.unlink(missing_ok=True)
        all_results[parcel_id] = sensor_results

    return all_results
