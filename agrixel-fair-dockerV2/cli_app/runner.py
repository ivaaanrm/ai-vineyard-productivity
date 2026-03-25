"""Docker execution and parallel orchestration for CLI runs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .cleanup import cleanup_sensor
from .config import (
    ERA5Config,
    MODISConfig,
    RunConfig,
    S1Config,
    S2Config,
    S3Config,
)

IMAGE_NAME = "agrixel-fair"

ZARR_CLEANUP_SENSORS = {"S1", "S2", "S3"}


def split_date_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    """Divide [start, end] into n equal sub-windows. Last window absorbs remainder."""
    total_days = (end - start).days + 1
    window_size = total_days // n
    windows = []
    for i in range(n):
        win_start = start + timedelta(days=i * window_size)
        if i == n - 1:
            win_end = end
        else:
            win_end = start + timedelta(days=(i + 1) * window_size - 1)
        windows.append((win_start, win_end))
    return windows


def _build_params(
    sensor_key: str,
    sensor_cfg: Any,
    date_from: str,
    date_to: str,
    run_cfg: RunConfig,
    max_items: int | None,
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
            "path": "/workspace/data/input/aoi_table.csv",
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
        prefix="params_cli_",
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
    """Launch a Docker container and return (exit_code, log_lines)."""
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

    log_lines: list[str] = []
    assert proc.stdout is not None
    assert proc.stderr is not None

    for line in proc.stdout:
        stripped = line.rstrip("\n")
        log_lines.append(stripped)
        print(stripped)

    proc.wait()

    for line in proc.stderr:
        stripped = line.rstrip("\n")
        log_lines.append(f"STDERR: {stripped}")
        print(f"STDERR: {stripped}")

    return proc.returncode, log_lines


SENSOR_FOLDER_NAMES = {
    "S1": "SENTINEL-1",
    "S2": "SENTINEL-2",
    "S3": "SENTINEL-3",
    "MODIS": "MODIS",
    "ERA5": "ERA5",
}


def _run_sensor(
    sensor_key: str,
    sensor_cfg: Any,
    run_cfg: RunConfig,
    data_dir: Path,
    dry_run: bool,
    env: dict[str, str],
) -> dict:
    """Run all date windows for a single sensor. Returns result dict for metadata."""
    print(f"\n{'='*60}")
    print(f"  SENSOR: {sensor_key}")
    print(f"{'='*60}")

    if run_cfg.equidistant:
        windows = split_date_range(run_cfg.start_date, run_cfg.end_date, run_cfg.num_samples)
        max_items = 1
    else:
        windows = [(run_cfg.start_date, run_cfg.end_date)]
        max_items = None

    input_dir = data_dir / "input"
    window_results = []

    for i, (win_start, win_end) in enumerate(windows):
        label = f"[{sensor_key}] window {i+1}/{len(windows)}: {win_start} -> {win_end}"
        print(f"\n--- {label} ---")

        params = _build_params(
            sensor_key, sensor_cfg,
            str(win_start), str(win_end),
            run_cfg, max_items,
        )

        if dry_run:
            print(f"  [DRY RUN] Would run Docker with params:")
            print(f"  {json.dumps(params, indent=2, default=str)}")
            window_results.append({
                "date_from": str(win_start),
                "date_to": str(win_end),
                "exit_code": 0,
                "dry_run": True,
            })
            continue

        params_file = _write_params_file(params, input_dir)
        exit_code, log_lines = _run_docker(params_file, data_dir, sensor_key, env)

        # Clean up temp params file
        params_file.unlink(missing_ok=True)

        window_results.append({
            "date_from": str(win_start),
            "date_to": str(win_end),
            "exit_code": exit_code,
        })

        if exit_code != 0:
            print(f"  WARNING: Docker exited with code {exit_code}")

    # Cleanup for S1/S2/S3: keep only cube.zarr
    cleanup_stats = None
    if sensor_key in ZARR_CLEANUP_SENSORS and not dry_run:
        sensor_folder = SENSOR_FOLDER_NAMES[sensor_key]
        files_dir = data_dir / "output" / "files" / sensor_folder
        if files_dir.is_dir():
            cleanup_stats = cleanup_sensor(files_dir)
            print(f"  Cleanup {sensor_key}: {cleanup_stats}")

    succeeded = sum(1 for w in window_results if w.get("exit_code") == 0)
    failed = len(window_results) - succeeded

    result = {
        "status": "completed" if failed == 0 else ("partial" if succeeded > 0 else "failed"),
        "windows": window_results,
        "windows_total": len(window_results),
        "windows_succeeded": succeeded,
        "windows_failed": failed,
    }
    if cleanup_stats is not None:
        result["cleanup"] = cleanup_stats

    return result


def run_all(
    run_cfg: RunConfig,
    data_dir: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Run all enabled sensors in parallel. Returns per-sensor result dicts."""
    if env is None:
        env = {}
    enabled = run_cfg.enabled_sensors()
    if not enabled:
        print("No sensors enabled in config.")
        return {}

    print(f"Sensors to run: {list(enabled.keys())}")
    print(f"Max parallel: {run_cfg.max_parallel}")
    print(f"Date range: {run_cfg.start_date} -> {run_cfg.end_date}")
    if run_cfg.equidistant:
        print(f"Equidistant: {run_cfg.num_samples} samples")

    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=min(run_cfg.max_parallel, len(enabled))) as pool:
        futures = {
            pool.submit(
                _run_sensor, sensor_key, sensor_cfg, run_cfg, data_dir, dry_run, env
            ): sensor_key
            for sensor_key, sensor_cfg in enabled.items()
        }

        for future in as_completed(futures):
            sensor_key = futures[future]
            try:
                results[sensor_key] = future.result()
            except Exception as exc:
                print(f"ERROR: Sensor {sensor_key} failed with exception: {exc}")
                results[sensor_key] = {
                    "status": "error",
                    "error": str(exc),
                }

    return results
