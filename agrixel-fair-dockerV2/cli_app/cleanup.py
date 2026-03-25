"""Cleanup utilities: keep only cube.zarr for S1/S2/S3 parcel directories."""

from __future__ import annotations

import shutil
from pathlib import Path


def cleanup_sensor(sensor_files_dir: Path) -> dict:
    """Remove everything except cube.zarr inside each parcel subdirectory.

    Args:
        sensor_files_dir: e.g. data/output/files/SENTINEL-2

    Returns:
        Stats dict with files_removed, dirs_removed, zarr_kept counts.
    """
    files_removed = 0
    dirs_removed = 0
    zarr_kept = 0

    for parcel_dir in sorted(sensor_files_dir.iterdir()):
        if not parcel_dir.is_dir():
            continue

        zarr_path = parcel_dir / "cube.zarr"
        has_zarr = zarr_path.is_dir()

        if has_zarr:
            zarr_kept += 1

        for item in sorted(parcel_dir.iterdir()):
            if item.name == "cube.zarr":
                continue
            if item.is_dir():
                shutil.rmtree(item)
                dirs_removed += 1
            else:
                item.unlink()
                files_removed += 1

    return {
        "files_removed": files_removed,
        "dirs_removed": dirs_removed,
        "zarr_kept": zarr_kept,
    }
