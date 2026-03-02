"""Docker container management for agrixel-fair downloads."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

_DOCKER_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _DOCKER_ROOT / "data"
IMAGE_NAME = "agrixel-fair"


def build_params(
    *,
    sensor: str,
    date_from: str,
    date_to: str,
    parcel_ids: list[str] | None = None,
    bands_s2: list[str] | None = None,
    cloud_cover: int = 20,
    pols_s1: list[str] | None = None,
    buffer_m: int = 100,
    target_res_m: int | None = None,
    max_items: int | None = None,
) -> dict:
    """Build a params dict suitable for the agrixel-fair runner."""
    params: dict = {
        "sensor": sensor,
        "date_from": date_from,
        "date_to": date_to,
        "buffer_m": buffer_m,
        "max_items": max_items,
        "target_res_m": target_res_m,
        "token_ttl_days": 7,
        "enable_crop_links": True,
        "out_dir": "/workspace/data/output/xml",
        "download_dir": "/workspace/data/output/files",
    }

    # AOI: always batch mode via aoi_table
    if parcel_ids:
        params["aoi_table"] = {
            "path": "/workspace/data/input/aoi_table.csv",
            "id_column": "parcel_id",
            "geometry_column": "geometry",
            "geometry_format": "wkt",
            "geometry_epsg": 3857,
            "filter_ids": parcel_ids,
        }
    else:
        params["aoi_table"] = {
            "path": "/workspace/data/input/aoi_table.csv",
            "id_column": "parcel_id",
            "geometry_column": "geometry",
            "geometry_format": "wkt",
            "geometry_epsg": 3857,
        }

    # Sensor-specific
    if sensor == "S2":
        params["select_products_s2"] = bands_s2 or ["B04", "B08", "NDVI"]
        params["cloud_cover_lte"] = cloud_cover
        params["save_ndvi"] = "NDVI" in (bands_s2 or ["NDVI"])
    elif sensor == "S1":
        params["select_products_s1"] = pols_s1 or ["VV", "VH"]

    return params


def write_params_file(params: dict) -> Path:
    """Write params dict to a temporary JSON file inside data/input/."""
    input_dir = DATA_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=input_dir,
        prefix="params_streamlit_",
        suffix=".json",
        mode="w",
        delete=False,
    )
    json.dump(params, tmp, indent=2)
    tmp.close()
    return Path(tmp.name)


def run_docker(params_file: Path):
    """Launch the Docker container and yield stdout/stderr lines.

    Yields (stream, line) tuples where stream is 'stdout' or 'stderr'.
    Returns the process exit code via a final ('exit', code) tuple.
    """
    container_params = "/workspace/data/input/" + params_file.name
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{DATA_DIR}:/workspace/data",
        "-e",
        f"PARAMS_FILE={container_params}",
        IMAGE_NAME,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    # Read stdout line-by-line (stderr is collected after)
    for line in proc.stdout:
        yield ("stdout", line.rstrip("\n"))

    proc.wait()

    for line in proc.stderr:
        yield ("stderr", line.rstrip("\n"))

    yield ("exit", str(proc.returncode))
