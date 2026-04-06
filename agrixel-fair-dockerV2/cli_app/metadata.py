"""Generate metadata_run.json for reproducibility and summary."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import RunConfig


def build_metadata(
    run_cfg: RunConfig,
    parcel_results: dict[str, dict[str, dict]],
    started_at: datetime,
    finished_at: datetime,
    dry_run: bool = False,
) -> dict:
    """Build the metadata dict for the run.

    Args:
        parcel_results: {parcel_id: {sensor_key: result_dict}}
    """
    return {
        "run_id": started_at.strftime("%Y%m%d_%H%M%S"),
        "config": json.loads(run_cfg.model_dump_json()),
        "dry_run": dry_run,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 1),
        "parcels_total": len(parcel_results),
        "parcels": parcel_results,
    }


def write_metadata(metadata: dict, output_dir: Path) -> Path:
    """Write metadata_run.json to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata_run.json"
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path
