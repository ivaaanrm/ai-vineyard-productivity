"""CLI entry point: uv run python -m cli_app config.yml

Run from the agrixel-fair-dockerV2 directory:
    cd agrixel-fair-dockerV2
    uv run python -m cli_app cli_app/run_config.yml --dry-run
"""

from __future__ import annotations

import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .config import load_config, load_env
from .metadata import build_metadata, write_metadata
from .runner import request_shutdown, run_all

app = typer.Typer(help="CLI tool for batch satellite data downloads via Docker.")


def _install_signal_handlers() -> None:
    """Install handlers so Ctrl+C / SIGTERM stops Docker containers gracefully."""
    def _handler(signum, frame):
        request_shutdown()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _parse_batch(value: str) -> tuple[int, int]:
    """Parse 'N/M' into (batch_index, batch_total)."""
    try:
        index, total = value.split("/")
        index, total = int(index), int(total)
    except ValueError:
        raise typer.BadParameter("Must be N/M, e.g. 1/3")
    if total < 1 or index < 1 or index > total:
        raise typer.BadParameter(f"Need 1 <= N <= M, got {index}/{total}")
    return index, total


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to YAML config file.", exists=True),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Data directory (contains input/ and output/). Defaults to ../data relative to cli_app.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would run without launching Docker."),
    batch: Optional[str] = typer.Option(
        None,
        "--batch", "-b",
        help="Process a slice of parcels: N/M (e.g. 1/3 = first third, 2/3 = second third).",
    ),
) -> None:
    """Run satellite data downloads based on a YAML config."""
    cfg = load_config(config_path)

    batch_index, batch_total = _parse_batch(batch) if batch else (None, None)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"

    output_dir = output_dir.resolve()

    # Load .env for sensor credentials (ERA5, MODIS)
    env_path = Path(__file__).resolve().parent / ".env"
    env = load_env(env_path)

    if not (output_dir / "input" / "aoi_table.csv").exists():
        typer.echo(f"ERROR: aoi_table.csv not found at {output_dir / 'input' / 'aoi_table.csv'}")
        raise typer.Exit(code=1)

    _install_signal_handlers()

    typer.echo(f"Config: {config_path}")
    typer.echo(f"Data dir: {output_dir}")
    if env:
        typer.echo(f"Env: loaded {len(env)} vars from {env_path}")
    if dry_run:
        typer.echo("DRY RUN MODE")
    if batch:
        typer.echo(f"Batch: {batch}")
    typer.echo("Press Ctrl+C to stop all running downloads.\n")

    started_at = datetime.now()
    parcel_results = run_all(
        cfg, output_dir, dry_run=dry_run, env=env,
        batch_index=batch_index, batch_total=batch_total,
    )
    finished_at = datetime.now()

    metadata = build_metadata(cfg, parcel_results, started_at, finished_at, dry_run=dry_run)
    meta_path = write_metadata(metadata, output_dir / "output")

    typer.echo(f"\nMetadata written to: {meta_path}")

    # Summary
    typer.echo(f"\n--- Summary ({len(parcel_results)} parcels) ---")
    for parcel_id, sensors in parcel_results.items():
        parts = []
        for sensor_key, result in sensors.items():
            status = result.get("status", "unknown")
            succeeded = result.get("windows_succeeded", 0)
            total = result.get("windows_total", 0)
            parts.append(f"{sensor_key}:{status}({succeeded}/{total})")
        typer.echo(f"  {parcel_id}: {', '.join(parts)}")


if __name__ == "__main__":
    app()
