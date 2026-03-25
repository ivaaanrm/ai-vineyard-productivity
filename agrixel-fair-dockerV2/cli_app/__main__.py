"""CLI entry point: uv run python -m cli_app config.yml

Run from the agrixel-fair-dockerV2 directory:
    cd agrixel-fair-dockerV2
    uv run python -m cli_app cli_app/run_config.yml --dry-run
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .config import load_config, load_env
from .metadata import build_metadata, write_metadata
from .runner import run_all

app = typer.Typer(help="CLI tool for batch satellite data downloads via Docker.")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to YAML config file.", exists=True),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Data directory (contains input/ and output/). Defaults to ../data relative to cli_app.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would run without launching Docker."),
) -> None:
    """Run satellite data downloads based on a YAML config."""
    cfg = load_config(config_path)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"

    output_dir = output_dir.resolve()

    # Load .env for sensor credentials (ERA5, MODIS)
    env_path = Path(__file__).resolve().parent / ".env"
    env = load_env(env_path)

    if not dry_run and not (output_dir / "input" / "aoi_table.csv").exists():
        typer.echo(f"ERROR: aoi_table.csv not found at {output_dir / 'input' / 'aoi_table.csv'}")
        raise typer.Exit(code=1)

    typer.echo(f"Config: {config_path}")
    typer.echo(f"Data dir: {output_dir}")
    if env:
        typer.echo(f"Env: loaded {len(env)} vars from {env_path}")
    if dry_run:
        typer.echo("DRY RUN MODE")
    typer.echo("")

    started_at = datetime.now()
    sensor_results = run_all(cfg, output_dir, dry_run=dry_run, env=env)
    finished_at = datetime.now()

    metadata = build_metadata(cfg, sensor_results, started_at, finished_at, dry_run=dry_run)
    meta_path = write_metadata(metadata, output_dir / "output")

    typer.echo(f"\nMetadata written to: {meta_path}")

    # Summary
    typer.echo("\n--- Summary ---")
    for sensor_key, result in sensor_results.items():
        status = result.get("status", "unknown")
        succeeded = result.get("windows_succeeded", 0)
        total = result.get("windows_total", 0)
        typer.echo(f"  {sensor_key}: {status} ({succeeded}/{total} windows)")
        if "cleanup" in result:
            c = result["cleanup"]
            typer.echo(f"    cleanup: {c['dirs_removed']} dirs removed, {c['zarr_kept']} zarr kept")


if __name__ == "__main__":
    app()
