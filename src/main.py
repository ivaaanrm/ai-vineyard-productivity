"""Full pipeline runner — dataset → features → training.

Usage:
    # Run all steps from an experiment directory
    uv run -m src.main experiments/EXP_20260330

    # Run a single step
    uv run -m src.main experiments/EXP_20260330 --step features
    uv run -m src.main experiments/EXP_20260330 --step training

    # Use default configs (src/config/)
    uv run -m src.main --step training
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = typer.Typer(help="Vineyard productivity pipeline runner")


class Step(str, Enum):
    dataset = "dataset"
    features = "features"
    training = "training"


STEPS = list(Step)


def _resolve_config(experiment_dir: Path | None, step: Step) -> str | None:
    """Find the config file for a step: experiment dir first, then src/config/."""
    if experiment_dir:
        path = experiment_dir / f"{step.value}.yml"
        if path.exists():
            return str(path)
    default = ROOT / "src" / "config" / f"{step.value}.yml"
    if default.exists():
        return str(default)
    return None


def run_dataset(config_path: str) -> None:
    from src.dataset.run import main as dataset_main

    dataset_main(config_path)


def run_features(config_path: str) -> None:
    from src.features.run import main as features_main

    features_main(config_path)


def run_training(config_path: str) -> None:
    from src.train.run import main as training_main

    training_main(config_path)


RUNNERS = {
    Step.dataset: run_dataset,
    Step.features: run_features,
    Step.training: run_training,
}


def run_step(step: Step, config: str) -> None:
    try:
        RUNNERS.get(step)(config)
    except Exception as e:
        typer.echo(e)

@app.command()
def main(
    experiment_dir: Annotated[
        Optional[Path],
        typer.Argument(
            help="Experiment directory with dataset.yml, features.yml, training.yml"
        ),
    ] = None,
    step: Annotated[
        Optional[Step],
        typer.Option(help="Run a single step (default: run all steps)"),
    ] = None,
) -> None:
    """Run the vineyard productivity pipeline."""
    steps = [step] if step else STEPS

    if experiment_dir:
        typer.echo(f"Experiment: {experiment_dir}")
    typer.echo(f"Steps: {' → '.join(s.value for s in steps)}\n")

    for i, s in enumerate[Step](steps, 1):
        config_path = _resolve_config(experiment_dir, s)
        if config_path is None:
            typer.echo(f"[{i}/{len(steps)}] {s.value}: no config found, skipping")
            continue

        typer.echo(f"{'=' * 60}")
        typer.echo(f"[{i}/{len(steps)}] {s.value}")
        typer.echo(f"{'=' * 60}")
        typer.echo(f"Config: {config_path}\n")

        t0 = time.time()
        run_step(step=s, config=config_path)
        elapsed = time.time() - t0

        typer.echo(f"\n  {s.value} completed in {elapsed:.1f}s\n")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
