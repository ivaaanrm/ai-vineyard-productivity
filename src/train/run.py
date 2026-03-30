"""Training CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.train.config import TrainingConfig
from src.train.pipeline import TrainingPipeline


def main(config_path: str | Path | None = None, *, dry_run: bool = False) -> None:
    """Run the training pipeline from a YAML config."""
    config_path = Path(
        config_path or Path(__file__).parent.parent / "config" / "training.yml"
    )
    config = TrainingConfig.from_yaml(config_path)
    if dry_run:
        config.dry_run = True

    pipeline = TrainingPipeline(config)
    results = pipeline.execute()

    for r in results:
        if r.get("dry_run"):
            _print_dry_run(r)
        else:
            _print_result(r)


def _print_dry_run(r: dict) -> None:
    print(f"\n{'=' * 60}")
    print("DRY RUN — no models trained, no files written")
    print(f"{'=' * 60}")
    print(f"  Dataset rows:    {r['dataset_rows']}")
    print(f"  Features:        {r['n_features']}")
    print(f"  Targets found:   {r['targets_found']}")
    if r["targets_missing"]:
        print(f"  Targets MISSING: {r['targets_missing']}")
    print(f"  Split:           {r['split_info']}")
    print(f"  Models:          {r['enabled_models']}")
    print(f"  Grid sizes:      {r['grid_sizes']}")
    if r["nan_features"]:
        print(f"  NaN features:    {r['nan_features']}")
    else:
        print("  NaN features:    none")


def _print_result(r: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Model: {r['model_name']}")
    print(f"Experiment: {r['experiment_dir']}")
    print(f"Split: {r['split_info']}")
    for target, info in r["targets"].items():
        print(f"\n  {target} (grid: {info['grid_size']} combos):")
        print(f"  Best params: {info['best_params']}")

        vm = info["val_metrics"]
        print(f"  Validation:")
        print(f"    MAE  = {vm['mae']:.1f} kg/ha")
        print(f"    RMSE = {vm['rmse']:.1f} kg/ha")
        print(f"    R²   = {vm['r2']:.4f}")
        print(f"    MAPE = {vm['mape']:.2%}")

        if info.get("test_metrics"):
            tm = info["test_metrics"]
            print(f"  Test:")
            print(f"    MAE  = {tm['mae']:.1f} kg/ha")
            print(f"    RMSE = {tm['rmse']:.1f} kg/ha")
            print(f"    R²   = {tm['r2']:.4f}")
            print(f"    MAPE = {tm['mape']:.2%}")


if __name__ == "__main__":
    main()
