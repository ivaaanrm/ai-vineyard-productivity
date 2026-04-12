"""MLflow experiment tracking integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def log_run(
    config: Any,
    model_name: str,
    target: str,
    target_info: Dict[str, Any],
    feature_cols: List[str],
    split_info: Dict[str, Any],
    n_combos: int,
    experiment_dir: Path,
    models_dir: Path,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"{model_name}-{target}"):
        mlflow.set_tags({
            "model_name": model_name,
            "target": target,
            "split_method": split_info["method"],
            "grid_size": str(n_combos),
        })

        mlflow.log_params({
            **{f"hp_{k}": v for k, v in target_info["best_params"].items()},
            "n_features": len(feature_cols),
            "train_size": split_info["train_size"],
            "val_size": split_info["val_size"],
            "test_size": split_info.get("test_size", 0),
        })

        mlflow.log_metrics({f"val_{k}": v for k, v in target_info["val_metrics"].items()})
        mlflow.log_metric("val_within_10_pct", target_info["val_within_10_pct"])

        if target_info.get("test_metrics"):
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in target_info["test_metrics"].items()}
            )
        if target_info.get("test_within_10_pct") is not None:
            mlflow.log_metric("test_within_10_pct", target_info["test_within_10_pct"])

        if config.mlflow.log_artifacts:
            candidates = [
                experiment_dir / f"grid_results_{target}.csv",
                experiment_dir / f"predictions_val_{target}.csv",
                experiment_dir / f"predictions_test_{target}.csv",
                experiment_dir / f"error_by_variety_val_{target}.csv",
                experiment_dir / f"error_by_variety_test_{target}.csv",
                experiment_dir / f"error_by_group_val_{target}.png",
                experiment_dir / f"error_by_group_test_{target}.png",
                experiment_dir / f"feature_importance_{target}.csv",
                experiment_dir / f"feature_importance_{target}.png",
                experiment_dir / f"tree_{target}.png",
                experiment_dir / f"performance_val_{target}.png",
                experiment_dir / f"performance_test_{target}.png",
                experiment_dir / f"learning_curve_{target}.csv",
                experiment_dir / f"learning_curve_{target}.png",
                experiment_dir / f"correlation_heatmap_{target}.png",
                models_dir / f"{target}.joblib",
                experiment_dir / "config.json",
                experiment_dir / "summary.csv",
                experiment_dir / "run_log.json",
            ]
            for path in candidates:
                if path.exists():
                    mlflow.log_artifact(str(path))
