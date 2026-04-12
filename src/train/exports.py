"""Artifact export utilities — save CSVs, JSON logs, and model files."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def save_predictions(
    experiment_dir: Path,
    df_split: pd.DataFrame,
    target: str,
    y_pred: np.ndarray,
    split_name: str,
) -> None:
    pred_df = pd.DataFrame({
        "parcel_id": df_split["parcel_id"].values,
        "year": df_split["year"].values,
        f"gt_{target}": df_split[target].values,
        f"predicted_{target}": np.round(y_pred, 2),
    })
    if "variety" in df_split.columns:
        pred_df.insert(2, "variety", df_split["variety"].values)
    pred_df["error"] = pred_df[f"predicted_{target}"] - pred_df[f"gt_{target}"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df["pct_error"] = (pred_df["abs_error"] / pred_df[f"gt_{target}"].replace(0, np.nan) * 100).round(2)
    pred_df.to_csv(experiment_dir / f"predictions_{split_name}_{target}.csv", index=False)

    if "variety" in pred_df.columns:
        variety_df = (
            pred_df.groupby("variety")
            .agg(
                n=("abs_error", "count"),
                mae=(f"abs_error", "mean"),
                me=("error", "mean"),
                rmse=("error", lambda x: float(np.sqrt((x**2).mean()))),
                mape=("pct_error", "mean"),
            )
            .round(3)
            .sort_values("mae", ascending=False)
            .reset_index()
        )
        variety_df.to_csv(
            experiment_dir / f"error_by_variety_{split_name}_{target}.csv", index=False
        )


def save_feature_importance(
    experiment_dir: Path,
    model: Any,
    feature_cols: List[str],
    target: str,
) -> pd.DataFrame:
    """Save feature importance CSV and return the sorted DataFrame."""
    importances = model.feature_importances
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(experiment_dir / f"feature_importance_{target}.csv", index=False)
    return imp_df


def save_summary_csv(experiment_dir: Path, result: Dict[str, Any]) -> None:
    rows = []
    for target, info in result["targets"].items():
        row: Dict[str, Any] = {
            "model": result["model_name"],
            "target": target,
            "grid_size": info["grid_size"],
            "best_params": json.dumps(info["best_params"]),
            "train_size": result["split_info"]["train_size"],
            "val_size": result["split_info"]["val_size"],
            "test_size": result["split_info"]["test_size"],
            "n_features": len(result["feature_columns"]),
        }
        for k, v in info["val_metrics"].items():
            row[f"val_{k}"] = v
        if info.get("test_metrics"):
            for k, v in info["test_metrics"].items():
                row[f"test_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(experiment_dir / "summary.csv", index=False)


def save_experiment(
    experiment_dir: Path,
    df: pd.DataFrame,
    result: Dict[str, Any],
    config: Any,
) -> None:
    df.to_csv(experiment_dir / "dataset.csv", index=False)

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)

    run_log = {"timestamp": datetime.now().isoformat(), **result}
    with open(experiment_dir / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    src_csv = Path(config.features_csv)
    if (
        src_csv.exists()
        and src_csv.resolve() != (experiment_dir / "dataset.csv").resolve()
    ):
        shutil.copy2(src_csv, experiment_dir / "features_source.csv")
