"""Training pipeline — train models, evaluate, save experiments."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import TrainingConfig
from .models import build_model


class TrainingPipeline:
    """Train regression models per target, evaluate, and persist experiments.

    For each enabled model × target combination the pipeline:
    1. Splits data (using ``split`` column or random split).
    2. Trains the model.
    3. Evaluates on the test set.
    4. Saves everything under ``experiments/<id>_<model>_<date>/``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ── public API ───────────────────────────────────────────────────────

    def execute(self, df: pd.DataFrame | None = None) -> List[Dict[str, Any]]:
        """Run the full training pipeline.

        Returns a list of experiment result dicts (one per model).
        In dry-run mode, validates data and config without training.
        """
        if df is None:
            df = pd.read_csv(self.config.features_csv)

        if self.config.dry_run:
            return [self._dry_run(df)]

        results: List[Dict[str, Any]] = []
        for model_name, model_cfg in self.config.models.items():
            if not model_cfg.enabled:
                continue
            result = self._train_model(model_name, model_cfg.params, df)
            results.append(result)

        return results

    # ── internals ────────────────────────────────────────────────────────

    def _dry_run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data, config, and split without training."""
        feature_cols = self._get_feature_columns(df)
        X_train, X_test, y_train_dict, y_test_dict, split_info = self._split_data(
            df, feature_cols
        )

        enabled_models = [
            name for name, cfg in self.config.models.items() if cfg.enabled
        ]

        nan_counts = {
            col: int(np.isnan(X_train[:, i]).sum() + np.isnan(X_test[:, i]).sum())
            for i, col in enumerate(feature_cols)
        }
        nan_features = {k: v for k, v in nan_counts.items() if v > 0}

        return {
            "dry_run": True,
            "dataset_rows": len(df),
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "split_info": split_info,
            "targets_found": list(y_train_dict.keys()),
            "targets_missing": [
                t for t in self.config.targets if t not in y_train_dict
            ],
            "enabled_models": enabled_models,
            "nan_features": nan_features,
        }

    def _train_model(
        self, model_name: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train one model on all targets and save the experiment."""
        experiment_dir = self._make_experiment_dir(model_name)
        feature_cols = self._get_feature_columns(df)

        X_train, X_test, y_train_dict, y_test_dict, split_info = self._split_data(
            df, feature_cols
        )

        target_results: Dict[str, Any] = {}
        models_dir = experiment_dir / "models"
        models_dir.mkdir()

        for target in self.config.targets:
            if target not in y_train_dict:
                continue

            model = build_model(model_name, params)
            y_tr = y_train_dict[target]
            y_te = y_test_dict[target]

            model.fit(X_train, y_tr)
            y_pred = model.predict(X_test)

            metrics = self._compute_metrics(y_te, y_pred)
            model_path = model.save(models_dir / f"{target}.joblib")

            target_results[target] = {
                "metrics": metrics,
                "model_path": str(model_path),
            }

        # Save experiment artifacts
        result = {
            "model_name": model_name,
            "experiment_dir": str(experiment_dir),
            "params": params,
            "feature_columns": feature_cols,
            "split_info": split_info,
            "targets": target_results,
        }

        self._save_experiment(experiment_dir, df, result)
        return result

    def _split_data(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Any],
    ]:
        """Split data into train/test using split column or random split."""
        split_col = self.config.split_column

        if split_col and split_col in df.columns:
            train_mask = df[split_col] == "train"
            test_mask = df[split_col].isin(["test", "val"])
            df_train = df.loc[train_mask]
            df_test = df.loc[test_mask]
            split_info = {
                "method": "column",
                "column": split_col,
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
            }
        else:
            df_train, df_test = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            split_info = {
                "method": "random",
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "train_size": len(df_train),
                "test_size_actual": len(df_test),
            }

        X_train = df_train[feature_cols].values.astype(float)
        X_test = df_test[feature_cols].values.astype(float)

        y_train_dict = {}
        y_test_dict = {}
        for target in self.config.targets:
            if target in df.columns:
                y_train_dict[target] = df_train[target].values.astype(float)
                y_test_dict[target] = df_test[target].values.astype(float)

        return X_train, X_test, y_train_dict, y_test_dict, split_info

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """All numeric columns except targets, identifiers, and split."""
        exclude = set(self.config.targets) | {"parcel_id", "year", "split"}
        return [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }

    def _make_experiment_dir(self, model_name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{model_name}_{ts}"
        exp_dir = Path(self.config.experiments_dir) / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _save_experiment(
        self,
        experiment_dir: Path,
        df: pd.DataFrame,
        result: Dict[str, Any],
    ) -> None:
        """Persist dataset copy, config snapshot, and run log."""
        # Copy dataset
        df.to_csv(experiment_dir / "dataset.csv", index=False)

        # Save training config
        with open(experiment_dir / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        # Save run log (config + metrics + timestamps)
        run_log = {
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        with open(experiment_dir / "run_log.json", "w") as f:
            json.dump(run_log, f, indent=2, default=str)

        # Copy source features CSV if it exists
        src_csv = Path(self.config.features_csv)
        if (
            src_csv.exists()
            and src_csv.resolve() != (experiment_dir / "dataset.csv").resolve()
        ):
            shutil.copy2(src_csv, experiment_dir / "features_source.csv")
