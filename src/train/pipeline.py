"""Training pipeline — train models, evaluate, save experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

from .config import TrainingConfig
from .models import build_model
from . import exports, tracking, visualizations


class TrainingPipeline:
    """Train regression models with optional grid search.

    Flow:
    1. Load data, split into train / val / test.
    2. For each enabled model, expand param lists into a grid.
    3. Train every combination on train, score on val.
    4. Keep the best model (lowest val MAE).
    5. Run inference on test with the best model.
    6. Save all artifacts under ``experiment_directory/<model>_<ts>/``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ── public API ───────────────────────────────────────────────────────

    def execute(self, df: pd.DataFrame | None = None) -> List[Dict[str, Any]]:
        print("[1/5] Loading data...")
        if df is None:
            df = pd.read_csv(self.config.features_csv)
        print(
            f"       Loaded {self.config.features_csv}"
            f" ({df.shape[0]} rows, {df.shape[1]} cols)"
        )

        feature_cols = self._get_feature_columns(df)
        nan_rows = df[feature_cols].isna().any(axis=1)
        if nan_rows.any():
            dropped = df[nan_rows][["parcel_id", "year"]].values.tolist()
            print(f"       Dropping {nan_rows.sum()} rows with unrecoverable NaN: {dropped}")
            df = df[~nan_rows].reset_index(drop=True)

        if self.config.dry_run:
            return [self._dry_run(df)]

        results: List[Dict[str, Any]] = []
        enabled = [(n, c) for n, c in self.config.models.items() if c.enabled]
        for i, (model_name, model_cfg) in enumerate(enabled, 1):
            print(f"\n[2/5] Training model {i}/{len(enabled)}: {model_name}")
            result = self._train_model(model_name, model_cfg.params, df)
            results.append(result)

        return results

    # ── core training ────────────────────────────────────────────────────

    def _dry_run(self, df: pd.DataFrame) -> Dict[str, Any]:
        feature_cols = self._get_feature_columns(df)
        df_train, df_val, df_test, split_info = self._split_data(df)

        X_train = df_train[feature_cols].values.astype(float)
        X_val = df_val[feature_cols].values.astype(float) if len(df_val) > 0 else None

        nan_total = np.isnan(X_train).sum(axis=0)
        if X_val is not None:
            nan_total += np.isnan(X_val).sum(axis=0)
        nan_counts = {col: int(nan_total[i]) for i, col in enumerate(feature_cols)}
        nan_features = {k: v for k, v in nan_counts.items() if v > 0}

        enabled_models = [n for n, c in self.config.models.items() if c.enabled]
        grid_sizes = {
            name: len(self._expand_param_grid(cfg.params))
            for name, cfg in self.config.models.items()
            if cfg.enabled
        }

        return {
            "dry_run": True,
            "dataset_rows": len(df),
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "split_info": split_info,
            "targets_found": [t for t in self.config.targets if t in df.columns],
            "targets_missing": [t for t in self.config.targets if t not in df.columns],
            "enabled_models": enabled_models,
            "grid_sizes": grid_sizes,
            "nan_features": nan_features,
        }

    def _train_model(
        self, model_name: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> Dict[str, Any]:
        experiment_dir = self._make_experiment_dir(model_name)
        feature_cols = self._get_feature_columns(df)
        df_train, df_val, df_test, split_info = self._split_data(df)

        X_train = df_train[feature_cols].values.astype(float)
        X_val = df_val[feature_cols].values.astype(float) if len(df_val) > 0 else None
        X_test = df_test[feature_cols].values.astype(float) if len(df_test) > 0 else None

        param_combos = self._expand_param_grid(params)
        n_combos = len(param_combos)

        print(f"       Features: {len(feature_cols)}")
        print(
            f"       Split: train={len(df_train)}, val={len(df_val)},"
            f" test={len(df_test)}"
        )
        print(f"       Grid search: {n_combos} combination(s)")

        target_results: Dict[str, Any] = {}
        models_dir = experiment_dir / "models"
        models_dir.mkdir()

        for target in self.config.targets:
            if target not in df.columns:
                continue

            y_tr = df_train[target].values.astype(float)
            y_val = df_val[target].values.astype(float) if len(df_val) > 0 else None
            y_test = df_test[target].values.astype(float) if len(df_test) > 0 else None

            if self.config.log_transform_target:
                y_tr = np.log(y_tr)
                y_val = np.log(y_val) if y_val is not None else None
                y_test = np.log(y_test) if y_test is not None else None

            print(f"\n[3/5] Grid search for target: {target}")
            best_model, best_combo, grid_rows = self._run_grid_search(
                model_name, param_combos, X_train, y_tr, X_val, y_val
            )
            pd.DataFrame(grid_rows).sort_values("val_mae").to_csv(
                experiment_dir / f"grid_results_{target}.csv", index=False
            )

            print(f"\n       Best params: {best_combo}")
            print(f"\n[4/5] Saving validation artifacts...")
            val_metrics, pct_within, model_path = self._run_validation(
                experiment_dir, models_dir, best_model, feature_cols,
                df_val, X_val, y_val, target, df,
                log_transform=self.config.log_transform_target,
            )

            test_metrics, pct_within_test = self._run_test_inference(
                experiment_dir, best_model, df_test, X_test, y_test, target,
                log_transform=self.config.log_transform_target,
            )

            target_results[target] = {
                "val_metrics": val_metrics,
                "val_within_10_pct": pct_within,
                "test_metrics": test_metrics,
                "test_within_10_pct": pct_within_test,
                "best_params": best_combo,
                "grid_size": n_combos,
                "model_path": str(model_path),
            }

        result = {
            "model_name": model_name,
            "experiment_dir": str(experiment_dir),
            "feature_columns": feature_cols,
            "split_info": split_info,
            "targets": target_results,
        }

        print(f"\n[5/5] Saving experiment to {experiment_dir}")
        exports.save_experiment(experiment_dir, df, result, self.config)
        exports.save_summary_csv(experiment_dir, result)

        if self.config.mlflow.enabled:
            for target, info in target_results.items():
                tracking.log_run(
                    self.config, model_name, target, info,
                    feature_cols, split_info, n_combos, experiment_dir, models_dir,
                )

        return result

    # ── training phases ──────────────────────────────────────────────────

    def _run_grid_search(
        self,
        model_name: str,
        param_combos: List[Dict[str, Any]],
        X_train: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
        n_combos = len(param_combos)
        grid_rows: List[Dict[str, Any]] = []
        best_mae = float("inf")
        best_model = None
        best_combo: Dict[str, Any] = {}

        for i, combo in enumerate(param_combos, 1):
            model = build_model(model_name, combo)

            fit_kwargs: Dict[str, Any] = {}
            if X_val is not None:
                fit_kwargs["eval_set"] = [(X_train, y_tr), (X_val, y_val)]
                fit_kwargs["verbose"] = False

            model.fit(X_train, y_tr, **fit_kwargs)
            y_pred_val = model.predict(X_val)
            val_metrics = self._compute_metrics(y_val, y_pred_val)

            grid_rows.append(
                {"combo": i, **combo, **{f"val_{k}": v for k, v in val_metrics.items()}}
            )

            tag = " *best*" if val_metrics["mae"] < best_mae else ""
            print(
                f"       [{i:>{len(str(n_combos))}}/{n_combos}]"
                f" val_MAE={val_metrics['mae']:.1f}{tag}"
            )

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                best_model = model
                best_combo = combo

        return best_model, best_combo, grid_rows

    def _run_validation(
        self,
        experiment_dir: Path,
        models_dir: Path,
        best_model: Any,
        feature_cols: List[str],
        df_val: pd.DataFrame,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target: str,
        df_full: pd.DataFrame,
        log_transform: bool = False,
    ) -> tuple[Dict[str, float], float, Path]:
        y_pred_val = best_model.predict(X_val)
        if log_transform:
            y_pred_val = np.exp(y_pred_val)
            y_val = np.exp(y_val)
        val_metrics = self._compute_metrics(y_val, y_pred_val)
        model_path = best_model.save(models_dir / f"{target}.joblib")

        pct_within = float(np.mean(
            np.abs(y_pred_val - y_val) / np.where(y_val == 0, 1, y_val) <= 0.10
        ))

        print(f"       Validation results ({target}):")
        print(f"         MAE  = {val_metrics['mae']:.1f} kg/ha")
        print(f"         RMSE = {val_metrics['rmse']:.1f} kg/ha")
        print(f"         R2   = {val_metrics['r2']:.4f}")
        print(f"         MAPE = {val_metrics['mape']:.2%}")
        print(f"         Within 10% = {pct_within:.0%}")

        exports.save_predictions(experiment_dir, df_val, target, y_pred_val, "val")
        visualizations.save_error_by_group_plot(experiment_dir, df_val, y_pred_val, target, "val")
        imp_df = exports.save_feature_importance(experiment_dir, best_model, feature_cols, target)
        visualizations.save_feature_importance_plot(experiment_dir, imp_df, target)
        visualizations.save_performance_plots(
            experiment_dir, df_val, y_pred_val, target, "val", val_metrics
        )
        visualizations.save_correlation_heatmap(
            experiment_dir, df_full, best_model, feature_cols, target
        )
        visualizations.save_learning_curve(experiment_dir, best_model, target)
        if hasattr(best_model, "_model") and hasattr(best_model._model, "estimators_"):
            visualizations.save_rf_tree_plot(experiment_dir, best_model, feature_cols, target)

        return val_metrics, pct_within, model_path

    def _run_test_inference(
        self,
        experiment_dir: Path,
        best_model: Any,
        df_test: pd.DataFrame,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        target: str,
        log_transform: bool = False,
    ) -> tuple[Dict[str, float] | None, float | None]:
        if X_test is None or y_test is None or len(df_test) == 0:
            return None, None

        y_pred_test = best_model.predict(X_test)
        if log_transform:
            y_pred_test = np.exp(y_pred_test)
            y_test = np.exp(y_test)
        test_metrics = self._compute_metrics(y_test, y_pred_test)

        pct_within_test = float(np.mean(
            np.abs(y_pred_test - y_test) / np.where(y_test == 0, 1, y_test) <= 0.10
        ))

        print(f"\n       Test inference ({target}):")
        print(f"         MAE  = {test_metrics['mae']:.1f} kg/ha")
        print(f"         RMSE = {test_metrics['rmse']:.1f} kg/ha")
        print(f"         R2   = {test_metrics['r2']:.4f}")
        print(f"         MAPE = {test_metrics['mape']:.2%}")
        print(f"         Within 10% = {pct_within_test:.0%}")

        exports.save_predictions(experiment_dir, df_test, target, y_pred_test, "test")
        visualizations.save_error_by_group_plot(experiment_dir, df_test, y_pred_test, target, "test")
        visualizations.save_performance_plots(
            experiment_dir, df_test, y_pred_test, target, "test", test_metrics
        )

        return test_metrics, pct_within_test

    # ── data splitting ───────────────────────────────────────────────────

    def _split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Split data into train / val / test DataFrames."""
        split_col = self.config.split_column

        if split_col and split_col in df.columns:
            df_train = df.loc[df[split_col] == "train"].copy()
            df_val = df.loc[df[split_col] == "val"].copy()
            df_test = df.loc[df[split_col] == "test"].copy()
            split_info = {
                "method": "column",
                "column": split_col,
                "train_size": len(df_train),
                "val_size": len(df_val),
                "test_size": len(df_test),
            }
        else:
            df_trainval, df_test = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            df_train, df_val = train_test_split(
                df_trainval,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            split_info = {
                "method": "random",
                "random_state": self.config.random_state,
                "train_size": len(df_train),
                "val_size": len(df_val),
                "test_size": len(df_test),
            }

        return df_train, df_val, df_test, split_info

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return all numeric columns that are not targets or metadata."""
        exclude = set(self.config.targets) | {"parcel_id", "year", "split"}
        return [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }

    @staticmethod
    def _expand_param_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand params with list values into all combinations."""
        grid_spec: Dict[str, list] = {}
        for k, v in params.items():
            grid_spec[k] = v if isinstance(v, list) else [v]
        return list(ParameterGrid(grid_spec))

    def _make_experiment_dir(self, model_name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(self.config.experiment_directory) / f"{model_name}_{ts}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
