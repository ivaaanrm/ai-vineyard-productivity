"""Training pipeline — train models, evaluate, save experiments."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from .config import TrainingConfig
from .models import build_model


class TrainingPipeline:
    """Train regression models per target, evaluate, and persist experiments.

    For each enabled model × target combination the pipeline:
    1. Splits data (using ``split`` column or random split).
    2. Trains the model.
    3. Evaluates on the validation set.
    4. Saves everything under ``experiments/<model>_<date>/``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ── public API ───────────────────────────────────────────────────────

    def execute(self, df: pd.DataFrame | None = None) -> List[Dict[str, Any]]:
        """Run the full training pipeline.

        Returns a list of experiment result dicts (one per model).
        In dry-run mode, validates data and config without training.
        """
        print("[1/4] Loading data...")
        if df is None:
            df = pd.read_csv(self.config.features_csv)
        print(f"       Loaded {self.config.features_csv} ({df.shape[0]} rows, {df.shape[1]} cols)")

        if self.config.dry_run:
            return [self._dry_run(df)]

        results: List[Dict[str, Any]] = []
        enabled = [(n, c) for n, c in self.config.models.items() if c.enabled]
        for i, (model_name, model_cfg) in enumerate(enabled, 1):
            print(f"\n[2/4] Training model {i}/{len(enabled)}: {model_name}")
            result = self._train_model(model_name, model_cfg.params, df)
            results.append(result)

        return results

    # ── internals ────────────────────────────────────────────────────────

    def _dry_run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data, config, and split without training."""
        feature_cols = self._get_feature_columns(df)
        df_train, df_val, split_info = self._split_data(df)

        X_train = df_train[feature_cols].values.astype(float)
        X_val = df_val[feature_cols].values.astype(float) if len(df_val) > 0 else None

        enabled_models = [
            name for name, cfg in self.config.models.items() if cfg.enabled
        ]

        nan_total = np.isnan(X_train).sum(axis=0)
        if X_val is not None:
            nan_total += np.isnan(X_val).sum(axis=0)
        nan_counts = {
            col: int(nan_total[i]) for i, col in enumerate(feature_cols)
        }
        nan_features = {k: v for k, v in nan_counts.items() if v > 0}

        y_train_dict = {
            t: df_train[t].values
            for t in self.config.targets
            if t in df_train.columns
        }

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

        print(f"       Features: {len(feature_cols)}")
        print(f"       Params: {params}")

        df_train, df_val, split_info = self._split_data(df)
        print(f"       Split: train={len(df_train)}, val={len(df_val)}, "
              f"test_excluded={split_info.get('test_excluded', 'n/a')}")

        X_train = df_train[feature_cols].values.astype(float)
        X_val = df_val[feature_cols].values.astype(float) if len(df_val) > 0 else None

        target_results: Dict[str, Any] = {}
        models_dir = experiment_dir / "models"
        models_dir.mkdir()

        for target in self.config.targets:
            if target not in df.columns:
                continue

            print(f"       Fitting target: {target}...")
            model = build_model(model_name, params)
            y_tr = df_train[target].values.astype(float)
            y_val = df_val[target].values.astype(float)

            fit_kwargs: Dict[str, Any] = {}
            if X_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False

            model.fit(X_train, y_tr, **fit_kwargs)
            y_pred = model.predict(X_val)

            metrics = self._compute_metrics(y_val, y_pred)
            model_path = model.save(models_dir / f"{target}.joblib")

            pct_within = np.mean(np.abs(y_pred - y_val) / np.where(y_val == 0, 1, y_val) <= 0.10)
            print(f"       Results ({target}):")
            print(f"         MAE  = {metrics['mae']:.1f} kg/ha")
            print(f"         RMSE = {metrics['rmse']:.1f} kg/ha")
            print(f"         R2   = {metrics['r2']:.4f}")
            print(f"         MAPE = {metrics['mape']:.2%}")
            print(f"         Within 10% = {pct_within:.0%}")

            # Save per-target artifacts
            print(f"\n[3/4] Saving artifacts...")
            self._save_predictions(experiment_dir, df_val, target, y_pred)
            self._save_feature_importance(
                experiment_dir, model, feature_cols, target
            )
            self._save_performance_plots(experiment_dir, y_val, y_pred, target)
            self._save_correlation_heatmap(
                experiment_dir, df, model, feature_cols, target
            )

            target_results[target] = {
                "metrics": metrics,
                "model_path": str(model_path),
            }

        result = {
            "model_name": model_name,
            "experiment_dir": str(experiment_dir),
            "params": params,
            "feature_columns": feature_cols,
            "split_info": split_info,
            "targets": target_results,
        }

        print(f"\n[4/4] Saving experiment to {experiment_dir}")
        self._save_experiment(experiment_dir, df, result)
        self._save_summary_csv(experiment_dir, result)
        return result

    # ── data splitting ───────────────────────────────────────────────────

    def _split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Split data into train/val DataFrames. Test rows excluded."""
        split_col = self.config.split_column

        if split_col and split_col in df.columns:
            df_train = df.loc[df[split_col] == "train"].copy()
            df_val = df.loc[df[split_col] == "val"].copy()
            n_test = int((df[split_col] == "test").sum())
            split_info = {
                "method": "column",
                "column": split_col,
                "train_size": len(df_train),
                "val_size": len(df_val),
                "test_excluded": n_test,
            }
        else:
            df_train, df_val = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            split_info = {
                "method": "random",
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "train_size": len(df_train),
                "val_size": len(df_val),
            }

        return df_train, df_val, split_info

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
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }

    # ── artifact saving ─────────────────────────────────────────────────

    def _save_predictions(
        self,
        experiment_dir: Path,
        df_val: pd.DataFrame,
        target: str,
        y_pred: np.ndarray,
    ) -> None:
        """Save CSV with parcel_id, year, ground truth, and predictions."""
        pred_df = pd.DataFrame(
            {
                "parcel_id": df_val["parcel_id"].values,
                "year": df_val["year"].values,
                f"gt_{target}": df_val[target].values,
                f"predicted_{target}": np.round(y_pred, 2),
            }
        )
        pred_df[f"error"] = pred_df[f"predicted_{target}"] - pred_df[f"gt_{target}"]
        pred_df[f"abs_error"] = pred_df[f"error"].abs()
        pred_df.to_csv(experiment_dir / f"predictions_{target}.csv", index=False)

    def _save_feature_importance(
        self,
        experiment_dir: Path,
        model: Any,
        feature_cols: List[str],
        target: str,
    ) -> None:
        """Save feature importance bar chart and CSV."""
        importances = model.feature_importances
        imp_df = (
            pd.DataFrame({"feature": feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df.to_csv(experiment_dir / f"feature_importance_{target}.csv", index=False)

        top_n = min(20, len(imp_df))
        top = imp_df.head(top_n).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["feature"], top["importance"], color="skyblue")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Feature Importance — {target}")
        fig.tight_layout()
        fig.savefig(experiment_dir / f"feature_importance_{target}.png", dpi=150)
        plt.close(fig)

    def _save_performance_plots(
        self,
        experiment_dir: Path,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target: str,
    ) -> None:
        """Save predicted-vs-actual scatter and residual histogram."""
        residuals = y_pred - y_true
        metrics = self._compute_metrics(y_true, y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter: predicted vs actual, colored by 10% error threshold
        ax = axes[0]
        pct_error = np.abs(y_pred - y_true) / np.where(y_true == 0, 1, y_true)
        within = pct_error <= 0.10
        colors = np.where(within, "#90ee90", "#f08080")  # light green / light red
        ax.scatter(
            y_true, y_pred, c=colors, alpha=0.7, edgecolors="k", linewidths=0.5
        )
        lims = [
            min(y_true.min(), y_pred.min()) * 0.9,
            max(y_true.max(), y_pred.max()) * 1.1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="1:1")
        # 10% error band
        x_line = np.array(lims)
        ax.fill_between(
            x_line, x_line * 0.9, x_line * 1.1,
            alpha=0.10, color="green", label="\u00b110% band",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual (kg/ha)")
        ax.set_ylabel("Predicted (kg/ha)")
        ax.set_title("Predicted vs Actual")
        n_within = int(within.sum())
        ax.legend()
        ax.text(
            0.05,
            0.95,
            f"MAE={metrics['mae']:.0f}\nR\u00b2={metrics['r2']:.3f}\n"
            f"within 10%: {n_within}/{len(y_true)} ({n_within/len(y_true):.0%})",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Residual histogram
        ax = axes[1]
        ax.hist(residuals, bins=20, color="skyblue", edgecolor="k", alpha=0.7)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (kg/ha)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution")
        ax.text(
            0.05,
            0.95,
            f"mean={residuals.mean():.0f}\nstd={residuals.std():.0f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.suptitle(f"{target} — Validation Performance", fontsize=13)
        fig.tight_layout()
        fig.savefig(experiment_dir / f"performance_{target}.png", dpi=150)
        plt.close(fig)

    def _save_correlation_heatmap(
        self,
        experiment_dir: Path,
        df: pd.DataFrame,
        model: Any,
        feature_cols: List[str],
        target: str,
    ) -> None:
        """Save correlation heatmap of top 30 features by importance + target."""
        importances = model.feature_importances
        imp_order = np.argsort(importances)[::-1]
        top_n = min(30, len(feature_cols))
        top_cols = [feature_cols[i] for i in imp_order[:top_n]] + [target]

        corr = df[top_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)

        # Annotate cells
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=color)

        ax.set_title(f"Correlation Matrix — Top {top_n} Features + {target}", fontsize=13)
        fig.tight_layout()
        fig.savefig(experiment_dir / f"correlation_heatmap_{target}.png", dpi=150)
        plt.close(fig)

    def _save_summary_csv(
        self, experiment_dir: Path, result: Dict[str, Any]
    ) -> None:
        """Save a single-row summary CSV with model info and metrics."""
        rows = []
        for target, info in result["targets"].items():
            row = {
                "model": result["model_name"],
                "target": target,
                "train_size": result["split_info"]["train_size"],
                "val_size": result["split_info"]["val_size"],
                "n_features": len(result["feature_columns"]),
                **info["metrics"],
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(experiment_dir / "summary.csv", index=False)

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
        df.to_csv(experiment_dir / "dataset.csv", index=False)

        with open(experiment_dir / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        run_log = {
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        with open(experiment_dir / "run_log.json", "w") as f:
            json.dump(run_log, f, indent=2, default=str)

        src_csv = Path(self.config.features_csv)
        if (
            src_csv.exists()
            and src_csv.resolve() != (experiment_dir / "dataset.csv").resolve()
        ):
            shutil.copy2(src_csv, experiment_dir / "features_source.csv")
