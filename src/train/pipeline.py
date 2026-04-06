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
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

from .config import TrainingConfig
from .models import build_model


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

        if self.config.dry_run:
            return [self._dry_run(df)]

        results: List[Dict[str, Any]] = []
        enabled = [(n, c) for n, c in self.config.models.items() if c.enabled]
        for i, (model_name, model_cfg) in enumerate(enabled, 1):
            print(f"\n[2/5] Training model {i}/{len(enabled)}: {model_name}")
            result = self._train_model(model_name, model_cfg.params, df)
            results.append(result)

        return results

    # ── grid helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _expand_param_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand params with list values into all combinations."""
        grid_spec: Dict[str, list] = {}
        for k, v in params.items():
            grid_spec[k] = v if isinstance(v, list) else [v]
        return list(ParameterGrid(grid_spec))

    # ── core training ────────────────────────────────────────────────────

    def _dry_run(self, df: pd.DataFrame) -> Dict[str, Any]:
        feature_cols = self._get_feature_columns(df)
        df_train, df_val, df_test, split_info = self._split_data(df)

        X_train = df_train[feature_cols].values.astype(float)
        X_val = df_val[feature_cols].values.astype(float) if len(df_val) > 0 else None

        nan_total = np.isnan(X_train).sum(axis=0)
        if X_val is not None:
            nan_total += np.isnan(X_val).sum(axis=0)
        nan_counts = {
            col: int(nan_total[i]) for i, col in enumerate(feature_cols)
        }
        nan_features = {k: v for k, v in nan_counts.items() if v > 0}

        enabled_models = [n for n, c in self.config.models.items() if c.enabled]
        grid_sizes = {}
        for name, cfg in self.config.models.items():
            if cfg.enabled:
                grid_sizes[name] = len(self._expand_param_grid(cfg.params))

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

            # ── grid search on validation ────────────────────────────
            print(f"\n[3/5] Grid search for target: {target}")
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

            # ── save grid results ────────────────────────────────────
            grid_df = pd.DataFrame(grid_rows).sort_values("val_mae")
            grid_df.to_csv(experiment_dir / f"grid_results_{target}.csv", index=False)

            # ── best model — val metrics & artifacts ─────────────────
            y_pred_val = best_model.predict(X_val)
            val_metrics = self._compute_metrics(y_val, y_pred_val)
            model_path = best_model.save(models_dir / f"{target}.joblib")

            pct_within = float(np.mean(
                np.abs(y_pred_val - y_val)
                / np.where(y_val == 0, 1, y_val)
                <= 0.10
            ))

            print(f"\n       Best params: {best_combo}")
            print(f"       Validation results ({target}):")
            print(f"         MAE  = {val_metrics['mae']:.1f} kg/ha")
            print(f"         RMSE = {val_metrics['rmse']:.1f} kg/ha")
            print(f"         R2   = {val_metrics['r2']:.4f}")
            print(f"         MAPE = {val_metrics['mape']:.2%}")
            print(f"         Within 10% = {pct_within:.0%}")

            print(f"\n[4/5] Saving validation artifacts...")
            self._save_predictions(experiment_dir, df_val, target, y_pred_val, "val")
            self._save_feature_importance(experiment_dir, best_model, feature_cols, target)
            self._save_performance_plots(experiment_dir, df_val, y_pred_val, target, "val")
            self._save_correlation_heatmap(experiment_dir, df, best_model, feature_cols, target)
            self._save_learning_curve(experiment_dir, best_model, target)

            # ── test inference ───────────────────────────────────────
            test_metrics = None
            if X_test is not None and y_test is not None and len(df_test) > 0:
                y_pred_test = best_model.predict(X_test)
                test_metrics = self._compute_metrics(y_test, y_pred_test)

                pct_within_test = float(np.mean(
                    np.abs(y_pred_test - y_test)
                    / np.where(y_test == 0, 1, y_test)
                    <= 0.10
                ))

                print(f"\n       Test inference ({target}):")
                print(f"         MAE  = {test_metrics['mae']:.1f} kg/ha")
                print(f"         RMSE = {test_metrics['rmse']:.1f} kg/ha")
                print(f"         R2   = {test_metrics['r2']:.4f}")
                print(f"         MAPE = {test_metrics['mape']:.2%}")
                print(f"         Within 10% = {pct_within_test:.0%}")

                self._save_predictions(experiment_dir, df_test, target, y_pred_test, "test")
                self._save_performance_plots(experiment_dir, df_test, y_pred_test, target, "test")

            target_results[target] = {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
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
        self._save_experiment(experiment_dir, df, result)
        self._save_summary_csv(experiment_dir, result)
        return result

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
        """Return feature columns: from file if configured, else auto-detect."""
        if self.config.feature_columns_file:
            path = Path(self.config.feature_columns_file)
            cols = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            present = [c for c in cols if c in df.columns]
            missing = set(cols) - set(present)
            if missing:
                print(
                    f"       WARNING: {len(missing)} columns from"
                    f" {path.name} not found in data: {sorted(missing)}"
                )
            print(f"       Using {len(present)}/{len(cols)} columns from {path.name}")
            return present

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
        df_split: pd.DataFrame,
        target: str,
        y_pred: np.ndarray,
        split_name: str,
    ) -> None:
        pred_df = pd.DataFrame(
            {
                "parcel_id": df_split["parcel_id"].values,
                "year": df_split["year"].values,
                f"gt_{target}": df_split[target].values,
                f"predicted_{target}": np.round(y_pred, 2),
            }
        )
        pred_df["error"] = pred_df[f"predicted_{target}"] - pred_df[f"gt_{target}"]
        pred_df["abs_error"] = pred_df["error"].abs()
        pred_df.to_csv(
            experiment_dir / f"predictions_{split_name}_{target}.csv", index=False
        )

    def _save_feature_importance(
        self,
        experiment_dir: Path,
        model: Any,
        feature_cols: List[str],
        target: str,
    ) -> None:
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
        df_split: pd.DataFrame,
        y_pred: np.ndarray,
        target: str,
        split_name: str,
    ) -> None:
        y_true = df_split[target].values.astype(float)
        metrics = self._compute_metrics(y_true, y_pred)

        # Determine year(s) for title and mean line
        years = sorted(df_split["year"].unique())
        year_label = ", ".join(str(int(y)) for y in years)
        gt_mean = float(y_true.mean())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: Predicted vs Actual scatter ────────────────────
        ax = axes[0]
        pct_error = np.abs(y_pred - y_true) / np.where(y_true == 0, 1, y_true)
        within = pct_error <= 0.10
        colors = np.where(within, "#90ee90", "#f08080")
        ax.scatter(y_true, y_pred, c=colors, alpha=0.7, edgecolors="k", linewidths=0.5)
        lims = [
            min(y_true.min(), y_pred.min()) * 0.9,
            max(y_true.max(), y_pred.max()) * 1.1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="1:1")
        x_line = np.array(lims)
        ax.fill_between(
            x_line, x_line * 0.9, x_line * 1.1,
            alpha=0.10, color="green", label="\u00b110% band",
        )
        # Mean production line
        ax.axhline(gt_mean, color="navy", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"mean={gt_mean:.0f}")
        ax.axvline(gt_mean, color="navy", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual (kg/ha)")
        ax.set_ylabel("Predicted (kg/ha)")
        ax.set_title(f"Predicted vs Actual")
        n_within = int(within.sum())
        ax.legend(fontsize=8)
        ax.text(
            0.05, 0.95,
            f"MAE={metrics['mae']:.0f}\nR\u00b2={metrics['r2']:.3f}\n"
            f"within 10%: {n_within}/{len(y_true)} ({n_within / len(y_true):.0%})",
            transform=ax.transAxes, verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # ── Right: Distribution of predicted vs ground truth ─────
        ax = axes[1]
        sns.kdeplot(y_true, ax=ax, fill=True, alpha=0.4, color="steelblue",
                    label="Ground truth", linewidth=1.5)
        sns.kdeplot(y_pred, ax=ax, fill=True, alpha=0.4, color="coral",
                    label="Predicted", linewidth=1.5)
        ax.axvline(gt_mean, color="steelblue", linestyle="--", linewidth=1,
                   label=f"GT mean={gt_mean:.0f}")
        ax.axvline(float(y_pred.mean()), color="coral", linestyle="--", linewidth=1,
                   label=f"Pred mean={y_pred.mean():.0f}")
        ax.set_xlabel("Yield (kg/ha)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution: Predicted vs Ground Truth")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"{target} — {split_name.title()} Performance ({year_label})",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(
            experiment_dir / f"performance_{split_name}_{target}.png", dpi=150
        )
        plt.close(fig)

    def _save_learning_curve(
        self,
        experiment_dir: Path,
        model: Any,
        target: str,
    ) -> None:
        """Save train vs val loss per boosting round (XGBoost only)."""
        if not hasattr(model, "evals_result"):
            return
        evals = model.evals_result
        if not evals:
            return

        # evals_result keys: 'validation_0' (train), 'validation_1' (val)
        eval_keys = list(evals.keys())
        if len(eval_keys) < 2:
            return

        train_key, val_key = eval_keys[0], eval_keys[1]
        # Get the metric name (first metric available)
        metric_name = list(evals[train_key].keys())[0]
        train_scores = evals[train_key][metric_name]
        val_scores = evals[val_key][metric_name]
        rounds = list(range(1, len(train_scores) + 1))

        # Save CSV
        curve_df = pd.DataFrame({
            "round": rounds,
            f"train_{metric_name}": train_scores,
            f"val_{metric_name}": val_scores,
        })
        curve_df.to_csv(
            experiment_dir / f"learning_curve_{target}.csv", index=False
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rounds, train_scores, label=f"Train {metric_name}", color="steelblue",
                linewidth=1.5)
        ax.plot(rounds, val_scores, label=f"Val {metric_name}", color="coral",
                linewidth=1.5)

        # Mark best val round
        best_round = int(np.argmin(val_scores)) + 1
        best_val = float(min(val_scores))
        ax.axvline(best_round, color="gray", linestyle=":", linewidth=1,
                   label=f"Best round={best_round}")
        ax.plot(best_round, best_val, "o", color="coral", markersize=8, zorder=5)

        ax.set_xlabel("Boosting Round")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"Learning Curve — {target}")
        ax.legend()
        ax.text(
            0.95, 0.95,
            f"Best val {metric_name}={best_val:.1f}\nat round {best_round}/{len(rounds)}",
            transform=ax.transAxes, verticalalignment="top", horizontalalignment="right",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout()
        fig.savefig(experiment_dir / f"learning_curve_{target}.png", dpi=150)
        plt.close(fig)

    def _save_correlation_heatmap(
        self,
        experiment_dir: Path,
        df: pd.DataFrame,
        model: Any,
        feature_cols: List[str],
        target: str,
    ) -> None:
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

        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color,
                )

        ax.set_title(
            f"Correlation Matrix — Top {top_n} Features + {target}", fontsize=13
        )
        fig.tight_layout()
        fig.savefig(experiment_dir / f"correlation_heatmap_{target}.png", dpi=150)
        plt.close(fig)

    def _save_summary_csv(
        self, experiment_dir: Path, result: Dict[str, Any]
    ) -> None:
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

    def _make_experiment_dir(self, model_name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{model_name}_{ts}"
        exp_dir = Path(self.config.experiment_directory) / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _save_experiment(
        self,
        experiment_dir: Path,
        df: pd.DataFrame,
        result: Dict[str, Any],
    ) -> None:
        df.to_csv(experiment_dir / "dataset.csv", index=False)

        with open(experiment_dir / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        run_log = {"timestamp": datetime.now().isoformat(), **result}
        with open(experiment_dir / "run_log.json", "w") as f:
            json.dump(run_log, f, indent=2, default=str)

        src_csv = Path(self.config.features_csv)
        if (
            src_csv.exists()
            and src_csv.resolve() != (experiment_dir / "dataset.csv").resolve()
        ):
            shutil.copy2(src_csv, experiment_dir / "features_source.csv")
