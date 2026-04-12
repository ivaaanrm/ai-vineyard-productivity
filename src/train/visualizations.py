"""Visualization utilities — generate and save experiment plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree


def save_feature_importance_plot(
    experiment_dir: Path,
    imp_df: pd.DataFrame,
    target: str,
) -> None:
    top_n = min(20, len(imp_df))
    top = imp_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"], color="skyblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importance — {target}")
    fig.tight_layout()
    fig.savefig(experiment_dir / f"feature_importance_{target}.png", dpi=150)
    plt.close(fig)


def save_performance_plots(
    experiment_dir: Path,
    df_split: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    split_name: str,
    metrics: Dict[str, float],
) -> None:
    y_true = df_split[target].values.astype(float)
    years = sorted(df_split["year"].unique())
    year_label = ", ".join(str(int(y)) for y in years)
    gt_mean = float(y_true.mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Predicted vs Actual scatter ────────────────────────────────
    ax = axes[0]
    pct_error = np.abs(y_pred - y_true) / np.where(y_true == 0, 1, y_true)
    within = pct_error <= 0.10
    colors = np.where(within, "#90ee90", "#f08080")
    ax.scatter(y_true, y_pred, c=colors, alpha=0.7, edgecolors="k", linewidths=0.5)
    x_min = max(min(y_true.min(), y_pred.min()) * 0.9, 1e-3)
    x_max = max(y_true.max(), y_pred.max()) * 1.1
    y_min = min(y_true.min(), y_pred.min()) * 0.9
    y_max = x_max
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    ax.plot(x_line, x_line, "r--", linewidth=1, label="1:1")
    ax.fill_between(
        x_line, x_line * 0.9, x_line * 1.1,
        alpha=0.10, color="green", label="\u00b110% band",
    )
    # ax.axhline(gt_mean, color="navy", linestyle="--", linewidth=1, alpha=0.7,
    #            label=f"mean={gt_mean:.2f}")
    # ax.axvline(gt_mean, color="navy", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(mticker.LogLocator(subs=np.arange(2, 10)))
        axis.set_minor_formatter(mticker.ScalarFormatter())
    ax.tick_params(which="minor", length=3, labelsize=7, color="gray")
    ax.set_xlabel("Actual (T/ha)")
    ax.set_ylabel("Predicted (T/ha)")
    ax.set_title("Predicted vs Actual")
    n_within = int(within.sum())
    ax.legend(fontsize=8)
    ax.text(
        0.05, 0.95,
        f"MAE={metrics['mae']:.2f}\nR\u00b2={metrics['r2']:.3f}\n"
        f"within 10%: {n_within}/{len(y_true)} ({n_within / len(y_true):.0%})",
        transform=ax.transAxes, verticalalignment="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ── Right: Distribution of predicted vs ground truth ─────────────────
    ax = axes[1]
    sns.kdeplot(y_true, ax=ax, fill=True, alpha=0.4, color="steelblue",
                label="Ground truth", linewidth=1.5)
    sns.kdeplot(y_pred, ax=ax, fill=True, alpha=0.4, color="coral",
                label="Predicted", linewidth=1.5)
    ax.axvline(gt_mean, color="steelblue", linestyle="--", linewidth=1,
               label=f"GT mean={gt_mean:.2f}")
    ax.axvline(float(y_pred.mean()), color="coral", linestyle="--", linewidth=1,
               label=f"Pred mean={y_pred.mean():.2f}")
    ax.set_xlim(left=0)
    ax.set_xlabel("Yield (T/ha)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution: Predicted vs Ground Truth")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{target} — {split_name.title()} Performance ({year_label})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(experiment_dir / f"performance_{split_name}_{target}.png", dpi=150)
    plt.close(fig)


def save_learning_curve(
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

    eval_keys = list(evals.keys())
    if len(eval_keys) < 2:
        return

    train_key, val_key = eval_keys[0], eval_keys[1]
    metric_name = list(evals[train_key].keys())[0]
    train_scores = evals[train_key][metric_name]
    val_scores = evals[val_key][metric_name]
    rounds = list(range(1, len(train_scores) + 1))

    pd.DataFrame({
        "round": rounds,
        f"train_{metric_name}": train_scores,
        f"val_{metric_name}": val_scores,
    }).to_csv(experiment_dir / f"learning_curve_{target}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, train_scores, label=f"Train {metric_name}", color="steelblue", linewidth=1.5)
    ax.plot(rounds, val_scores, label=f"Val {metric_name}", color="coral", linewidth=1.5)

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


def save_correlation_heatmap(
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
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    ax.set_title(f"Correlation Matrix — Top {top_n} Features + {target}", fontsize=13)
    fig.tight_layout()
    fig.savefig(experiment_dir / f"correlation_heatmap_{target}.png", dpi=150)
    plt.close(fig)


def save_error_by_group_plot(
    experiment_dir: Path,
    df_split: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    split_name: str,
) -> None:
    """Bar subplots of MAE grouped by variety and province (whichever columns exist)."""
    variety_col = "variety" if "variety" in df_split.columns else (
        "variety_enc" if "variety_enc" in df_split.columns else None
    )
    province_col = "province" if "province" in df_split.columns else None

    groups = [(c, c.replace("_enc", "").replace("_", " ").title())
              for c in [variety_col, province_col] if c is not None]
    if not groups:
        return

    y_true = df_split[target].values.astype(float)
    abs_err = np.abs(y_pred - y_true)
    signed_err = y_pred - y_true

    n_panels = len(groups)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, groups):
        grp = (
            pd.DataFrame({"group": df_split[col].values, "abs_err": abs_err, "err": signed_err})
            .groupby("group")
            .agg(n=("abs_err", "count"), mae=("abs_err", "mean"), me=("err", "mean"))
            .sort_values("mae", ascending=True)
            .reset_index()
        )
        bars = ax.barh(grp["group"].astype(str), grp["mae"], color="steelblue", alpha=0.8)
        # overlay signed error as a dot
        ax.scatter(grp["me"], grp["group"].astype(str), color="coral",
                   zorder=5, s=40, label="Mean error (signed)")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        for bar, n in zip(bars, grp["n"]):
            ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"n={n}", va="center", fontsize=8)
        ax.set_xlabel("MAE")
        ax.set_title(f"Mean Error by {label}")
        ax.legend(fontsize=8)

    fig.suptitle(f"{target} — Error by group ({split_name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(experiment_dir / f"error_by_group_{split_name}_{target}.png", dpi=150)
    plt.close(fig)


def save_rf_tree_plot(
    experiment_dir: Path,
    rf_model: Any,
    feature_cols: List[str],
    target: str,
    tree_index: int = 0,
    max_depth: int = 4,
) -> None:
    """Plot one tree from a fitted RandomForestRegressor and save as PNG."""
    estimators = rf_model._model.estimators_
    tree = estimators[tree_index % len(estimators)]

    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        tree,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=max_depth,
        ax=ax,
    )
    n_trees = len(estimators)
    ax.set_title(
        f"Tree #{tree_index} of {n_trees} — {target} (max_depth shown: {max_depth})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(experiment_dir / f"tree_{target}.png", dpi=120)
    plt.close(fig)
