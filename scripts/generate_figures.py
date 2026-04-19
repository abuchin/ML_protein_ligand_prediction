#!/usr/bin/env python3
"""Generate summary figures from a completed training run.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --output_dir outputs/figures
    python scripts/generate_figures.py --output_dir outputs/figures --dpi 150

Reads from:
    outputs/model_comparison.csv        — per-split test metrics
    outputs/results.json                — confusion matrices + CV results
    outputs/predictions/cold_start_summary.csv  — drug-discovery metrics
    outputs/predictions/predictions_<model>_<split>.csv  — per-row scores

Produces (saved to outputs/figures/ by default):
    1_metric_comparison.png         — grouped bar chart: ROC-AUC / PR-AUC / F1
    2_cold_start_metrics.png        — cold-start ROC-AUC / PR-AUC / BEDROC / EF
    3_roc_curves.png                — ROC curve per model (cold-protein split)
    4_pr_curves.png                 — Precision-Recall curve per model
    5_confusion_matrices.png        — confusion matrix grid
    6_cv_results.png                — cross-validation ROC-AUC with error bars
    7_score_distributions.png       — predicted probability histograms by true class
    summary_table.csv               — machine-readable table of all key metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — safe on servers and in CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# ── Aesthetics ────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "interaction_mlp",
]
MODEL_LABELS = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "interaction_mlp": "MLP",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
MODEL_COLOR = dict(zip(MODEL_ORDER, PALETTE))

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1_binary": "F1 (binary)",
    "ef_at_1pct": "EF@1%",
    "ef_at_5pct": "EF@5%",
    "bedroc": "BEDROC (α=20)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _present_models(df_or_keys, model_order: List[str]) -> List[str]:
    """Return ordered list of models that actually appear in the data."""
    if isinstance(df_or_keys, pd.DataFrame):
        available = set(df_or_keys["model"].unique()) if "model" in df_or_keys.columns else set(df_or_keys.index)
    else:
        available = set(df_or_keys)
    return [m for m in model_order if m in available]


# ── Figure 1: Metric comparison bar chart ────────────────────────────────────

def fig_metric_comparison(
    comparison: pd.DataFrame,
    out: Path,
    dpi: int,
    metrics: List[str] = ("roc_auc", "pr_auc", "f1_binary"),
) -> None:
    models = _present_models(comparison, MODEL_ORDER)
    metrics = [m for m in metrics if m in comparison.columns]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = [comparison.loc[m, metric] if m in comparison.index else float("nan") for m in models]
        colors = [MODEL_COLOR.get(m, "#888888") for m in models]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in models], vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0, 1.08)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Performance (hold-out test set)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out, dpi)


# ── Figure 2: Cold-start metrics ──────────────────────────────────────────────

def fig_cold_start_metrics(
    cold_df: pd.DataFrame,
    out: Path,
    dpi: int,
    metrics: List[str] = ("roc_auc", "pr_auc", "bedroc", "ef_at_1pct"),
) -> None:
    pivot = cold_df.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    metrics = [m for m in metrics if m in pivot.columns]
    models = _present_models(pivot, MODEL_ORDER)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = [pivot.loc[m, metric] if m in pivot.index else float("nan") for m in models]
        colors = [MODEL_COLOR.get(m, "#888888") for m in models]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in models], vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        # scale axis to data range with small padding
        vmax = max((v for v in vals if not np.isnan(v)), default=1.0)
        ax.set_ylim(0, vmax * 1.12)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + vmax * 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    split_label = cold_df["split_type"].iloc[0].replace("_", " ").title() if len(cold_df) > 0 else "Cold-start"
    fig.suptitle(f"Cold-start Metrics ({split_label} split)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out, dpi)


# ── Figure 3 & 4: ROC and PR curves ──────────────────────────────────────────

def _load_prediction_files(pred_dir: Path, split: str) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for path in sorted(pred_dir.glob(f"predictions_*_{split}.csv")):
        # filename: predictions_<model_name>_<split>.csv
        stem = path.stem  # predictions_<model>_<split>
        suffix = f"_{split}"
        model_name = stem[len("predictions_"):][: -len(suffix)]
        dfs[model_name] = pd.read_csv(path)
    return dfs


def fig_roc_curves(pred_dfs: Dict[str, pd.DataFrame], out: Path, dpi: int, split: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")
    models = _present_models(list(pred_dfs.keys()), MODEL_ORDER)

    for model in models:
        df = pred_dfs[model]
        if "y_true" not in df or "y_proba" not in df:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLOR.get(model, "#888888"),
                linewidth=2, label=f"{MODEL_LABELS.get(model, model)} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {split.replace('_', ' ').title()} split")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out, dpi)


def fig_pr_curves(pred_dfs: Dict[str, pd.DataFrame], out: Path, dpi: int, split: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    models = _present_models(list(pred_dfs.keys()), MODEL_ORDER)

    baseline = None
    for model in models:
        df = pred_dfs[model]
        if "y_true" not in df or "y_proba" not in df:
            continue
        pos_rate = df["y_true"].mean()
        if baseline is None:
            baseline = pos_rate
        prec, rec, _ = precision_recall_curve(df["y_true"], df["y_proba"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=MODEL_COLOR.get(model, "#888888"),
                linewidth=2, label=f"{MODEL_LABELS.get(model, model)} (AUC={pr_auc:.3f})")

    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8,
                   label=f"Random (precision={baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {split.replace('_', ' ').title()} split")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out, dpi)


# ── Figure 5: Confusion matrices ──────────────────────────────────────────────

def fig_confusion_matrices(
    pred_dfs: Dict[str, pd.DataFrame],
    out: Path,
    dpi: int,
    split: str,
) -> None:
    models = _present_models(list(pred_dfs.keys()), MODEL_ORDER)
    n = len(models)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]

    for ax, model in zip(axes, models):
        df = pred_dfs[model]
        if "y_true" not in df or "y_pred" not in df:
            ax.set_visible(False)
            continue
        cm = confusion_matrix(df["y_true"], df["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(MODEL_LABELS.get(model, model))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Confusion Matrices — {split.replace('_', ' ').title()} split",
                 fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out, dpi)


# ── Figure 6: Cross-validation results ───────────────────────────────────────

def fig_cv_results(results: dict, out: Path, dpi: int) -> None:
    cv_entries = {k: v for k, v in results.items() if k.startswith("cv_")}
    if not cv_entries:
        print("  No CV results found — skipping figure 6.")
        return

    cv_metrics = ["roc_auc", "average_precision", "f1_macro"]

    rows = []
    for key, cv in cv_entries.items():
        model = key[len("cv_"):]
        for metric in cv_metrics:
            if metric in cv:
                rows.append({
                    "model": model,
                    "metric": metric,
                    "mean": cv[metric]["mean"],
                    "std": cv[metric]["std"],
                })
    if not rows:
        return

    cv_df = pd.DataFrame(rows)
    metrics_present = [m for m in cv_metrics if m in cv_df["metric"].values]
    models = _present_models(cv_df, MODEL_ORDER)

    fig, axes = plt.subplots(1, len(metrics_present), figsize=(4.5 * len(metrics_present), 5), sharey=False)
    if len(metrics_present) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_present):
        sub = cv_df[cv_df["metric"] == metric].set_index("model")
        vals = [sub.loc[m, "mean"] if m in sub.index else float("nan") for m in models]
        errs = [sub.loc[m, "std"] if m in sub.index else 0.0 for m in models]
        colors = [MODEL_COLOR.get(m, "#888888") for m in models]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in models], vals,
                      yerr=errs, capsize=4, color=colors, edgecolor="white", linewidth=0.8,
                      error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
        ax.set_ylim(0, 1.08)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("CV Score (mean ± std)")
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)

    fig.suptitle("Cross-validation Results (protein-aware groups)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out, dpi)


# ── Figure 7: Score distributions ────────────────────────────────────────────

def fig_score_distributions(pred_dfs: Dict[str, pd.DataFrame], out: Path, dpi: int) -> None:
    models = _present_models(list(pred_dfs.keys()), MODEL_ORDER)
    n = len(models)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]

    for ax, model in zip(axes, models):
        df = pred_dfs[model]
        if "y_true" not in df or "y_proba" not in df:
            ax.set_visible(False)
            continue
        neg = df.loc[df["y_true"] == 0, "y_proba"]
        pos = df.loc[df["y_true"] == 1, "y_proba"]
        bins = np.linspace(0, 1, 30)
        ax.hist(neg, bins=bins, alpha=0.6, color="#4C72B0", label="Non-binder (0)", density=True)
        ax.hist(pos, bins=bins, alpha=0.6, color="#DD8452", label="Binder (1)", density=True)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Predicted Score Distributions by True Class", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out, dpi)


# ── Summary table ─────────────────────────────────────────────────────────────

def save_summary_table(
    comparison: pd.DataFrame,
    cold_df: pd.DataFrame,
    results: dict,
    out: Path,
) -> None:
    rows = []
    models = _present_models(comparison, MODEL_ORDER)
    cold_pivot = cold_df.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")

    for model in models:
        row = {"model": model}
        for col in ("roc_auc", "pr_auc", "f1_binary", "accuracy"):
            row[f"test_{col}"] = comparison.loc[model, col] if model in comparison.index and col in comparison.columns else float("nan")
        for metric in ("roc_auc", "pr_auc", "bedroc", "ef_at_1pct", "ef_at_5pct"):
            row[f"cold_{metric}"] = cold_pivot.loc[model, metric] if model in cold_pivot.index and metric in cold_pivot.columns else float("nan")
        cv_key = f"cv_{model}"
        if cv_key in results:
            cv = results[cv_key]
            row["cv_roc_auc_mean"] = cv.get("roc_auc", {}).get("mean", float("nan"))
            row["cv_roc_auc_std"] = cv.get("roc_auc", {}).get("std", float("nan"))
        rows.append(row)

    summary = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False, float_format="%.4f")
    print(f"  Saved → {out}")
    print(summary.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate summary figures from a training run")
    p.add_argument("--output_dir", default="outputs/figures", help="Where to save figures")
    p.add_argument("--results_dir", default="outputs", help="Directory containing outputs")
    p.add_argument("--split", default=None,
                   help="Prediction split to plot ROC/PR/CM for (auto-detected if omitted)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    pred_dir = results_dir / "predictions"
    dpi = args.dpi

    # ── Load data ──────────────────────────────────────────────────────────────
    comparison_path = results_dir / "model_comparison.csv"
    cold_path = pred_dir / "cold_start_summary.csv"
    results_path = results_dir / "results.json"

    if not comparison_path.exists():
        sys.exit(f"ERROR: {comparison_path} not found. Run run_training.py first.")

    comparison = pd.read_csv(comparison_path, index_col="model")
    cold_df = pd.read_csv(cold_path) if cold_path.exists() else pd.DataFrame()
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    # Auto-detect split from prediction files
    split = args.split
    if split is None and pred_dir.exists():
        pred_files = list(pred_dir.glob("predictions_*.csv"))
        if pred_files:
            # Extract split name from filenames (last segment after final underscore grouping)
            splits = set()
            for f in pred_files:
                parts = f.stem.split("_")  # predictions_model_name_split
                # Split is everything after "predictions_" and the model name
                # We'll use a heuristic: join last 1-2 parts
                for i in range(1, len(parts)):
                    candidate = "_".join(parts[i:])
                    if candidate in ("cold_protein", "cold_ligand", "random", "scaffold", "cold_both"):
                        splits.add(candidate)
            split = sorted(splits)[0] if splits else "cold_protein"
    split = split or "cold_protein"

    pred_dfs = _load_prediction_files(pred_dir, split) if pred_dir.exists() else {}

    print(f"\nGenerating figures → {out_dir}/")
    print(f"  Models found : {list(comparison.index)}")
    print(f"  Prediction split: {split}")
    print(f"  Prediction files: {list(pred_dfs.keys())}")
    print()

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_metric_comparison(comparison, out_dir / "1_metric_comparison.png", dpi)
    if not cold_df.empty:
        fig_cold_start_metrics(cold_df, out_dir / "2_cold_start_metrics.png", dpi)
    if pred_dfs:
        fig_roc_curves(pred_dfs, out_dir / "3_roc_curves.png", dpi, split)
        fig_pr_curves(pred_dfs, out_dir / "4_pr_curves.png", dpi, split)
        fig_confusion_matrices(pred_dfs, out_dir / "5_confusion_matrices.png", dpi, split)
        fig_score_distributions(pred_dfs, out_dir / "7_score_distributions.png", dpi)
    fig_cv_results(results, out_dir / "6_cv_results.png", dpi)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\nSummary table:")
    save_summary_table(comparison, cold_df, results, out_dir / "summary_table.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
