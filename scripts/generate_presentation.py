#!/usr/bin/env python3
"""Generate a PDF presentation from training results.

Reads results.json and per-model prediction CSVs to generate all figures
and slides inline — no dependency on pre-existing PNG files.

Usage:
    python scripts/generate_presentation.py
    python scripts/generate_presentation.py --results_dir outputs --output Presentation/protein_ligand_results.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)

# ── Constants ──────────────────────────────────────────────────────────────────
W, H = 16, 9

BG      = "#0f1923"
ACCENT  = "#00c9a7"
TEXT    = "#e8edf2"
SUBTEXT = "#8fa3b8"
CARD    = "#1c2b3a"
GRID    = "#243447"
RED     = "#ff6b6b"
YELLOW  = "#ffd93d"
BLUE    = "#4dc9f6"
PURPLE  = "#9b59b6"
GREEN   = "#2ecc71"

MODEL_COLORS = {
    "logistic_regression": BLUE,
    "random_forest":       GREEN,
    "xgboost":             YELLOW,
    "lightgbm":            ACCENT,
    "interaction_mlp":     PURPLE,
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Reg.",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "interaction_mlp":     "MLP",
}

SKLEARN_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
ALL_MODELS     = SKLEARN_MODELS + ["interaction_mlp"]

RUN_LABEL     = "1000-protein cold_both run"
SPLIT_LABEL   = "Cold-Both"
PROTEIN_LABEL = "1 000 proteins"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict:
    path = results_dir / "results.json"
    with path.open() as f:
        return json.load(f)


def load_predictions(pred_dir: Path, split: str = "cold_both") -> dict[str, pd.DataFrame]:
    """Load per-model prediction CSVs. Returns {model_name: df}."""
    preds = {}
    for model in ALL_MODELS:
        p = pred_dir / f"predictions_{model}_{split}.csv"
        if p.exists():
            preds[model] = pd.read_csv(p)
    return preds


def build_summary_df(results: dict, split: str = "cold_both") -> pd.DataFrame:
    """Build summary DataFrame from results.json."""
    rows = []
    for model in ALL_MODELS:
        m = results.get(model)
        if m is None:
            continue
        row: dict = {"model": model}
        for metric in ("roc_auc", "pr_auc", "f1_binary", "accuracy"):
            row[f"test_{metric}"] = m.get(metric, float("nan"))

        # Cold-start metrics from the cold_start block
        cs = results.get("cold_start", {})
        cs_df = pd.DataFrame(cs) if cs else pd.DataFrame()
        if not cs_df.empty:
            mask = (cs_df.get("model", pd.Series()) == model) & \
                   (cs_df.get("split_type", pd.Series()) == split)
            for metric_name, col_name in [("bedroc", "cold_bedroc"),
                                           ("ef_at_1pct", "cold_ef_at_1pct"),
                                           ("ef_at_5pct", "cold_ef_at_5pct")]:
                subset = cs_df[mask & (cs_df.get("metric", pd.Series()) == metric_name)]
                row[col_name] = float(subset["value"].values[0]) if not subset.empty else float("nan")

        # CV metrics
        cv = results.get(f"cv_{model}")
        if cv:
            row["cv_roc_auc_mean"] = cv.get("roc_auc", {}).get("mean", float("nan"))
            row["cv_roc_auc_std"]  = cv.get("roc_auc", {}).get("std", float("nan"))
            row["cv_prauc_mean"]   = cv.get("average_precision", {}).get("mean", float("nan"))
            row["cv_prauc_std"]    = cv.get("average_precision", {}).get("std", float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_slide(fig=None):
    if fig is not None:
        plt.close(fig)
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    return fig


def header(fig, title: str, subtitle: str = ""):
    fig.text(0.05, 0.92, title, fontsize=28, fontweight="bold",
             color=TEXT, va="top", ha="left")
    if subtitle:
        fig.text(0.05, 0.87, subtitle, fontsize=13, color=SUBTEXT,
                 va="top", ha="left")
    ax = fig.add_axes([0.05, 0.855, 0.9, 0.003])
    ax.set_facecolor(ACCENT)
    ax.axis("off")


def footer(fig, page: int, total: int):
    fig.text(0.95, 0.025, f"{page} / {total}", fontsize=10,
             color=SUBTEXT, ha="right", va="bottom")
    fig.text(0.05, 0.025,
             f"Protein–Ligand Binding Prediction  |  KIBA Dataset  |  {RUN_LABEL}",
             fontsize=10, color=SUBTEXT, ha="left", va="bottom")


def _style_ax(ax):
    ax.set_facecolor(CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SUBTEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


# ── Slides ─────────────────────────────────────────────────────────────────────

def slide_title(pdf):
    fig = new_slide()
    n_models = len(MODEL_LABELS)
    fig.text(0.5, 0.60, "Protein–Ligand Binding Prediction",
             fontsize=40, fontweight="bold", color=TEXT, ha="center", va="center")
    fig.text(0.5, 0.50, "Machine Learning on the KIBA Dataset",
             fontsize=22, color=ACCENT, ha="center", va="center")
    fig.text(0.5, 0.42,
             f"{PROTEIN_LABEL} benchmark  ·  {SPLIT_LABEL} evaluation  ·  {n_models} models + CV",
             fontsize=14, color=SUBTEXT, ha="center", va="center")
    ax = fig.add_axes([0.3, 0.37, 0.4, 0.003])
    ax.set_facecolor(ACCENT); ax.axis("off")
    fig.text(0.5, 0.28,
             "ESM-2 protein embeddings  ·  Morgan + MACCS fingerprints  ·  RDKit descriptors",
             fontsize=12, color=SUBTEXT, ha="center", va="center")
    footer(fig, 1, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_overview(pdf):
    fig = new_slide()
    header(fig, "Pipeline Overview")

    steps = [
        ("1  Data",       f"KIBA kinase activity\ndataset · binder\nthreshold < 12.1\n{PROTEIN_LABEL}"),
        ("2  Proteins",   "UniProt sequences\n→ ESM-2 (35M)\nmean+max pooling\ndim = 960"),
        ("3  Ligands",    "PubChem SMILES\n→ Morgan+MACCS+\natom-pair FPs\n+ RDKit  dim=2229"),
        ("4  Split",      f"{SPLIT_LABEL}\n(unseen proteins\nAND unseen ligands\nin test set)"),
        ("5  Models",     "Logistic Reg.\nRandom Forest\nXGBoost · LightGBM\nInteractionMLP\n+ 5-fold CV"),
        ("6  Evaluation", "ROC-AUC · PR-AUC\nBEDROC · EF@1%/5%\n5-fold protein-\naware CV"),
    ]

    n = len(steps)
    xs = np.linspace(0.08, 0.92, n)
    box_w, box_h = 0.13, 0.40

    for i, (title, body) in enumerate(steps):
        left = xs[i] - box_w / 2
        ax = fig.add_axes([left, 0.26, box_w, box_h])
        ax.set_facecolor(CARD); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.90, title, fontsize=11, fontweight="bold",
                color=ACCENT, ha="center", va="top")
        ax.text(0.5, 0.72, body, fontsize=9, color=TEXT,
                ha="center", va="top", linespacing=1.5)
        if i < n - 1:
            fig.text(xs[i] + box_w / 2 + 0.005, 0.46, "→",
                     fontsize=18, color=SUBTEXT, ha="left", va="center")

    footer(fig, 2, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_metrics_comparison(pdf, summary: pd.DataFrame):
    fig = new_slide()
    header(fig, f"Model Comparison — Test Set ({SPLIT_LABEL} Split)",
           f"Unseen proteins AND unseen ligands in test · {PROTEIN_LABEL}")

    models = summary["model"].tolist()
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, BLUE) for m in models]

    metrics = [
        ("test_roc_auc",   "ROC-AUC"),
        ("test_pr_auc",    "PR-AUC"),
        ("test_f1_binary", "F1 (binary)"),
        ("test_accuracy",  "Accuracy"),
    ]

    for i, (col, label) in enumerate(metrics):
        ax = fig.add_axes([0.05 + i * 0.235, 0.17, 0.20, 0.62])
        ax.set_facecolor(CARD)
        vals = summary[col].tolist()
        bars = ax.barh(labels, vals, color=colors, height=0.55)
        ax.set_xlim(0, 1.15)
        ax.set_facecolor(CARD)
        ax.set_title(label, color=ACCENT, fontsize=13, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(GRID)
        ax.tick_params(axis="y", colors=TEXT, labelsize=9)
        ax.tick_params(axis="x", colors=SUBTEXT, labelsize=9)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9, color=TEXT)

    footer(fig, 3, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_roc_pr(pdf, preds: dict[str, pd.DataFrame]):
    fig = new_slide()
    header(fig, f"ROC and PR Curves — {SPLIT_LABEL} Test Set",
           "Test proteins and test ligands are entirely unseen during training")

    ax_roc = fig.add_axes([0.05, 0.12, 0.42, 0.68])
    ax_pr  = fig.add_axes([0.54, 0.12, 0.42, 0.68])

    for ax in (ax_roc, ax_pr):
        _style_ax(ax)

    # Diagonal reference
    ax_roc.plot([0, 1], [0, 1], color=SUBTEXT, lw=1, linestyle="--", alpha=0.5)
    ax_pr.axhline(y=0.5, color=SUBTEXT, lw=1, linestyle="--", alpha=0.5)

    for model, df in preds.items():
        if "y_proba" not in df.columns or "y_true" not in df.columns:
            continue
        y_true  = df["y_true"].values
        y_proba = df["y_proba"].values
        color   = MODEL_COLORS.get(model, BLUE)
        label   = MODEL_LABELS.get(model, model)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label}  (AUC={roc_auc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax_pr.plot(rec, prec, color=color, lw=2,
                   label=f"{label}  (AP={ap:.3f})")

    ax_roc.set_xlabel("False Positive Rate", color=SUBTEXT, fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", color=TEXT, fontsize=11)
    ax_roc.set_title("ROC Curves", color=ACCENT, fontsize=14)
    ax_roc.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.05)

    ax_pr.set_xlabel("Recall", color=SUBTEXT, fontsize=11)
    ax_pr.set_ylabel("Precision", color=TEXT, fontsize=11)
    ax_pr.set_title("Precision-Recall Curves", color=ACCENT, fontsize=14)
    ax_pr.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(-0.02, 1.05)

    footer(fig, 4, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_confusion(pdf, preds: dict[str, pd.DataFrame]):
    models_present = [m for m in ALL_MODELS if m in preds]
    n = len(models_present)
    if n == 0:
        return

    fig = new_slide()
    header(fig, f"Confusion Matrices — {SPLIT_LABEL} Test Set",
           f"Evaluation samples from the {RUN_LABEL}")

    # Lay out n matrices in one row
    margin_l, margin_r = 0.04, 0.04
    gap = 0.01
    total_w = 1 - margin_l - margin_r - gap * (n - 1)
    cell_w = total_w / n
    cell_h = 0.62
    top_y  = 0.15

    for i, model in enumerate(models_present):
        df = preds[model]
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values if "y_pred" in df.columns else (df["y_proba"].values > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        x0 = margin_l + i * (cell_w + gap)
        ax = fig.add_axes([x0, top_y, cell_w * 0.85, cell_h])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_facecolor(CARD)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], color=TEXT, fontsize=9)
        ax.set_yticklabels(["Neg", "Pos"], color=TEXT, fontsize=9, rotation=90, va="center")
        ax.set_xlabel("Predicted", color=SUBTEXT, fontsize=9)
        ax.set_ylabel("Actual", color=TEXT, fontsize=9)
        ax.set_title(MODEL_LABELS.get(model, model),
                     color=MODEL_COLORS.get(model, TEXT), fontsize=11, fontweight="bold")

        total = cm.sum()
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                pct = 100 * val / max(total, 1)
                txt_color = "white" if val > cm.max() * 0.5 else TEXT
                ax.text(c, r, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, color=txt_color,
                        fontweight="bold")

    footer(fig, 5, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_score_dist(pdf, preds: dict[str, pd.DataFrame]):
    models_present = [m for m in ALL_MODELS if m in preds]
    n = len(models_present)
    if n == 0:
        return

    fig = new_slide()
    header(fig, "Predicted Score Distributions")

    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    margin_l, margin_r = 0.04, 0.04
    margin_b, top_y = 0.10, 0.12
    h_space, v_space = 0.02, 0.06
    total_w = 1 - margin_l - margin_r - h_space * (cols - 1)
    total_h = 0.72 - v_space * (rows - 1)
    cell_w = total_w / cols
    cell_h = total_h / rows

    for idx, model in enumerate(models_present):
        df = preds[model]
        if "y_proba" not in df.columns:
            continue
        col_i = idx % cols
        row_i = idx // cols
        x0 = margin_l + col_i * (cell_w + h_space)
        y0 = 1 - top_y - (row_i + 1) * cell_h - row_i * v_space

        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        _style_ax(ax)
        color = MODEL_COLORS.get(model, BLUE)
        label = MODEL_LABELS.get(model, model)

        pos_mask = df["y_true"] == 1
        neg_mask = ~pos_mask
        bins = np.linspace(0, 1, 31)
        ax.hist(df.loc[neg_mask, "y_proba"], bins=bins, alpha=0.65,
                color=RED, label="Non-binder", density=True)
        ax.hist(df.loc[pos_mask, "y_proba"], bins=bins, alpha=0.65,
                color=color, label="Binder", density=True)
        ax.set_title(label, color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted probability", color=SUBTEXT, fontsize=8)
        ax.set_ylabel("Density", color=TEXT, fontsize=8)
        ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

    footer(fig, 6, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_cv(pdf, results: dict, summary: pd.DataFrame):
    fig = new_slide()
    header(fig, "5-Fold Protein-Aware Cross-Validation",
           "StratifiedGroupKFold — no protein appears in both train and validation folds")

    cv_models = [m for m in SKLEARN_MODELS if f"cv_{m}" in results]

    # Left: bar chart of CV ROC-AUC
    ax = fig.add_axes([0.05, 0.14, 0.52, 0.66])
    _style_ax(ax)
    means = [results[f"cv_{m}"]["roc_auc"]["mean"] for m in cv_models]
    stds  = [results[f"cv_{m}"]["roc_auc"]["std"]  for m in cv_models]
    labels_cv = [MODEL_LABELS.get(m, m) for m in cv_models]
    colors_cv  = [MODEL_COLORS.get(m, BLUE) for m in cv_models]
    y_pos = np.arange(len(cv_models))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors_cv, height=0.55,
                   error_kw={"ecolor": TEXT, "capsize": 4, "linewidth": 1.5})
    ax.set_yticks(y_pos); ax.set_yticklabels(labels_cv, color=TEXT, fontsize=10)
    ax.set_xlabel("ROC-AUC", color=SUBTEXT, fontsize=11)
    ax.set_title("Cross-Validation ROC-AUC (mean ± std)", color=ACCENT, fontsize=12)
    ax.set_xlim(0.80, 1.05)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}±{std:.3f}", va="center", ha="left", fontsize=9, color=TEXT)

    # Right: table
    ax2 = fig.add_axes([0.62, 0.14, 0.35, 0.66])
    ax2.set_facecolor(CARD); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.97, "CV Summary", fontsize=13, fontweight="bold",
             color=ACCENT, ha="center", va="top")
    ax2.axhline(0.90, color=ACCENT, linewidth=1, xmin=0.03, xmax=0.97)
    ax2.text(0.05, 0.87, "Model", fontsize=9, color=SUBTEXT, va="top", fontweight="bold")
    ax2.text(0.50, 0.87, "ROC-AUC", fontsize=9, color=SUBTEXT, va="top", ha="center", fontweight="bold")
    ax2.text(0.95, 0.87, "PR-AUC", fontsize=9, color=SUBTEXT, va="top", ha="right", fontweight="bold")

    for i, model in enumerate(cv_models):
        cv = results[f"cv_{model}"]
        roc_mean = cv["roc_auc"]["mean"]
        roc_std  = cv["roc_auc"]["std"]
        pr_mean  = cv.get("average_precision", {}).get("mean", float("nan"))
        label    = MODEL_LABELS.get(model, model)
        color    = MODEL_COLORS.get(model, TEXT)
        y = 0.80 - i * 0.175
        ax2.text(0.05, y, label, fontsize=9.5, color=color, va="center", fontweight="bold")
        ax2.text(0.50, y, f"{roc_mean:.3f}±{roc_std:.3f}", fontsize=9, color=TEXT, va="center", ha="center")
        ax2.text(0.95, y, f"{pr_mean:.3f}", fontsize=9, color=TEXT, va="center", ha="right")

    ax2.text(0.5, 0.05,
             "CV is consistent with test set —\nmodels generalise well to unseen proteins.",
             fontsize=8.5, color=SUBTEXT, ha="center", va="bottom", linespacing=1.4)

    footer(fig, 7, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_summary_table(pdf, summary: pd.DataFrame):
    fig = new_slide()
    header(fig, "Results Summary — All Models",
           f"{SPLIT_LABEL} split · {PROTEIN_LABEL} · 5-fold protein-aware CV")

    has_cv = "cv_roc_auc_mean" in summary.columns
    cols_show  = ["model", "test_roc_auc", "test_pr_auc", "test_f1_binary",
                  "cold_bedroc", "cold_ef_at_1pct"]
    col_labels = ["Model", "ROC-AUC", "PR-AUC", "F1", "BEDROC", "EF@1%"]
    if has_cv:
        cols_show += ["cv_roc_auc_mean", "cv_roc_auc_std"]
        col_labels += ["CV ROC-AUC", "CV ± Std"]

    avail_cols   = [c for c in cols_show if c in summary.columns]
    avail_labels = [col_labels[cols_show.index(c)] for c in avail_cols]
    df = summary[avail_cols].copy()
    df["model"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))

    n_rows, n_cols = len(df), len(avail_labels)
    ax = fig.add_axes([0.05, 0.18, 0.90, 0.64])
    ax.set_facecolor(CARD); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for j, lbl in enumerate(avail_labels):
        ax.text((j + 0.5) / n_cols, 0.96, lbl,
                fontsize=13, fontweight="bold", color=ACCENT, ha="center", va="top")
    ax.axhline(0.88, color=ACCENT, linewidth=1.2, xmin=0.01, xmax=0.99)

    for i, row in enumerate(df.itertuples(index=False)):
        y = 0.84 - i * (0.80 / n_rows)
        bg = GRID if i % 2 == 0 else CARD
        rect = plt.Rectangle((0, y - 0.035), 1, 0.08,
                              facecolor=bg, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)
        orig_model = summary["model"].iloc[i]
        for j, val in enumerate(row):
            if j == 0:
                txt   = str(val)
                color = MODEL_COLORS.get(orig_model, TEXT)
            else:
                try:
                    txt = f"{float(val):.4f}"
                except (ValueError, TypeError):
                    txt = "—"
                color = TEXT
            ax.text((j + 0.5) / n_cols, y + 0.005, txt,
                    fontsize=11, color=color, ha="center", va="center")

    footer(fig, 8, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_limitations(pdf):
    fig = new_slide()
    header(fig, "Limitations & Next Steps")

    panels = [
        ("Current Limitations", [
            "Negatives are property-matched decoys, not true",
            "measured non-binders — may inflate performance",
            "",
            f"{PROTEIN_LABEL} subset used for this benchmark",
            "results may vary at full dataset scale",
            "",
            "KIBA scores are partially estimated (not all measured)",
            "",
            "Ligand features are fingerprint-based — no 3-D geometry",
        ]),
        ("Next Steps", [
            "Graph Neural Networks on molecular graphs (GNN-DTI)",
            "",
            "Structure-aware models: AlphaFold2 binding site features",
            "",
            "Multi-task learning across kinase families",
            "",
            "Active learning loop: prioritise wet-lab validation hits",
            "",
            "Larger benchmark: full KIBA + BindingDB + ChEMBL",
        ]),
    ]

    for k, (title, lines) in enumerate(panels):
        ax = fig.add_axes([0.04 + k * 0.50, 0.12, 0.44, 0.70])
        ax.set_facecolor(CARD); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, title, fontsize=14, fontweight="bold",
                color=ACCENT, ha="center", va="top")
        ax.axhline(0.88, color=ACCENT, linewidth=1, xmin=0.05, xmax=0.95)
        for i, line in enumerate(lines):
            y = 0.83 - i * 0.095
            ax.text(0.06, y, ("• " + line) if line else "",
                    fontsize=10.5, color=TEXT, va="top")

    footer(fig, 9, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",   default="outputs")
    p.add_argument("--output",        default="Presentation/protein_ligand_results.pdf")
    p.add_argument("--split",         default="cold_both")
    p.add_argument("--run_label",     default=RUN_LABEL)
    p.add_argument("--split_label",   default=SPLIT_LABEL)
    p.add_argument("--protein_label", default=PROTEIN_LABEL)
    return p.parse_args()


def main():
    global RUN_LABEL, SPLIT_LABEL, PROTEIN_LABEL
    args = parse_args()
    RUN_LABEL     = args.run_label
    SPLIT_LABEL   = args.split_label
    PROTEIN_LABEL = args.protein_label

    results_dir = Path(args.results_dir)
    pred_dir    = results_dir / "predictions"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    preds   = load_predictions(pred_dir, split=args.split)
    summary = build_summary_df(results, split=args.split)

    # Save updated summary_table.csv for reference
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(figures_dir / "summary_table.csv", index=False)
    print(f"Summary table saved → {figures_dir / 'summary_table.csv'}")
    print(f"Models in summary: {summary['model'].tolist()}")

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": BG,
        "axes.facecolor":   CARD,
        "axes.edgecolor":   GRID,
        "axes.labelcolor":  TEXT,
        "xtick.color":      SUBTEXT,
        "ytick.color":      TEXT,
        "text.color":       TEXT,
        "grid.color":       GRID,
    })

    with PdfPages(output_path) as pdf:
        d = pdf.infodict()
        d["Title"]   = f"Protein-Ligand Binding Prediction — {RUN_LABEL}"
        d["Author"]  = "abuchin"
        d["Subject"] = "ML results on KIBA dataset, cold-both evaluation with CV"

        slide_title(pdf)
        slide_overview(pdf)
        slide_metrics_comparison(pdf, summary)
        slide_roc_pr(pdf, preds)
        slide_confusion(pdf, preds)
        slide_score_dist(pdf, preds)
        slide_cv(pdf, results, summary)
        slide_summary_table(pdf, summary)
        slide_limitations(pdf)

    size_kb = output_path.stat().st_size // 1024
    print(f"Saved → {output_path}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
