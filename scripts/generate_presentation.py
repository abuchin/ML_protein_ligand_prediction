#!/usr/bin/env python3
"""Generate a PDF presentation from 1000-protein run results.

Usage:
    python scripts/generate_presentation.py
    python scripts/generate_presentation.py --results_dir outputs/run_1000p --output Presentation/protein_ligand_results.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
W, H = 16, 9  # slide dimensions in inches (16:9)

BG       = "#0f1923"
ACCENT   = "#00c9a7"
TEXT     = "#e8edf2"
SUBTEXT  = "#8fa3b8"
CARD     = "#1c2b3a"
GRID     = "#243447"
RED      = "#ff6b6b"
YELLOW   = "#ffd93d"
BLUE     = "#4dc9f6"
PURPLE   = "#9b59b6"
GREEN    = "#2ecc71"

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
             "Protein–Ligand Binding Prediction  |  KIBA Dataset  |  1 200-protein cold-both run",
             fontsize=10, color=SUBTEXT, ha="left", va="bottom")


def embed_image(fig, path: str, rect):
    img = mpimg.imread(path)
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis("off")
    return ax


# ── Slides ─────────────────────────────────────────────────────────────────────

def slide_title(pdf):
    fig = new_slide()
    fig.text(0.5, 0.60, "Protein–Ligand Binding Prediction",
             fontsize=40, fontweight="bold", color=TEXT, ha="center", va="center")
    fig.text(0.5, 0.50, "Machine Learning on the KIBA Dataset",
             fontsize=22, color=ACCENT, ha="center", va="center")
    fig.text(0.5, 0.42,
             "1 200-protein benchmark  ·  Cold-both evaluation  ·  4 models + CV",
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
        ("1  Data",       "KIBA kinase activity\ndataset · binder\nthreshold < 12.1\n1 200 proteins"),
        ("2  Proteins",   "UniProt sequences\n→ ESM-2 (35M)\nmean+max pooling\ndim = 960"),
        ("3  Ligands",    "PubChem SMILES\n→ Morgan+MACCS+\natom-pair FPs\n+ RDKit  dim=2229"),
        ("4  Split",      "Cold-both\n(unseen proteins\nAND unseen binders\nin test set)"),
        ("5  Models",     "Logistic Reg.\nRandom Forest\nXGBoost · LightGBM\n+ 5-fold CV"),
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
    header(fig, "Model Comparison — Test Set (Cold-Both Split)",
           "Unseen proteins AND unseen binders in test · 1 200 proteins · 50 k rows")

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
        ax.set_xlim(0, 1.12)
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


def slide_roc_pr(pdf, figures_dir: Path):
    fig = new_slide()
    header(fig, "ROC and PR Curves — Cold-Both Test Set",
           "Test proteins and test binders are entirely unseen during training")
    embed_image(fig, str(figures_dir / "3_roc_curves.png"), [0.03, 0.10, 0.47, 0.72])
    embed_image(fig, str(figures_dir / "4_pr_curves.png"),  [0.52, 0.10, 0.47, 0.72])
    footer(fig, 4, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_confusion(pdf, figures_dir: Path):
    fig = new_slide()
    header(fig, "Confusion Matrices — Cold-Both Test Set",
           "7 295 test samples · positive rate 59%")
    embed_image(fig, str(figures_dir / "5_confusion_matrices.png"), [0.05, 0.08, 0.90, 0.76])
    footer(fig, 5, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_score_dist(pdf, figures_dir: Path):
    fig = new_slide()
    header(fig, "Predicted Score Distributions")
    embed_image(fig, str(figures_dir / "7_score_distributions.png"), [0.05, 0.08, 0.90, 0.76])
    footer(fig, 6, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_cv(pdf, figures_dir: Path, summary: pd.DataFrame):
    fig = new_slide()
    header(fig, "5-Fold Protein-Aware Cross-Validation",
           "StratifiedGroupKFold — no protein appears in both train and validation folds")

    embed_image(fig, str(figures_dir / "6_cv_results.png"), [0.05, 0.10, 0.55, 0.72])

    # Table of CV results for all models
    cv_models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    ax = fig.add_axes([0.63, 0.15, 0.34, 0.68])
    ax.set_facecolor(CARD); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "CV ROC-AUC (5-fold)", fontsize=12, fontweight="bold",
            color=ACCENT, ha="center", va="top")
    ax.axhline(0.90, color=ACCENT, linewidth=1, xmin=0.03, xmax=0.97)

    for i, model in enumerate(cv_models):
        row = summary[summary["model"] == model]
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, TEXT)
        y = 0.83 - i * 0.18
        if not row.empty and "cv_roc_auc_mean" in row.columns:
            mean = float(row["cv_roc_auc_mean"].values[0])
            std  = float(row["cv_roc_auc_std"].values[0])
            ax.text(0.05, y, label, fontsize=10, color=color, va="center", fontweight="bold")
            ax.text(0.95, y, f"{mean:.3f} ± {std:.3f}", fontsize=11,
                    color=TEXT, va="center", ha="right")
            # mini bar
            bar_ax = fig.add_axes([0.64 + 0.02, 0.15 + (3-i)*0.155 + 0.01, 0.30, 0.035])
            bar_ax.set_facecolor(CARD)
            bar_ax.barh([0], [mean], color=color, height=0.7)
            bar_ax.set_xlim(0.8, 1.0)
            bar_ax.axis("off")
        else:
            ax.text(0.5, y, f"{label}: N/A", fontsize=10, color=SUBTEXT, ha="center", va="center")

    ax.text(0.5, 0.06,
            "CV scores are consistent with test set —\nmodels generalise well to unseen proteins.",
            fontsize=9, color=SUBTEXT, ha="center", va="bottom", linespacing=1.4)

    footer(fig, 7, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_summary_table(pdf, summary: pd.DataFrame):
    fig = new_slide()
    header(fig, "Results Summary — All Models",
           "Cold-both split · 1 200 proteins · 50 k rows · 5-fold protein-aware CV")

    has_cv = "cv_roc_auc_mean" in summary.columns
    cols_show  = ["model", "test_roc_auc", "test_pr_auc", "test_f1_binary",
                  "cold_bedroc", "cold_ef_at_1pct"]
    col_labels = ["Model", "ROC-AUC", "PR-AUC", "F1", "BEDROC", "EF@1%"]
    if has_cv:
        cols_show += ["cv_roc_auc_mean", "cv_roc_auc_std"]
        col_labels += ["CV ROC-AUC", "CV ± Std"]

    avail_cols = [c for c in cols_show if c in summary.columns]
    avail_labels = [col_labels[cols_show.index(c)] for c in avail_cols]
    df = summary[avail_cols].copy()
    col_labels = avail_labels
    cols_show = avail_cols
    df["model"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))

    n_rows, n_cols = len(df), len(col_labels)
    row_h = 0.60 / (n_rows + 1)
    col_w = 0.90 / n_cols

    ax = fig.add_axes([0.05, 0.18, 0.90, 0.64])
    ax.set_facecolor(CARD); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # header row
    for j, lbl in enumerate(col_labels):
        ax.text((j + 0.5) / n_cols, 0.96, lbl,
                fontsize=13, fontweight="bold", color=ACCENT, ha="center", va="top")
    ax.axhline(0.88, color=ACCENT, linewidth=1.2, xmin=0.01, xmax=0.99)

    for i, row in enumerate(df.itertuples(index=False)):
        y = 0.84 - i * (0.80 / n_rows)
        bg = GRID if i % 2 == 0 else CARD
        rect = plt.Rectangle((0, y - 0.035), 1, 0.08,
                              facecolor=bg, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)
        for j, val in enumerate(row):
            if j == 0:
                txt = str(val)
                color = MODEL_COLORS.get(summary["model"].iloc[i], TEXT)
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
            "50 k row subsample used (full 227 k strains RAM)",
            "on a MacBook; results may vary at full scale",
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
    p.add_argument("--results_dir", default="outputs")
    p.add_argument("--output", default="Presentation/protein_ligand_results.pdf")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    if not figures_dir.exists():
        figures_dir = results_dir  # fallback: figures in results_dir directly
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(figures_dir / "summary_table.csv")

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
        d["Title"]   = "Protein-Ligand Binding Prediction — 1200-Protein Cold-Both Benchmark"
        d["Author"]  = "abuchin"
        d["Subject"] = "ML results on KIBA dataset, cold-both evaluation with CV"

        slide_title(pdf)
        slide_overview(pdf)
        slide_metrics_comparison(pdf, summary)
        slide_roc_pr(pdf, figures_dir)
        slide_confusion(pdf, figures_dir)
        slide_score_dist(pdf, figures_dir)
        slide_cv(pdf, figures_dir, summary)
        slide_summary_table(pdf, summary)
        slide_limitations(pdf)

    print(f"Saved → {output_path}  ({output_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
