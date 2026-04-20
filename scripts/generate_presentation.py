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
             "Protein–Ligand Binding Prediction  |  KIBA Dataset  |  1 000-protein run",
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
             "1 000-protein benchmark  ·  Cold-protein evaluation  ·  5 models",
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
        ("1  Data",       "KIBA kinase activity\ndataset · binder\nthreshold < 12.1"),
        ("2  Proteins",   "UniProt sequences\n→ ESM-2 (650M)\nmean+max pooling\ndim = 960"),
        ("3  Ligands",    "PubChem SMILES\n→ Morgan+MACCS+\natom-pair FPs\n+ RDKit  dim=2229"),
        ("4  Split",      "Cold-protein\n(no UniProt_ID\noverlap between\ntrain & test)"),
        ("5  Models",     "Logistic Reg.\nRandom Forest\nXGBoost · LightGBM\nInteraction MLP"),
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
    header(fig, "Model Comparison — Test Set (Cold-Protein Split)")

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
    header(fig, "ROC and PR Curves — Cold-Protein Test Set")
    embed_image(fig, str(figures_dir / "3_roc_curves.png"), [0.03, 0.10, 0.47, 0.72])
    embed_image(fig, str(figures_dir / "4_pr_curves.png"),  [0.52, 0.10, 0.47, 0.72])
    footer(fig, 4, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_confusion(pdf, figures_dir: Path):
    fig = new_slide()
    header(fig, "Confusion Matrices — Cold-Protein Test Set")
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

    embed_image(fig, str(figures_dir / "6_cv_results.png"), [0.05, 0.10, 0.58, 0.72])

    ax = fig.add_axes([0.66, 0.38, 0.31, 0.40])
    ax.set_facecolor(CARD); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.93, "Key Finding", fontsize=13, fontweight="bold",
            color=ACCENT, ha="center", va="top")
    ax.text(0.5, 0.72, "CV ROC-AUC ≈ 0.52\n(near chance)",
            fontsize=18, fontweight="bold", color=RED, ha="center", va="top")
    ax.text(0.5, 0.44,
            "Honest estimate for unseen proteins.\nSingle-split 0.77 is optimistic —\nproteins leak between folds.",
            fontsize=10, color=TEXT, ha="center", va="top", linespacing=1.5)

    lr_row = summary[summary["model"] == "logistic_regression"]
    if not lr_row.empty and "cv_roc_auc_mean" in lr_row.columns:
        mean = float(lr_row["cv_roc_auc_mean"].values[0])
        std  = float(lr_row["cv_roc_auc_std"].values[0])
        fig.text(0.685, 0.36, f"LR CV: {mean:.3f} ± {std:.3f}",
                 fontsize=11, color=SUBTEXT)

    footer(fig, 7, 9)
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def slide_summary_table(pdf, summary: pd.DataFrame):
    fig = new_slide()
    header(fig, "Results Summary — All Models")

    cols_show  = ["model", "test_roc_auc", "test_pr_auc", "test_f1_binary",
                  "test_accuracy", "cold_bedroc", "cold_ef_at_1pct"]
    col_labels = ["Model", "ROC-AUC", "PR-AUC", "F1", "Accuracy", "BEDROC", "EF@1%"]

    df = summary[cols_show].copy()
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
            "Cold-protein CV ROC-AUC ≈ 0.52 — generalisation",
            "to novel proteins is near chance-level",
            "",
            "KIBA scores are partially estimated (not all measured)",
            "",
            "Ligand features are fingerprint-based — no 3-D geometry",
            "",
            "No cross-docking or structure-aware features used",
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
    p.add_argument("--results_dir", default="outputs/run_1000p")
    p.add_argument("--output", default="Presentation/protein_ligand_results.pdf")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
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
        d["Title"]   = "Protein-Ligand Binding Prediction — 1000-Protein Benchmark"
        d["Author"]  = "abuchin"
        d["Subject"] = "ML results on KIBA dataset, cold-protein evaluation"

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
