#!/usr/bin/env python3
"""Generate a PDF presentation summarising the protein-ligand binding prediction project.

Usage:
    python scripts/generate_presentation.py
    python scripts/generate_presentation.py --out Presentation/results.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# ── Colour palette ────────────────────────────────────────────────────────────
DARK   = "#1a1a2e"
ACCENT = "#16213e"
BLUE   = "#0f3460"
GOLD   = "#e94560"
LIGHT  = "#f5f5f5"
WHITE  = "#ffffff"

MODEL_COLOR = {
    "logistic_regression": "#4C72B0",
    "random_forest":       "#DD8452",
    "xgboost":             "#55A868",
    "lightgbm":            "#C44E52",
    "interaction_mlp":     "#8172B2",
}
MODEL_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "interaction_mlp":     "InteractionMLP",
}

SLIDE_W, SLIDE_H = 13, 7.5   # inches (16:9-ish)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": DARK,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blank(bg=LIGHT):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(bg)
    return fig


def _header(fig, title: str, subtitle: str = "", bg=BLUE, fg=WHITE):
    ax = fig.add_axes([0, 0.82, 1, 0.18])
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.62, title,   color=fg, fontsize=18, fontweight="bold", va="center")
    ax.text(0.04, 0.22, subtitle, color=GOLD, fontsize=11, va="center")


def _footer(fig, text="Protein-Ligand Binding Prediction  |  KIBA Dataset  |  Cold-Both Benchmark"):
    ax = fig.add_axes([0, 0, 1, 0.05])
    ax.set_facecolor(ACCENT)
    ax.axis("off")
    ax.text(0.5, 0.5, text, color=WHITE, fontsize=8, ha="center", va="center")


def _img_ax(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    return ax


def _paste_image(fig, path: Path, rect):
    ax = _img_ax(fig, rect)
    img = plt.imread(str(path))
    ax.imshow(img, aspect="auto")


# ── Slides ────────────────────────────────────────────────────────────────────

def slide_title(pdf):
    fig = _blank(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK); ax.axis("off")

    # gradient-ish bar
    bar = fig.add_axes([0, 0.42, 1, 0.16])
    bar.set_facecolor(BLUE); bar.axis("off")
    bar.text(0.5, 0.65, "Protein-Ligand Binding Prediction",
             color=WHITE, fontsize=26, fontweight="bold", ha="center", va="center")
    bar.text(0.5, 0.18, "Machine Learning Benchmark on the KIBA Dataset",
             color=GOLD, fontsize=14, ha="center", va="center")

    ax.text(0.5, 0.30, "748 proteins  ·  13,361 ligands  ·  17,788 interactions",
            color="#aaaacc", fontsize=13, ha="center", va="center")
    ax.text(0.5, 0.20, "Cold-Both Split  ·  5 Models  ·  LightGBM ROC-AUC = 0.776",
            color=GOLD, fontsize=13, ha="center", va="center")
    ax.text(0.5, 0.08, "github.com/abuchin/ML_protein_ligand_prediction",
            color="#666688", fontsize=10, ha="center", va="center")

    _footer(fig)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_problem(pdf):
    fig = _blank()
    _header(fig, "Problem Statement", "Drug discovery: predicting which small molecules bind to a target protein")
    _footer(fig)

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.68])
    ax.set_facecolor(LIGHT); ax.axis("off")

    blocks = [
        ("Task",
         "Binary classification: does compound X bind to protein Y?\n"
         "Positive (binder): KIBA score < 12.1   |   Negative: KIBA score ≥ 12.1"),
        ("Dataset",
         "KIBA (Kinase Inhibitor BioActivity) — 1.13M measured & imputed\n"
         "binding affinities across 4,480 kinases and 683,413 compounds.\n"
         "This run: 748 proteins · 13,361 ligands · 17,788 interactions (measured only)"),
        ("Evaluation challenge",
         "Random split → inflated scores (models memorise protein/ligand combos).\n"
         "Cold-both split → proteins AND ligands unseen at test time → honest estimate\n"
         "of real-world virtual screening performance."),
        ("Key metrics",
         "ROC-AUC (threshold-free ranking)  ·  PR-AUC (imbalance-robust)\n"
         "EF@1% — Enrichment Factor (drug-discovery standard): how many more\n"
         "actives in the top 1% than expected by chance?"),
    ]

    y = 0.93
    for title, body in blocks:
        ax.text(0.01, y, f"▸  {title}", fontsize=12, fontweight="bold", color=BLUE, va="top")
        ax.text(0.04, y - 0.07, body, fontsize=10, color=DARK, va="top",
                linespacing=1.5)
        y -= 0.28

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_architecture(pdf):
    fig = _blank()
    _header(fig, "System Architecture", "End-to-end pipeline from raw KIBA scores to trained models")
    _footer(fig)

    # Pipeline flow diagram
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.70])
    ax.set_facecolor(LIGHT); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    stages = [
        (0.5, "Raw Data\n(KIBA CSV)",        "#d0e4f7"),
        (2.5, "Preprocessor\n(labels + decoys)", "#d0f0d0"),
        (4.5, "Feature Encoding\n(ESM-2 + RDKit)", "#f7e0d0"),
        (6.5, "Train/Val/Test\nSplit",        "#f0d0f0"),
        (8.5, "Model Training\n& Evaluation", "#fff0c0"),
    ]

    for x, label, color in stages:
        rect = FancyBboxPatch((x - 0.85, 1.5), 1.7, 1.0,
                              boxstyle="round,pad=0.08", facecolor=color,
                              edgecolor=BLUE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 2.0, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=DARK)
        if x < 8.5:
            ax.annotate("", xy=(x + 0.85 + 0.3, 2.0), xytext=(x + 0.85, 2.0),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

    # Sub-labels
    sublabels = [
        (0.5,  "1.13M rows\n4,480 proteins"),
        (2.5,  "KIBA < 12.1\n= binder"),
        (4.5,  "960-dim protein\n2,214-dim ligand"),
        (6.5,  "cold_protein\ncold_both"),
        (8.5,  "LR · RF · XGB\nLGB · MLP"),
    ]
    for x, txt in sublabels:
        ax.text(x, 1.2, txt, ha="center", va="center", fontsize=8,
                color="#555555", style="italic")

    # Code modules
    ax.text(0.5, 3.6, "Code structure  —  src/plbind/", fontsize=11,
            fontweight="bold", color=BLUE)
    modules = [
        "data/preprocessor.py\ndata/protein_encoder.py  (ESM-2)\ndata/ligand_encoder.py  (RDKit)",
        "training/splitter.py\n(random / cold_protein /\ncold_ligand / scaffold / cold_both)",
        "models/  (LR, RF, XGBoost,\nLightGBM, InteractionMLP)\nmodels/base.py  (BaseModel API)",
        "evaluation/evaluator.py\nevaluation/cold_start.py\n(EF, BEDROC, ROC, PR)",
    ]
    xs = [1.5, 3.5, 6.0, 8.5]
    for x, txt in zip(xs, modules):
        ax.text(x, 3.55, txt, fontsize=7.5, color=DARK, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                          edgecolor="#cccccc", linewidth=0.8))

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_features(pdf):
    fig = _blank()
    _header(fig, "Feature Engineering",
            "Protein: ESM-2 language model embeddings  ·  Ligand: Morgan + MACCS + atom-pair + descriptors")
    _footer(fig)

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.70])
    ax.set_facecolor(LIGHT); ax.axis("off")

    # Two columns
    left = [
        ("Protein features  (960 dim)", [
            "ESM-2  esm2_t12_35M_UR50D  (Meta, 2022)",
            "12-layer transformer, 35M parameters",
            "mean + max pooling over residues → 2 × 480 = 960 dim",
            "Pre-trained on 250M UniRef50 sequences",
            "Cached per UniProt_ID to avoid re-encoding",
        ]),
        ("Auxiliary protein features  (~95 dim)", [
            "GO terms (top 50), Pfam domains (top 30)",
            "Subcellular location, organism one-hot",
            "Fetched from UniProt REST API",
        ]),
    ]
    right = [
        ("Ligand features  (2,214 dim fingerprint  +  15 descriptors)", [
            "Morgan count fingerprint  radius-2, 1024 bits  (ECFP4 counts)",
            "MACCS keys  166 bits  (pharmacophore patterns)",
            "Atom-pair fingerprint  1024 bits  (inter-atom distances)",
            "15 physicochemical descriptors: MW, LogP, TPSA, HBD/HBA,",
            "  aromaticity, QED, Gasteiger charges, stereocentres …",
            "Stored as scipy.sparse (saves ~8 GB vs dense at 500k rows)",
        ]),
        ("InteractionMLP architecture", [
            "Protein → Linear(960, 256) → LayerNorm → ReLU  =  prot_h",
            "Ligand  → Linear(2229, 256) → LayerNorm → ReLU  =  lig_h",
            "interaction_h = prot_h  ×  lig_h   (element-wise product)",
            "concat([prot_h, lig_h, interaction_h]) → 768 → 512 → 256 → 2",
            "FocalLoss (γ=2, α=0.25), Adam, ReduceLROnPlateau, MPS/CUDA",
        ]),
    ]

    for col, items, x0 in [(left, None, 0.01), (right, None, 0.50)]:
        y = 0.97
        for heading, lines in col:
            ax.text(x0, y, heading, fontsize=10, fontweight="bold",
                    color=BLUE, va="top")
            y -= 0.10
            for line in lines:
                ax.text(x0 + 0.01, y, f"• {line}", fontsize=8.5,
                        color=DARK, va="top", linespacing=1.4)
                y -= 0.09
            y -= 0.05

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_split_strategy(pdf):
    fig = _blank()
    _header(fig, "Split Strategy",
            "cold_both is the honest drug-discovery benchmark — proteins AND ligands unseen at test time")
    _footer(fig)

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.70])
    ax.set_facecolor(LIGHT); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    strategies = [
        ("random",        "Random row split",       "Train/test share\nproteins & ligands",   "#ffcccc", "~1.00"),
        ("cold_protein",  "Cold Protein",            "Test proteins\nnever seen in train",      "#ffe0b0", "~0.70"),
        ("cold_ligand",   "Cold Ligand",             "Test ligands\nnever seen in train",       "#fffab0", "~0.65"),
        ("cold_both",     "Cold Both  ★",            "Both proteins AND\nligands unseen",        "#c8f0c8", "~0.73"),
    ]

    for i, (key, label, desc, color, roc) in enumerate(strategies):
        x = 0.5 + i * 2.3
        rect = FancyBboxPatch((x - 0.95, 1.6), 1.9, 1.8,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor=BLUE if key == "cold_both" else "#aaaaaa",
                              linewidth=2.5 if key == "cold_both" else 1.0)
        ax.add_patch(rect)
        ax.text(x, 3.2, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color=DARK)
        ax.text(x, 2.55, desc, ha="center", va="center", fontsize=8.5,
                color=DARK, linespacing=1.4)
        ax.text(x, 1.85, f"Typical\nROC-AUC ≈ {roc}",
                ha="center", va="center", fontsize=8, color="#333333",
                style="italic")

    ax.annotate("", xy=(9.35, 2.5), xytext=(0.1, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=1.0, linestyle="dashed"))
    ax.text(5.0, 1.2, "← optimistic  ·  increasingly honest  ·  most realistic →",
            ha="center", fontsize=9, color="#666666", style="italic")

    ax.text(0.1, 0.75,
            "This run used cold_both: 748 proteins · 13,361 ligands · 17,788 interactions → "
            "2,646 test rows  (both protein and ligand IDs unseen during training)",
            fontsize=9, color=DARK)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_results_table(pdf, summary: pd.DataFrame):
    fig = _blank()
    _header(fig, "Model Results — Cold-Both Split",
            "Both proteins and ligands unseen at test time  ·  748 proteins  ·  2,646 test rows")
    _footer(fig)

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.70])
    ax.set_facecolor(LIGHT); ax.axis("off")

    models = ["lightgbm", "random_forest", "xgboost", "interaction_mlp", "logistic_regression"]
    cols   = ["test_roc_auc", "test_pr_auc", "test_f1_binary", "test_accuracy",
              "cold_ef_at_1pct", "cold_ef_at_5pct"]
    col_labels = ["ROC-AUC", "PR-AUC", "F1", "Accuracy", "EF@1%", "EF@5%"]

    sub = summary.set_index("model")

    # Table header
    xs = [0.28, 0.40, 0.52, 0.62, 0.72, 0.82, 0.92]
    y0 = 0.92
    ax.text(0.05, y0, "Model", fontsize=10, fontweight="bold", color=BLUE, va="top")
    for x, lbl in zip(xs[1:], col_labels):
        ax.text(x, y0, lbl, fontsize=9, fontweight="bold", color=BLUE,
                ha="center", va="top")

    ax.axhline(y0 - 0.06, color=BLUE, linewidth=1.2, xmin=0.02, xmax=0.98)

    for rank, model in enumerate(models):
        y = y0 - 0.12 - rank * 0.13
        bg = "#eef4ff" if rank % 2 == 0 else LIGHT
        rect = FancyBboxPatch((0.03, y - 0.03), 0.93, 0.11,
                              boxstyle="round,pad=0.01", facecolor=bg,
                              edgecolor="none")
        ax.add_patch(rect)

        medal = ["  1.", "  2.", "  3.", "  4.", "  5."][rank]
        ax.text(0.05, y + 0.03, f"{medal}  {MODEL_LABEL[model]}",
                fontsize=10, color=DARK, va="center",
                fontweight="bold" if rank == 0 else "normal")

        for x, col in zip(xs[1:], cols):
            val = sub.loc[model, col] if model in sub.index and col in sub.columns else float("nan")
            txt = f"{val:.3f}" if not np.isnan(val) else "—"
            best_col = sub[col].max() if col in sub.columns else float("nan")
            is_best = not np.isnan(val) and abs(val - best_col) < 1e-6
            ax.text(x, y + 0.03, txt, fontsize=9.5, ha="center", va="center",
                    color=GOLD if is_best else DARK,
                    fontweight="bold" if is_best else "normal")

    # CV note
    cv_mean = summary.loc[summary["model"] == "logistic_regression", "cv_roc_auc_mean"].values
    cv_std  = summary.loc[summary["model"] == "logistic_regression", "cv_roc_auc_std"].values
    if len(cv_mean) and not np.isnan(cv_mean[0]):
        ax.text(0.05, 0.04,
                f"Cross-validation ROC-AUC (LR, protein-aware StratifiedGroupKFold, 5-fold): "
                f"{cv_mean[0]:.3f} ± {cv_std[0]:.3f}",
                fontsize=8.5, color="#555555", style="italic")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_with_figure(pdf, img_path: Path, title: str, subtitle: str, notes: str = ""):
    fig = _blank()
    _header(fig, title, subtitle)
    _footer(fig)

    h = 0.62 if notes else 0.70
    _paste_image(fig, img_path, [0.03, 0.08, 0.94, h])

    if notes:
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.05])
        ax.set_facecolor("#e8eef8"); ax.axis("off")
        ax.text(0.01, 0.5, notes, fontsize=8.5, color=DARK, va="center",
                style="italic")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_two_figures(pdf, left: Path, right: Path, title: str, subtitle: str):
    fig = _blank()
    _header(fig, title, subtitle)
    _footer(fig)
    _paste_image(fig, left,  [0.01, 0.08, 0.48, 0.70])
    _paste_image(fig, right, [0.51, 0.08, 0.48, 0.70])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def slide_takeaways(pdf):
    fig = _blank(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK); ax.axis("off")

    bar = fig.add_axes([0, 0.78, 1, 0.22])
    bar.set_facecolor(BLUE); bar.axis("off")
    bar.text(0.5, 0.60, "Key Takeaways", color=WHITE, fontsize=22,
             fontweight="bold", ha="center", va="center")
    bar.text(0.5, 0.18, "Cold-Both Benchmark  ·  748 Proteins  ·  13,361 Ligands",
             color=GOLD, fontsize=12, ha="center", va="center")

    points = [
        ("[1]  LightGBM wins cold-both",
         "ROC-AUC = 0.776  |  PR-AUC = 0.748  |  EF@1% = 2.38x\n"
         "Tree models outperform LR and MLP on this dataset -- interaction features are non-linear\n"
         "but the tabular fingerprint representation favours ensemble trees."),
        ("[2]  Enrichment Factor confirms screening utility",
         "EF@1% of 2.2-2.4x means the top-ranked 1% of compounds contains\n"
         "2x more binders than expected by chance -- practically useful for virtual screening."),
        ("[3]  Logistic Regression fails on cold-both",
         "ROC-AUC ~ 0.50 -- essentially random for completely unseen pairs.\n"
         "The binding signal is fundamentally non-linear; linear models cannot generalise."),
        ("[4]  InteractionMLP is competitive but not dominant",
         "ROC-AUC = 0.726 with only 50 epochs on a 35M-parameter ESM-2 backbone.\n"
         "A larger ESM-2 (650M, t33) or more training data would likely close the gap with LightGBM."),
        ("[5]  Cold-both is the honest benchmark",
         "Random-split ROC-AUC ~ 1.0  ->  cold-both ~ 0.77: the 23-point drop is the\n"
         "realistic estimate of how much harder truly novel protein-ligand pairs are."),
    ]

    y = 0.70
    for heading, body in points:
        ax.text(0.04, y, heading, color=GOLD, fontsize=11, fontweight="bold", va="top")
        ax.text(0.06, y - 0.06, body, color="#ccccdd", fontsize=9,
                va="top", linespacing=1.5)
        y -= 0.175

    _footer(fig, "github.com/abuchin/ML_protein_ligand_prediction")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="Presentation/protein_ligand_results.pdf")
    p.add_argument("--figures_dir", default="outputs/figures")
    p.add_argument("--summary", default="outputs/figures/summary_table.csv")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    figs = Path(args.figures_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)

    print(f"Building presentation → {out}")
    with PdfPages(out) as pdf:
        slide_title(pdf)
        slide_problem(pdf)
        slide_architecture(pdf)
        slide_features(pdf)
        slide_split_strategy(pdf)
        slide_results_table(pdf, summary)
        slide_with_figure(pdf, figs / "1_metric_comparison.png",
                          "Metric Comparison — Hold-out Test Set",
                          "ROC-AUC / PR-AUC / F1 across all models on the cold-both test partition")
        slide_two_figures(pdf, figs / "3_roc_curves.png", figs / "4_pr_curves.png",
                          "ROC & Precision-Recall Curves",
                          "Cold-both split — 2,646 test rows — completely unseen proteins and ligands")
        slide_with_figure(pdf, figs / "2_cold_start_metrics.png",
                          "Drug-Discovery Metrics",
                          "ROC-AUC · PR-AUC · BEDROC · EF@1% — cold-both partition",
                          notes="EF@1% > 1.0 means the model retrieves more binders in the top 1% than random chance. "
                                "BEDROC weights early retrieval exponentially (α=20).")
        slide_with_figure(pdf, figs / "7_score_distributions.png",
                          "Predicted Score Distributions",
                          "Histogram of model output probabilities split by true class (binder vs non-binder)")
        slide_with_figure(pdf, figs / "5_confusion_matrices.png",
                          "Confusion Matrices",
                          "Threshold = 0.5  ·  cold-both test set")
        slide_with_figure(pdf, figs / "6_cv_results.png",
                          "Cross-Validation Results",
                          "Protein-aware StratifiedGroupKFold (5-fold) — no protein leakage between folds")
        slide_takeaways(pdf)

        d = pdf.infodict()
        d["Title"]   = "Protein-Ligand Binding Prediction Results"
        d["Author"]  = "abuchin"
        d["Subject"] = "KIBA cold-both benchmark — 748 proteins, 5 models"

    print(f"Done — {out}  ({out.stat().st_size // 1024} KB,  "
          f"{sum(1 for _ in PdfPages.__mro__)} slides approx)")
    print(f"Slides: title + problem + architecture + features + split strategy + "
          f"results table + 6 figure slides + takeaways  =  13 slides")


if __name__ == "__main__":
    main()
