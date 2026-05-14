"""
Generate plots from tumor_eval_analysis outputs.

Usage (defaults match the standard output paths):
    python eval/plot_analysis.py
    python eval/plot_analysis.py \
        --binary_dir     outputs/analysis/binary_tumor_tumor_eval \
        --multiclass_dir outputs/analysis/multiclass_tumor_tumor_eval \
        --out_dir        outputs/analysis/plots

Produces 6 PNGs:
    binary_model_accuracy.png          - all 4 metrics per model (binary)
    multiclass_model_accuracy.png      - accuracy + F1 per model (multiclass)
    binary_vs_multiclass_accuracy.png  - accuracy side-by-side across datasets
    binary_field_accuracy.png          - MedGemma field accuracy (binary)
    multiclass_field_accuracy.png      - MedGemma field accuracy (multiclass)
    diagnosis_detailed_comparison.png  - subtype accuracy binary vs multiclass
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.3,
})

BINARY_DIR     = Path("outputs/analysis/binary_tumor_tumor_eval")
MULTICLASS_DIR = Path("outputs/analysis/multiclass_tumor_tumor_eval")
OUT_DIR        = Path("outputs/analysis/plots")

MODEL_ORDER = ["cnn", "biomedclip", "medgemma_initial", "medgemma_final", "pipeline_final"]
MODEL_LABELS = {
    "cnn":              "CNN",
    "biomedclip":       "BiomedCLIP",
    "medgemma_initial": "MedGemma (triage)",
    "medgemma_final":   "MedGemma (final)",
    "pipeline_final":   "Pipeline (fused)",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _label_bar(ax, bar, v, fmt=".2f", offset=0.006, fontsize=8):
    if not np.isnan(v) and v > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{v:{fmt}}",
            ha="center", va="bottom", fontsize=fontsize,
        )


def _ordered_models(df):
    return [m for m in MODEL_ORDER if m in df["model"].values]


# ── Plot 1 — Binary model accuracy ───────────────────────────────────────────

def plot_binary_model_accuracy(df: pd.DataFrame, out: Path):
    metrics = ["accuracy", "f1_macro", "sensitivity", "specificity"]
    m_labels = ["Accuracy", "F1-macro", "Sensitivity", "Specificity"]
    models  = _ordered_models(df)
    x       = np.arange(len(models))
    width   = 0.18
    offsets = np.linspace(-(len(metrics) - 1) / 2 * width,
                          (len(metrics) - 1) / 2 * width, len(metrics))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (metric, mlabel, offset) in enumerate(zip(metrics, m_labels, offsets)):
        vals = [
            float(df.loc[df["model"] == m, metric].values[0])
            if metric in df.columns and m in df["model"].values else 0.0
            for m in models
        ]
        bars = ax.bar(x + offset, vals, width, label=mlabel,
                      color=PALETTE[i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            _label_bar(ax, bar, v)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Binary Tumor Detection — Per-model Metrics  (Br35H, n≈1 000)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="Random baseline")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    path = out / "binary_model_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ── Plot 2 — Multiclass model accuracy ───────────────────────────────────────

def plot_multiclass_model_accuracy(df: pd.DataFrame, out: Path):
    metrics  = ["accuracy", "f1_macro"]
    m_labels = ["Accuracy", "F1-macro"]
    models   = _ordered_models(df)
    x        = np.arange(len(models))
    width    = 0.3
    offsets  = [-width / 2, width / 2]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, mlabel, offset) in enumerate(zip(metrics, m_labels, offsets)):
        vals = [
            float(df.loc[df["model"] == m, metric].values[0])
            if m in df["model"].values else 0.0
            for m in models
        ]
        bars = ax.bar(x + offset, vals, width, label=mlabel,
                      color=PALETTE[i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            _label_bar(ax, bar, v, fmt=".3f")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Multiclass Tumor Subtype Classification — Per-model Metrics\n"
                 "(Figshare3: meningioma / glioma / pituitary, n≈1 000)")
    ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Random baseline (1/3)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    path = out / "multiclass_model_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ── Plot 3 — Binary vs multiclass accuracy ────────────────────────────────────

def plot_binary_vs_multiclass(bin_df: pd.DataFrame, mc_df: pd.DataFrame, out: Path):
    models  = [m for m in MODEL_ORDER
               if m in bin_df["model"].values and m in mc_df["model"].values]
    x       = np.arange(len(models))
    width   = 0.35

    bin_acc = [float(bin_df.loc[bin_df["model"] == m, "accuracy"].values[0]) for m in models]
    mc_acc  = [float(mc_df.loc[mc_df["model"]  == m, "accuracy"].values[0]) for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, bin_acc, width,
                   label="Binary  (Br35H)", color=PALETTE[0], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, mc_acc,  width,
                   label="Multiclass  (Figshare3)", color=PALETTE[1], alpha=0.85, edgecolor="white")

    for bar, v in zip(bars1, bin_acc):
        _label_bar(ax, bar, v)
    for bar, v in zip(bars2, mc_acc):
        _label_bar(ax, bar, v)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy — Binary vs Multiclass  (all models)")
    ax.axhline(1 / 3, color=PALETTE[1], linestyle=":", linewidth=0.9, alpha=0.5,
               label="Multiclass random (0.33)")
    ax.axhline(0.5,   color=PALETTE[0], linestyle=":", linewidth=0.9, alpha=0.5,
               label="Binary random (0.50)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    path = out / "binary_vs_multiclass_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ── Plot 4 & 5 — MedGemma field accuracy (one function, two calls) ────────────

def plot_field_accuracy(df: pd.DataFrame, title: str, path: Path, is_multiclass: bool):
    diag_fields = ["diagnosis_name", "diagnosis_detailed"]
    fact_fields = ["modality", "plane"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12)

    # ── left panel: diagnosis fields ─────────────────────────────────────────
    ax = axes[0]
    rows_left = []
    for field in diag_fields:
        for stage in ["initial", "final"]:
            row = df[(df["field"] == field) & (df["stage"] == stage)]
            if not row.empty:
                rows_left.append({
                    "label":  f"{field}\n({stage})",
                    "acc":    float(row["accuracy"].values[0]),
                    "color":  PALETTE[0] if stage == "initial" else PALETTE[1],
                    "n":      int(row["n"].values[0]),
                })

    if rows_left:
        ldf = pd.DataFrame(rows_left)
        y   = np.arange(len(ldf))
        bars = ax.barh(y, ldf["acc"], color=ldf["color"], alpha=0.85,
                       edgecolor="white", height=0.5)
        for bar, row in zip(bars, ldf.itertuples()):
            v = row.acc
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}  (n={row.n})", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(ldf["label"])
        baseline = 1 / 3 if is_multiclass else 0.5
        ax.axvline(baseline, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label=f"Random ({baseline:.2f})")
        init_p  = mpatches.Patch(color=PALETTE[0], alpha=0.85, label="initial")
        final_p = mpatches.Patch(color=PALETTE[1], alpha=0.85, label="final")
        ax.legend(handles=[init_p, final_p, ], loc="lower right", fontsize=9)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy")
    ax.set_title("Diagnosis fields")

    # ── right panel: factual fields ───────────────────────────────────────────
    ax = axes[1]
    rows_right = []
    for field in fact_fields:
        for stage in ["initial", "final"]:
            row = df[(df["field"] == field) & (df["stage"] == stage)]
            if not row.empty:
                expected = row["expected_value"].values[0] if "expected_value" in row.columns else ""
                rows_right.append({
                    "label": f"{field} = {expected}\n({stage})",
                    "acc":   float(row["accuracy"].values[0]),
                    "color": PALETTE[2] if stage == "initial" else PALETTE[3],
                })

    if rows_right:
        rdf = pd.DataFrame(rows_right)
        y   = np.arange(len(rdf))
        bars = ax.barh(y, rdf["acc"], color=rdf["color"], alpha=0.85,
                       edgecolor="white", height=0.5)
        for bar, v in zip(bars, rdf["acc"]):
            ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(rdf["label"])
        init_p  = mpatches.Patch(color=PALETTE[2], alpha=0.85, label="initial")
        final_p = mpatches.Patch(color=PALETTE[3], alpha=0.85, label="final")
        ax.legend(handles=[init_p, final_p], loc="lower right", fontsize=9)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy")
    ax.set_title("Factual fields  (modality → MRI, plane → axial)")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ── Plot 6 — diagnosis_detailed comparison binary vs multiclass ───────────────

def plot_diagnosis_detailed_comparison(bin_df: pd.DataFrame, mc_df: pd.DataFrame, out: Path):
    rows = []
    for stage in ["initial", "final"]:
        b = bin_df[(bin_df["field"] == "diagnosis_detailed") & (bin_df["stage"] == stage)]
        m = mc_df[(mc_df["field"]  == "diagnosis_detailed") & (mc_df["stage"] == stage)]
        rows.append({
            "stage":      stage.capitalize(),
            "bin_acc":    float(b["accuracy"].values[0])  if not b.empty else float("nan"),
            "bin_f1":     float(b["f1_macro"].values[0])  if not b.empty and "f1_macro" in b.columns else float("nan"),
            "mc_acc":     float(m["accuracy"].values[0])  if not m.empty else float("nan"),
            "mc_f1":      float(m["f1_macro"].values[0])  if not m.empty and "f1_macro" in m.columns else float("nan"),
        })
    cdf = pd.DataFrame(rows)

    metrics = ["bin_acc", "bin_f1", "mc_acc", "mc_f1"]
    labels  = ["Binary acc", "Binary F1-macro", "Multiclass acc", "Multiclass F1-macro"]
    colors  = [PALETTE[0], PALETTE[0], PALETTE[1], PALETTE[1]]
    alphas  = [0.9, 0.5, 0.9, 0.5]

    x       = np.arange(len(cdf))
    width   = 0.18
    offsets = np.linspace(-(len(metrics) - 1) / 2 * width,
                          (len(metrics) - 1) / 2 * width, len(metrics))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label, color, alpha) in enumerate(zip(metrics, labels, colors, alphas)):
        vals = cdf[metric].fillna(0).values
        bars = ax.bar(x + offsets[i], vals, width, label=label,
                      color=color, alpha=alpha, edgecolor="white")
        for bar, v in zip(bars, cdf[metric].values):
            if not np.isnan(v):
                _label_bar(ax, bar, v, fmt=".3f")

    ax.set_xticks(x)
    ax.set_xticklabels(cdf["stage"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("diagnosis_detailed Accuracy & F1-macro\n"
                 "Binary (Br35H, among tumors only)  vs  Multiclass (Figshare3, all 3 subtypes)")
    ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Multiclass random (0.33)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    path = out / "diagnosis_detailed_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_dir",     default=str(BINARY_DIR))
    parser.add_argument("--multiclass_dir", default=str(MULTICLASS_DIR))
    parser.add_argument("--out_dir",        default=str(OUT_DIR))
    args = parser.parse_args()

    bin_dir = Path(args.binary_dir)
    mc_dir  = Path(args.multiclass_dir)
    out     = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bin_model = pd.read_csv(bin_dir / "model_accuracy_summary.csv")
    mc_model  = pd.read_csv(mc_dir  / "model_accuracy_summary.csv")
    bin_field = pd.read_csv(bin_dir / "medgemma_field_accuracy.csv")
    mc_field  = pd.read_csv(mc_dir  / "medgemma_field_accuracy.csv")

    print("Generating plots →")
    plot_binary_model_accuracy(bin_model, out)
    plot_multiclass_model_accuracy(mc_model, out)
    plot_binary_vs_multiclass(bin_model, mc_model, out)
    plot_field_accuracy(
        bin_field,
        "MedGemma Field Accuracy — Binary  (Br35H)",
        out / "binary_field_accuracy.png",
        is_multiclass=False,
    )
    plot_field_accuracy(
        mc_field,
        "MedGemma Field Accuracy — Multiclass  (Figshare3)",
        out / "multiclass_field_accuracy.png",
        is_multiclass=True,
    )
    plot_diagnosis_detailed_comparison(bin_field, mc_field, out)
    print(f"\nDone. All plots in {out}/")


if __name__ == "__main__":
    main()
