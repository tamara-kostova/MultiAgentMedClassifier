"""
General analysis for any binary or multiclass task eval JSONL.
Supports: ms, stroke, binary_tumor, multiclass_tumor.
Generates CSVs and matplotlib visualizations, focused on initial vs. final MedGemma.

Usage:
    python eval/eval_analysis.py --jsonl outputs/eval/ms_dataset_eval.jsonl
    python eval/eval_analysis.py --jsonl outputs/eval/stroke_dataset_eval.jsonl
    python eval/eval_analysis.py --jsonl outputs/eval/binary_tumor_tumor_eval.jsonl
    python eval/eval_analysis.py --jsonl outputs/eval/multiclass_tumor_tumor_eval.jsonl

Outputs (outputs/analysis/<stem>/)
────────────────────────────────────
CSVs:
  model_accuracy_summary.csv
  confusion_matrix_<model>.csv
  medgemma_shift_analysis.csv
  confidence_calibration_summary.csv
  calibration_bins_<model>.csv
  latency_stats.csv
Plots:
  model_accuracy.png
  confusion_matrices.png
  medgemma_initial_vs_final.png
  calibration_plot.png
  confidence_by_correctness.png
  latency.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── label normalization ────────────────────────────────────────────────────────

_TUMOR_SUBTYPES = (
    "carcinoma", "germinoma", "glioma", "granuloma", "medulloblastoma",
    "meningioma", "neurocytoma", "papilloma", "schwannoma", "tuberculoma",
)
_STROKE_TOKENS = ("stroke", "ischemic", "ischemia", "hemorrhagic", "bleeding", "infarct")


def canonical_label(value: object, task: str | None = None) -> str:
    text = str(value or "").strip().lower()
    if not text or text in ("none", "null", "nan"):
        return ""
    n = (text.replace("_", " ").replace("-", " ")
              .replace("/", " ").replace("tumour", "tumor"))
    if "normal" in n or "control" in n:
        return "normal"
    if n == "ms" or n.startswith("ms ") or "multiple sclerosis" in n:
        return "ms"
    if any(tok in n for tok in _STROKE_TOKENS):
        return "stroke"
    subtype = next((s for s in _TUMOR_SUBTYPES if s in n), None)
    tumorish = (subtype is not None or "pituitary" in n or "brain tumor" in n
                or n == "tumor" or " tumor" in n)
    if task == "binary_tumor" and tumorish:
        return "tumor"
    if "pituitary" in n:
        return "pituitary_tumor"
    if subtype:
        return subtype
    if tumorish:
        return "tumor"
    return n


def is_multiclass(task: str) -> bool:
    return task == "multiclass_tumor"


def pos_class(task: str) -> str:
    return {"ms": "ms", "stroke": "stroke"}.get(task, "tumor")


def prep(label: str, task: str) -> str:
    """Normalize for comparison. Multiclass: keep subtype. Binary: collapse to (pos|normal|unknown)."""
    if is_multiclass(task):
        cl = canonical_label(label, task)
        return cl if cl else "unknown"
    # Binary: any non-empty non-normal → positive class
    cl = canonical_label(label, task)
    if not cl:
        return "unknown"
    if cl == "normal":
        return "normal"
    return pos_class(task)


def mg_diag_pred(diag: object, task: str) -> str:
    """Extract MedGemma's prediction from a diagnosis dict.

    Multiclass tumor uses diagnosis_detailed (name is always 'tumor' per schema).
    All other tasks use diagnosis_name.
    """
    if not isinstance(diag, dict):
        return ""
    field = "diagnosis_detailed" if is_multiclass(task) else "diagnosis_name"
    return canonical_label(diag.get(field) or "", task)


def mg_conf(diag: object) -> float:
    if not isinstance(diag, dict):
        return 0.5
    v = diag.get("diagnosis_confidence")
    return float(v) if v is not None else 0.5


# ── loader ─────────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    for col in ("medgemma_diagnosis", "final_medgemma_diagnosis"):
        df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})

    # Older JSONL files (binary/multiclass tumor) omit canonical columns — compute them.
    task = str(df["task"].iloc[0]) if "task" in df.columns and len(df) else "unknown"

    def _fill_canonical(col, raw_col, fallback_col=None):
        if col not in df.columns or df[col].isna().all():
            src = df[raw_col].fillna(
                df[fallback_col].fillna("") if fallback_col else ""
            )
            df[col] = src.apply(lambda v: canonical_label(str(v or ""), task))

    _fill_canonical("true_label_canonical",       "true_label_name",       "true_label")
    _fill_canonical("predicted_class_canonical",  "predicted_class")
    _fill_canonical("cnn_predicted_class_canonical", "cnn_predicted_class")

    return df


# ── metric helpers ─────────────────────────────────────────────────────────────

def binary_metrics_row(y_true: list, y_pred: list, pos: str, name: str) -> dict | None:
    valid = [(t, p) for t, p in zip(y_true, y_pred)
             if t in (pos, "normal") and p in (pos, "normal")]
    if not valid:
        return None
    yt, yp = zip(*valid)
    cm = confusion_matrix(list(yt), list(yp), labels=["normal", pos])
    tn, fp, fn, tp_ = cm.ravel()
    return {
        "model":       name,
        "n":           len(yt),
        "accuracy":    round(accuracy_score(list(yt), list(yp)), 4),
        "f1_macro":    round(f1_score(list(yt), list(yp), average="macro", zero_division=0), 4),
        "sensitivity": round(tp_ / (tp_ + fn) if (tp_ + fn) else float("nan"), 4),
        "specificity": round(tn  / (tn  + fp) if (tn  + fp) else float("nan"), 4),
        "precision":   round(precision_score(list(yt), list(yp), pos_label=pos, zero_division=0), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp_),
    }


def multiclass_metrics_row(y_true: list, y_pred: list, name: str) -> dict | None:
    valid = [(t, p) for t, p in zip(y_true, y_pred)
             if t not in ("", "unknown") and p not in ("", "unknown")]
    if not valid:
        return None
    yt, yp = zip(*valid)
    classes = sorted(set(list(yt) + list(yp)))
    per_class = {}
    for cls in classes:
        tp_ = sum(1 for t, p in zip(yt, yp) if t == cls and p == cls)
        fn_ = sum(1 for t, p in zip(yt, yp) if t == cls and p != cls)
        fp_ = sum(1 for t, p in zip(yt, yp) if t != cls and p == cls)
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else float("nan")
        pre = tp_ / (tp_ + fp_) if (tp_ + fp_) else float("nan")
        per_class[f"recall_{cls}"]    = round(rec, 4)
        per_class[f"precision_{cls}"] = round(pre, 4)
    return {
        "model":       name,
        "n":           len(yt),
        "accuracy":    round(accuracy_score(list(yt), list(yp)), 4),
        "f1_macro":    round(f1_score(list(yt), list(yp), average="macro",     zero_division=0), 4),
        "f1_weighted": round(f1_score(list(yt), list(yp), average="weighted",  zero_division=0), 4),
        **per_class,
    }


def compute_ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(confs)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum():
            ece += (m.sum() / n) * abs(confs[m].mean() - correct[m].mean())
    return float(ece)


def calibration_bins_df(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        n = int(m.sum())
        rows.append({
            "bin_lo":    round(lo, 2),
            "bin_hi":    round(hi, 2),
            "n":         n,
            "mean_conf": round(float(confs[m].mean()), 4) if n else float("nan"),
            "mean_acc":  round(float(correct[m].mean()), 4) if n else float("nan"),
        })
    return pd.DataFrame(rows)


# ── prediction builder ─────────────────────────────────────────────────────────

def build_model_preds(df: pd.DataFrame, task: str) -> dict[str, tuple[list, list]]:
    """Return {model_name: ([prep'd predictions], [confidences])}.

    Binary tasks: predictions are binarized to (pos | normal | unknown).
    Multiclass:   predictions are canonical subtype labels.
    """
    def _prep_col(col: str) -> list:
        return [prep(str(v or ""), task) for v in df[col].fillna("")]

    return {
        "cnn": (
            _prep_col("cnn_predicted_class_canonical"),
            df["cnn_confidence"].fillna(0.5).tolist(),
        ),
        "biomedclip": (
            [prep(canonical_label(str(v or ""), task), task)
             for v in df["biomedclip_top_label"].fillna("")],
            df["biomedclip_top_score"].fillna(0.5).tolist(),
        ),
        "medgemma_initial": (
            [prep(mg_diag_pred(d, task), task) for d in df["medgemma_diagnosis"]],
            [mg_conf(d) for d in df["medgemma_diagnosis"]],
        ),
        "medgemma_final": (
            [prep(mg_diag_pred(d, task), task) for d in df["final_medgemma_diagnosis"]],
            [mg_conf(d) for d in df["final_medgemma_diagnosis"]],
        ),
        "pipeline_final": (
            _prep_col("predicted_class_canonical"),
            df["final_confidence"].fillna(0.5).tolist(),
        ),
    }


# ── sections ───────────────────────────────────────────────────────────────────

def section_model_accuracy(df: pd.DataFrame, task: str) -> pd.DataFrame:
    gt    = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    rows  = []
    multi = is_multiclass(task)
    pos   = pos_class(task)

    for name, (preds, _) in build_model_preds(df, task).items():
        row = (multiclass_metrics_row(gt, preds, name)
               if multi else binary_metrics_row(gt, preds, pos, name))
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def section_confusion_matrices(df: pd.DataFrame, task: str) -> dict[str, pd.DataFrame]:
    multi  = is_multiclass(task)
    pos    = pos_class(task)
    gt     = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    valid_labels = sorted(set(gt) - {"", "unknown"})

    result = {}
    for name, (preds, _) in build_model_preds(df, task).items():
        if name == "biomedclip":
            continue
        valid = [(t, p) for t, p in zip(gt, preds)
                 if t in valid_labels and p in valid_labels]
        if not valid:
            continue
        yt, yp = zip(*valid)
        pred_labels = sorted(set(list(yt) + list(yp)))
        cm = confusion_matrix(list(yt), list(yp), labels=pred_labels)
        result[name] = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in pred_labels],
            columns=[f"pred_{l}" for l in pred_labels],
        )
    return result


def section_medgemma_shift(df: pd.DataFrame, task: str) -> pd.DataFrame:
    gt   = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    init = [prep(mg_diag_pred(d, task), task) for d in df["medgemma_diagnosis"]]
    fin  = [prep(mg_diag_pred(d, task), task) for d in df["final_medgemma_diagnosis"]]
    rows = []
    for g, i, f in zip(gt, init, fin):
        if not g or g == "unknown" or not i or i == "unknown" or not f or f == "unknown":
            continue
        rows.append({
            "true":         g,
            "initial":      i,
            "final":        f,
            "init_correct": int(i == g),
            "fin_correct":  int(f == g),
            "changed":      int(i != f),
            "recovered":    int(i != g and f == g),
            "degraded":     int(i == g and f != g),
        })
    return pd.DataFrame(rows)


def section_calibration(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    gt       = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    valid_gt = set(gt) - {"", "unknown"}
    summary_rows, bins_dict = [], {}

    for name, (preds, confs) in build_model_preds(df, task).items():
        valid = [(g, p, c) for g, p, c in zip(gt, preds, confs)
                 if g in valid_gt and p in valid_gt]
        if not valid:
            continue
        yt, yp, yc = zip(*valid)
        correct   = np.array([int(t == p) for t, p in zip(yt, yp)], dtype=float)
        confs_arr = np.clip(np.array(yc, dtype=float), 0, 1)
        ece = compute_ece(confs_arr, correct)
        bins_dict[name] = calibration_bins_df(confs_arr, correct)
        summary_rows.append({
            "model":          name,
            "n":              len(yt),
            "ece":            round(ece, 4),
            "mean_conf":      round(float(confs_arr.mean()), 4),
            "mean_acc":       round(float(correct.mean()), 4),
            "overconfidence": round(float(confs_arr.mean() - correct.mean()), 4),
        })
    return pd.DataFrame(summary_rows), bins_dict


def section_latency(df: pd.DataFrame) -> pd.DataFrame:
    lat   = df["latency_s"].dropna().values
    paths = df["routing_path"].fillna("unknown").tolist()

    def _stats(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if not len(a):
            return {}
        return {
            "n": len(a), "mean_s": round(float(a.mean()), 2),
            "median_s": round(float(np.median(a)), 2),
            "std_s": round(float(a.std()), 2),
            "min_s": round(float(a.min()), 2), "max_s": round(float(a.max()), 2),
        }

    rows = [{"routing_path": "ALL", **_stats(lat)}]
    for path, _ in Counter(paths).most_common(6):
        l = df["latency_s"][df["routing_path"] == path].dropna().values
        rows.append({"routing_path": path[:80], **_stats(l)})
    return pd.DataFrame(rows)


def section_forest_voting(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Agent Forest voting quality — dissent rate vs. accuracy.

    Empty unless the JSONL carries a `dissent_rate` column (i.e. it was produced by
    `--pipeline_mode forest`). Shows whether agent disagreement flags harder cases:
    accuracy on unanimous vs. split votes.
    """
    if "dissent_rate" not in df.columns or df["dissent_rate"].isna().all():
        return pd.DataFrame()
    sub = df[df["dissent_rate"].notna()].copy()
    gt = [prep(str(v or ""), task) for v in sub["true_label_canonical"].fillna("")]
    pr = [prep(str(v or ""), task) for v in sub["predicted_class_canonical"].fillna("")]
    correct = np.array([int(t == p) for t, p in zip(gt, pr)], dtype=float)
    dissent = sub["dissent_rate"].astype(float).values
    unan, split = dissent == 0.0, dissent > 0.0

    def _acc(mask, min_n=1):
        return round(float(correct[mask].mean()), 4) if mask.sum() >= min_n else float("nan")

    return pd.DataFrame([{
        "n": len(sub),
        "mean_dissent_rate": round(float(dissent.mean()), 4),
        "unanimous_pct": round(float(unan.mean()) * 100, 1),
        "accuracy_unanimous": _acc(unan),
        "accuracy_split": _acc(split, min_n=5),
        "accuracy_overall": round(float(correct.mean()), 4),
    }])


def section_debate_rounds(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Multi-Agent Debate — verdict stability vs. accuracy/ECE.

    Empty unless the JSONL carries a `debate_rounds_completed` column (i.e. it was
    produced by `--pipeline_mode debate`). Shows whether cases whose verdict flipped
    across rounds are less accurate / worse calibrated than stable ones.
    """
    if "debate_rounds_completed" not in df.columns or df["debate_rounds_completed"].isna().all():
        return pd.DataFrame()
    sub = df[df["debate_rounds_completed"].notna()].copy()
    gt = [prep(str(v or ""), task) for v in sub["true_label_canonical"].fillna("")]
    pr = [prep(str(v or ""), task) for v in sub["predicted_class_canonical"].fillna("")]
    correct = np.array([int(t == p) for t, p in zip(gt, pr)], dtype=float)
    confs = np.clip(sub["final_confidence"].astype(float).fillna(0.0).values, 0, 1)
    changed = sub["debate_round_changed"].fillna(False).astype(bool).values

    def _acc(mask):
        return round(float(correct[mask].mean()), 4) if mask.sum() >= 5 else float("nan")

    def _ece(mask):
        return round(compute_ece(confs[mask], correct[mask]), 4) if mask.sum() >= 5 else float("nan")

    return pd.DataFrame([{
        "n": len(sub),
        "pct_verdict_changed": round(float(changed.mean()) * 100, 1),
        "accuracy_changed": _acc(changed),
        "accuracy_unchanged": _acc(~changed),
        "ece_changed": _ece(changed),
        "ece_unchanged": _ece(~changed),
        "ece_overall": round(compute_ece(confs, correct), 4),
    }])


# ── plot helpers ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_model_accuracy(accuracy_df: pd.DataFrame, task: str, out: Path) -> None:
    multi   = is_multiclass(task)
    metrics = (["accuracy", "f1_macro", "f1_weighted"]
               if multi else ["accuracy", "sensitivity", "specificity", "f1_macro"])
    models  = accuracy_df["model"].tolist()
    x       = np.arange(len(models))
    width   = 0.8 / len(metrics)
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.2), 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric not in accuracy_df.columns:
            continue
        vals = []
        for mod in models:
            row = accuracy_df[accuracy_df["model"] == mod]
            vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.82)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=30)

    offset = width * (len(metrics) - 1) / 2
    ax.set_xticks(x + offset)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-model metrics — {task} task")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out / "model_accuracy.png")


def plot_confusion_matrices(cm_dict: dict, task: str, out: Path) -> None:
    items = [(k, v) for k, v in cm_dict.items() if not v.empty]
    if not items:
        return
    cols  = 2
    nrows = (len(items) + 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 5.5, nrows * 4.5))
    axes = np.array(axes).flatten()

    for ax, (name, cm_df) in zip(axes, items):
        vals = cm_df.values.astype(float)
        im = ax.imshow(vals, cmap="Blues")
        ax.set_xticks(range(vals.shape[1]))
        ax.set_yticks(range(vals.shape[0]))
        ax.set_xticklabels(cm_df.columns, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(cm_df.index, fontsize=8)
        threshold = vals.max() * 0.55 if vals.max() else 1
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                color = "white" if vals[i, j] > threshold else "black"
                ax.text(j, i, str(int(vals[i, j])), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
        total = vals.sum()
        acc   = np.trace(vals) / total if total else 0
        ax.set_title(f"{name.replace('_', ' ').title()}\nAcc = {acc:.3f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(items):]:
        ax.set_visible(False)

    fig.suptitle(f"Confusion Matrices — {task} task", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, out / "confusion_matrices.png")


def plot_medgemma_initial_vs_final(shift_df: pd.DataFrame, task: str, out: Path) -> None:
    if shift_df.empty:
        return
    multi = is_multiclass(task)

    if multi:
        # For multiclass: show per-class init/final accuracy as grouped bars
        classes = sorted(shift_df["true"].unique())
        x = np.arange(len(classes))
        width = 0.35
        init_accs = [shift_df[shift_df["true"] == c]["init_correct"].mean() for c in classes]
        fin_accs  = [shift_df[shift_df["true"] == c]["fin_correct"].mean()  for c in classes]
        counts    = [len(shift_df[shift_df["true"] == c]) for c in classes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.bar(x - width / 2, init_accs, width, label="MedGemma initial", color="#FF9800", alpha=0.82)
        ax1.bar(x + width / 2, fin_accs,  width, label="MedGemma final",   color="#4CAF50", alpha=0.82)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{c}\n(n={cnt})" for c, cnt in zip(classes, counts)], fontsize=9)
        ax1.set_ylim(0, 1.15)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Per-class accuracy: initial vs final MedGemma")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        n = len(shift_df)
        overall = {
            "Always correct":           int(((shift_df["init_correct"] == 1) & (shift_df["fin_correct"] == 1)).sum()),
            "Always wrong":             int(((shift_df["init_correct"] == 0) & (shift_df["fin_correct"] == 0)).sum()),
            "Recovered\n(wrong→right)": int(shift_df["recovered"].sum()),
            "Degraded\n(right→wrong)":  int(shift_df["degraded"].sum()),
        }
        bar_colors = ["#4CAF50", "#F44336", "#8BC34A", "#FF5722"]
        bars = ax2.bar(list(overall.keys()), [v / n for v in overall.values()],
                       color=bar_colors, alpha=0.85)
        for bar, (k, v) in zip(bars, overall.items()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v}\n({v/n:.1%})", ha="center", va="bottom", fontsize=9)
        ax2.axhline(shift_df["init_correct"].mean(), color="#FF9800", linestyle="--",
                    linewidth=1.8, label=f"Init acc = {shift_df['init_correct'].mean():.3f}")
        ax2.axhline(shift_df["fin_correct"].mean(), color="#4CAF50", linestyle="--",
                    linewidth=1.8, label=f"Final acc = {shift_df['fin_correct'].mean():.3f}")
        ax2.set_ylim(0, 1.25)
        ax2.set_ylabel("Rate")
        ax2.set_title(f"Overall transition (n={n})")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

    else:
        pos     = pos_class(task)
        classes = [c for c in ["normal", pos] if c in shift_df["true"].values]
        bar_colors_map = {
            "Always\ncorrect":          "#4CAF50",
            "Always\nwrong":            "#F44336",
            "Recovered\n(wrong→right)": "#8BC34A",
            "Degraded\n(right→wrong)":  "#FF5722",
        }
        fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 6))
        if len(classes) == 1:
            axes = [axes]

        for ax, cls in zip(axes, classes):
            sub = shift_df[shift_df["true"] == cls]
            if sub.empty:
                ax.set_visible(False)
                continue
            n = len(sub)
            data = {
                "Always\ncorrect":          int(((sub["init_correct"] == 1) & (sub["fin_correct"] == 1)).sum()),
                "Always\nwrong":            int(((sub["init_correct"] == 0) & (sub["fin_correct"] == 0)).sum()),
                "Recovered\n(wrong→right)": int(sub["recovered"].sum()),
                "Degraded\n(right→wrong)":  int(sub["degraded"].sum()),
            }
            bars = ax.bar(list(data.keys()), [v / n for v in data.values()],
                          color=[bar_colors_map[k] for k in data], alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for bar, (k, v) in zip(bars, data.items()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v}\n({v/n:.1%})", ha="center", va="bottom", fontsize=9)
            ax.axhline(sub["init_correct"].mean(), color="#FF9800", linestyle="--",
                       linewidth=1.8, label=f"Init acc = {sub['init_correct'].mean():.3f}")
            ax.axhline(sub["fin_correct"].mean(), color="#2196F3", linestyle="--",
                       linewidth=1.8, label=f"Final acc = {sub['fin_correct'].mean():.3f}")
            ax.set_ylim(0, 1.25)
            ax.set_title(f"True class: {cls}  (n={n})")
            ax.set_ylabel("Rate")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=10)

    fig.suptitle(f"MedGemma Initial→Final Transition — {task}", fontsize=13)
    plt.tight_layout()
    _save(fig, out / "medgemma_initial_vs_final.png")


def plot_calibration(bins_dict: dict, task: str, out: Path) -> None:
    if not bins_dict:
        return
    n     = len(bins_dict)
    cols  = min(n, 3)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 5, nrows * 4.5))
    axes = np.array(axes).flatten()

    model_colors = {
        "cnn": "#2196F3", "medgemma_initial": "#FF9800",
        "medgemma_final": "#4CAF50", "pipeline_final": "#9C27B0",
        "biomedclip": "#00BCD4",
    }

    for ax, (name, bdf) in zip(axes, bins_dict.items()):
        color = model_colors.get(name, "steelblue")
        valid = bdf.dropna(subset=["mean_conf", "mean_acc"])

        ax.bar(valid["bin_lo"], valid["mean_acc"], width=0.1, align="edge",
               alpha=0.4, color=color, label="Bin accuracy")
        ax.plot(valid["mean_conf"], valid["mean_acc"], "o-", color=color,
                markersize=5, linewidth=1.5, label="Observed")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Perfect")

        c_arr = valid["mean_conf"].values
        a_arr = valid["mean_acc"].values
        mask  = ~(np.isnan(c_arr) | np.isnan(a_arr))
        if mask.sum():
            ece_approx = np.average(np.abs(c_arr[mask] - a_arr[mask]),
                                    weights=valid["n"].values[mask])
            ax.text(0.05, 0.92, f"ECE ≈ {ece_approx:.3f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Calibration Reliability Diagrams — {task}", fontsize=13)
    plt.tight_layout()
    _save(fig, out / "calibration_plot.png")


def plot_confidence_by_correctness(df: pd.DataFrame, task: str, out: Path) -> None:
    gt     = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    mpreds = build_model_preds(df, task)
    names  = list(mpreds.keys())

    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 6), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        preds, confs = mpreds[name]
        valid_gt = set(gt) - {"", "unknown"}
        correct_c = [c for g, p, c in zip(gt, preds, confs)
                     if g in valid_gt and p in valid_gt and g == p]
        wrong_c   = [c for g, p, c in zip(gt, preds, confs)
                     if g in valid_gt and p in valid_gt and g != p]

        bp = ax.boxplot([correct_c, wrong_c],
                        tick_labels=["correct", "wrong"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#4CAF50", "#F44336"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        if correct_c:
            ax.axhline(np.mean(correct_c), color="#4CAF50", linestyle="--",
                       linewidth=0.9, alpha=0.7)
        if wrong_c:
            ax.axhline(np.mean(wrong_c), color="#F44336", linestyle="--",
                       linewidth=0.9, alpha=0.7)

        ax.set_title(name.replace("_", "\n"), fontsize=9)
        ax.set_ylabel("Confidence" if name == names[0] else "")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.5, -0.12, f"n={len(correct_c)}/{len(wrong_c)}",
                transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    fig.suptitle(f"Confidence Distribution: Correct vs Wrong — {task}", fontsize=12)
    plt.tight_layout()
    _save(fig, out / "confidence_by_correctness.png")


def plot_latency(df: pd.DataFrame, task: str, out: Path) -> None:
    lat   = df["latency_s"].dropna().values
    paths = df["routing_path"].fillna("unknown").tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(lat, bins=40, color="#2196F3", alpha=0.75, edgecolor="white")
    ax1.axvline(lat.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {lat.mean():.1f}s")
    ax1.axvline(np.median(lat), color="orange", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(lat):.1f}s")
    ax1.set_xlabel("Latency (s)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Latency Distribution — {task}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    path_counts = Counter(paths)
    top5 = [p for p, _ in path_counts.most_common(5)]
    lat_data = [df["latency_s"][df["routing_path"] == p].dropna().values for p in top5]
    short_labels = [
        (p[:40] + "…" if len(p) > 41 else p).replace(" → ", "→")
        + f"\n(n={path_counts[p]})"
        for p in top5
    ]
    ax2.boxplot(lat_data, vert=True)
    ax2.set_xticks(range(1, len(top5) + 1))
    ax2.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=7)
    ax2.set_ylabel("Latency (s)")
    ax2.set_title("Latency by Routing Path (top 5)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, out / "latency.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval JSONL analysis — CSVs + matplotlib plots (binary + multiclass)"
    )
    parser.add_argument("--jsonl",      required=True, help="Path to eval JSONL file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: outputs/analysis/<jsonl-stem>)")
    args = parser.parse_args()

    df   = load(args.jsonl)
    task = str(df["task"].iloc[0]) if "task" in df.columns and len(df) else "unknown"
    pos  = pos_class(task)
    stem = Path(args.jsonl).stem
    out  = Path(args.output_dir) if args.output_dir else Path("outputs/analysis") / stem
    out.mkdir(parents=True, exist_ok=True)

    multi = is_multiclass(task)
    print(f"\nTask: {task}  |  {'multiclass' if multi else f'positive class: {pos}'}  |  n={len(df)}")
    print(f"Output: {out}/\n")

    # ── compute ────────────────────────────────────────────────────────────────
    print("══ Model accuracy ══")
    accuracy_df = section_model_accuracy(df, task)
    print(accuracy_df.to_string(index=False))

    print("\n══ Confusion matrices ══")
    cm_dict = section_confusion_matrices(df, task)
    for name, cm_df in cm_dict.items():
        if not cm_df.empty:
            print(f"\n  [{name}]")
            print(cm_df.to_string())

    print("\n══ MedGemma initial→final shift ══")
    shift_df = section_medgemma_shift(df, task)
    if multi:
        n = len(shift_df)
        print(f"  n={n}  init_acc={shift_df['init_correct'].mean():.4f}  "
              f"fin_acc={shift_df['fin_correct'].mean():.4f}  "
              f"changed={shift_df['changed'].sum()} ({shift_df['changed'].mean():.1%})  "
              f"recovered={shift_df['recovered'].sum()}  degraded={shift_df['degraded'].sum()}")
    else:
        for cls in ["normal", pos]:
            sub = shift_df[shift_df["true"] == cls]
            if not sub.empty:
                n = len(sub)
                print(f"  true={cls} (n={n}): "
                      f"init_acc={sub['init_correct'].mean():.4f}  "
                      f"fin_acc={sub['fin_correct'].mean():.4f}  "
                      f"changed={sub['changed'].sum()} ({sub['changed'].mean():.1%})  "
                      f"recovered={sub['recovered'].sum()}  degraded={sub['degraded'].sum()}")

    print("\n══ Calibration ══")
    calib_df, bins_dict = section_calibration(df, task)
    print(calib_df.to_string(index=False))

    print("\n══ Latency ══")
    lat_df = section_latency(df)
    print(lat_df.to_string(index=False))

    forest_df = section_forest_voting(df, task)
    if not forest_df.empty:
        print("\n══ Agent Forest — voting quality (dissent vs. accuracy) ══")
        print(forest_df.to_string(index=False))

    debate_df = section_debate_rounds(df, task)
    if not debate_df.empty:
        print("\n══ Multi-Agent Debate — round analysis (verdict stability vs. ECE) ══")
        print(debate_df.to_string(index=False))

    # ── save CSVs ──────────────────────────────────────────────────────────────
    print("\nSaving CSVs...")
    accuracy_df.to_csv(out / "model_accuracy_summary.csv", index=False)
    shift_df.to_csv(out / "medgemma_shift_analysis.csv",   index=False)
    calib_df.to_csv(out / "confidence_calibration_summary.csv", index=False)
    lat_df.to_csv(out / "latency_stats.csv", index=False)
    if not forest_df.empty:
        forest_df.to_csv(out / "forest_voting_quality.csv", index=False)
    if not debate_df.empty:
        debate_df.to_csv(out / "debate_round_analysis.csv", index=False)
    for name, bins in bins_dict.items():
        bins.to_csv(out / f"calibration_bins_{name}.csv", index=False)
    for name, cm_df in cm_dict.items():
        if not cm_df.empty:
            cm_df.to_csv(out / f"confusion_matrix_{name}.csv")
    print("  Done.")

    # ── save plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_model_accuracy(accuracy_df, task, out)
    plot_confusion_matrices(cm_dict, task, out)
    plot_medgemma_initial_vs_final(shift_df, task, out)
    plot_calibration(bins_dict, task, out)
    plot_confidence_by_correctness(df, task, out)
    plot_latency(df, task, out)

    print(f"\nAll outputs saved to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
