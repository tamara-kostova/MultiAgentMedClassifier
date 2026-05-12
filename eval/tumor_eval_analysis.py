"""
Comprehensive analysis of a binary-tumor tumor_eval JSONL produced by run_pipeline.py.

Usage:
    python eval/tumor_eval_analysis.py \
        --jsonl outputs/eval/binary_tumor_tumor_eval.jsonl \
        [--output_dir outputs/analysis]

Outputs (outputs/analysis/)
────────────────────────────
Tables (CSV):
  model_accuracy_summary.csv           - per-model binary accuracy + F1 + AUC
  medgemma_field_distributions.csv     - value counts per structured-output field, initial vs final
  medgemma_field_agreement.csv         - initial/final agreement rate per field
  medgemma_field_accuracy.csv          - accuracy vs ground truth for diagnosis fields
  medgemma_diagnosis_confusion.csv     - confusion matrices for initial/final MedGemma
  cnn_metrics.csv                      - CNN per-class metrics + AUC
  cnn_calibration_bins.csv             - CNN 10-bin reliability diagram data
  clip_metrics.csv                     - BiomedCLIP metrics + collapse note
  clip_score_distribution.csv          - BiomedCLIP score percentiles by true class
  sam3_bbox_stats.csv                  - bbox coverage stats by true class
  sam3_iou_percentiles.csv             - IoU percentile table by true class
  sam3_false_positive.csv              - SAM3 activation analysis on normal scans
  confidence_calibration_summary.csv   - ECE + mean conf per model
  calibration_bins_<model>.csv         - per-model 10-bin calibration tables
  initial_vs_final_disagreements.csv   - rows where MedGemma changed its binary diagnosis
  human_review_analysis.csv            - flagging rate + per-subset accuracy
  latency_stats.csv                    - latency summary + by routing path
  severity_score_analysis.csv          - severity_score distribution + accuracy correlation
Narrative:
  analysis_report.md                   - full markdown report with embedded tables
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── label helpers ─────────────────────────────────────────────────────────────

GT_MAP = {"yes": "tumor", "no": "normal"}   # Br35H raw label → binary

TUMOR_WORDS = {
    "tumor", "glioma", "meningioma", "pituitary", "mass", "granuloma",
    "cyst", "schwannoma", "vestibular schwannoma", "hemorrhagic",
    "multiple sclerosis", "brain tumor", "brain tumor mri",
    "liver mass", "other abnormalities",
}


def to_binary(label) -> str:
    if label is None or str(label).lower() in ("null", "none", "nan", ""):
        return "unknown"
    s = str(label).lower().strip()
    if s in ("normal", "normal brain mri"):
        return "normal"
    if s == "brain tumor mri":
        return "tumor"
    for w in TUMOR_WORDS:
        if w in s:
            return "tumor"
    return "unknown"


def mg_field(diag, field):
    if isinstance(diag, dict):
        return diag.get(field)
    return None


def mg_to_binary(diag) -> str:
    return to_binary(mg_field(diag, "diagnosis_name"))


# ── metric helpers ────────────────────────────────────────────────────────────

def binary_metrics_dict(y_true, y_pred) -> dict:
    pairs = [(t, p) for t, p in zip(y_true, y_pred)
             if t not in ("unknown",) and p not in ("unknown",)]
    if not pairs:
        return {}
    yt, yp = zip(*pairs)
    cm = confusion_matrix(yt, yp, labels=["normal", "tumor"])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": len(yt),
        "accuracy":      round(accuracy_score(yt, yp), 4),
        "f1_macro":      round(f1_score(yt, yp, average="macro", zero_division=0,
                                        labels=["normal", "tumor"]), 4),
        "prec_tumor":    round(precision_score(yt, yp, pos_label="tumor",  zero_division=0), 4),
        "rec_tumor":     round(recall_score(yt, yp, pos_label="tumor",    zero_division=0), 4),
        "prec_normal":   round(precision_score(yt, yp, pos_label="normal", zero_division=0), 4),
        "rec_normal":    round(recall_score(yt, yp, pos_label="normal",   zero_division=0), 4),
        "specificity":   round(tn / (tn + fp) if (tn + fp) > 0 else float("nan"), 4),
        "sensitivity":   round(tp / (tp + fn) if (tp + fn) > 0 else float("nan"), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece)


def calibration_bins(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n = int(mask.sum())
        rows.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "n": n,
            "mean_conf": round(float(confidences[mask].mean()), 4) if n > 0 else float("nan"),
            "mean_acc":  round(float(correct[mask].mean()),  4) if n > 0 else float("nan"),
            "gap":       round(float(abs(confidences[mask].mean() - correct[mask].mean())), 4) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def roc_auc_binary(y_true, y_scores, pos_label="tumor"):
    pairs = [(t, s) for t, s in zip(y_true, y_scores) if t not in ("unknown",)]
    if not pairs:
        return float("nan")
    yt, ys = zip(*pairs)
    y_bin = [1 if t == pos_label else 0 for t in yt]
    try:
        return round(roc_auc_score(y_bin, ys), 4)
    except ValueError:
        return float("nan")


# ── multiclass helpers ────────────────────────────────────────────────────────

_BINARY_CLASS_SET = frozenset({"tumor", "normal"})


def _is_binary(df: pd.DataFrame) -> bool:
    return set(df["gt"].dropna().unique()).issubset(_BINARY_CLASS_SET | {"unknown"})


def _unique_classes(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df["gt"].dropna().unique() if c != "unknown")


def _normalize_pred(pred_str, classes: list[str]) -> str:
    if pred_str is None or str(pred_str).lower() in ("null", "none", "nan", ""):
        return "unknown"
    s = str(pred_str).lower().strip()
    for c in classes:
        if c.lower() == s:
            return c
    for c in classes:
        if c.lower() in s:
            return c
    return "unknown"


def multiclass_metrics_dict(y_true, y_pred) -> dict:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "unknown" and p != "unknown"]
    if not pairs:
        return {}
    yt, yp = zip(*pairs)
    return {
        "n": len(yt),
        "accuracy": round(accuracy_score(yt, yp), 4),
        "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
    }


def _mg_norm_series(diag_series: "pd.Series", binary: bool, classes: list) -> "pd.Series":
    """Normalise a MedGemma diagnosis-dict series to a prediction label.

    Binary mode  → binary label via mg_to_binary (uses diagnosis_name).
    Multiclass   → subtype via diagnosis_detailed (glioma/meningioma/pituitary_tumor),
                   because the prompt constrains diagnosis_name to 'tumor' for all tumors.
    """
    if binary:
        return diag_series.apply(mg_to_binary)
    return diag_series.apply(lambda d: _normalize_pred(mg_field(d, "diagnosis_detailed"), classes))


# ── end multiclass helpers ────────────────────────────────────────────────────


def percentile_table(values: np.ndarray, name: str) -> dict:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return {}
    return {
        "name": name,
        "n": len(v),
        "mean": round(float(v.mean()), 4),
        "std":  round(float(v.std()),  4),
        "min":  round(float(v.min()),  4),
        "p10":  round(float(np.percentile(v, 10)), 4),
        "p25":  round(float(np.percentile(v, 25)), 4),
        "p50":  round(float(np.percentile(v, 50)), 4),
        "p75":  round(float(np.percentile(v, 75)), 4),
        "p90":  round(float(np.percentile(v, 90)), 4),
        "max":  round(float(v.max()),  4),
    }


def df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def print_section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print("═" * 72)


# ── loader ────────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["medgemma_diagnosis"] = df["medgemma_diagnosis"].apply(
        lambda x: x if isinstance(x, dict) else {})
    df["final_medgemma_diagnosis"] = df["final_medgemma_diagnosis"].apply(
        lambda x: x if isinstance(x, dict) else {})
    df["gt"] = df["true_label"].map(GT_MAP)
    # Figshare3 labels ("1","2","3") are not in GT_MAP — use true_label_name
    missing = df["gt"].isna()
    if missing.any():
        df.loc[missing, "gt"] = (
            df.loc[missing, "true_label_name"].fillna(df.loc[missing, "true_label"])
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# §1  Dataset split
# ══════════════════════════════════════════════════════════════════════════════

def section_dataset(df: pd.DataFrame) -> dict:
    print_section("1. Dataset split")
    counts = df["true_label"].value_counts()
    split = {GT_MAP.get(k, k): int(v) for k, v in counts.items()}
    print(f"  Total : {len(df)}")
    for label, n in split.items():
        print(f"    {label:8s} : {n}  ({n/len(df)*100:.1f}%)")
    return split


# ══════════════════════════════════════════════════════════════════════════════
# §2  Per-model binary accuracy summary
# ══════════════════════════════════════════════════════════════════════════════

def section_model_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    binary = _is_binary(df)
    classes = _unique_classes(df)
    title = "2. Per-model binary accuracy summary" if binary else "2. Per-model accuracy summary (multiclass)"
    print_section(title)

    gt = df["gt"]

    def _norm(series):
        if binary:
            return series.apply(to_binary)
        return series.apply(lambda x: _normalize_pred(x, classes))

    models = {
        "pipeline_final":   (_norm(df["predicted_class"]),     df["final_confidence"]),
        "cnn":              (_norm(df["cnn_predicted_class"]),  df["cnn_confidence"]),
        "biomedclip":       (_norm(df["biomedclip_top_label"]), df["biomedclip_top_score"]),
        "medgemma_initial": (_mg_norm_series(df["medgemma_diagnosis"], binary, classes),
                             df["medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0)),
        "medgemma_final":   (_mg_norm_series(df["final_medgemma_diagnosis"], binary, classes),
                             df["final_medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0)),
    }

    rows = []
    for name, (preds, scores) in models.items():
        if binary:
            m = binary_metrics_dict(gt, preds)
            if not m:
                continue
            valid_mask = (gt != "unknown") & (preds != "unknown")
            yt_valid  = gt[valid_mask].tolist()
            yp_valid  = preds[valid_mask].tolist()
            ys_valid  = scores[valid_mask].tolist()
            auc_oriented = [s if p == "tumor" else 1 - s for p, s in zip(yp_valid, ys_valid)]
            auc = roc_auc_binary(yt_valid, auc_oriented)
            rows.append({
                "model": name, "n": m["n"],
                "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "roc_auc": auc,
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "prec_tumor": m["prec_tumor"],   "rec_tumor": m["rec_tumor"],
                "prec_normal": m["prec_normal"], "rec_normal": m["rec_normal"],
            })
        else:
            m = multiclass_metrics_dict(gt, preds)
            if not m:
                continue
            rows.append({"model": name, "n": m["n"],
                         "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §3  MedGemma per-field analysis
# ══════════════════════════════════════════════════════════════════════════════

STRUCTURED_FIELDS = [
    "modality", "specialized_sequence", "plane",
    "diagnosis_name", "diagnosis_detailed",
    "icd10_code", "severity_score", "diagnosis_confidence",
]

# Fields where we can derive binary ground truth
GT_APPLICABLE_FIELDS = {"diagnosis_name", "diagnosis_detailed"}

# Expected correct value for factual fields (Br35H dataset)
FACTUAL_EXPECTED = {"modality": "MRI", "plane": "axial"}


def _extract_field_series(diag_col: pd.Series, field: str) -> pd.Series:
    return diag_col.apply(lambda d: mg_field(d, field))


def section_medgemma_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("3. MedGemma structured-output per-field analysis")

    binary  = _is_binary(df)
    classes = _unique_classes(df)
    gt = df["gt"]

    distribution_rows = []   # value counts per (field, stage, value)
    agreement_rows    = []   # initial/final agreement per field
    accuracy_rows     = []   # accuracy vs GT where applicable

    for field in STRUCTURED_FIELDS:
        init_vals = _extract_field_series(df["medgemma_diagnosis"],       field)
        fin_vals  = _extract_field_series(df["final_medgemma_diagnosis"], field)

        # ── distribution ──────────────────────────────────────────────────────
        init_counts = Counter(str(v) for v in init_vals)
        fin_counts  = Counter(str(v) for v in fin_vals)
        all_vals    = sorted(set(list(init_counts) + list(fin_counts)))
        for val in all_vals:
            distribution_rows.append({
                "field": field,
                "value": val,
                "initial_n": init_counts.get(val, 0),
                "final_n":   fin_counts.get(val, 0),
            })

        # ── agreement ─────────────────────────────────────────────────────────
        agree_mask = (init_vals.astype(str) == fin_vals.astype(str))
        both_null  = (init_vals.isna() & fin_vals.isna())
        agreement_rate = agree_mask.mean()
        agreement_rows.append({
            "field": field,
            "agreement_rate": round(float(agreement_rate), 4),
            "n_agree": int(agree_mask.sum()),
            "n_differ": int((~agree_mask).sum()),
            "both_null": int(both_null.sum()),
            "null_initial": int(init_vals.isna().sum()),
            "null_final":   int(fin_vals.isna().sum()),
        })

        # ── accuracy vs GT ─────────────────────────────────────────────────────
        if field in GT_APPLICABLE_FIELDS:
            for stage, vals in [("initial", init_vals), ("final", fin_vals)]:
                if binary:
                    preds = vals.apply(to_binary)
                    m = binary_metrics_dict(gt, preds)
                elif field == "diagnosis_name":
                    # In multiclass context, diagnosis_name is always "tumor" per schema;
                    # gt for all figshare3 images is also tumor at this level.
                    gt_tumor = pd.Series(["tumor"] * len(df), index=df.index)
                    preds = vals.astype(str).str.lower().str.strip()
                    m = multiclass_metrics_dict(gt_tumor, preds)
                else:
                    # diagnosis_detailed → compare against actual subtypes
                    preds = vals.apply(lambda x: _normalize_pred(x, classes))
                    m = multiclass_metrics_dict(gt, preds)
                if m:
                    accuracy_rows.append({
                        "field": field, "stage": stage, **m
                    })

        elif field in FACTUAL_EXPECTED:
            expected = FACTUAL_EXPECTED[field]
            for stage, vals in [("initial", init_vals), ("final", fin_vals)]:
                acc = (vals.astype(str).str.lower() == expected.lower()).mean()
                accuracy_rows.append({
                    "field": field, "stage": stage,
                    "expected_value": expected,
                    "accuracy": round(float(acc), 4),
                    "n": len(vals),
                })

    dist_df  = pd.DataFrame(distribution_rows)
    agree_df = pd.DataFrame(agreement_rows)
    acc_df   = pd.DataFrame(accuracy_rows)

    # ── print summary tables ──────────────────────────────────────────────────
    print("\n  [Agreement rates — initial vs final]")
    agree_print = agree_df[["field", "agreement_rate", "n_agree", "n_differ", "null_initial", "null_final"]]
    print(agree_print.to_string(index=False))

    print("\n  [Accuracy vs ground truth — diagnosis fields]")
    diag_acc = acc_df[acc_df["field"].isin(GT_APPLICABLE_FIELDS)]
    if not diag_acc.empty:
        cols = [c for c in ["field","stage","n","accuracy","f1_macro","sensitivity","specificity"] if c in diag_acc.columns]
        print(diag_acc[cols].to_string(index=False))

    print("\n  [Factual field accuracy (modality→MRI, plane→axial)]")
    fact_acc = acc_df[acc_df["field"].isin(FACTUAL_EXPECTED)]
    if not fact_acc.empty:
        cols = [c for c in ["field","stage","expected_value","accuracy","n"] if c in fact_acc.columns]
        print(fact_acc[cols].to_string(index=False))

    print("\n  [Detailed distributions — top values per field]")
    for field in STRUCTURED_FIELDS:
        sub = dist_df[dist_df["field"] == field].sort_values("initial_n", ascending=False)
        top = sub.head(10)
        if top.empty:
            continue
        print(f"\n  {field}")
        print(top[["value", "initial_n", "final_n"]].to_string(index=False))

    return dist_df, agree_df, acc_df


# ══════════════════════════════════════════════════════════════════════════════
# §4  Initial vs final MedGemma — binary diagnosis shift
# ══════════════════════════════════════════════════════════════════════════════

def section_initial_vs_final(df: pd.DataFrame) -> pd.DataFrame:
    print_section("4. Initial vs final MedGemma diagnosis shift")

    binary  = _is_binary(df)
    classes = _unique_classes(df)
    gt      = df["gt"]

    init_pred = _mg_norm_series(df["medgemma_diagnosis"],       binary, classes)
    fin_pred  = _mg_norm_series(df["final_medgemma_diagnosis"], binary, classes)
    if binary:
        cnn_pred = df["cnn_predicted_class"].apply(to_binary)
    else:
        cnn_pred = df["cnn_predicted_class"].apply(lambda x: _normalize_pred(x, classes))

    agree_mask = (init_pred == fin_pred)
    label = "binary" if binary else "class"
    print(f"  Agreement ({label}) : {agree_mask.sum()} / {len(df)}  ({agree_mask.mean()*100:.1f}%)")
    print(f"  Changed             : {(~agree_mask).sum()} ({(~agree_mask).mean()*100:.1f}%)")

    shift_df = pd.DataFrame({
        "init_pred": init_pred, "fin_pred": fin_pred, "gt": gt,
        "cnn_pred": cnn_pred, "cnn_conf": df["cnn_confidence"],
    })
    shift_counts = shift_df.groupby(["init_pred", "fin_pred"]).size().reset_index(name="n")
    print(f"\n  Shift breakdown (init → fin):")
    print(shift_counts.to_string(index=False))

    print()
    for stage, preds in [("initial", init_pred), ("final", fin_pred)]:
        both = (gt != "unknown") & (preds != "unknown")
        if binary:
            m = binary_metrics_dict(gt[both], preds[both])
            if m:
                print(f"  {stage:8s}  n={m['n']}  acc={m['accuracy']}  f1={m['f1_macro']}  "
                      f"sens={m['sensitivity']}  spec={m['specificity']}")
        else:
            m = multiclass_metrics_dict(gt[both], preds[both])
            if m:
                print(f"  {stage:8s}  n={m['n']}  acc={m['accuracy']}  f1={m['f1_macro']}")

    init_correct = (init_pred == gt) & (gt != "unknown") & (init_pred != "unknown")
    fin_correct  = (fin_pred  == gt) & (gt != "unknown") & (fin_pred  != "unknown")
    print(f"\n  Final recovered cases (init wrong → final right) : {((~init_correct) & fin_correct).sum()}")
    print(f"  Final degraded  cases (init right → final wrong) : {(init_correct & (~fin_correct)).sum()}")

    disagree = df[~agree_mask].copy()
    disagree["gt_val"]    = gt[~agree_mask].values
    disagree["init_pred"] = init_pred[~agree_mask].values
    disagree["fin_pred"]  = fin_pred[~agree_mask].values
    return disagree[["image_path","true_label","gt_val","init_pred","fin_pred",
                      "cnn_predicted_class","cnn_confidence"]]


# ══════════════════════════════════════════════════════════════════════════════
# §5  CNN detailed metrics + calibration
# ══════════════════════════════════════════════════════════════════════════════

def section_cnn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("5. CNN detailed metrics + calibration")

    binary  = _is_binary(df)
    classes = _unique_classes(df)
    gt      = df["gt"]

    if binary:
        pred = df["cnn_predicted_class"].apply(to_binary)
    else:
        pred = df["cnn_predicted_class"].apply(lambda x: _normalize_pred(x, classes))

    conf    = df["cnn_confidence"].values
    valid   = (gt != "unknown") & (pred != "unknown")
    gt_v    = gt[valid]
    pred_v  = pred[valid]
    conf_v  = conf[valid.values]
    correct = (pred_v == gt_v).values.astype(float)
    ece     = compute_ece(conf_v, correct)
    cal_df  = calibration_bins(conf_v, correct)

    per_class_rows = []
    for cls in classes:
        cls_mask = (gt_v == cls).values
        if cls_mask.sum() == 0:
            continue
        c = conf_v[cls_mask]
        per_class_rows.append({
            "true_class": cls,
            "n": int(cls_mask.sum()),
            "conf_mean": round(float(c.mean()), 4),
            "conf_std":  round(float(c.std()),  4),
            "conf_min":  round(float(c.min()),  4),
            "conf_p50":  round(float(np.percentile(c, 50)), 4),
            "conf_max":  round(float(c.max()),  4),
            "accuracy":  round(float((pred_v[gt_v == cls] == cls).mean()), 4),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    if binary:
        m = binary_metrics_dict(gt, pred)
        auc_scores = [s if p == "tumor" else 1 - s for p, s in zip(pred_v, conf_v)]
        auc = roc_auc_binary(gt_v.tolist(), auc_scores)
        metrics_row = {**m, "roc_auc": auc, "ece": round(ece, 4)}
        print(f"  Accuracy:    {m['accuracy']}    F1-macro: {m['f1_macro']}    AUC: {auc}    ECE: {round(ece,4)}")
        print(f"  Sensitivity: {m['sensitivity']}   Specificity: {m['specificity']}")
        print(f"  Confusion matrix (normal/tumor):  TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    else:
        m = multiclass_metrics_dict(gt, pred)
        metrics_row = {**m, "ece": round(ece, 4)}
        print(f"  Accuracy: {m['accuracy']}    F1-macro: {m['f1_macro']}    ECE: {round(ece,4)}")
        cm = confusion_matrix(gt_v.tolist(), pred_v.tolist(), labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                             columns=[f"pred_{c}" for c in classes])
        print("\n  Confusion matrix:")
        print(cm_df.to_string())

    print("\n  Per-class confidence:")
    print(per_class_df.to_string(index=False))
    print("\n  Calibration bins (10 equal-width):")
    print(cal_df.to_string(index=False))

    return per_class_df, cal_df, pd.DataFrame([metrics_row])


# ══════════════════════════════════════════════════════════════════════════════
# §6  BiomedCLIP detailed metrics
# ══════════════════════════════════════════════════════════════════════════════

def section_clip(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("6. BiomedCLIP detailed metrics")

    binary  = _is_binary(df)
    classes = _unique_classes(df)
    gt      = df["gt"]

    if binary:
        pred = df["biomedclip_top_label"].apply(to_binary)
    else:
        pred = df["biomedclip_top_label"].apply(lambda x: _normalize_pred(x, classes))

    scores = df["biomedclip_top_score"].values

    unique_preds = pred.unique()
    print(f"  Unique predictions : {unique_preds.tolist()}")
    if len(unique_preds) == 1:
        print(f"  ⚠  BiomedCLIP predicted '{unique_preds[0]}' for ALL {len(df)} samples (label collapse).")

    if binary:
        m = binary_metrics_dict(gt, pred)
        print(f"\n  Accuracy: {m.get('accuracy','N/A')}  F1: {m.get('f1_macro','N/A')}  "
              f"Sens: {m.get('sensitivity','N/A')}  Spec: {m.get('specificity','N/A')}")
    else:
        m = multiclass_metrics_dict(gt, pred)
        print(f"\n  Accuracy: {m.get('accuracy','N/A')}  F1: {m.get('f1_macro','N/A')}")

    # Score distribution by true class
    pct_rows = []
    for cls in classes + ["all"]:
        mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
        s = scores[mask.values]
        row = percentile_table(s, cls)
        pct_rows.append(row)
    pct_df = pd.DataFrame(pct_rows)
    print("\n  Top-score percentiles by true class:")
    print(pct_df.to_string(index=False))

    # Score difference between classes
    if binary:
        score_tumor  = scores[(gt == "tumor").values]
        score_normal = scores[(gt == "normal").values]
        if len(score_tumor) and len(score_normal):
            diff = score_tumor.mean() - score_normal.mean()
            print(f"\n  Mean score: tumor={score_tumor.mean():.4f}  normal={score_normal.mean():.4f}  "
                  f"diff={diff:.4f}  (positive = BiomedCLIP scores tumors higher)")
    else:
        score_parts = []
        for cls in classes:
            s = scores[(gt == cls).values]
            if len(s):
                score_parts.append(f"{cls}={s.mean():.4f}")
        if score_parts:
            print(f"\n  Mean top-score by class:  {',  '.join(score_parts)}")

    valid = (gt != "unknown") & (pred != "unknown")
    conf_v   = scores[valid.values]
    correct  = (pred[valid] == gt[valid]).values.astype(float)
    ece      = compute_ece(conf_v, correct)
    cal_df   = calibration_bins(conf_v, correct)
    print(f"\n  ECE: {round(ece,4)}")
    print("\n  Calibration bins:")
    print(cal_df.to_string(index=False))

    metrics_row = pd.DataFrame([{**m, "ece": round(ece, 4)}])
    return pct_df, cal_df, metrics_row


# ══════════════════════════════════════════════════════════════════════════════
# §7  SAM3 detailed analysis
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_coverage(bboxes, img_size=512) -> np.ndarray:
    img_area = img_size * img_size
    result = []
    for b in bboxes:
        if isinstance(b, list) and len(b) >= 4:
            x1, y1, x2, y2 = b[:4]
            result.append(abs((x2 - x1) * (y2 - y1)) / img_area)
        else:
            result.append(float("nan"))
    return np.array(result)


def section_sam3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("7. SAM3 detailed analysis")

    classes  = _unique_classes(df)
    gt       = df["gt"]
    iou      = df["saliency_sam3_iou"].values
    coverage = _bbox_coverage(df["sam3_bbox"].tolist())
    skipped  = df["sam3_skipped"].values

    print(f"  sam3_skipped : {skipped.sum()} / {len(df)}")

    # ── IoU percentile table by class ─────────────────────────────────────────
    print("\n  [IoU (GradCAM++ ∩ SAM3 mask) — percentiles by true class]")
    iou_rows = []
    for cls in classes + ["all"]:
        mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
        iou_cls = iou[mask.values]
        row = percentile_table(iou_cls, cls)
        if row:
            row["zero_rate"] = round(float((iou_cls == 0).mean()), 4)
        iou_rows.append(row)
    iou_df = pd.DataFrame(iou_rows)
    print(iou_df.to_string(index=False))

    # ── bbox coverage by class ─────────────────────────────────────────────────
    print("\n  [Bbox coverage (bbox_area / 512²) by true class]")
    bbox_rows = []
    for cls in classes + ["all"]:
        mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
        cov = coverage[mask.values]
        row = percentile_table(cov, cls)
        if row:
            row["full_image_rate"] = round(float((cov >= 0.99).mean()), 4)
        bbox_rows.append(row)
    bbox_df = pd.DataFrame(bbox_rows)
    print(bbox_df.to_string(index=False))

    # ── per-class SAM3 activation (coverage bins) ──────────────────────────────
    print("\n  [SAM3 bbox coverage distribution by true class]")
    cov_bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.01]
    fp_rows = []
    for cls in classes:
        cls_mask = (gt == cls).values
        cov_cls  = coverage[cls_mask]
        if len(cov_cls) == 0:
            continue
        for lo, hi in zip(cov_bins[:-1], cov_bins[1:]):
            n_in = int(((cov_cls >= lo) & (cov_cls < hi)).sum())
            fp_rows.append({
                "true_class": cls,
                "coverage_range": f"[{lo:.2f}, {hi:.2f})",
                "n": n_in,
                "pct": round(n_in / len(cov_cls) * 100, 1),
            })
    fp_df = pd.DataFrame(fp_rows)
    not_full = (coverage < 0.99) & ~np.isnan(coverage)
    print(f"  SAM3 localised (bbox <99% of image): {not_full.sum()} / {len(df)} ({not_full.mean()*100:.1f}%)")
    print(fp_df.to_string(index=False))

    # ── IoU vs CNN confidence cross-analysis ──────────────────────────────────
    print("\n  [IoU > 0 vs CNN confidence — do they correlate?]")
    iou_nonzero = iou > 0
    cnn_conf = df["cnn_confidence"].values
    for label, mask in [("IoU=0", ~iou_nonzero), ("IoU>0", iou_nonzero)]:
        c = cnn_conf[mask]
        print(f"  {label:8s}  n={mask.sum():4d}  cnn_conf_mean={c.mean():.4f}  std={c.std():.4f}")

    return iou_df, bbox_df, fp_df


# ══════════════════════════════════════════════════════════════════════════════
# §8  Confidence calibration summary across all models
# ══════════════════════════════════════════════════════════════════════════════

def section_confidence_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print_section("8. Confidence calibration summary (all models)")

    binary  = _is_binary(df)
    classes = _unique_classes(df)
    gt      = df["gt"]
    valid_mask = (gt != "unknown")

    def get_conf_correct(pred_series, conf_series):
        both = valid_mask & (pred_series != "unknown")
        pv   = pred_series[both]
        gv   = gt[both]
        cv   = conf_series[both].values if hasattr(conf_series[both], "values") else np.array(conf_series[both])
        c    = (pv == gv).values.astype(float)
        return cv, c

    def _norm(series):
        if binary:
            return series.apply(to_binary)
        return series.apply(lambda x: _normalize_pred(x, classes))

    models_conf = {
        "cnn": (
            _norm(df["cnn_predicted_class"]),
            df["cnn_confidence"],
        ),
        "medgemma_initial": (
            _mg_norm_series(df["medgemma_diagnosis"], binary, classes),
            df["medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0),
        ),
        "medgemma_final": (
            _mg_norm_series(df["final_medgemma_diagnosis"], binary, classes),
            df["final_medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0),
        ),
        "biomedclip": (
            _norm(df["biomedclip_top_label"]),
            df["biomedclip_top_score"],
        ),
    }

    summary_rows = []
    cal_bins_dict = {}

    for name, (pred, conf) in models_conf.items():
        cv, corr = get_conf_correct(pred, conf)
        if len(cv) == 0:
            continue
        ece = compute_ece(cv, corr)
        cal = calibration_bins(cv, corr)
        cal_bins_dict[name] = cal

        mean_all   = float(cv.mean())
        mean_corr  = float(cv[corr == 1].mean()) if (corr == 1).any() else float("nan")
        mean_wrong = float(cv[corr == 0].mean()) if (corr == 0).any() else float("nan")
        summary_rows.append({
            "model": name,
            "n": len(cv),
            "ece": round(ece, 4),
            "mean_conf_all":   round(mean_all,   4),
            "mean_conf_correct": round(mean_corr,  4) if not np.isnan(mean_corr)  else float("nan"),
            "mean_conf_wrong":   round(mean_wrong, 4) if not np.isnan(mean_wrong) else float("nan"),
            "overconfidence":  round(mean_all - float(corr.mean()), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n  Calibration bins per model:")
    for name, cal in cal_bins_dict.items():
        print(f"\n  [{name}]")
        print(cal.to_string(index=False))

    return summary_df, cal_bins_dict


# ══════════════════════════════════════════════════════════════════════════════
# §9  Human-review flag analysis
# ══════════════════════════════════════════════════════════════════════════════

def section_human_review(df: pd.DataFrame) -> pd.DataFrame:
    print_section("9. Human-review flag analysis")

    binary  = _is_binary(df)
    classes = _unique_classes(df)
    gt      = df["gt"]
    flagged = df["requires_human_review"]
    iou     = df["saliency_sam3_iou"].values

    total_flagged = int(flagged.sum())
    print(f"  Flagged : {total_flagged} / {len(df)}  ({total_flagged/len(df)*100:.1f}%)")

    if binary:
        pred_models = [
            ("cnn",           df["cnn_predicted_class"].apply(to_binary)),
            ("medgemma_init", _mg_norm_series(df["medgemma_diagnosis"], binary, classes)),
        ]
    else:
        pred_models = [
            ("cnn",           df["cnn_predicted_class"].apply(lambda x: _normalize_pred(x, classes))),
            ("medgemma_init", _mg_norm_series(df["medgemma_diagnosis"], binary, classes)),
        ]

    rows = []
    for label, mask in [("flagged", flagged), ("not_flagged", ~flagged)]:
        n = int(mask.sum())
        if n == 0:
            rows.append({"subset": label, "n": 0})
            continue
        for model_name, pred in pred_models:
            sub_gt   = gt[mask]
            sub_pred = pred[mask]
            both     = (sub_gt != "unknown") & (sub_pred != "unknown")
            if both.sum() == 0:
                continue
            if binary:
                m = binary_metrics_dict(sub_gt[both], sub_pred[both])
                rows.append({
                    "subset": label, "model": model_name, "n": n,
                    "accuracy": m.get("accuracy"), "f1_macro": m.get("f1_macro"),
                    "sensitivity": m.get("sensitivity"), "specificity": m.get("specificity"),
                })
            else:
                m = multiclass_metrics_dict(sub_gt[both], sub_pred[both])
                rows.append({
                    "subset": label, "model": model_name, "n": n,
                    "accuracy": m.get("accuracy"), "f1_macro": m.get("f1_macro"),
                })

    review_df = pd.DataFrame(rows)
    print(review_df.dropna(subset=["accuracy"]).to_string(index=False))

    # IoU correlation with flag
    for label, mask in [("flagged", flagged.values), ("not_flagged", (~flagged).values)]:
        iou_sub = iou[mask]
        if len(iou_sub) == 0:
            print(f"  IoU {label}: N/A")
        else:
            print(f"  IoU {label:12s}: n={len(iou_sub)}  mean={iou_sub.mean():.4f}  zero%={(iou_sub==0).mean()*100:.1f}%")

    return review_df


# ══════════════════════════════════════════════════════════════════════════════
# §10  Severity score analysis
# ══════════════════════════════════════════════════════════════════════════════

def section_severity(df: pd.DataFrame) -> pd.DataFrame:
    print_section("10. Severity score analysis")

    classes = _unique_classes(df)
    gt      = df["gt"]

    init_sev = _extract_field_series(df["medgemma_diagnosis"],       "severity_score")
    fin_sev  = _extract_field_series(df["final_medgemma_diagnosis"], "severity_score")
    init_sev_num = pd.to_numeric(init_sev, errors="coerce")
    fin_sev_num  = pd.to_numeric(fin_sev,  errors="coerce")

    rows = []
    for stage, vals_num in [("initial", init_sev_num), ("final", fin_sev_num)]:
        for cls in classes + ["all"]:
            mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
            v = vals_num[mask].dropna().values
            if len(v) == 0:
                continue
            rows.append({
                "stage": stage, "true_class": cls,
                "n_with_score": len(v),
                "n_null": int(mask.sum()) - len(v),
                "mean":  round(float(v.mean()), 4),
                "std":   round(float(v.std()),  4),
                "p25":   round(float(np.percentile(v, 25)), 4),
                "p50":   round(float(np.percentile(v, 50)), 4),
                "p75":   round(float(np.percentile(v, 75)), 4),
            })

    sev_df = pd.DataFrame(rows)
    print(sev_df.to_string(index=False))
    return sev_df


# ══════════════════════════════════════════════════════════════════════════════
# §11  Latency
# ══════════════════════════════════════════════════════════════════════════════

def section_latency(df: pd.DataFrame) -> pd.DataFrame:
    print_section("11. Latency analysis")

    classes = _unique_classes(df)
    lat     = df["latency_s"].values
    gt      = df["gt"]

    summary_rows = [percentile_table(lat, "all")]
    for cls in classes:
        summary_rows.append(percentile_table(lat[(gt == cls).values], cls))
    lat_df = pd.DataFrame(summary_rows)
    print(lat_df.to_string(index=False))

    # By routing path
    paths = Counter(df["routing_path"].tolist())
    print("\n  By routing path:")
    path_rows = []
    for path, n in paths.most_common():
        mask  = df["routing_path"] == path
        lp    = df["latency_s"][mask].values
        path_rows.append({
            "path": path, "n": n,
            "mean_s": round(float(lp.mean()), 2),
            "median_s": round(float(np.median(lp)), 2),
            "std_s": round(float(lp.std()), 2),
        })
        print(f"  [{n:4d}]  {path}")
        print(f"         mean={lp.mean():.2f}s  median={np.median(lp):.2f}s  std={lp.std():.2f}s")
    return pd.DataFrame(path_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_multiclass_report(
    jsonl_path: str,
    df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    agree_df: pd.DataFrame,
    field_acc_df: pd.DataFrame,
    cnn_class_df: pd.DataFrame,
    cnn_cal_df: pd.DataFrame,
    cnn_metrics_df: pd.DataFrame,
    clip_pct_df: pd.DataFrame,
    iou_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    conf_summary_df: pd.DataFrame,
    review_df: pd.DataFrame,
    sev_df: pd.DataFrame,
    lat_df: pd.DataFrame,
    out_dir: Path,
):
    """Markdown report for multiclass (Figshare3 meningioma/glioma/pituitary)."""
    classes = _unique_classes(df)
    n       = len(df)
    gt      = df["gt"]

    def get_metric(model, col):
        row = accuracy_df[accuracy_df["model"] == model]
        if row.empty:
            return "N/A"
        v = row.iloc[0].get(col, "N/A")
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    class_counts = {c: int((gt == c).sum()) for c in classes}
    counts_str   = "  ".join(f"{c}={n2}" for c, n2 in class_counts.items())

    total_flagged = int(df["requires_human_review"].sum()) if "requires_human_review" in df.columns else n
    flag_pct      = total_flagged / n * 100 if n > 0 else 0.0
    iou_vals      = df["saliency_sam3_iou"].values if "saliency_sam3_iou" in df.columns else np.array([])
    iou_zero_rate = float((iou_vals == 0).mean()) if len(iou_vals) > 0 else float("nan")

    lines = [
        "# Multiclass Tumor Evaluation — Analysis Report",
        "",
        f"**Source:** `{jsonl_path}`  ",
        f"**Total samples:** {n}  ({counts_str})  ",
        f"**Task:** multiclass tumor subtype classification (Figshare3: {', '.join(classes)})  ",
        "",
        "---",
        "",
        "## 1. Per-model accuracy summary",
        "",
        accuracy_df.to_markdown(index=False),
        "",
        "### Key observations",
        f"- **CNN**: acc={get_metric('cnn','accuracy')},  F1-macro={get_metric('cnn','f1_macro')}",
        f"- **BiomedCLIP**: acc={get_metric('biomedclip','accuracy')},  F1-macro={get_metric('biomedclip','f1_macro')}",
        f"- **MedGemma initial**: acc={get_metric('medgemma_initial','accuracy')},  F1-macro={get_metric('medgemma_initial','f1_macro')}",
        f"- **MedGemma final**: acc={get_metric('medgemma_final','accuracy')},  F1-macro={get_metric('medgemma_final','f1_macro')}",
        f"- **Pipeline final**: acc={get_metric('pipeline_final','accuracy')},  F1-macro={get_metric('pipeline_final','f1_macro')}",
        "",
        "---",
        "",
        "## 2. MedGemma structured-output field agreement",
        "",
        agree_df[["field", "agreement_rate", "n_agree", "n_differ", "null_initial", "null_final"]].to_markdown(index=False),
        "",
        "---",
        "",
        "## 3. CNN per-class confidence + calibration",
        "",
        cnn_class_df.to_markdown(index=False),
        "",
        cnn_cal_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 4. BiomedCLIP top-score percentiles by class",
        "",
        clip_pct_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 5. SAM3 IoU percentiles by class",
        "",
        iou_df.to_markdown(index=False),
        "",
        "### Bbox coverage by class",
        "",
        bbox_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 6. Confidence calibration (ECE)",
        "",
        conf_summary_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 7. Human-review flag analysis",
        "",
        f"- **{flag_pct:.1f}%** of samples flagged ({total_flagged}/{n})",
        "",
        review_df.dropna(subset=["accuracy"]).to_markdown(index=False) if not review_df.dropna(subset=["accuracy"]).empty else "_No data._",
        "",
        "---",
        "",
        "## 8. Latency",
        "",
        lat_df.to_markdown(index=False),
        "",
    ]

    report = "\n".join(lines)
    report_path = out_dir / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report written to {report_path}")
    return report_path


def generate_report(
    jsonl_path: str,
    df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    agree_df: pd.DataFrame,
    field_acc_df: pd.DataFrame,
    disagree_df: pd.DataFrame,
    cnn_class_df: pd.DataFrame,
    cnn_cal_df: pd.DataFrame,
    cnn_metrics_df: pd.DataFrame,
    clip_pct_df: pd.DataFrame,
    clip_cal_df: pd.DataFrame,
    clip_metrics_df: pd.DataFrame,
    iou_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    conf_summary_df: pd.DataFrame,
    cal_bins_dict: dict,
    review_df: pd.DataFrame,
    sev_df: pd.DataFrame,
    lat_df: pd.DataFrame,
    out_dir: Path,
):
    if not _is_binary(df):
        return _generate_multiclass_report(
            jsonl_path, df, accuracy_df, dist_df, agree_df, field_acc_df,
            cnn_class_df, cnn_cal_df, cnn_metrics_df,
            clip_pct_df, iou_df, bbox_df, fp_df,
            conf_summary_df, review_df, sev_df, lat_df, out_dir,
        )

    gt = df["gt"]
    n = len(df)
    n_tumor  = int((gt == "tumor").sum())
    n_normal = int((gt == "normal").sum())

    # Pull key metrics
    def get_metric(model, col):
        row = accuracy_df[accuracy_df["model"] == model]
        if row.empty:
            return "N/A"
        v = row.iloc[0].get(col, "N/A")
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def _float_metric(model, col):
        v = get_metric(model, col)
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("nan")

    # Detect BiomedCLIP label collapse
    clip_preds = df["biomedclip_top_label"].apply(to_binary)
    clip_unique = clip_preds.unique()
    clip_collapsed = len(clip_unique) == 1

    # Compute human-review rate from data
    total_flagged = int(df["requires_human_review"].sum()) if "requires_human_review" in df.columns else n
    flag_pct = total_flagged / n * 100 if n > 0 else 0.0

    # Compute SAM3 zero-IoU rate from data
    iou_vals = df["saliency_sam3_iou"].values if "saliency_sam3_iou" in df.columns else np.array([])
    iou_zero_rate = float((iou_vals == 0).mean()) if len(iou_vals) > 0 else float("nan")

    # BiomedCLIP key observation
    if clip_collapsed:
        clip_obs = (
            f"- **BiomedCLIP** collapsed to a single label ('{clip_unique[0]}') for all {n} samples, "
            f"giving {_float_metric('biomedclip','accuracy')*100:.1f}% accuracy (chance level). Specificity = 0."
        )
    else:
        clip_obs = (
            f"- **BiomedCLIP** (linear probe): acc={get_metric('biomedclip','accuracy')}, "
            f"F1={get_metric('biomedclip','f1_macro')}, "
            f"sensitivity={get_metric('biomedclip','sensitivity')}, "
            f"specificity={get_metric('biomedclip','specificity')}."
        )

    # Pipeline final observation
    pf_spec = _float_metric('pipeline_final', 'specificity')
    cnn_spec = _float_metric('cnn', 'specificity')
    if not (np.isnan(pf_spec) or np.isnan(cnn_spec)) and pf_spec < cnn_spec - 0.05:
        pipeline_obs = (
            f"- The **pipeline final** result has specificity={get_metric('pipeline_final','specificity')} — "
            f"degraded from CNN ({get_metric('cnn','specificity')}) due to false positives introduced by "
            f"MedGemma and/or BiomedCLIP on normal scans."
        )
    else:
        pipeline_obs = (
            f"- The **pipeline final** result: acc={get_metric('pipeline_final','accuracy')}, "
            f"sensitivity={get_metric('pipeline_final','sensitivity')}, "
            f"specificity={get_metric('pipeline_final','specificity')}."
        )

    lines = [
        "# Binary Tumor Evaluation — Analysis Report",
        "",
        f"**Source:** `{jsonl_path}`  ",
        f"**Total samples:** {n} ({n_tumor} tumor / {n_normal} normal)  ",
        f"**Task:** binary tumor detection (Br35H dataset)  ",
        "",
        "---",
        "",
        "## 1. Per-model accuracy summary",
        "",
        accuracy_df.to_markdown(index=False),
        "",
        "### Key observations",
        (
            f"- **CNN** achieves perfect accuracy ({get_metric('cnn','accuracy')}) and AUC ({get_metric('cnn','roc_auc')}) — likely memorised Br35H patterns."
            if _float_metric('cnn', 'accuracy') >= 0.9999
            else f"- **CNN**: acc={get_metric('cnn','accuracy')}, F1={get_metric('cnn','f1_macro')}, "
                 f"sensitivity={get_metric('cnn','sensitivity')}, specificity={get_metric('cnn','specificity')}."
        ),
        clip_obs,
        f"- **MedGemma initial triage**: acc={get_metric('medgemma_initial','accuracy')}, F1={get_metric('medgemma_initial','f1_macro')}, sensitivity={get_metric('medgemma_initial','sensitivity')}, specificity={get_metric('medgemma_initial','specificity')}.",
        f"- **MedGemma final report**: acc={get_metric('medgemma_final','accuracy')}, F1={get_metric('medgemma_final','f1_macro')}, sensitivity={get_metric('medgemma_final','sensitivity')}, specificity={get_metric('medgemma_final','specificity')}.",
        pipeline_obs,
        "",
        "---",
        "",
        "## 2. MedGemma structured-output per-field analysis",
        "",
        "### 2a. Initial vs final agreement rates",
        "",
        agree_df[["field", "agreement_rate", "n_agree", "n_differ", "null_initial", "null_final"]].to_markdown(index=False),
        "",
        "### 2b. Accuracy vs ground truth — diagnosis fields",
        "",
    ]

    diag_acc = field_acc_df[field_acc_df["field"].isin(GT_APPLICABLE_FIELDS)]
    if not diag_acc.empty:
        cols = [c for c in ["field","stage","n","accuracy","f1_macro","sensitivity","specificity"] if c in diag_acc.columns]
        lines.append(diag_acc[cols].to_markdown(index=False))
    else:
        lines.append("_No per-field accuracy data available._")

    fact_acc = field_acc_df[field_acc_df["field"].isin(FACTUAL_EXPECTED)]
    if not fact_acc.empty:
        lines += ["", "### 2c. Factual field accuracy (modality → MRI, plane → axial)", ""]
        cols = [c for c in ["field","stage","expected_value","accuracy","n"] if c in fact_acc.columns]
        lines.append(fact_acc[cols].to_markdown(index=False))

    lines += [
        "",
        "### 2d. Distribution highlights",
        "",
        "**diagnosis_name top values (initial → final):**",
    ]
    dn = dist_df[dist_df["field"] == "diagnosis_name"].sort_values("initial_n", ascending=False).head(10)
    lines.append(dn[["value","initial_n","final_n"]].to_markdown(index=False))

    lines += [
        "",
        "**specialized_sequence distribution:**",
    ]
    sq = dist_df[dist_df["field"] == "specialized_sequence"].sort_values("initial_n", ascending=False).head(8)
    lines.append(sq[["value","initial_n","final_n"]].to_markdown(index=False))

    lines += [
        "",
        "---",
        "",
        "## 3. Initial vs final MedGemma diagnosis shift",
        "",
        f"- Agreement (binary tumor/normal): {int((df['medgemma_diagnosis'].apply(mg_to_binary) == df['final_medgemma_diagnosis'].apply(mg_to_binary)).sum())} / {n}",
        f"- Disagree rows saved to `initial_vs_final_disagreements.csv`  ",
        "",
        "---",
        "",
        "## 4. CNN detailed metrics + calibration",
        "",
        f"**Accuracy:** {get_metric('cnn','accuracy')}  **AUC:** {get_metric('cnn','roc_auc')}  **ECE:** {cnn_metrics_df.iloc[0].get('ece','N/A') if not cnn_metrics_df.empty else 'N/A'}",
        "",
        "### Per-class confidence:",
        "",
        cnn_class_df.to_markdown(index=False),
        "",
        "### Calibration bins:",
        "",
        cnn_cal_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 5. BiomedCLIP metrics",
        "",
    ]

    if not clip_metrics_df.empty:
        lines.append(clip_metrics_df.to_markdown(index=False))

    clip_score_note: list[str] = []
    if clip_collapsed:
        clip_score_note = [
            f"> **Note:** BiomedCLIP predicted '{clip_unique[0]}' for ALL {n} samples (label collapse). ",
            "> Score distributions for tumor vs normal are nearly identical — label embedding dominates visual features.",
        ]
    else:
        clip_score_mean_diff = (
            df["biomedclip_top_score"][(gt == "tumor").values].mean()
            - df["biomedclip_top_score"][(gt == "normal").values].mean()
        )
        clip_score_note = [
            f"> Mean score: tumor={df['biomedclip_top_score'][(gt=='tumor').values].mean():.4f}  "
            f"normal={df['biomedclip_top_score'][(gt=='normal').values].mean():.4f}  "
            f"diff={clip_score_mean_diff:+.4f}",
        ]

    lines += [
        "",
        "### Score percentiles by true class:",
        "",
        clip_pct_df.to_markdown(index=False),
        "",
        *clip_score_note,
        "",
        "---",
        "",
        "## 6. SAM3 segmentation analysis",
        "",
        "### IoU (GradCAM++ ∩ SAM3 mask) percentiles by true class:",
        "",
        iou_df.to_markdown(index=False),
        "",
        "### Bbox coverage percentiles by true class:",
        "",
        bbox_df.to_markdown(index=False),
        "",
        "### False-positive activation on normal scans (bbox coverage bins):",
        "",
        fp_df.to_markdown(index=False),
        "",
        "### Key SAM3 findings:",
        f"- IoU with GradCAM++ is zero for {iou_zero_rate*100:.1f}% of samples (especially normals).",
        (
            f"- Near-zero IoU triggers the confidence penalty → `final_confidence` collapse → {flag_pct:.1f}% of samples flagged for human review."
            if flag_pct > 50
            else f"- {flag_pct:.1f}% of samples flagged for human review (IoU-driven confidence penalty)."
        ),
        "",
        "---",
        "",
        "## 7. Confidence calibration",
        "",
        conf_summary_df.to_markdown(index=False),
        "",
    ]

    for name, cal in cal_bins_dict.items():
        lines += [f"### Calibration bins — {name}", "", cal.to_markdown(index=False), ""]

    lines += [
        "---",
        "",
        "## 8. Human-review flag analysis",
        "",
        f"- **{flag_pct:.1f}%** of samples flagged for human review ({total_flagged}/{n}) — caused by low IoU → confidence penalty → threshold trigger.",
        "",
        review_df.dropna(subset=["accuracy"]).to_markdown(index=False),
        "",
        "---",
        "",
        "## 9. Severity score analysis",
        "",
        sev_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 10. Latency",
        "",
        lat_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## Summary of issues identified",
        "",
        "| Issue | Component | Impact |",
        "|-------|-----------|--------|",
        *(
            [f"| BiomedCLIP label collapse — predicts '{clip_unique[0]}' for all samples | BiomedCLIP | Specificity = 0 |"]
            if clip_collapsed else
            [f"| BiomedCLIP linear probe: acc={get_metric('biomedclip','accuracy')}, spec={get_metric('biomedclip','specificity')} | BiomedCLIP | — |"]
        ),
        f"| SAM3 IoU = 0 for {iou_zero_rate*100:.1f}% of samples → `final_confidence` zeroed out | SAM3 + report_node | {flag_pct:.1f}% human-review flag rate |",
        *(
            ["| MedGemma overconfident on wrong predictions (conf_wrong > conf_correct) | MedGemma | High ECE |"]
            if any(
                (r.get("mean_conf_wrong", 0) or 0) > (r.get("mean_conf_correct", 1) or 1)
                for r in conf_summary_df.to_dict("records")
                if "medgemma" in str(r.get("model", ""))
            ) else []
        ),
        *(
            [f"| Pipeline specificity ({get_metric('pipeline_final','specificity')}) degrades from CNN ({get_metric('cnn','specificity')}) — MedGemma/CLIP FPs on normal scans | Pipeline fusion | Specificity collapse |"]
            if not np.isnan(pf_spec) and not np.isnan(cnn_spec) and pf_spec < cnn_spec - 0.05 else []
        ),
        *(
            [f"| {flag_pct:.1f}% human-review flag rate makes triage signal useless | Pipeline | No filtering signal |"]
            if flag_pct > 90 else []
        ),
        "",
    ]

    report = "\n".join(lines)
    report_path = out_dir / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report written to {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Comprehensive binary tumor_eval analysis")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: outputs/analysis/<jsonl-stem>)")
    args = parser.parse_args()

    df  = load(args.jsonl)
    # Default: outputs/analysis/<stem> so binary and multiclass never clobber each other
    if args.output_dir is None:
        stem = Path(args.jsonl).stem  # e.g. binary_tumor_tumor_eval
        out  = Path("outputs/analysis") / stem
    else:
        out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    section_dataset(df)
    accuracy_df                             = section_model_accuracy(df)
    dist_df, agree_df, field_acc_df        = section_medgemma_fields(df)
    disagree_df                             = section_initial_vs_final(df)
    cnn_class_df, cnn_cal_df, cnn_met_df   = section_cnn(df)
    clip_pct_df, clip_cal_df, clip_met_df  = section_clip(df)
    iou_df, bbox_df, fp_df                 = section_sam3(df)
    conf_summary_df, cal_bins_dict         = section_confidence_summary(df)
    review_df                              = section_human_review(df)
    sev_df                                 = section_severity(df)
    lat_df                                 = section_latency(df)

    # ── save all CSVs ──────────────────────────────────────────────────────────
    saves = {
        "model_accuracy_summary.csv":           accuracy_df,
        "medgemma_field_distributions.csv":     dist_df,
        "medgemma_field_agreement.csv":         agree_df,
        "medgemma_field_accuracy.csv":          field_acc_df,
        "cnn_per_class_confidence.csv":         cnn_class_df,
        "cnn_calibration_bins.csv":             cnn_cal_df,
        "cnn_metrics.csv":                      cnn_met_df,
        "clip_score_distribution.csv":          clip_pct_df,
        "clip_calibration_bins.csv":            clip_cal_df,
        "clip_metrics.csv":                     clip_met_df,
        "sam3_iou_percentiles.csv":             iou_df,
        "sam3_bbox_stats.csv":                  bbox_df,
        "sam3_false_positive.csv":              fp_df,
        "confidence_calibration_summary.csv":   conf_summary_df,
        "initial_vs_final_disagreements.csv":   disagree_df,
        "human_review_analysis.csv":            review_df,
        "severity_score_analysis.csv":          sev_df,
        "latency_stats.csv":                    lat_df,
    }
    for filename, data_df in saves.items():
        if data_df is not None and not data_df.empty:
            data_df.to_csv(out / filename, index=False)

    for name, cal in cal_bins_dict.items():
        cal.to_csv(out / f"calibration_bins_{name}.csv", index=False)

    # ── generate markdown report ───────────────────────────────────────────────
    generate_report(
        jsonl_path=args.jsonl,
        df=df,
        accuracy_df=accuracy_df,
        dist_df=dist_df,
        agree_df=agree_df,
        field_acc_df=field_acc_df,
        disagree_df=disagree_df,
        cnn_class_df=cnn_class_df,
        cnn_cal_df=cnn_cal_df,
        cnn_metrics_df=cnn_met_df,
        clip_pct_df=clip_pct_df,
        clip_cal_df=clip_cal_df,
        clip_metrics_df=clip_met_df,
        iou_df=iou_df,
        bbox_df=bbox_df,
        fp_df=fp_df,
        conf_summary_df=conf_summary_df,
        cal_bins_dict=cal_bins_dict,
        review_df=review_df,
        sev_df=sev_df,
        lat_df=lat_df,
        out_dir=out,
    )

    print(f"\n{'─'*72}")
    print(f"All outputs in {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
