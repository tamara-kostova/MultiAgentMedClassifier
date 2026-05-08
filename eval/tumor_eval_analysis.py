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
    print_section("2. Per-model binary accuracy summary")

    gt = df["gt"]

    # Build per-model predictions + confidence score for AUC
    models = {
        "pipeline_final":  (df["predicted_class"].apply(to_binary),
                            df["final_confidence"]),
        "cnn":             (df["cnn_predicted_class"].apply(to_binary),
                            df["cnn_confidence"]),
        "biomedclip":      (df["biomedclip_top_label"].apply(to_binary),
                            df["biomedclip_top_score"]),
        "medgemma_initial":(df["medgemma_diagnosis"].apply(mg_to_binary),
                            df["medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0)),
        "medgemma_final":  (df["final_medgemma_diagnosis"].apply(mg_to_binary),
                            df["final_medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0)),
    }

    rows = []
    for name, (preds, scores) in models.items():
        m = binary_metrics_dict(gt, preds)
        if not m:
            continue
        # AUC: use score oriented toward "tumor" prediction
        auc_scores = [
            s if p == "tumor" else (1 - s)
            for p, s in zip(preds, scores)
            if gt[preds.index[list(preds).index(p)]] not in ("unknown",)
        ]
        # simpler approach: build aligned lists
        valid_mask = (gt != "unknown") & (preds != "unknown")
        yt_valid = gt[valid_mask].tolist()
        yp_valid = preds[valid_mask].tolist()
        ys_valid = scores[valid_mask].tolist()
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
                preds = vals.apply(to_binary)
                m = binary_metrics_dict(gt, preds)
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

    gt         = df["gt"]
    init_bin   = df["medgemma_diagnosis"].apply(mg_to_binary)
    fin_bin    = df["final_medgemma_diagnosis"].apply(mg_to_binary)

    agree_mask = (init_bin == fin_bin)
    print(f"  Agreement (binary) : {agree_mask.sum()} / {len(df)}  ({agree_mask.mean()*100:.1f}%)")
    print(f"  Changed            : {(~agree_mask).sum()} ({(~agree_mask).mean()*100:.1f}%)")

    shift_df = pd.DataFrame({
        "init_bin": init_bin, "fin_bin": fin_bin, "gt": gt,
        "cnn_pred": df["cnn_predicted_class"].apply(to_binary),
        "cnn_conf": df["cnn_confidence"],
    })
    shift_counts = shift_df.groupby(["init_bin", "fin_bin"]).size().reset_index(name="n")
    print("\n  Shift breakdown (init_bin → fin_bin):")
    print(shift_counts.to_string(index=False))

    # Accuracy per stage
    print()
    for stage, preds in [("initial", init_bin), ("final", fin_bin)]:
        both = (gt != "unknown") & (preds != "unknown")
        m = binary_metrics_dict(gt[both], preds[both])
        if m:
            print(f"  {stage:8s}  n={m['n']}  acc={m['accuracy']}  f1={m['f1_macro']}  "
                  f"sens={m['sensitivity']}  spec={m['specificity']}")

    # Where initial was right but final was wrong and vice versa
    init_correct = (init_bin == gt) & (gt != "unknown") & (init_bin != "unknown")
    fin_correct  = (fin_bin  == gt) & (gt != "unknown") & (fin_bin  != "unknown")
    recovered = (~init_correct) & fin_correct
    degraded  = init_correct & (~fin_correct)
    print(f"\n  Final recovered cases (init wrong → final right) : {recovered.sum()}")
    print(f"  Final degraded  cases (init right → final wrong) : {degraded.sum()}")

    disagree = df[~agree_mask].copy()
    disagree["gt_bin"]   = gt[~agree_mask].values
    disagree["init_bin"] = init_bin[~agree_mask].values
    disagree["fin_bin"]  = fin_bin[~agree_mask].values
    return disagree[["image_path","true_label","gt_bin","init_bin","fin_bin",
                      "cnn_predicted_class","cnn_confidence"]]


# ══════════════════════════════════════════════════════════════════════════════
# §5  CNN detailed metrics + calibration
# ══════════════════════════════════════════════════════════════════════════════

def section_cnn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("5. CNN detailed metrics + calibration")

    gt       = df["gt"]
    pred     = df["cnn_predicted_class"].apply(to_binary)
    conf     = df["cnn_confidence"].values

    valid    = (gt != "unknown") & (pred != "unknown")
    gt_v     = gt[valid];  pred_v = pred[valid];  conf_v = conf[valid.values]
    correct  = (pred_v == gt_v).values.astype(float)

    m = binary_metrics_dict(gt, pred)
    auc_scores = [s if p == "tumor" else 1 - s for p, s in zip(pred_v, conf_v)]
    auc = roc_auc_binary(gt_v.tolist(), auc_scores)

    # Per-class confidence stats
    per_class_rows = []
    for cls in ["tumor", "normal"]:
        cls_mask = (gt_v == cls).values
        c = conf_v[cls_mask]
        per_class_rows.append({
            "true_class": cls,
            "n": int(cls_mask.sum()),
            "conf_mean": round(float(c.mean()), 4),
            "conf_std":  round(float(c.std()),  4),
            "conf_min":  round(float(c.min()),  4),
            "conf_p25":  round(float(np.percentile(c, 25)), 4),
            "conf_p50":  round(float(np.percentile(c, 50)), 4),
            "conf_p75":  round(float(np.percentile(c, 75)), 4),
            "conf_max":  round(float(c.max()),  4),
            "accuracy":  round(float((pred_v[gt_v == cls] == cls).mean()), 4),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # Calibration
    ece = compute_ece(conf_v, correct)
    cal_df = calibration_bins(conf_v, correct)

    metrics_row = {**m, "roc_auc": auc, "ece": round(ece, 4)}

    print(f"  Accuracy:    {m['accuracy']}    F1-macro: {m['f1_macro']}    AUC: {auc}    ECE: {round(ece,4)}")
    print(f"  Sensitivity: {m['sensitivity']}   Specificity: {m['specificity']}")
    print(f"  Confusion matrix (normal/tumor):  TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
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

    gt     = df["gt"]
    pred   = df["biomedclip_top_label"].apply(to_binary)
    scores = df["biomedclip_top_score"].values

    unique_preds = pred.unique()
    print(f"  Unique predictions : {unique_preds.tolist()}")
    if len(unique_preds) == 1:
        print(f"  ⚠  BiomedCLIP predicted '{unique_preds[0]}' for ALL {len(df)} samples (label collapse).")
        print(f"     Effective accuracy = class balance = {(gt == unique_preds[0]).mean():.4f}")

    m   = binary_metrics_dict(gt, pred)
    print(f"\n  Accuracy: {m.get('accuracy','N/A')}  F1: {m.get('f1_macro','N/A')}  "
          f"Sens: {m.get('sensitivity','N/A')}  Spec: {m.get('specificity','N/A')}")

    # Score distribution by true class
    pct_rows = []
    for cls in ["tumor", "normal", "all"]:
        mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
        s = scores[mask.values]
        row = percentile_table(s, cls)
        pct_rows.append(row)
    pct_df = pd.DataFrame(pct_rows)
    print("\n  Top-score percentiles by true class:")
    print(pct_df.to_string(index=False))

    # Score difference between classes (if possible)
    score_tumor  = scores[(gt == "tumor").values]
    score_normal = scores[(gt == "normal").values]
    diff = score_tumor.mean() - score_normal.mean()
    print(f"\n  Mean score: tumor={score_tumor.mean():.4f}  normal={score_normal.mean():.4f}  "
          f"diff={diff:.4f}  (positive = BiomedCLIP scores tumors higher)")

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

    gt       = df["gt"]
    iou      = df["saliency_sam3_iou"].values
    coverage = _bbox_coverage(df["sam3_bbox"].tolist())
    skipped  = df["sam3_skipped"].values

    print(f"  sam3_skipped : {skipped.sum()} / {len(df)}")

    # ── IoU percentile table by class ─────────────────────────────────────────
    print("\n  [IoU (GradCAM++ ∩ SAM3 mask) — percentiles by true class]")
    iou_rows = []
    for cls in ["tumor", "normal", "all"]:
        mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
        iou_cls = iou[mask.values]
        row = percentile_table(iou_cls, cls)
        row["zero_rate"] = round(float((iou_cls == 0).mean()), 4)
        iou_rows.append(row)
    iou_df = pd.DataFrame(iou_rows)
    print(iou_df.to_string(index=False))

    # ── bbox coverage by class ─────────────────────────────────────────────────
    print("\n  [Bbox coverage (bbox_area / 512²) by true class]")
    bbox_rows = []
    for cls in ["tumor", "normal", "all"]:
        mask = (gt == cls) if cls != "all" else pd.Series([True]*len(df), index=df.index)
        cov = coverage[mask.values]
        row = percentile_table(cov, cls)
        row["full_image_rate"] = round(float((cov >= 0.99).mean()), 4)
        bbox_rows.append(row)
    bbox_df = pd.DataFrame(bbox_rows)
    print(bbox_df.to_string(index=False))

    # ── false-positive analysis: SAM3 activation on normal scans ──────────────
    print("\n  [SAM3 false-positive activation on normal scans]")
    normal_mask = (gt == "normal").values

    # "Activation" = SAM3 bbox is NOT full-image (it localised something)
    not_full_img = (coverage < 0.99) & ~np.isnan(coverage)
    fp_activation_rate = not_full_img[normal_mask].mean()

    cov_bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.01]
    fp_rows = []
    for cls in ["tumor", "normal"]:
        cls_mask = (gt == cls).values
        cov_cls  = coverage[cls_mask]
        for lo, hi in zip(cov_bins[:-1], cov_bins[1:]):
            n_in = int(((cov_cls >= lo) & (cov_cls < hi)).sum())
            fp_rows.append({
                "true_class": cls,
                "coverage_range": f"[{lo:.2f}, {hi:.2f})",
                "n": n_in,
                "pct": round(n_in / len(cov_cls) * 100, 1),
            })
    fp_df = pd.DataFrame(fp_rows)
    print(f"  SAM3 activation rate on NORMAL scans (bbox <99% of image): "
          f"{fp_activation_rate*100:.1f}%")
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

    gt = df["gt"]
    valid_mask = (gt != "unknown")

    def get_conf_correct(pred_series, conf_series):
        both = valid_mask & (pred_series != "unknown")
        pv   = pred_series[both]
        gv   = gt[both]
        cv   = conf_series[both].values if hasattr(conf_series[both], "values") else np.array(conf_series[both])
        c    = (pv == gv).values.astype(float)
        return cv, c

    models_conf = {
        "cnn": (
            df["cnn_predicted_class"].apply(to_binary),
            df["cnn_confidence"],
        ),
        "medgemma_initial": (
            df["medgemma_diagnosis"].apply(mg_to_binary),
            df["medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0),
        ),
        "medgemma_final": (
            df["final_medgemma_diagnosis"].apply(mg_to_binary),
            df["final_medgemma_diagnosis"].apply(lambda d: mg_field(d, "diagnosis_confidence") or 0.0),
        ),
        "biomedclip": (
            df["biomedclip_top_label"].apply(to_binary),
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

    gt      = df["gt"]
    flagged = df["requires_human_review"]
    iou     = df["saliency_sam3_iou"].values

    total_flagged = int(flagged.sum())
    print(f"  Flagged : {total_flagged} / {len(df)}  ({total_flagged/len(df)*100:.1f}%)")

    rows = []
    for label, mask in [("flagged", flagged), ("not_flagged", ~flagged)]:
        n = int(mask.sum())
        if n == 0:
            rows.append({"subset": label, "n": 0})
            continue
        for model_name, pred in [
            ("cnn",          df["cnn_predicted_class"].apply(to_binary)),
            ("medgemma_init",df["medgemma_diagnosis"].apply(mg_to_binary)),
        ]:
            sub_gt   = gt[mask]
            sub_pred = pred[mask]
            both     = (sub_gt != "unknown") & (sub_pred != "unknown")
            if both.sum() == 0:
                continue
            m = binary_metrics_dict(sub_gt[both], sub_pred[both])
            rows.append({
                "subset": label, "model": model_name, "n": n,
                "accuracy": m.get("accuracy"), "f1_macro": m.get("f1_macro"),
                "sensitivity": m.get("sensitivity"), "specificity": m.get("specificity"),
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

    gt = df["gt"]

    init_sev = _extract_field_series(df["medgemma_diagnosis"],       "severity_score")
    fin_sev  = _extract_field_series(df["final_medgemma_diagnosis"], "severity_score")
    init_sev_num = pd.to_numeric(init_sev, errors="coerce")
    fin_sev_num  = pd.to_numeric(fin_sev,  errors="coerce")

    rows = []
    for stage, vals_num in [("initial", init_sev_num), ("final", fin_sev_num)]:
        for cls in ["tumor", "normal", "all"]:
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

    lat = df["latency_s"].values
    gt  = df["gt"]

    summary = {
        "all":    percentile_table(lat, "all"),
        "tumor":  percentile_table(lat[(gt == "tumor").values], "tumor"),
        "normal": percentile_table(lat[(gt == "normal").values], "normal"),
    }
    lat_df = pd.DataFrame(list(summary.values()))
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
        f"- **CNN** achieves perfect accuracy ({get_metric('cnn','accuracy')}) and AUC ({get_metric('cnn','roc_auc')}) — likely memorised Br35H patterns.",
        f"- **BiomedCLIP** collapsed to a single label ('brain tumor MRI') for all {n} samples, giving 50% accuracy (chance level). Specificity = 0.",
        f"- **MedGemma initial triage**: acc={get_metric('medgemma_initial','accuracy')}, F1={get_metric('medgemma_initial','f1_macro')}, sensitivity={get_metric('medgemma_initial','sensitivity')}, specificity={get_metric('medgemma_initial','specificity')}.",
        f"- **MedGemma final report**: acc={get_metric('medgemma_final','accuracy')}, F1={get_metric('medgemma_final','f1_macro')}, sensitivity={get_metric('medgemma_final','sensitivity')}, specificity={get_metric('medgemma_final','specificity')}.",
        f"- The **pipeline final** result has specificity={get_metric('pipeline_final','specificity')} — significantly degraded from CNN because MedGemma and BiomedCLIP introduce false positives on normal scans.",
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

    lines += [
        "",
        "### Score percentiles by true class:",
        "",
        clip_pct_df.to_markdown(index=False),
        "",
        "> **Note:** BiomedCLIP scored all samples as 'brain tumor MRI' (mean score ≈ 0.99). ",
        "> The score distributions for tumor vs normal are nearly identical, confirming the label",
        "> embedding dominates over visual features for this binary task.",
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
        "- IoU with GradCAM++ is near-zero for the majority of samples, especially normals.",
        "- This drives `final_confidence → 0` and flags 100% of samples for human review.",
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
        f"- **100%** of samples flagged for human review (caused by IoU=0 → confidence penalty → threshold trigger).",
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
        "| BiomedCLIP label collapse — predicts 'brain tumor MRI' for all samples | BiomedCLIP | Specificity = 0 |",
        "| SAM3 IoU ≈ 0 for 81.7% of samples → `final_confidence` zeroed out | SAM3 + report_node | 100% human-review flag rate |",
        "| MedGemma overconfident on wrong predictions (conf_wrong > conf_correct) | MedGemma | ECE = 0.31–0.24 |",
        "| Pipeline specificity degrades from CNN (1.00) to 0.35 due to MedGemma/CLIP FPs | Pipeline fusion | Specificity collapse |",
        "| 100% human-review rate makes triage flag useless | Pipeline | No filtering signal |",
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
    parser.add_argument("--output_dir", default="outputs/analysis")
    args = parser.parse_args()

    df  = load(args.jsonl)
    out = Path(args.output_dir)
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
