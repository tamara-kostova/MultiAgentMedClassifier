"""
General analysis for any binary or multiclass task eval JSONL.
Supports: ms, stroke, binary_tumor, multiclass_tumor.
Generates CSVs and matplotlib visualizations, focused on initial vs. final MedGemma.

Usage:
    python eval/eval_analysis.py --jsonl outputs/eval/ms_dataset_eval.jsonl
    python eval/eval_analysis.py --jsonl outputs/eval/stroke_dataset_eval.jsonl
    python eval/eval_analysis.py --jsonl outputs/eval/binary_tumor_tumor_eval.jsonl
    python eval/eval_analysis.py --jsonl outputs/eval/multiclass_tumor_tumor_eval.jsonl
    python eval/eval_analysis.py --jsonl ... --abstain drop   # reproduce pre-2026-10 tables

Conventions (shared with paired_system_comparison): JSONLs load through
eval.jsonl_io (last non-error row per image; error-only images scored wrong);
abstentions count as WRONG by default (--abstain wrong|drop; n and n_abstained are
always reported); ECE from eval.metrics (first bin [0, 0.1]); forest votes are
recomputed from canonical ballots (eval.forest_votes); debate "verdict changed"
prefers per-round verdicts over the judge's self-report.

Outputs (outputs/analysis/<stem>/)
────────────────────────────────────
CSVs:
  model_accuracy_summary.csv
  confusion_matrix_<model>.csv
  medgemma_shift_analysis.csv
  confidence_calibration_summary.csv
  calibration_bins_<model>.csv
  latency_stats.csv
  forest_voting_quality.csv       (forest runs only)
  forest_agent_accuracy.csv       (forest runs only)
  debate_round_analysis.csv       (debate runs only)
  load_summary.csv                (rows / duplicates / unparsable / error-only / parse failures)
Plots:
  model_accuracy.png
  confusion_matrices.png
  medgemma_initial_vs_final.png
  calibration_plot.png
  confidence_by_correctness.png
  latency.png
  forest_agent_accuracy.png       (forest runs only)
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

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # run as `python eval/<script>.py`: make `eval` importable
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

# ── label normalization ────────────────────────────────────────────────────────
# canonical_label lives in eval/labels.py (single source of truth); re-exported
# here because paper scripts import it as eval.eval_analysis.canonical_label.
from eval.labels import (  # noqa: E402,F401
    MS_TOKENS as _MS_TOKENS,
    STROKE_TOKENS as _STROKE_TOKENS,
    TUMOR_SUBTYPES as _TUMOR_SUBTYPES,
    canonical_label,
)
from eval.forest_votes import ballots as forest_ballots  # noqa: E402
from eval.jsonl_io import ERROR_FLAG, load_df  # noqa: E402
from eval.metrics import calibration_bins, check_ece_bound, compute_ece, parse_bool  # noqa: E402,F401

# Sentinel prediction for an image whose every logged attempt errored. Always
# scored wrong (under both --abstain modes) and counted separately.
ERROR_PRED = "error"


def is_multiclass(task: str) -> bool:
    return task == "multiclass_tumor"


def pos_class(task: str) -> str:
    return {"ms": "ms", "stroke": "stroke"}.get(task, "tumor")


# Binary scoring convention. Both are defensible but they measure different things,
# so the choice must be stated explicitly alongside any reported number.
#   "strict"   — the task's own question ("is this stroke?"). A prediction naming a
#                *different* pathology is an assertion that the target pathology is
#                absent, so it scores as a negative. This is what "MS detection" and
#                "stroke detection" mean, and it is the primary convention.
#   "abnormal" — abnormality screening ("is this scan not-normal?"). Any pathology
#                label counts as positive. Inflates sensitivity and deflates
#                specificity whenever the model names an off-task pathology.
SCORING_MODES = ("strict", "abnormal")
SCORING = "strict"


def set_scoring(mode: str) -> None:
    if mode not in SCORING_MODES:
        raise ValueError(f"scoring must be one of {SCORING_MODES}, got {mode!r}")
    global SCORING
    SCORING = mode


# Abstention convention. An abstention is an empty / unparseable / "null"
# prediction (prep -> "unknown").
#   "wrong" — (default, and what paired_system_comparison and the paper's
#             tab:comparison use) the row stays in n and counts as incorrect:
#             a miss for its true class (FN for a positive, FP-side miss for a normal).
#   "drop"  — the row is removed from the denominator (the old eval_analysis
#             behaviour; kept for reproducing earlier tables).
# Image paths whose every attempt errored are always scored wrong, in both modes.
ABSTAIN_MODES = ("wrong", "drop")
ABSTAIN = "wrong"


def set_abstain(mode: str) -> None:
    if mode not in ABSTAIN_MODES:
        raise ValueError(f"abstain must be one of {ABSTAIN_MODES}, got {mode!r}")
    global ABSTAIN
    ABSTAIN = mode


def _keep(true: str, pred: str) -> bool:
    """Row enters the denominator? Unknown ground truth never does."""
    if true in ("", "unknown"):
        return False
    if pred in ("", "unknown"):
        return ABSTAIN == "wrong"
    return True


def prep(label: str, task: str, scoring: str | None = None) -> str:
    """Normalize for comparison. Multiclass: keep subtype. Binary: collapse to (pos|normal|unknown).

    "unknown" marks an abstention (empty label / parse failure / "null" sentinel).
    How it is scored is decided by ABSTAIN (see set_abstain), not here.
    """
    if label == ERROR_PRED:
        return ERROR_PRED
    if is_multiclass(task):
        cl = canonical_label(label, task)
        return cl if cl else "unknown"
    cl = canonical_label(label, task)
    if not cl:
        return "unknown"
    if cl == "normal":
        return "normal"
    pos = pos_class(task)
    if (scoring or SCORING) == "abnormal":
        return pos
    # strict: only the task's own pathology counts as positive; naming another
    # pathology asserts the target pathology is absent.
    return pos if cl == pos else "normal"


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
    """MedGemma confidence; only a missing/non-numeric value becomes 0.5 (0.0 stays 0.0)."""
    if not isinstance(diag, dict):
        return 0.5
    v = diag.get("diagnosis_confidence")
    try:
        return float(v) if v is not None else 0.5
    except (TypeError, ValueError):
        return 0.5


# ── loader ─────────────────────────────────────────────────────────────────────

# Optional columns written by newer tumor_eval.py versions. Older JSONLs lack them;
# every section must tolerate their absence.
_OPTIONAL_COLS = ("forest_votes", "dissent_rate", "vote_fraction", "debate_rounds_completed",
                  "debate_round_changed", "debate_round_verdicts", "run_config",
                  "confidence_penalty_factor", "biomedclip_top_label", "biomedclip_top_score",
                  "cnn_predicted_class", "cnn_confidence", "final_confidence",
                  "latency_s", "routing_path", "predicted_class", "true_label_name", "true_label")


def load(path: str) -> pd.DataFrame:
    """Load an eval JSONL through eval.jsonl_io (dedupe: last non-error row per image)."""
    df, _stats = load_df(path)
    for col in _OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None

    for col in ("medgemma_diagnosis", "final_medgemma_diagnosis"):
        if col not in df.columns:
            df[col] = None
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


def parse_failure_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count True values of every *_parse_failed column (newer JSONLs only)."""
    out = {}
    for col in sorted(c for c in df.columns if c.endswith("_parse_failed")):
        out[col] = int(sum(parse_bool(v) is True for v in df[col]))
    return out


# ── metric helpers ─────────────────────────────────────────────────────────────

def _abstain_counts(pairs) -> tuple[int, int]:
    n_abst = sum(1 for _, p in pairs if p in ("", "unknown"))
    n_err = sum(1 for _, p in pairs if p == ERROR_PRED)
    return n_abst, n_err


def binary_metrics_row(y_true: list, y_pred: list, pos: str, name: str) -> dict | None:
    """Binary metrics over rows with known ground truth.

    Abstentions follow ABSTAIN: under "wrong" they stay in n as a miss for their
    true class (lower sensitivity for a positive, lower specificity for a normal),
    exactly like paired_system_comparison.sens_spec. Error-only rows are always misses.
    tn/fp/fn/tp count actual normal/pos predictions only.
    """
    valid = [(t, p) for t, p in zip(y_true, y_pred)
             if t in (pos, "normal") and _keep(t, p)]
    if not valid:
        return None
    yt = [t for t, _ in valid]
    yp = [p for _, p in valid]
    tp_ = sum(1 for t, p in valid if t == pos and p == pos)
    fn = sum(1 for t, p in valid if t == pos and p == "normal")
    tn = sum(1 for t, p in valid if t == "normal" and p == "normal")
    fp = sum(1 for t, p in valid if t == "normal" and p == pos)
    n_pos = sum(1 for t in yt if t == pos)
    n_neg = len(yt) - n_pos
    # abstentions are always reported, also under --abstain drop (where they are not in n)
    n_abst, n_err = _abstain_counts(
        [(t, p) for t, p in zip(y_true, y_pred) if t in (pos, "normal")])
    return {
        "model":       name,
        "n":           len(yt),
        "n_abstained": n_abst,
        "n_error":     n_err,
        "accuracy":    round(sum(t == p for t, p in valid) / len(valid), 4),
        "f1_macro":    round(f1_score(yt, yp, labels=["normal", pos], average="macro",
                                      zero_division=0), 4),
        "sensitivity": round(tp_ / n_pos if n_pos else float("nan"), 4),
        "specificity": round(tn / n_neg if n_neg else float("nan"), 4),
        "precision":   round(tp_ / (tp_ + fp) if (tp_ + fp) else 0.0, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp_),
    }


def multiclass_metrics_row(y_true: list, y_pred: list, name: str) -> dict | None:
    valid = [(t, p) for t, p in zip(y_true, y_pred) if _keep(t, p)]
    if not valid:
        return None
    yt = [t for t, _ in valid]
    yp = [p for _, p in valid]
    real = {p for p in yp if p not in ("", "unknown", ERROR_PRED)}
    classes = sorted(set(yt) | real)
    per_class = {}
    for cls in classes:
        tp_ = sum(1 for t, p in valid if t == cls and p == cls)
        fn_ = sum(1 for t, p in valid if t == cls and p != cls)
        fp_ = sum(1 for t, p in valid if t != cls and p == cls)
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else float("nan")
        pre = tp_ / (tp_ + fp_) if (tp_ + fp_) else float("nan")
        per_class[f"recall_{cls}"]    = round(rec, 4)
        per_class[f"precision_{cls}"] = round(pre, 4)
    n_abst, n_err = _abstain_counts(
        [(t, p) for t, p in zip(y_true, y_pred) if t not in ("", "unknown")])
    return {
        "model":       name,
        "n":           len(yt),
        "n_abstained": n_abst,
        "n_error":     n_err,
        "accuracy":    round(sum(t == p for t, p in valid) / len(valid), 4),
        "f1_macro":    round(f1_score(yt, yp, labels=classes, average="macro",    zero_division=0), 4),
        "f1_weighted": round(f1_score(yt, yp, labels=classes, average="weighted", zero_division=0), 4),
        **per_class,
    }


def calibration_bins_df(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    return pd.DataFrame(calibration_bins(confs, correct, n_bins))


# ── prediction builder ─────────────────────────────────────────────────────────

def build_model_preds(
    df: pd.DataFrame, task: str, scoring: str | None = None
) -> dict[str, tuple[list, list]]:
    """Return {model_name: ([prep'd predictions], [confidences])}.

    Binary tasks: predictions are binarized to (pos | normal | unknown).
    Multiclass:   predictions are canonical subtype labels.
    """
    err = (df[ERROR_FLAG].fillna(False).astype(bool).tolist()
           if ERROR_FLAG in df.columns else [False] * len(df))

    def _mask_err(preds: list) -> list:
        return [ERROR_PRED if e else p for p, e in zip(preds, err)]

    def _prep_col(col: str) -> list:
        return [prep(str(v or ""), task, scoring) for v in df[col].fillna("")]

    def _one_conf(v) -> float:
        # Only a missing/non-numeric value becomes 0.5; a logged 0.0 stays 0.0.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return 0.5 if np.isnan(f) else f

    def _conf_col(col: str) -> list:
        return [_one_conf(v) for v in df[col].tolist()]

    raw = {
        "cnn": (
            _prep_col("cnn_predicted_class_canonical"),
            _conf_col("cnn_confidence"),
        ),
        "biomedclip": (
            [prep(canonical_label(str(v or ""), task), task, scoring)
             for v in df["biomedclip_top_label"].fillna("")],
            _conf_col("biomedclip_top_score"),
        ),
        "medgemma_initial": (
            [prep(mg_diag_pred(d, task), task, scoring) for d in df["medgemma_diagnosis"]],
            [mg_conf(d) for d in df["medgemma_diagnosis"]],
        ),
        "medgemma_final": (
            [prep(mg_diag_pred(d, task), task, scoring) for d in df["final_medgemma_diagnosis"]],
            [mg_conf(d) for d in df["final_medgemma_diagnosis"]],
        ),
        "pipeline_final": (
            _prep_col("predicted_class_canonical"),
            _conf_col("final_confidence"),
        ),
    }
    return {name: (_mask_err(p), c) for name, (p, c) in raw.items()}


# ── sections ───────────────────────────────────────────────────────────────────

def section_model_accuracy(df: pd.DataFrame, task: str) -> pd.DataFrame:
    gt    = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    rows  = []
    multi = is_multiclass(task)
    pos   = pos_class(task)

    # Same predictions scored under the alternate binary convention, so a reader can
    # see how much of any number is the scoring choice rather than model behaviour.
    alt = next(m for m in SCORING_MODES if m != SCORING) if not multi else None
    alt_preds = (
        {n: p for n, (p, _) in build_model_preds(df, task, scoring=alt).items()}
        if alt else {}
    )
    alt_gt = [prep(str(v or ""), task, scoring=alt) for v in
              df["true_label_canonical"].fillna("")] if alt else []

    for name, (preds, _) in build_model_preds(df, task).items():
        row = (multiclass_metrics_row(gt, preds, name)
               if multi else binary_metrics_row(gt, preds, pos, name))
        if not row:
            continue
        row["scoring"] = "multiclass" if multi else SCORING
        row["abstain"] = ABSTAIN
        if alt:
            alt_row = binary_metrics_row(alt_gt, alt_preds[name], pos, name)
            if alt_row:
                row[f"accuracy_{alt}"]    = alt_row["accuracy"]
                row[f"sensitivity_{alt}"] = alt_row["sensitivity"]
                row[f"specificity_{alt}"] = alt_row["specificity"]
                row[f"n_{alt}"]           = alt_row["n"]
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
        valid = [(t, p if p not in ("",) else "unknown") for t, p in zip(gt, preds)
                 if t in valid_labels and _keep(t, p)]
        if not valid:
            continue
        yt, yp = zip(*valid)
        # Rows: ground-truth classes. Columns: the same classes in the same order
        # (so the diagonal is the hits), then any off-label predictions and, under
        # --abstain wrong, "unknown"/"error" columns.
        row_labels = sorted(set(yt))
        extra = sorted(set(yp) - set(row_labels))
        col_labels = row_labels + extra
        cm = pd.crosstab(pd.Series(yt, name="t"), pd.Series(yp, name="p"))
        cm = cm.reindex(index=row_labels, columns=col_labels, fill_value=0)
        result[name] = pd.DataFrame(
            cm.values,
            index=[f"true_{l}" for l in row_labels],
            columns=[f"pred_{l}" for l in col_labels],
        )
    return result


def section_medgemma_shift(df: pd.DataFrame, task: str) -> pd.DataFrame:
    gt   = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    init = [prep(mg_diag_pred(d, task), task) for d in df["medgemma_diagnosis"]]
    fin  = [prep(mg_diag_pred(d, task), task) for d in df["final_medgemma_diagnosis"]]
    rows = []
    for g, i, f in zip(gt, init, fin):
        if not _keep(g, i) or not _keep(g, f):
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


_CONF_SOURCE = {"cnn": "cnn_confidence", "biomedclip": "biomedclip_top_score",
                "pipeline_final": "final_confidence",
                "medgemma_initial": "medgemma_diagnosis", "medgemma_final": "final_medgemma_diagnosis"}


def _n_conf_substituted(df: pd.DataFrame, name: str, mask: list[bool]) -> int:
    """Rows (among mask) whose confidence was missing and replaced by 0.5.

    E.g. debate rows whose judge failed to parse log final_confidence=None.
    """
    col = _CONF_SOURCE.get(name)
    if col is None or col not in df.columns:
        return 0
    n = 0
    for v, m in zip(df[col].tolist(), mask):
        if not m:
            continue
        if isinstance(v, dict):
            v = v.get("diagnosis_confidence")
        try:
            missing = v is None or np.isnan(float(v))
        except (TypeError, ValueError):
            missing = True
        n += int(missing)
    return n


def section_calibration(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    gt       = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    valid_gt = set(gt) - {"", "unknown"}
    summary_rows, bins_dict = [], {}

    for name, (preds, confs) in build_model_preds(df, task).items():
        valid = [(g, p, c) for g, p, c in zip(gt, preds, confs)
                 if g in valid_gt and _keep(g, p)]
        if not valid:
            continue
        yt, yp, yc = zip(*valid)
        correct   = np.array([int(t == p) for t, p in zip(yt, yp)], dtype=float)
        confs_arr = np.clip(np.array(yc, dtype=float), 0, 1)
        ece = check_ece_bound(confs_arr, correct)
        bins_dict[name] = calibration_bins_df(confs_arr, correct)
        summary_rows.append({
            "model":          name,
            "n":              len(yt),
            "n_abstained":    sum(1 for g, p in zip(gt, preds)
                                  if g in valid_gt and p in ("", "unknown")),
            "abstain":        ABSTAIN,
            "n_conf_substituted": _n_conf_substituted(
                df, name, [g in valid_gt and _keep(g, p) for g, p in zip(gt, preds)]),
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


def _row_correct(df: pd.DataFrame, task: str, col: str = "predicted_class_canonical") -> tuple[np.ndarray, np.ndarray]:
    """(correct, keep) arrays for a prediction column under the current ABSTAIN mode."""
    gt = [prep(str(v or ""), task) for v in df["true_label_canonical"].fillna("")]
    pr = [prep(str(v or ""), task) for v in df[col].fillna("")]
    if ERROR_FLAG in df.columns:
        pr = [ERROR_PRED if e else p for p, e in zip(pr, df[ERROR_FLAG].fillna(False))]
    keep = np.array([_keep(t, p) for t, p in zip(gt, pr)], dtype=bool)
    correct = np.array([int(t == p) for t, p in zip(gt, pr)], dtype=float)
    return correct, keep


def _has_votes(v) -> bool:
    return isinstance(v, list) and len(v) > 0


def section_forest_voting(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Agent Forest voting quality — dissent rate vs. accuracy.

    Empty unless the JSONL is a forest run. When per-agent ballots (`forest_votes`)
    are logged, dissent and vote_fraction are RECOMPUTED from canonical ballots
    (eval.forest_votes): the stored values of older multiclass runs were computed
    over the raw diagnosis_name and do not match the ballots. Accuracy columns are
    the final pipeline prediction; triage_accuracy_* score the recomputed vote.
    """
    from eval.forest_votes import forest_vote

    has_votes = "forest_votes" in df.columns and df["forest_votes"].apply(_has_votes).any()
    has_stored = "dissent_rate" in df.columns and df["dissent_rate"].notna().any()
    if not has_votes and not has_stored:
        return pd.DataFrame()

    if has_votes:
        sub = df[df["forest_votes"].apply(_has_votes)].copy()
        fv = [forest_vote(v, task) for v in sub["forest_votes"]]
        sub["_vote_label"] = [f["label"] for f in fv]
        dissent = np.array([1.0 - f["vote_fraction"] for f in fv], dtype=float)
        source = "recomputed_from_ballots"
        stored = pd.to_numeric(sub["vote_fraction"], errors="coerce").values
        n_mismatch = int(np.sum(~np.isnan(stored) & (np.abs(stored - (1 - dissent)) > 1e-9)))
    else:
        sub = df[df["dissent_rate"].notna()].copy()
        dissent = pd.to_numeric(sub["dissent_rate"], errors="coerce").astype(float).values
        source = "stored"
        n_mismatch = float("nan")

    correct, keep = _row_correct(sub, task)
    correct, d = correct[keep], dissent[keep]
    unan, split = d == 0.0, d > 0.0

    def _acc(c, mask, min_n=1):
        return round(float(c[mask].mean()), 4) if mask.sum() >= min_n else float("nan")

    row = {
        "n": int(keep.sum()),
        "dissent_source": source,
        "n_stored_vote_fraction_mismatch": n_mismatch,
        "mean_dissent_rate": round(float(d.mean()), 4) if len(d) else float("nan"),
        "unanimous_pct": round(float(unan.mean()) * 100, 1) if len(d) else float("nan"),
        "accuracy_unanimous": _acc(correct, unan),
        "accuracy_split": _acc(correct, split, min_n=5),
        "accuracy_overall": round(float(correct.mean()), 4) if len(correct) else float("nan"),
    }
    if has_votes:
        tc, tk = _row_correct(sub, task, "_vote_label")
        tc, td = tc[tk], dissent[tk]
        row.update({
            "triage_n": int(tk.sum()),
            "triage_accuracy_unanimous": _acc(tc, td == 0.0),
            "triage_accuracy_split": _acc(tc, td > 0.0, min_n=5),
            "triage_accuracy_overall": round(float(tc.mean()), 4) if len(tc) else float("nan"),
        })
    return pd.DataFrame([row])


def section_forest_agent_accuracy(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Agent Forest — per-agent accuracy breakdown.

    Empty unless the JSONL carries `forest_votes`. Agents are keyed by agent_idx
    (or list position), never by role, since homogeneous forests repeat roles; the
    row is named by its role when roles are unique, else "<role>#<agent>". Ballots
    use eval.forest_votes.ballot_label (multiclass: diagnosis_detailed, falling
    back to diagnosis_name) and are scored like the rest of model_accuracy.
    """
    if "forest_votes" not in df.columns:
        return pd.DataFrame()
    sub = df[df["forest_votes"].apply(_has_votes)]
    if sub.empty:
        return pd.DataFrame()

    multi = is_multiclass(task)
    pos = pos_class(task)

    gt_list = [prep(str(v or ""), task) for v in sub["true_label_canonical"].fillna("")]
    ballot_list = [{b["agent"]: b for b in forest_ballots(v, task)} for v in sub["forest_votes"]]
    agents = sorted({a for bl in ballot_list for a in bl})
    role_of = {}
    for bl in ballot_list:
        for a, b in bl.items():
            role_of.setdefault(a, b["role"])
    unique_roles = len(set(role_of.values())) == len(role_of)

    rows = []
    for agent in agents:
        name = role_of[agent] if unique_roles else f"{role_of[agent]}#{agent}"
        preds, confs = [], []
        for bl in ballot_list:
            b = bl.get(agent)
            preds.append(prep(b["label"], task) if b else "unknown")
            confs.append(b["conf"] if b else None)
        row = (multiclass_metrics_row(gt_list, preds, name) if multi
               else binary_metrics_row(gt_list, preds, pos, name))
        if row:
            valid_confs = [c for c in confs if c is not None]
            row["agent"] = agent
            row["mean_conf"] = (
                round(sum(valid_confs) / len(valid_confs), 4) if valid_confs else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _verdict_label(verdict: object, task: str) -> str:
    """Canonical label of one judge verdict (multiclass prefers winner_detailed, as debate_node does)."""
    if not isinstance(verdict, dict):
        return ""
    keys = ("winner_detailed", "winner") if is_multiclass(task) else ("winner", "winner_detailed")
    for k in keys:
        lbl = canonical_label(verdict.get(k) or "", task)
        if lbl:
            return lbl
    return ""


def debate_verdict_changed(row: dict | pd.Series, task: str) -> tuple[bool | None, str]:
    """(changed, source) for one debate record.

    Preferred: `debate_round_verdicts` (list of per-round verdict dicts) — changed if
    the canonical label of any round r differs from round r-1. Fallback for old
    files: the judge's self-reported `debate_round_changed` flag, parsed as a real
    boolean (the string "false" is False).
    """
    verdicts = row.get("debate_round_verdicts") if hasattr(row, "get") else None
    if isinstance(verdicts, list) and len(verdicts) >= 1:
        labels = [_verdict_label(v, task) for v in verdicts]
        changed = any(labels[i] != labels[i - 1] for i in range(1, len(labels)))
        return changed, "per-round verdicts"
    if not hasattr(row, "get"):
        return None, "judge-reported (legacy)"
    flag = parse_bool(row.get("debate_judge_claimed_round_changed"))
    if flag is None:
        flag = parse_bool(row.get("debate_round_changed"))
    return flag, "judge-reported (legacy)"


def section_debate_rounds(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Multi-Agent Debate — verdict stability vs. accuracy/ECE.

    Empty unless the JSONL carries a `debate_rounds_completed` column (i.e. it was
    produced by `--pipeline_mode debate`). "Changed" comes from per-round verdicts
    when logged; older files fall back to the judge's own `round_changed` flag,
    labelled "judge-reported (legacy)" in the `changed_source` column.
    """
    if "debate_rounds_completed" not in df.columns or df["debate_rounds_completed"].isna().all():
        return pd.DataFrame()
    sub = df[df["debate_rounds_completed"].notna()].copy()
    correct, keep = _row_correct(sub, task)
    raw_conf = pd.to_numeric(sub["final_confidence"], errors="coerce").astype(float)
    n_conf_sub = int(raw_conf[keep].isna().sum())  # e.g. judge parse failures log None
    confs = np.clip(raw_conf.fillna(0.5).values, 0, 1)
    ch = [debate_verdict_changed(r, task) for _, r in sub.iterrows()]
    changed = np.array([bool(c) for c, _ in ch], dtype=bool)
    sources = Counter(src for _, src in ch)
    correct, confs, changed = correct[keep], confs[keep], changed[keep]

    def _acc(mask):
        return round(float(correct[mask].mean()), 4) if mask.sum() >= 5 else float("nan")

    def _ece(mask):
        return round(compute_ece(confs[mask], correct[mask]), 4) if mask.sum() >= 5 else float("nan")

    pf = parse_failure_counts(sub)
    return pd.DataFrame([{
        "n": int(keep.sum()),
        "changed_source": " + ".join(f"{k} (n={v})" for k, v in sources.most_common()),
        "pct_verdict_changed": round(float(changed.mean()) * 100, 1) if len(changed) else float("nan"),
        "accuracy_changed": _acc(changed),
        "accuracy_unchanged": _acc(~changed),
        "ece_changed": _ece(changed),
        "ece_unchanged": _ece(~changed),
        "ece_overall": round(check_ece_bound(confs, correct), 4) if len(confs) else float("nan"),
        "n_conf_substituted": n_conf_sub,
        **{f"n_{k}": v for k, v in pf.items()},
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


def plot_forest_agent_accuracy(
    forest_agent_df: pd.DataFrame, accuracy_df: pd.DataFrame, task: str, out: Path
) -> None:
    """Per-role forest agent metrics, grouped bars, with the majority-vote
    (pipeline_final) accuracy overlaid as a reference line."""
    if forest_agent_df.empty:
        return
    multi   = is_multiclass(task)
    metrics = (["accuracy", "f1_macro", "f1_weighted"]
               if multi else ["accuracy", "sensitivity", "specificity", "f1_macro"])
    roles   = forest_agent_df["model"].tolist()
    x       = np.arange(len(roles))
    width   = 0.8 / len(metrics)
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(9, len(roles) * 2.2), 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric not in forest_agent_df.columns:
            continue
        vals = forest_agent_df[metric].astype(float).tolist()
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.82)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=30)

    offset = width * (len(metrics) - 1) / 2
    ax.set_xticks(x + offset)
    ax.set_xticklabels([r.replace("_", "\n") for r in roles], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title(f"Agent Forest — per-role vs. majority-vote accuracy ({task} task)")

    final_row = accuracy_df[accuracy_df["model"] == "pipeline_final"]
    if not final_row.empty and not np.isnan(final_row["accuracy"].iloc[0]):
        final_acc = float(final_row["accuracy"].iloc[0])
        ax.axhline(final_acc, color="#333333", linestyle="--", linewidth=1.4, alpha=0.85,
                   label=f"majority vote (pipeline_final) = {final_acc:.3f}")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out / "forest_agent_accuracy.png")


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
    parser.add_argument("--scoring", default="strict", choices=list(SCORING_MODES),
                        help=(
                            "Binary scoring convention. 'strict' (default): only the "
                            "task's own pathology counts as positive; naming another "
                            "pathology scores as negative. 'abnormal': any pathology "
                            "label counts as positive (abnormality screening). "
                            "Both are written to model_accuracy_summary.csv."
                        ))
    parser.add_argument("--abstain", default="wrong", choices=list(ABSTAIN_MODES),
                        help=(
                            "Abstention convention. 'wrong' (default, paper convention, "
                            "same as paired_system_comparison): an empty/unparseable "
                            "prediction stays in n and is scored incorrect. 'drop': "
                            "remove it from the denominator (older eval_analysis tables). "
                            "n and n_abstained are always reported. Error-only images are "
                            "scored wrong in both modes."
                        ))
    args = parser.parse_args()
    set_scoring(args.scoring)
    set_abstain(args.abstain)

    df   = load(args.jsonl)
    stats = df.attrs.get("load_stats")
    task = str(df["task"].iloc[0]) if "task" in df.columns and len(df) else "unknown"
    pos  = pos_class(task)
    stem = Path(args.jsonl).stem
    out  = Path(args.output_dir) if args.output_dir else Path("outputs/analysis") / stem
    out.mkdir(parents=True, exist_ok=True)

    multi = is_multiclass(task)
    print(f"\nTask: {task}  |  {'multiclass' if multi else f'positive class: {pos}'}  |  n={len(df)}"
          f"{'' if multi else f'  |  scoring: {SCORING}'}")
    print(f"Abstentions: {ABSTAIN}  |  {stats.summary() if stats else ''}")
    pf = parse_failure_counts(df)
    if pf:
        print("Parse failures: " + ", ".join(f"{k}={v}" for k, v in pf.items()))
    if "confidence_penalty_factor" in df.columns and df["confidence_penalty_factor"].notna().any():
        cpf = pd.to_numeric(df["confidence_penalty_factor"], errors="coerce")
        print(f"Confidence penalty applied (factor < 1): {int((cpf < 1).sum())}/{int(cpf.notna().sum())} rows")
    if "run_config" in df.columns:
        rc = [r for r in df["run_config"] if isinstance(r, dict)]
        if rc:
            distinct = {json.dumps(r, sort_keys=True, default=str) for r in rc}
            print(f"run_config: {len(distinct)} distinct value(s) across {len(rc)} rows"
                  + (f": {next(iter(distinct))[:300]}" if len(distinct) == 1 else
                     "  [WARNING: rows were produced under different configs]"))
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
    if shift_df.empty:
        # debate mode replaces the report node, so final_medgemma_diagnosis is
        # always null and there is nothing to compare initial vs. final on.
        print("  (empty — no final_medgemma_diagnosis, likely a debate/forest run)")
    elif multi:
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

    forest_agent_df = section_forest_agent_accuracy(df, task)
    if not forest_agent_df.empty:
        print("\n══ Agent Forest — per-role accuracy ══")
        print(forest_agent_df.to_string(index=False))

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
    pd.DataFrame([{
        "jsonl": args.jsonl, "task": task, "scoring": SCORING if not multi else "multiclass",
        "abstain": ABSTAIN,
        **({k: getattr(stats, k) for k in ("n_rows", "n_unique", "n_duplicate_rows",
                                           "n_unparsable", "n_error_only")} if stats else {}),
        **parse_failure_counts(df),
    }]).to_csv(out / "load_summary.csv", index=False)
    if not forest_df.empty:
        forest_df.to_csv(out / "forest_voting_quality.csv", index=False)
    if not forest_agent_df.empty:
        forest_agent_df.to_csv(out / "forest_agent_accuracy.csv", index=False)
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
    plot_forest_agent_accuracy(forest_agent_df, accuracy_df, task, out)

    print(f"\nAll outputs saved to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
