"""
Base vs. Forest vs. Debate — paired-subset system comparison for Chapter 8 / Chapter 9
of the thesis (tab:debate_main, tab:debate_rounds, tab:synthesis, and the McNemar tests
of ssec:stats).

Debate and Forest were both run on the identical 500-image subset per task (verified:
same image_paths, same first image, same CNN confusion matrix). Base was run on a
larger pool that fully contains that subset. This script:

  1. Filters each task's Base JSONL down to the exact image_paths in its Forest/Debate
     JSONL, so all three systems are compared on identical images.
  2. Scores triage and final-stage accuracy with the abstain-as-wrong convention used
     for tab:comparison (an empty/unparseable prediction counts as incorrect, not
     dropped) — see memory `project-forest-triage-vote-vs-seed` and
     `project-scoring-conventions`.
  3. For Forest, recomputes the triage vote from `forest_votes` (majority over the
     task-appropriate field, ties broken by fixed role order), not from
     `medgemma_diagnosis`, which is a seeded object that can disagree with the ballot.
  4. Computes accuracy with Wilson 95% CIs, sensitivity/specificity (binary-style
     tasks), and ECE for triage and final stage, all three systems, all four tasks.
  5. Runs McNemar's exact test for Debate-vs-Base and Debate-vs-Forest at both stages,
     Holm-Bonferroni corrected across the 16-test family (4 tasks x 2 stages x 2
     comparisons), matching the methodology already used for tab:comparison
     (ssec:stats).

Usage:
    python eval/paired_system_comparison.py
Outputs (outputs/analysis/debate_vs_base_forest/):
    paired_accuracy.csv     — one row per (task, system, stage): n, accuracy + Wilson CI,
                              sensitivity, specificity, ece, mean_conf
    mcnemar_tests.csv       — one row per (task, stage, comparison): n, discordant pairs,
                              statistic, p_raw, p_holm, significant
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from eval.eval_analysis import (
    canonical_label,
    compute_ece,
    is_multiclass,
    pos_class,
    prep,
)

TASKS = {
    "binary_tumor": {
        "base":   "outputs/eval/binary_tumor_tumor_eval.jsonl",
        "forest": "outputs/eval/binary_forest_n4.jsonl",
        "debate": "outputs/eval/binary_debate_r2.jsonl",
    },
    "multiclass_tumor": {
        "base":   "outputs/eval/multiclass_tumor_tumor_eval.jsonl",
        "forest": "outputs/eval/multiclass_forest_n4.jsonl",
        "debate": "outputs/eval/multiclass_debate_r2.jsonl",
    },
    "ms": {
        "base":   "outputs/eval/ms_dataset_eval.jsonl",
        "forest": "outputs/eval/ms_forest_n4.jsonl",
        "debate": "outputs/eval/ms_debate_r2.jsonl",
    },
    "stroke": {
        "base":   "outputs/eval/stroke_dataset_eval.jsonl",
        "forest": "outputs/eval/stroke_forest_n4.jsonl",
        "debate": "outputs/eval/stroke_debate_r2.jsonl",
    },
}

OUT = Path("outputs/analysis/debate_vs_base_forest")

ROLE_ORDER = ["radiologist", "conservative", "emergency", "differential"]


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _diag_field(task: str) -> str:
    return "diagnosis_detailed" if is_multiclass(task) else "diagnosis_name"


def _score(pred_canonical: str, true_canonical: str, task: str) -> int:
    """1 if correct, 0 otherwise (empty/abstained predictions score 0, never dropped)."""
    p = prep(pred_canonical, task) if pred_canonical else "unknown"
    t = prep(true_canonical, task)
    if p == "unknown":
        return 0
    return int(p == t)


def _forest_vote_label(votes: list[dict], task: str) -> str:
    field = _diag_field(task)
    by_role = {v.get("role"): canonical_label(v.get(field) or "", task) for v in votes}
    labels = [by_role.get(r, "") for r in ROLE_ORDER if by_role.get(r)]
    if not labels:
        return ""
    counts = Counter(labels)
    best_n = max(counts.values())
    winners = {lbl for lbl, n in counts.items() if n == best_n}
    for r in ROLE_ORDER:
        lbl = by_role.get(r, "")
        if lbl in winners:
            return lbl
    return ""


def _true_canonical(row: dict, task: str) -> str:
    v = row.get("true_label_canonical") or row.get("true_label_name") or row.get("true_label") or ""
    return canonical_label(str(v), task)


def build_task_frame(task: str, paths: dict) -> pd.DataFrame:
    base = _load_jsonl(paths["base"])
    forest = _load_jsonl(paths["forest"])
    debate = _load_jsonl(paths["debate"])

    forest_paths = {r["image_path"] for r in forest}
    debate_paths = {r["image_path"] for r in debate}
    assert forest_paths == debate_paths, (
        f"{task}: Forest and Debate were not run on the same image set "
        f"({len(forest_paths)} vs {len(debate_paths)}, "
        f"symdiff={len(forest_paths ^ debate_paths)})"
    )
    paired = forest_paths
    base_by_path = {r["image_path"]: r for r in base if r["image_path"] in paired}
    missing = paired - set(base_by_path)
    assert not missing, f"{task}: {len(missing)} paired images missing from Base pool"

    field = _diag_field(task)
    rows = []
    for img in sorted(paired):
        b, fo, de = base_by_path[img], next(r for r in forest if r["image_path"] == img), \
                     next(r for r in debate if r["image_path"] == img)
        true_c = _true_canonical(b, task)
        assert true_c == _true_canonical(fo, task) == _true_canonical(de, task), \
            f"{task}: label mismatch on {img}"

        # ── triage stage ──
        base_triage = canonical_label((b.get("medgemma_diagnosis") or {}).get(field) or "", task)
        forest_triage = _forest_vote_label(fo.get("forest_votes") or [], task)
        debate_triage = canonical_label((de.get("medgemma_diagnosis") or {}).get(field) or "", task)

        # ── final stage ──
        base_final = canonical_label(b.get("predicted_class") or "", task)
        forest_final = canonical_label(fo.get("predicted_class") or "", task)
        debate_final = canonical_label(de.get("predicted_class") or "", task)

        rows.append({
            "task": task, "image_path": img, "true": true_c,
            "base_triage_correct": _score(base_triage, true_c, task),
            "forest_triage_correct": _score(forest_triage, true_c, task),
            "debate_triage_correct": _score(debate_triage, true_c, task),
            "base_final_correct": _score(base_final, true_c, task),
            "forest_final_correct": _score(forest_final, true_c, task),
            "debate_final_correct": _score(debate_final, true_c, task),
            "base_triage_pred": prep(base_triage, task) if base_triage else "unknown",
            "forest_triage_pred": prep(forest_triage, task) if forest_triage else "unknown",
            "debate_triage_pred": prep(debate_triage, task) if debate_triage else "unknown",
            "base_final_pred": prep(base_final, task) if base_final else "unknown",
            "forest_final_pred": prep(forest_final, task) if forest_final else "unknown",
            "debate_final_pred": prep(debate_final, task) if debate_final else "unknown",
            "base_triage_conf": ((b.get("medgemma_diagnosis") or {}).get("diagnosis_confidence")) or 0.5,
            "forest_triage_conf": fo.get("vote_fraction") if fo.get("vote_fraction") is not None else 0.5,
            "debate_triage_conf": ((de.get("medgemma_diagnosis") or {}).get("diagnosis_confidence")) or 0.5,
            "base_final_conf": b.get("final_confidence") if b.get("final_confidence") is not None else 0.5,
            "forest_final_conf": fo.get("final_confidence") if fo.get("final_confidence") is not None else 0.5,
            "debate_final_conf": de.get("final_confidence") if de.get("final_confidence") is not None else 0.5,
        })
    return pd.DataFrame(rows)


def wilson(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return round(k / n, 4), round(lo, 4), round(hi, 4)


def sens_spec(pred: list[str], true: list[str], pos: str) -> tuple[float, float]:
    pairs = [(t, p) for t, p in zip(true, pred) if t in (pos, "normal")]
    if not pairs:
        return float("nan"), float("nan")
    tp = sum(1 for t, p in pairs if t == pos and p == pos)
    fn = sum(1 for t, p in pairs if t == pos and p != pos)
    tn = sum(1 for t, p in pairs if t == "normal" and p == "normal")
    fp = sum(1 for t, p in pairs if t == "normal" and p != "normal")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return round(sens, 4), round(spec, 4)


def accuracy_rows(df: pd.DataFrame, task: str) -> list[dict]:
    n = len(df)
    pos = pos_class(task)
    out = []
    for system in ("base", "forest", "debate"):
        for stage in ("triage", "final"):
            correct_col = f"{system}_{stage}_correct"
            pred_col = f"{system}_{stage}_pred"
            conf_col = f"{system}_{stage}_conf"
            k = int(df[correct_col].sum())
            acc, lo, hi = wilson(k, n)
            correct = df[correct_col].values.astype(float)
            confs = np.clip(df[conf_col].astype(float).values, 0, 1)
            ece = round(compute_ece(confs, correct), 4)
            row = {
                "task": task, "system": system, "stage": stage, "n": n,
                "correct": k, "accuracy": acc, "wilson_lo": lo, "wilson_hi": hi,
                "ece": ece, "mean_conf": round(float(confs.mean()), 4),
            }
            if not is_multiclass(task):
                sens, spec = sens_spec(df[pred_col].tolist(), df["true"].tolist(), pos)
                row["sensitivity"] = sens
                row["specificity"] = spec
            out.append(row)
    return out


def holm_bonferroni(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def mcnemar_rows(df: pd.DataFrame, task: str) -> list[dict]:
    rows = []
    for stage in ("triage", "final"):
        debate_c = df[f"debate_{stage}_correct"].values
        for other in ("base", "forest"):
            other_c = df[f"{other}_{stage}_correct"].values
            b = int(((debate_c == 1) & (other_c == 0)).sum())
            c = int(((debate_c == 0) & (other_c == 1)).sum())
            table = [[int(((debate_c == 1) & (other_c == 1)).sum()), b],
                     [c, int(((debate_c == 0) & (other_c == 0)).sum())]]
            res = mcnemar(table, exact=True)
            rows.append({
                "task": task, "stage": stage, "comparison": f"debate_vs_{other}",
                "n": len(df), "debate_only_correct": b, "other_only_correct": c,
                "statistic": res.statistic, "p_raw": res.pvalue,
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc_rows, mc_rows, all_frames = [], [], []

    for task, paths in TASKS.items():
        df = build_task_frame(task, paths)
        all_frames.append(df)
        acc_rows.extend(accuracy_rows(df, task))
        mc_rows.extend(mcnemar_rows(df, task))

    acc_df = pd.DataFrame(acc_rows)
    mc_df = pd.DataFrame(mc_rows)
    mc_df["p_holm"] = holm_bonferroni(mc_df["p_raw"].tolist())
    mc_df["significant"] = mc_df["p_holm"] < 0.05

    acc_df.to_csv(OUT / "paired_accuracy.csv", index=False)
    mc_df.to_csv(OUT / "mcnemar_tests.csv", index=False)
    pd.concat(all_frames).to_csv(OUT / "paired_raw_scores.csv", index=False)

    print("== Paired accuracy (Base / Forest / Debate, identical images) ==")
    print(acc_df.to_string(index=False))
    print("\n== McNemar Debate-vs-Base / Debate-vs-Forest (Holm-Bonferroni over 16 tests) ==")
    print(mc_df.to_string(index=False))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
