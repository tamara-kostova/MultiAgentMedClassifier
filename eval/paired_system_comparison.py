"""
Base vs. Forest vs. Debate (and any added system) — paired-subset comparison for
Chapter 8 / Chapter 9 of the thesis (tab:debate_main, tab:debate_rounds,
tab:synthesis, and the McNemar tests of ssec:stats).

Debate and Forest were both run on the identical 500-image subset per task. Base was
run on a larger pool that fully contains that subset. This script:

  1. Loads every JSONL through eval.jsonl_io (per image_path the LAST non-error row;
     error-only images are kept and scored wrong) and filters each "pool" system
     (Base by default) down to the exact image_paths of the non-pool systems, which
     must all share one image set — systems are compared on identical images.
  2. Scores triage and final-stage accuracy with abstain-as-wrong (an
     empty/unparseable prediction counts as incorrect, never dropped).
  3. For a forest run (records carry `forest_votes`), recomputes the triage vote
     label AND its vote_fraction from canonical ballots (eval.forest_votes):
     multiclass ballots use diagnosis_detailed falling back to diagnosis_name, an
     empty ballot is an abstain vote (counted in n, never a label), ties go to the
     first ballot in list order, agents are keyed by agent_idx/position (roles may
     repeat). The stored `vote_fraction` and `medgemma_diagnosis` are NOT used:
     the stored fraction of multiclass_forest_n4.jsonl was computed over raw
     diagnosis_name and disagrees with the ballots on most rows.
  4. Confidence: only a missing (None) confidence is replaced by 0.5, and every
     substitution is counted (`n_conf_substituted`); a logged 0.0 stays 0.0.
  5. Computes accuracy with Wilson 95% CIs, sensitivity/specificity (binary-style
     tasks) and ECE (eval.metrics, first bin [0, 0.1]) per (task, system, stage).
  6. Runs McNemar's exact test for each configured comparison at both stages, Holm-
     Bonferroni corrected across the whole family, which is printed explicitly.
     Default family: {debate_vs_base, debate_vs_forest} x {triage, final} x 4
     tasks = 16 tests (the thesis family). Adding systems or comparisons changes
     the family size, and therefore every adjusted p-value.

Usage:
    python eval/paired_system_comparison.py
    # add a system (e.g. Base without explainability, homogeneous forest) and test it:
    python eval/paired_system_comparison.py \
        --system forest_homog binary_tumor outputs/eval/binary_forest_homog_n4.jsonl \
        --comparisons debate_vs_base debate_vs_forest forest_homog_vs_forest
    python eval/paired_system_comparison.py --system base_noexpl ms outputs/eval/ms_base_noexpl.jsonl \
        --pool_system base_noexpl --comparisons base_noexpl_vs_base

Outputs (--out_dir, default outputs/analysis/debate_vs_base_forest/):
    paired_accuracy.csv     — one row per (task, system, stage)
    mcnemar_tests.csv       — one row per (task, stage, comparison) incl. p_raw, p_holm
    holm_family.csv         — the exact list of tests in the Holm family
    paired_raw_scores.csv   — per-image correctness/predictions/confidences
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # run as `python eval/<script>.py`: make `eval` importable
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from eval.eval_analysis import ERROR_PRED, canonical_label, is_multiclass, mg_diag_pred, pos_class, prep
from eval.forest_votes import forest_vote
from eval.jsonl_io import ERROR_FLAG, by_image_path, load_records
from eval.metrics import check_ece_bound, holm_bonferroni, mcnemar_exact, wilson_ci

TASK_NAMES = ("binary_tumor", "multiclass_tumor", "ms", "stroke")
_PREFIX = {"binary_tumor": "binary", "multiclass_tumor": "multiclass", "ms": "ms", "stroke": "stroke"}
_BASE = {
    "binary_tumor": "outputs/eval/binary_tumor_tumor_eval.jsonl",
    "multiclass_tumor": "outputs/eval/multiclass_tumor_tumor_eval.jsonl",
    "ms": "outputs/eval/ms_dataset_eval.jsonl",
    "stroke": "outputs/eval/stroke_dataset_eval.jsonl",
}

# system -> task -> JSONL path
DEFAULT_SYSTEMS: dict[str, dict[str, str]] = {
    "base": dict(_BASE),
    "forest": {t: f"outputs/eval/{_PREFIX[t]}_forest_n4.jsonl" for t in TASK_NAMES},
    "debate": {t: f"outputs/eval/{_PREFIX[t]}_debate_r2.jsonl" for t in TASK_NAMES},
}
DEFAULT_POOL = ("base",)
DEFAULT_COMPARISONS = ("debate_vs_base", "debate_vs_forest")
STAGES = ("triage", "final")

# Back-compat for anything that imported the old per-task mapping.
TASKS = {t: {s: DEFAULT_SYSTEMS[s][t] for s in DEFAULT_SYSTEMS} for t in TASK_NAMES}

OUT = Path("outputs/analysis/debate_vs_base_forest")


def _score(pred: str, true_canonical: str, task: str) -> int:
    """1 if correct, 0 otherwise. Abstentions ("unknown") and error-only records score 0."""
    if pred in ("unknown", ERROR_PRED):
        return 0
    return int(pred == prep(true_canonical, task))


def _prep_pred(label: str, task: str) -> str:
    return prep(label, task) if label else "unknown"


def _true_canonical(row: dict, task: str) -> str:
    v = row.get("true_label_canonical") or row.get("true_label_name") or row.get("true_label") or ""
    return canonical_label(str(v), task)


def _forest_vote_label(votes, task: str) -> tuple[str, float | None]:
    """(canonical vote label, vote_fraction) recomputed from the ballots."""
    fv = forest_vote(votes, task)
    return fv["label"], fv["vote_fraction"]


def _conf(value, counter: list) -> float:
    """None -> 0.5 (counted); any number, including 0.0, is kept as-is."""
    if value is None:
        counter[0] += 1
        return 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        counter[0] += 1
        return 0.5
    if np.isnan(v):
        counter[0] += 1
        return 0.5
    return v


def _is_forest(records: list[dict]) -> bool:
    with_votes = sum(1 for r in records if isinstance(r.get("forest_votes"), list) and r["forest_votes"])
    return with_votes > len(records) / 2


def _system_rows(rec: dict, task: str, forest: bool, subs: dict) -> dict:
    """triage/final (pred, conf) for one record of one system."""
    if rec.get(ERROR_FLAG):
        subs["error"] += 1
        return {"triage": (ERROR_PRED, 0.5), "final": (ERROR_PRED, 0.5)}
    if forest:
        label, frac = _forest_vote_label(rec.get("forest_votes") or [], task)
        triage = (_prep_pred(label, task), _conf(frac, subs["triage"]))
    else:
        diag = rec.get("medgemma_diagnosis") or {}
        triage = (_prep_pred(mg_diag_pred(diag, task), task),
                  _conf(diag.get("diagnosis_confidence") if isinstance(diag, dict) else None,
                        subs["triage"]))
    final = (_prep_pred(canonical_label(rec.get("predicted_class") or "", task), task),
             _conf(rec.get("final_confidence"), subs["final"]))
    return {"triage": triage, "final": final}


def build_task_frame(task: str, paths: dict[str, str], pool: tuple[str, ...] = DEFAULT_POOL,
                     intersect: bool = False) -> tuple[pd.DataFrame, dict]:
    """Per-image frame for one task. Returns (df, meta) with meta[system] counts."""
    loaded = {}
    for system, path in paths.items():
        recs, stats = load_records(path)
        loaded[system] = (by_image_path(recs), stats)

    paired_systems = [s for s in paths if s not in pool]
    if not paired_systems:
        raise ValueError(f"{task}: every system is a pool system; nothing defines the paired set")
    sets = {s: set(loaded[s][0]) for s in paired_systems}
    ref = set.intersection(*sets.values())
    for s, ps in sets.items():
        if ps != ref:
            msg = (f"{task}: {s} image set differs from the paired set "
                   f"({len(ps)} vs {len(ref)} common)")
            if not intersect:
                raise AssertionError(msg + "; pass --intersect to compare on the intersection")
            print(f"[warn] {msg}; using the intersection")
    for s in pool:
        if s in loaded:
            missing = ref - set(loaded[s][0])
            assert not missing, f"{task}: {len(missing)} paired images missing from pool system {s}"

    meta = {}
    forest_kind = {s: _is_forest(list(loaded[s][0].values())) for s in paths}
    subs = {s: {"triage": [0], "final": [0], "error": 0} for s in paths}
    rows = []
    for img in sorted(ref):
        recs = {s: loaded[s][0][img] for s in paths}
        trues = {s: _true_canonical(r, task) for s, r in recs.items()}
        true_c = next(iter(trues.values()))
        assert len(set(trues.values())) == 1, f"{task}: label mismatch on {img}: {trues}"
        row = {"task": task, "image_path": img, "true": true_c}
        for s, rec in recs.items():
            out = _system_rows(rec, task, forest_kind[s], subs[s])
            for stage in STAGES:
                pred, conf = out[stage]
                row[f"{s}_{stage}_correct"] = _score(pred, true_c, task)
                row[f"{s}_{stage}_pred"] = pred
                row[f"{s}_{stage}_conf"] = conf
        rows.append(row)
    for s in paths:
        meta[s] = {"path": paths[s], "forest_vote_recomputed": forest_kind[s],
                   "n_error_only": subs[s]["error"],
                   "n_conf_substituted_triage": subs[s]["triage"][0],
                   "n_conf_substituted_final": subs[s]["final"][0],
                   "n_duplicate_rows": loaded[s][1].n_duplicate_rows,
                   "n_unparsable": loaded[s][1].n_unparsable}
    return pd.DataFrame(rows), meta


def wilson(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = wilson_ci(k, n)
    return round(k / n, 4), round(lo, 4), round(hi, 4)


def sens_spec(pred: list[str], true: list[str], pos: str) -> tuple[float, float]:
    """Abstentions/errors are misses: FN on a positive, a non-TN on a normal."""
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


def accuracy_rows(df: pd.DataFrame, task: str, systems: list[str], meta: dict | None = None) -> list[dict]:
    n = len(df)
    pos = pos_class(task)
    out = []
    for system in systems:
        for stage in STAGES:
            correct = df[f"{system}_{stage}_correct"].values.astype(float)
            preds = df[f"{system}_{stage}_pred"]
            confs = np.clip(df[f"{system}_{stage}_conf"].astype(float).values, 0, 1)
            k = int(correct.sum())
            acc, lo, hi = wilson(k, n)
            ece = check_ece_bound(confs, correct)
            row = {
                "task": task, "system": system, "stage": stage, "n": n,
                "n_abstained": int((preds == "unknown").sum()),
                "n_error": int((preds == ERROR_PRED).sum()),
                "correct": k, "accuracy": acc, "wilson_lo": lo, "wilson_hi": hi,
                "ece": round(ece, 4), "mean_conf": round(float(confs.mean()), 4),
                "n_conf_substituted": (meta or {}).get(system, {}).get(f"n_conf_substituted_{stage}", 0),
            }
            if not is_multiclass(task):
                sens, spec = sens_spec(preds.tolist(), df["true"].tolist(), pos)
                row["sensitivity"] = sens
                row["specificity"] = spec
            out.append(row)
    return out


def parse_comparison(comp: str, systems) -> tuple[str, str]:
    a, sep, b = comp.partition("_vs_")
    if not sep or a not in systems or b not in systems:
        raise ValueError(f"comparison {comp!r} must be '<A>_vs_<B>' with A, B in {sorted(systems)}")
    return a, b


def mcnemar_rows(df: pd.DataFrame, task: str, comparisons: list[str], systems) -> list[dict]:
    rows = []
    for stage in STAGES:
        for comp in comparisons:
            a, b = parse_comparison(comp, systems)
            res = mcnemar_exact(df[f"{a}_{stage}_correct"].values, df[f"{b}_{stage}_correct"].values)
            rows.append({
                "task": task, "stage": stage, "comparison": comp,
                "n": res["n"], "a_only_correct": res["a_only_correct"],
                "b_only_correct": res["b_only_correct"],
                "statistic": res["statistic"], "p_raw": res["p_raw"],
            })
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASK_NAMES), choices=list(TASK_NAMES))
    ap.add_argument("--system", nargs=3, action="append", default=[], metavar=("NAME", "TASK", "PATH"),
                    help="Add (or override) a system's JSONL for a task. Repeatable.")
    ap.add_argument("--pool_system", action="append", default=None,
                    help="Systems run on a larger pool and filtered to the paired set "
                         "(default: base). Repeatable.")
    ap.add_argument("--comparisons", nargs="+", default=list(DEFAULT_COMPARISONS),
                    help="McNemar comparisons '<A>_vs_<B>' forming the Holm family "
                         "(x stages x tasks where both systems exist).")
    ap.add_argument("--intersect", action="store_true",
                    help="Allow non-pool systems with different image sets; compare on the intersection.")
    ap.add_argument("--out_dir", default=str(OUT))
    args = ap.parse_args(argv)

    systems: dict[str, dict[str, str]] = {s: dict(p) for s, p in DEFAULT_SYSTEMS.items()}
    for name, task, path in args.system:
        if task not in TASK_NAMES:
            ap.error(f"--system task must be one of {TASK_NAMES}")
        systems.setdefault(name, {})[task] = path
    pool = tuple(args.pool_system) if args.pool_system else DEFAULT_POOL
    for comp in args.comparisons:
        parse_comparison(comp, systems)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    acc_rows, mc_rows, frames, meta_rows = [], [], [], []

    for task in args.tasks:
        paths = {s: p[task] for s, p in systems.items() if task in p and Path(p[task]).exists()}
        for s, p in systems.items():
            if task in p and s not in paths:
                print(f"[skip] {task}/{s}: {p[task]} not found")
        df, meta = build_task_frame(task, paths, pool=tuple(s for s in pool if s in paths),
                                    intersect=args.intersect)
        frames.append(df)
        acc_rows.extend(accuracy_rows(df, task, list(paths), meta))
        comps = [c for c in args.comparisons if all(s in paths for s in parse_comparison(c, systems))]
        skipped = sorted(set(args.comparisons) - set(comps))
        if skipped:
            print(f"[skip] {task}: comparisons {skipped} (a system is missing for this task)")
        mc_rows.extend(mcnemar_rows(df, task, comps, paths))
        for s, m in meta.items():
            meta_rows.append({"task": task, "system": s, **m})

    acc_df = pd.DataFrame(acc_rows)
    mc_df = pd.DataFrame(mc_rows)
    mc_df["p_holm"] = holm_bonferroni(mc_df["p_raw"].tolist())
    mc_df["significant"] = mc_df["p_holm"] < 0.05
    family = mc_df[["task", "stage", "comparison"]].copy()
    family.insert(0, "test", range(1, len(family) + 1))

    acc_df.to_csv(out / "paired_accuracy.csv", index=False)
    mc_df.to_csv(out / "mcnemar_tests.csv", index=False)
    family.to_csv(out / "holm_family.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out / "system_inputs.csv", index=False)
    pd.concat(frames).to_csv(out / "paired_raw_scores.csv", index=False)

    pd.set_option("display.width", 250)
    print("== Inputs ==")
    print(pd.DataFrame(meta_rows).drop(columns=["path"]).to_string(index=False))
    print("\n== Paired accuracy (identical images; abstain/error = wrong) ==")
    print(acc_df.to_string(index=False))
    print(f"\n== Holm-Bonferroni family: m = {len(family)} tests "
          f"({len(args.comparisons)} comparison(s) {list(args.comparisons)} x "
          f"{len(STAGES)} stages x {len(args.tasks)} task(s), minus skipped) ==")
    print(mc_df.to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
