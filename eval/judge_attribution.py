"""Which advocate does the debate judge say it believed, and was it right?

The debate judge writes a free-text ``Reason:`` into ``final_report`` for every
image. That text is the only surviving record of the judge's deliberation: the
per-round advocate arguments live in ``NeuroimagingState["debate_arguments"]``
but ``eval/tumor_eval.py`` never writes them to the eval JSONL, so the judge's
own characterisation of each argument is what can be measured after the fact.

This script classifies each judge reason by which advocate (if any) it singles
out as most compelling, and cross-tabulates that against correctness. It exists
to answer one question the thesis could not previously answer from the logs:
*why* \\Debate collapses on stroke.

The answer it produces is a defect rather than a property of debate. On MS and
stroke the SAM3 probe is ineligible (``RoutingConfig.sam3_eligible_tasks``), so
``agents/debate.py:_build_sam_prompt`` sets ``lesion_detected`` to
"No (SAM3 not applicable for this task)" and the advocate template still tells
it to "argue what the absence of segmentable pathology implies". The judge
template receives only the advocate's prose, never the ``skipped`` flag, so an
ineligible tool and a tool that looked and found nothing reach the judge in the
same words. See ``--sensitivity`` for what that costs.

Scoring goes through ``eval_analysis.prep``/``canonical_label`` so the numbers
are directly comparable with ``tab:debate_rounds`` in the thesis; run
``--verify`` to confirm that table reproduces.

Usage:
    python eval/judge_attribution.py
    python eval/judge_attribution.py --verify --sensitivity
    python eval/judge_attribution.py --task stroke --dump-endorsed sam3_reasons.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from eval.eval_analysis import canonical_label, prep, set_scoring

# Task -> default debate JSONL, and whether the SAM3 probe actually runs there.
TASKS: dict[str, tuple[str, bool]] = {
    "binary_tumor": ("outputs/eval/binary_debate_r2.jsonl", True),
    "multiclass_tumor": ("outputs/eval/multiclass_debate_r2.jsonl", True),
    "ms": ("outputs/eval/ms_debate_r2.jsonl", False),
    "stroke": ("outputs/eval/stroke_debate_r2.jsonl", False),
}

# Published tab:debate_rounds, for --verify: (% changed, acc stable, acc changed).
THESIS_ROUNDS = {
    "binary_tumor": (64.4, 0.972, 0.376),
    "multiclass_tumor": (80.0, 0.420, 0.193),
    "ms": (49.6, 0.790, 0.629),
    "stroke": (46.0, 0.641, 0.483),
}

ADVOCATES = {
    "SAM3": r"SAM[\s-]?3|segmentation (advocate|model)",
    "BiomedCLIP": r"BiomedCLIP",
    "CNN": r"\bCNN\b",
}
# The judge's stock phrasing for the argument it found decisive.
ENDORSE = r"most compelling|is compelling|strongest|persuasive|carries the most weight"
_SENT = re.compile(r"(?<=[.;])\s+")
_REASON = re.compile(r"Reason:\s*(.*)", re.S)


def judge_reason(rec: dict) -> str:
    m = _REASON.search(rec.get("final_report") or "")
    return m.group(1).strip() if m else ""


def endorsed(reason: str) -> set[str]:
    """Advocates named in a sentence that also calls an argument decisive."""
    out: set[str] = set()
    for sent in _SENT.split(reason):
        if re.search(ENDORSE, sent, re.I):
            for name, pat in ADVOCATES.items():
                if re.search(pat, sent, re.I):
                    out.add(name)
    return out


def mentions(reason: str, advocate: str) -> bool:
    return bool(re.search(ADVOCATES[advocate], reason, re.I))


def mentions_any(reason: str) -> bool:
    return any(re.search(p, reason, re.I) for p in ADVOCATES.values())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Unpaired two-proportion z-test; returns (z, two-sided p)."""
    if not n1 or not n2:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def accuracy(records: list[dict], task: str) -> tuple[int, int, float, float, float]:
    """Strict-scored accuracy. Abstentions are dropped, as eval_analysis does."""
    hits = []
    for r in records:
        gt = prep(canonical_label(r["true_label_canonical"], task), task)
        pr = prep(canonical_label(r["predicted_class_canonical"], task), task)
        if "unknown" not in (gt, pr):
            hits.append(gt == pr)
    k, n = sum(hits), len(hits)
    lo, hi = wilson(k, n)
    return k, n, (k / n if n else float("nan")), lo, hi


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def verify_rounds(task: str, recs: list[dict]) -> None:
    changed = [r for r in recs if r.get("debate_round_changed")]
    stable = [r for r in recs if not r.get("debate_round_changed")]
    pct = 100 * len(changed) / len(recs)
    _, _, a_st, _, _ = accuracy(stable, task)
    _, _, a_ch, _, _ = accuracy(changed, task)
    want = THESIS_ROUNDS[task]
    ok = lambda a, b: "ok" if abs(a - b) < 6e-4 else "MISMATCH"
    print(
        f"  {task:<17} changed {pct:5.1f}% [{want[0]:4.1f}] {ok(pct, want[0]):<8}"
        f" stable {a_st:.3f} [{want[1]:.3f}] {ok(a_st, want[1]):<8}"
        f" changed {a_ch:.3f} [{want[2]:.3f}] {ok(a_ch, want[2])}"
    )


def report(task: str, recs: list[dict], probe_runs: bool) -> None:
    _, _, overall, lo, hi = accuracy(recs, task)
    flag = "" if probe_runs else "   [SAM3 probe INELIGIBLE: null-evidence advocate]"
    print(f"\n=== {task}  n={len(recs)}  accuracy={overall:.3f} [{lo:.3f}, {hi:.3f}]{flag}")

    reasons = {id(r): judge_reason(r) for r in recs}
    named = [r for r in recs if mentions_any(reasons[id(r)])]
    unnamed = [r for r in recs if not mentions_any(reasons[id(r)])]
    _, n1, a1, _, _ = accuracy(named, task)
    _, n2, a2, _, _ = accuracy(unnamed, task)
    print(f"  reason names >=1 advocate : {len(named):4d} ({100*len(named)/len(recs):5.1f}%)  acc={a1:.3f}")
    print(f"  reason names no advocate  : {len(unnamed):4d} ({100*len(unnamed)/len(recs):5.1f}%)  acc={a2:.3f}")

    print(f"  {'judge calls decisive':<16} {'n':>5} {'share':>7} {'acc':>7}  {'95% CI':>16}")
    groups = {a: [r for r in recs if a in endorsed(reasons[id(r)])] for a in ADVOCATES}
    for name in ("SAM3", "BiomedCLIP", "CNN"):
        g = groups[name]
        if not g:
            continue
        k, n, a, lo, hi = accuracy(g, task)
        print(f"  {name:<16} {len(g):5d} {100*len(g)/len(recs):6.1f}% {a:7.3f}  [{lo:.3f}, {hi:.3f}]")

    sam, cnn = groups["SAM3"], groups["CNN"]
    if sam and cnn:
        ks, ns, as_, _, _ = accuracy(sam, task)
        kc, nc, ac, _, _ = accuracy(cnn, task)
        z, p = two_prop_z(kc, nc, ks, ns)
        print(f"  CNN-endorsed minus SAM3-endorsed: {100*(ac-as_):+.1f} pp  (z={z:.2f}, p={p:.2g})")


def sensitivity(task: str, recs: list[dict]) -> None:
    """How much of the gap sits on cases where the null advocate swayed the judge.

    This conditions on the judge's own stated reasoning, which is a
    post-treatment variable, so it bounds the defect's contribution. It is a
    diagnostic, not a corrected accuracy for a fixed system.
    """
    print(f"\n--- {task}: sensitivity to the null-evidence advocate")
    for label, subset in (
        ("all cases", recs),
        ("excl. SAM3-endorsed", [r for r in recs if "SAM3" not in endorsed(judge_reason(r))]),
        ("excl. any SAM3 mention", [r for r in recs if not mentions(judge_reason(r), "SAM3")]),
    ):
        _, n, a, lo, hi = accuracy(subset, task)
        print(f"  {label:<24} acc={a:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), action="append",
                    help="Restrict to one task (repeatable). Default: all four.")
    ap.add_argument("--scoring", default="strict", choices=("strict", "abnormal"))
    ap.add_argument("--verify", action="store_true",
                    help="Reproduce tab:debate_rounds as a consistency check.")
    ap.add_argument("--sensitivity", action="store_true",
                    help="Bound the null-evidence advocate's contribution (MS and stroke).")
    ap.add_argument("--dump-endorsed", metavar="PATH",
                    help="Write the SAM3-endorsed judge reasons to a file for inspection.")
    args = ap.parse_args()

    set_scoring(args.scoring)
    tasks = args.task or list(TASKS)
    loaded = {t: load(TASKS[t][0]) for t in tasks if Path(TASKS[t][0]).exists()}
    missing = [t for t in tasks if t not in loaded]
    for t in missing:
        print(f"[skip] {t}: {TASKS[t][0]} not found")
    if not loaded:
        raise SystemExit("no debate JSONLs found")

    if args.verify:
        print("Reproducing tab:debate_rounds (thesis values in brackets):")
        for t, recs in loaded.items():
            verify_rounds(t, recs)

    for t, recs in loaded.items():
        report(t, recs, TASKS[t][1])

    if args.sensitivity:
        for t, recs in loaded.items():
            if not TASKS[t][1]:
                sensitivity(t, recs)

    if args.dump_endorsed:
        out = Path(args.dump_endorsed)
        with out.open("w") as fh:
            for t, recs in loaded.items():
                for r in recs:
                    reason = judge_reason(r)
                    if "SAM3" in endorsed(reason):
                        fh.write(f"[{t}] true={r['true_label_canonical']} "
                                 f"pred={r['predicted_class_canonical']}\n{reason}\n\n")
        print(f"\nwrote SAM3-endorsed judge reasons to {out}")


if __name__ == "__main__":
    main()
