"""Every number, table and the main figure of the debate/forest paper, from the v2 run JSONLs.

    python eval/make_paper_results.py                              (repo root, in .venv)
    python eval/make_paper_results.py --paper_dir papers/icbinb_bio_v2 --runs outputs/eval/v2

Writes <paper_dir>/generated/:
    numbers.tex      \\res{key} macros for every number quoted in the text
    tab_*.tex        table bodies (\\input inside the table environments of main.tex)
    results.json     the same numbers, for checking
and <paper_dir>/main_result.png. main.tex reads them, so no number in the paper is typed by hand.

Scoring follows eval/paired_system_comparison.py (strict, abstain = wrong, error-only
images kept and scored wrong, forest triage recomputed from the ballots). Two Holm
families, fixed before the controls finished so the main p-values never move:
    main      12 final-stage tests {B vs A, B vs C, C vs A} x 4 tasks
              + 4 triage tests C vs A (the stage the forest replaces) = 16
    controls  triage {H vs C, R vs C} x 4 tasks = 8 (H = homogeneous sampled forest,
              R = role-diverse sampled forest); R is left out until its run is complete.
B vs A at triage is not tested: both systems run the identical single-agent triage node.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.ensemble_posthoc import advocate_labels, debate_analyses, forest_analyses, forest_table  # noqa: E402
from eval.eval_analysis import (  # noqa: E402
    _verdict_label, is_multiclass as is_multiclass_task, load, pos_class, section_medgemma_shift,
    section_model_accuracy,
)
from eval.forest_votes import forest_vote  # noqa: E402
from eval.jsonl_io import by_image_path, load_records  # noqa: E402
from eval.metrics import compute_ece, holm_bonferroni, mcnemar_exact, wilson_ci  # noqa: E402
from eval.paired_system_comparison import (  # noqa: E402
    _prep_pred, _score, accuracy_rows, build_task_frame, sens_spec,
)

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--paper_dir", default="papers/icbinb_bio_v2", help="LaTeX folder (relative to the repo root)")
_ap.add_argument("--runs", default="outputs/eval/v2", help="folder of <task>_<system>.jsonl runs")
_args = _ap.parse_args()
HERE = ROOT / _args.paper_dir
OUT = HERE / "generated"
V2 = ROOT / _args.runs
TASKS = ["binary_tumor", "multiclass_tumor", "ms", "stroke"]
TASK_KEY = {"binary_tumor": "bin", "multiclass_tumor": "multi", "ms": "ms", "stroke": "stroke"}
TASK_TEX = {"binary_tumor": r"\textsc{bin.\ tumor}", "multiclass_tumor": r"\textsc{multi.\ tumor}",
            "ms": r"\textsc{ms}", "stroke": r"\textsc{stroke}"}
SYS = {"A": "base", "B": "debate", "C": "forest", "H": "homog", "R": "rolesamp"}
N_EXPECTED = 500

R: dict[str, str] = {}   # \res{key} -> text
RAW: dict[str, object] = {}


def put(key: str, value, fmt: str | None = None) -> None:
    RAW[key] = value
    if fmt is None:
        text = str(value)
    elif fmt == "pct":      # 0.9404 -> 94.0
        text = f"{100 * value:.1f}"
    elif fmt == "pp":       # signed percentage points
        text = f"{100 * value:+.1f}".replace("+", "$+$").replace("-", "$-$")
    else:
        text = format(value, fmt)
    R[key] = text


def p_tex(p: float) -> str:
    if p < 0.001:
        exp = math.floor(math.log10(p)) if p > 0 else -300
        return rf"$<10^{{{exp + 1}}}$" if p > 0 else r"$<10^{-300}$"
    return f"${p:.3f}$"


def stars(p: float) -> str:
    return r"$^{***}$" if p < 0.001 else r"$^{**}$" if p < 0.01 else r"$^{*}$" if p < 0.05 else ""


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    recs, _ = load_records(path, verbose=False)
    return len(by_image_path(recs)) == N_EXPECTED


# ── load ──────────────────────────────────────────────────────────────────────

def paths_for(task: str) -> dict[str, str]:
    out = {}
    for code, name in SYS.items():
        p = V2 / f"{task}_{name}.jsonl"
        if complete(p):
            out[code] = str(p)
        elif code in "ABCH":
            raise SystemExit(f"missing or incomplete: {p}")
    return out


FRAMES, META, PATHS = {}, {}, {}
for t in TASKS:
    PATHS[t] = paths_for(t)
    FRAMES[t], META[t] = build_task_frame(t, PATHS[t], pool=())
HAVE_R = all("R" in PATHS[t] for t in TASKS)
put("rolesampDone", "yes" if HAVE_R else "no")


# ── accuracy, confidence, ECE, sensitivity/specificity ───────────────────────

ACC = {}
for t in TASKS:
    for row in accuracy_rows(FRAMES[t], t, list(PATHS[t]), META[t]):
        ACC[(t, row["system"], row["stage"])] = row
        k = f"{TASK_KEY[t]}.{row['system']}.{row['stage']}"
        put(f"{k}.acc", row["accuracy"], "pct")
        put(f"{k}.lo", row["wilson_lo"], "pct")
        put(f"{k}.hi", row["wilson_hi"], "pct")
        put(f"{k}.conf", row["mean_conf"], ".3f")
        put(f"{k}.ece", row["ece"], ".3f")
        put(f"{k}.correct", row["correct"])
        if "sensitivity" in row:
            put(f"{k}.sens", row["sensitivity"], ".3f")
            put(f"{k}.spec", row["specificity"], ".3f")
for t in TASKS:
    put(f"{TASK_KEY[t]}.n", len(FRAMES[t]))
    put(f"{TASK_KEY[t]}.B.errors", META[t]["B"]["n_error_only"])
    put(f"{TASK_KEY[t]}.B.confsub", META[t]["B"]["n_conf_substituted_final"])


# ── McNemar families ──────────────────────────────────────────────────────────

def family(tests: list[tuple[str, str, str, str]], name: str) -> list[dict]:
    rows = []
    for t, stage, a, b in tests:
        df = FRAMES[t]
        res = mcnemar_exact(df[f"{a}_{stage}_correct"].values, df[f"{b}_{stage}_correct"].values)
        rows.append({"task": t, "stage": stage, "a": a, "b": b, **res})
    for row, adj in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
        row["p_holm"] = adj
        k = f"{TASK_KEY[row['task']]}.{row['stage']}.{row['a']}vs{row['b']}"
        put(f"{k}.aonly", row["a_only_correct"])
        put(f"{k}.bonly", row["b_only_correct"])
        put(f"{k}.p", row["p_holm"], ".3g")
        R[f"{k}.ptex"] = p_tex(row["p_holm"])
    put(f"family.{name}.size", len(rows))
    return rows


MAIN = family([(t, "final", a, b) for t in TASKS for a, b in [("B", "A"), ("B", "C"), ("C", "A")]]
              + [(t, "triage", "C", "A") for t in TASKS], "main")
CTRL_SYSTEMS = ["H"] + (["R"] if HAVE_R else [])
CTRL = family([(t, "triage", s, "C") for t in TASKS for s in CTRL_SYSTEMS], "controls")
put("family.controls.planned", 8)
MAIN_P = {(r["task"], r["stage"], r["a"], r["b"]): r for r in MAIN}
CTRL_P = {(r["task"], r["stage"], r["a"], r["b"]): r for r in CTRL}


# ── debate: rounds from the saved per-round verdicts ──────────────────────────

def debate_records(t: str) -> dict[str, dict]:
    recs, _ = load_records(PATHS[t]["B"], verbose=False)
    return by_image_path(recs)


DEBATE = {}
for t in TASKS:
    df = FRAMES[t]
    recs = debate_records(t)
    r1_hit, changed, conf = [], [], []
    for img, fin_hit, c in zip(df["image_path"], df["B_final_correct"], df["B_final_conf"]):
        rec = recs[img]
        verdicts = rec.get("debate_round_verdicts") or []
        labels = [_verdict_label(v, t) for v in verdicts]
        if rec.get("error") or not labels:
            r1_hit.append(fin_hit)
            changed.append(False)
        else:
            r1_hit.append(_score(_prep_pred(labels[0], t), df.loc[df.image_path == img, "true"].iloc[0], t))
            changed.append(any(labels[i] != labels[i - 1] for i in range(1, len(labels))))
        conf.append(c)
    r1, fin = np.array(r1_hit), df["B_final_correct"].values
    ch, conf = np.array(changed), np.clip(np.array(conf, dtype=float), 0, 1)
    base_fin = df["A_final_correct"].values
    fixed = int(((r1 == 0) & (fin == 1)).sum())
    broke = int(((r1 == 1) & (fin == 0)).sum())
    mc = mcnemar_exact(fin, r1)
    k = TASK_KEY[t]
    put(f"{k}.deb.changedpct", ch.mean(), "pct")
    put(f"{k}.deb.nchanged", int(ch.sum()))
    put(f"{k}.deb.r1acc", r1.mean(), "pct")
    put(f"{k}.deb.fixed", fixed)
    put(f"{k}.deb.broke", broke)
    put(f"{k}.deb.net", fixed - broke)
    R[f"{k}.deb.ptex"] = p_tex(mc["p_raw"])
    put(f"{k}.deb.acc.stable", fin[~ch].mean(), ".3f")
    put(f"{k}.deb.acc.changed", fin[ch].mean() if ch.any() else float("nan"), ".3f")
    put(f"{k}.deb.ece.stable", compute_ece(conf[~ch], fin[~ch]), ".3f")
    put(f"{k}.deb.ece.changed", compute_ece(conf[ch], fin[ch]) if ch.any() else float("nan"), ".3f")
    put(f"{k}.deb.base.stable", base_fin[~ch].mean(), ".3f")
    put(f"{k}.deb.base.changed", base_fin[ch].mean() if ch.any() else float("nan"), ".3f")
    DEBATE[t] = dict(fixed=fixed, broke=broke, p=mc["p_raw"], n_changed=int(ch.sum()), r1=r1)


# Round 1 of an R=2 debate is what an R=1 debate returns (greedy decoding; round-1
# advocates see no prior verdict). Post hoc, so its tests form a third, exploratory family.
EXPLO = []
for t in TASKS:
    r1 = DEBATE[t]["r1"]
    put(f"{TASK_KEY[t]}.B1.final.acc", r1.mean(), "pct")
    for other in "AC":
        res = mcnemar_exact(r1, FRAMES[t][f"{other}_final_correct"].values)
        EXPLO.append({"task": t, "a": "B1", "b": other, **res})
for row, adj in zip(EXPLO, holm_bonferroni([r["p_raw"] for r in EXPLO])):
    k = f"{TASK_KEY[row['task']]}.final.B1vs{row['b']}"
    put(f"{k}.aonly", row["a_only_correct"])
    put(f"{k}.bonly", row["b_only_correct"])
    R[f"{k}.ptex"] = p_tex(adj)
    R[f"{k}.prawtex"] = p_tex(row["p_raw"])
put("family.exploratory.size", len(EXPLO))


# A dedicated R=1 run on the same images, made after the R=2 results, tests the exploratory
# finding. Its two comparisons per task (vs A and C, final stage) form the confirmatory family.
def debate1_path(t: str) -> Path:
    return V2 / f"{t}_debate1.jsonl"


HAVE_D1 = all(complete(debate1_path(t)) for t in TASKS)
put("family.confirmatory.size", 2 * len(TASKS) if HAVE_D1 else 0)
if HAVE_D1:
    CONF = []
    for t in TASKS:
        df, _ = build_task_frame(t, {"A": PATHS[t]["A"], "C": PATHS[t]["C"], "D": str(debate1_path(t))}, pool=())
        k = TASK_KEY[t]
        d1 = df["D_final_correct"].values
        put(f"{k}.D1.final.acc", d1.mean(), "pct")
        # agreement with the round-1 verdict saved by the R=2 run, aligned by image, not row order
        r1 = dict(zip(FRAMES[t]["image_path"], DEBATE[t]["r1"]))
        put(f"{k}.D1.agree", float(np.mean([h == r1[img] for img, h in zip(df["image_path"], d1)])), "pct")
        for other in "AC":
            res = mcnemar_exact(d1, df[f"{other}_final_correct"].values)
            CONF.append({"task": t, "b": other, **res})
    for row, adj in zip(CONF, holm_bonferroni([r["p_raw"] for r in CONF])):
        k = f"{TASK_KEY[row['task']]}.final.D1vs{row['b']}"
        put(f"{k}.aonly", row["a_only_correct"])
        put(f"{k}.bonly", row["b_only_correct"])
        R[f"{k}.ptex"] = p_tex(adj)                    # paper prose: no stars
        R[f"{k}.ptexs"] = p_tex(adj) + stars(adj)      # thesis prose: with stars
        RAW[f"{k}.p"] = adj
    put("thesis.d1.allagree", int(all(RAW[f"{TASK_KEY[t]}.D1.agree"] == 1.0 for t in TASKS)))


# A dedicated R=3 run on the same images, made after R=1 and R=2: does a third round undo
# round 2? Its rounds 1-2 should reproduce the R=2 run (greedy decoding). Tests {D3 vs B,
# D3 vs A} x 4 tasks at the final stage form their own exploratory Holm family.
def debate3_path(t: str) -> Path:
    return V2 / f"{t}_debate3.jsonl"


HAVE_D3 = all(complete(debate3_path(t)) for t in TASKS)
put("family.rounds.size", 2 * len(TASKS) if HAVE_D3 else 0)
if HAVE_D3:
    ROUNDS = []
    for t in TASKS:
        df, meta = build_task_frame(t, {"A": PATHS[t]["A"], "D": str(debate3_path(t))}, pool=())
        k = TASK_KEY[t]
        for row in accuracy_rows(df, t, ["D"], meta):
            if row["stage"] == "final":
                for f, src in [("acc", "accuracy"), ("lo", "wilson_lo"), ("hi", "wilson_hi")]:
                    put(f"{k}.D3.final.{f}", row[src], "pct")
                put(f"{k}.D3.final.ece", row["ece"], ".3f")
        recs3, recs2 = by_image_path(load_records(debate3_path(t), verbose=False)[0]), debate_records(t)
        prefix, changed, back, fixed, broke = 0, 0, 0, 0, 0
        for img, true in zip(df["image_path"], df["true"]):
            v3 = [_verdict_label(v, t) for v in recs3[img].get("debate_round_verdicts") or []]
            v2 = [_verdict_label(v, t) for v in recs2[img].get("debate_round_verdicts") or []]
            prefix += v3[:2] == v2[:2]
            if len(v3) == 3 and v3[2] != v3[1]:
                changed += 1
                back += v3[2] == v3[0]
                h2, h3 = _score(_prep_pred(v3[1], t), true, t), _score(_prep_pred(v3[2], t), true, t)
                fixed += h3 > h2
                broke += h2 > h3
        put(f"{k}.D3.prefix", prefix / len(df), "pct")
        put(f"{k}.D3.r3changed", changed)
        put(f"{k}.D3.r3changedpct", changed / len(df), "pct")
        put(f"{k}.D3.r3back", back)
        put(f"{k}.D3.r3fixed", fixed)
        put(f"{k}.D3.r3broke", broke)
        # B (R=2) aligned to this frame by image, not row order
        b_hit = dict(zip(FRAMES[t]["image_path"], FRAMES[t]["B_final_correct"]))
        d3 = df["D_final_correct"].values
        for other, hits in [("B", np.array([b_hit[img] for img in df["image_path"]])),
                            ("A", df["A_final_correct"].values)]:
            ROUNDS.append({"task": t, "b": other, **mcnemar_exact(d3, hits)})
    for row, adj in zip(ROUNDS, holm_bonferroni([r["p_raw"] for r in ROUNDS])):
        k = f"{TASK_KEY[row['task']]}.final.D3vs{row['b']}"
        put(f"{k}.aonly", row["a_only_correct"])
        put(f"{k}.bonly", row["b_only_correct"])
        R[f"{k}.ptex"] = p_tex(adj)
        R[f"{k}.ptexs"] = p_tex(adj) + stars(adj)
        RAW[f"{k}.p"] = adj
    put("thesis.d3.allprefix", int(all(RAW[f"{TASK_KEY[t]}.D3.prefix"] == 1.0 for t in TASKS)))


# Round-2 flips that break a correct verdict: whose position does the new verdict take?
for t in TASKS:
    df = FRAMES[t]
    recs = debate_records(t)
    k = TASK_KEY[t]
    r1_pred, broke_agree, n_broke = [], Counter(), 0
    for img, true, r1h, finh, fin_pred in zip(df["image_path"], df["true"], DEBATE[t]["r1"],
                                             df["B_final_correct"], df["B_final_pred"]):
        rec = recs[img]
        verdicts = rec.get("debate_round_verdicts") or []
        r1_pred.append(_prep_pred(_verdict_label(verdicts[0], t), t) if verdicts and not rec.get("error")
                       else fin_pred)
        if r1h == 1 and finh == 0:
            n_broke += 1
            adv = advocate_labels(rec, t)
            used = rec.get("debate_advocates") or []
            for name, code in [("CNN", "cnn"), ("BiomedCLIP", "clip"), ("SAM3", "sam")]:
                if code in used and _prep_pred(adv[name], t) == fin_pred:
                    broke_agree[code] += 1
            if not any(code in used and _prep_pred(adv[n], t) == fin_pred
                       for n, code in [("CNN", "cnn"), ("BiomedCLIP", "clip"), ("SAM3", "sam")]):
                broke_agree["none"] += 1
            if df.loc[df.image_path == img, "B_triage_pred"].iloc[0] == fin_pred:
                broke_agree["triage"] += 1
    sam_active = any("sam" in (r.get("debate_advocates") or []) for r in recs.values())
    for code in ["cnn", "clip", "sam", "none", "triage"]:
        if code == "sam" and not sam_active:
            continue
        put(f"{k}.broke.{code}", broke_agree[code] / n_broke if n_broke else float("nan"), "pct")
    put(f"{k}.broke.n", n_broke)
    if not is_multiclass_task(t):
        s1, p1 = sens_spec(r1_pred, df["true"].tolist(), pos_class(t))
        put(f"{k}.B1.final.sens", s1, ".3f")
        put(f"{k}.B1.final.spec", p1, ".3f")
    counts = df["true"].value_counts()
    put(f"{k}.classes", ", ".join(f"{n} {c.replace('_', ' ').replace('ms', 'MS')}" for c, n in counts.items()))
    put(f"{k}.majority", counts.max() / counts.sum(), "pct")


# ── debate: which advocate the judge follows ─────────────────────────────────

JUDGE = {}
for t in TASKS:
    recs, _ = load_records(PATHS[t]["B"], verbose=False)
    recs = list(by_image_path(recs).values())
    da = debate_analyses(recs, t)
    advocates_used = Counter(tuple(r.get("debate_advocates") or ()) for r in recs if not r.get("error"))
    sam_active = any("sam" in a for a in advocates_used)
    follows = da["debate_judge_follows"].set_index("advocate")
    claims = da["debate_advocate_claims"].set_index("advocate")
    k = TASK_KEY[t]
    put(f"{k}.sam.active", "yes" if sam_active else "no")
    JUDGE[t] = {}
    for adv, key in [("CNN", "cnn"), ("BiomedCLIP", "clip"), ("SAM3", "sam")]:
        if adv == "SAM3" and not sam_active:
            continue
        f, c = follows.loc[adv], claims.loc[adv]
        put(f"{k}.{key}.advacc", c["advocate_accuracy"], "pct")
        put(f"{k}.{key}.claimrate", c["positive_claim_rate"], "pct")
        put(f"{k}.{key}.follow", f["judge_follows_rate"], "pct")
        put(f"{k}.{key}.accfollow", f["final_acc_when_following"], "pct")
        put(f"{k}.{key}.accnotfollow", f["final_acc_when_not_following"], "pct")
        JUDGE[t][key] = (c["advocate_accuracy"], f["judge_follows_rate"])


# ── forest: vote shape, agreement as a signal ────────────────────────────────

def cochran_armitage(hits_by_group: list[np.ndarray]) -> tuple[float, float]:
    """Two-sided Cochran-Armitage trend test, scores 0..k-1 in group order."""
    scores = np.arange(len(hits_by_group), dtype=float)
    n = np.array([len(h) for h in hits_by_group], dtype=float)
    x = np.array([h.sum() for h in hits_by_group], dtype=float)
    N, p_bar = n.sum(), x.sum() / n.sum()
    t_stat = (scores * (x - n * p_bar)).sum()
    var = p_bar * (1 - p_bar) * ((scores ** 2 * n).sum() - (scores * n).sum() ** 2 / N)
    z = t_stat / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


SHAPES = {4: "40", 3: "31", 2: "22"}
for t in TASKS:
    recs = by_image_path(load_records(PATHS[t]["C"], verbose=False)[0])
    df = FRAMES[t]
    shape = []
    for img in df["image_path"]:
        v = forest_vote(recs[img].get("forest_votes") or [], t)
        top = round((v["vote_fraction"] or 0) * (v["n"] or 0)) if v["label"] else 0
        shape.append(top)
    shape = np.array(shape)
    fin = df["C_final_correct"].values
    k = TASK_KEY[t]
    groups = []
    for top, name in SHAPES.items():
        m = shape == top
        hits = fin[m]
        groups.append(hits)
        put(f"{k}.vote{name}.n", int(m.sum()))
        if m.sum():
            lo, hi = wilson_ci(int(hits.sum()), int(m.sum()))
            put(f"{k}.vote{name}.acc", hits.mean(), ".3f")
            put(f"{k}.vote{name}.lo", lo, ".3f")
            put(f"{k}.vote{name}.hi", hi, ".3f")
            pos = df["true"].values[m] != "normal"
            put(f"{k}.vote{name}.pospct", pos.mean(), ".2f")
    put(f"{k}.vote.nomaj", int((shape < 2).sum()))
    z, p = cochran_armitage([g for g in groups if len(g)])
    put(f"{k}.vote.trendz", z, "+.2f")
    R[f"{k}.vote.trendptex"] = p_tex(p)
    unan, split = fin[shape == 4], fin[(shape >= 2) & (shape < 4)]
    put(f"{k}.vote.unanpct", (shape == 4).mean(), "pct")
    put(f"{k}.vote.acc.unan", unan.mean(), ".3f")
    put(f"{k}.vote.acc.split", split.mean(), ".3f")


# ── forests compared: role-diverse greedy vs homogeneous / role sampled ──────

CONTROL_ROWS = {}
for t in TASKS:
    for code in ["C", "H"] + (["R"] if "R" in PATHS[t] else []):
        recs = list(by_image_path(load_records(PATHS[t][code], verbose=False)[0]).values())
        fa = forest_analyses(forest_table(recs, t), t)
        rules = fa["forest_vote_rules"].set_index("method")
        singles = rules[rules.index.str.startswith("single:")]
        phi = fa["forest_pairwise_phi"]
        dis = fa["forest_role_disagreement"].iloc[0]
        k = f"{TASK_KEY[t]}.{code}"
        put(f"{k}.phi", float(phi[phi.agent_a == "MEAN"].phi_correctness.iloc[0]), ".2f")
        put(f"{k}.labeldis", float(phi[phi.agent_a == "MEAN"].label_disagreement.iloc[0]), ".2f")
        put(f"{k}.majority", rules.loc["majority_vote", "accuracy"], ".3f")
        put(f"{k}.weighted", rules.loc["confidence_weighted_vote", "accuracy"], ".3f")
        put(f"{k}.bestsingle", singles.accuracy.max(), ".3f")
        put(f"{k}.oracle", rules.loc["oracle_any_agent", "accuracy"], ".3f")
        CONTROL_ROWS[(t, code)] = dict(phi=RAW[f"{k}.phi"], maj=RAW[f"{k}.majority"])


# ── baseline components and the consultation effect ─────────────────────────

for t in TASKS:
    bdf = load(PATHS[t]["A"])
    comp = section_model_accuracy(bdf, t).set_index("model")
    k = TASK_KEY[t]
    for m, key in [("cnn", "cnn"), ("biomedclip", "clip"), ("medgemma_initial", "init"),
                   ("pipeline_final", "final")]:
        put(f"{k}.comp.{key}", comp.loc[m, "accuracy"], ".3f")
    sh = section_medgemma_shift(bdf, t)
    put(f"{k}.shift.n", len(sh))
    put(f"{k}.shift.changed", int(sh.changed.sum()))
    put(f"{k}.shift.changedpct", sh.changed.mean(), "pct")
    put(f"{k}.shift.recovered", int(sh.recovered.sum()))
    put(f"{k}.shift.degraded", int(sh.degraded.sum()))
    put(f"{k}.shift.delta", sh.fin_correct.mean() - sh.init_correct.mean(), "pp")


# ── cost ──────────────────────────────────────────────────────────────────────

for t in TASKS:
    for code in "ABCH":
        recs = list(by_image_path(load_records(PATHS[t][code], verbose=False)[0]).values())
        lat = [r["latency_s"] for r in recs if not r.get("error") and r.get("latency_s")]
        put(f"{TASK_KEY[t]}.{code}.latency", float(np.median(lat)), ".1f")
for code in "ABCH":
    meds = [RAW[f"{TASK_KEY[t]}.{code}.latency"] for t in TASKS]
    ratios = [RAW[f"{TASK_KEY[t]}.{code}.latency"] / RAW[f"{TASK_KEY[t]}.A.latency"] for t in TASKS]
    put(f"{code}.latency.lo", min(meds), ".0f")
    put(f"{code}.latency.hi", max(meds), ".0f")
    put(f"{code}.ratio.lo", min(ratios), ".1f")
    put(f"{code}.ratio.hi", max(ratios), ".1f")
# the dedicated R=1 / R=3 debate runs, same images and GPU sharing as the others
for code, ok, path in [("D1", HAVE_D1, debate1_path), ("D3", HAVE_D3, debate3_path)]:
    if not ok:
        continue
    meds = []
    for t in TASKS:
        recs = list(by_image_path(load_records(path(t), verbose=False)[0]).values())
        meds.append(float(np.median([r["latency_s"] for r in recs if not r.get("error") and r.get("latency_s")])))
    put(f"{code}.latency.lo", min(meds), ".0f")
    put(f"{code}.latency.hi", max(meds), ".0f")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def cell(t, s, stage, field="acc"):
    return R[f"{TASK_KEY[t]}.{s}.{stage}.{field}"]


def tab_main() -> str:
    rows = []
    for s, name in [("A", "A (baseline)"), ("B", r"B ($R{=}2$)"), ("C", r"C ($N{=}4$)")]:
        cells = []
        for t in TASKS:
            mark = ""
            if s != "A":
                mark = stars(MAIN_P[(t, "final", s, "A")]["p_holm"])
            cells += [cell(t, s, "final") + mark, cell(t, s, "final", "conf"), cell(t, s, "final", "ece")]
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(rows) + "\n"


def tab_sensspec() -> str:
    out = []
    left, right = ["binary_tumor", "multiclass_tumor"], ["ms", "stroke"]
    names = {"binary_tumor": r"Bin.\ tumor", "multiclass_tumor": r"Multi.\ tumor", "ms": "MS", "stroke": "Stroke"}
    for lt, rt in zip(left, right):
        for i, s in enumerate("ABC"):
            def part(t):
                acc = f"{cell(t, s, 'final')} [{cell(t, s, 'final', 'lo')}, {cell(t, s, 'final', 'hi')}]"
                sens = R.get(f"{TASK_KEY[t]}.{s}.final.sens", "n/a")
                spec = R.get(f"{TASK_KEY[t]}.{s}.final.spec", "n/a")
                return f"{s} & {acc} & {sens} & {spec}"
            lhead = rf"\multirow{{3}}{{*}}{{{names[lt]}}}" if i == 0 else ""
            rhead = rf"\multirow{{3}}{{*}}{{{names[rt]}}}" if i == 0 else ""
            out.append(f"{lhead} & {part(lt)} & {rhead} & {part(rt)} \\\\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_mcnemar() -> str:
    names = {"binary_tumor": "Binary tumor", "multiclass_tumor": r"Multi.\ tumor", "ms": "MS", "stroke": "Stroke"}
    out = []
    for t in TASKS:
        rows = [r for r in MAIN if r["task"] == t]
        for i, r in enumerate(rows):
            stage = "final" if r["stage"] == "final" else "triage"
            better = r["a"] if r["a_only_correct"] > r["b_only_correct"] else r["b"]
            direction = "---" if r["p_holm"] >= 0.05 else rf"\system{better}{{}} better"
            head = rf"\multirow{{{len(rows)}}}{{*}}{{{names[t]}}}" if i == 0 else ""
            out.append(f"{head} & {r['a']} vs.\\ {r['b']} ({stage}) & "
                       f"{r['a_only_correct']}/{r['b_only_correct']} & {p_tex(r['p_holm'])} & {direction} \\\\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_transition() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        out.append(f"{TASK_TEX[t]} & {R[f'{k}.deb.nchanged']} & {R[f'{k}.deb.fixed']} & "
                   f"{R[f'{k}.deb.broke']} & ${RAW[f'{k}.deb.net']:+d}$ & {R[f'{k}.deb.ptex']} \\\\")
    return "\n".join(out) + "\n"


def tab_stability() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        out.append(" & ".join([TASK_TEX[t], R[f"{k}.deb.changedpct"],
                               R[f"{k}.deb.acc.stable"], R[f"{k}.deb.acc.changed"],
                               R[f"{k}.deb.ece.stable"], R[f"{k}.deb.ece.changed"],
                               R[f"{k}.deb.base.stable"], R[f"{k}.deb.base.changed"]]) + r" \\")
    return "\n".join(out) + "\n"


def tab_judge() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        for i, (key, name) in enumerate([("cnn", "CNN"), ("clip", "BiomedCLIP"), ("sam", "SAM3")]):
            head = rf"\multirow{{3}}{{*}}{{{TASK_TEX[t]}}}" if i == 0 else ""
            if f"{k}.{key}.follow" not in R:
                out.append(f"{head} & {name} & \\multicolumn{{4}}{{c}}{{not an advocate (SAM3 ineligible)}} \\\\")
                continue
            out.append(" & ".join([head, name, R[f"{k}.{key}.advacc"], R[f"{k}.{key}.follow"],
                                   R[f"{k}.{key}.accfollow"], R[f"{k}.{key}.accnotfollow"]]) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_voteshape() -> str:
    out = []
    for name, lab in [("40", "4--0"), ("31", "3--1"), ("22", "2--2")]:
        cells = []
        for t in TASKS:
            k = f"{TASK_KEY[t]}.vote{name}"
            cells.append(f"{R[k + '.acc']} [{R[k + '.lo']}, {R[k + '.hi']}]" if k + ".acc" in R else "---")
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    out.append(r"$n$ (4--0/3--1/2--2) & " + " & ".join(
        "/".join(R[f"{TASK_KEY[t]}.vote{n}.n"] for n in ("40", "31", "22")) for t in TASKS) + r" \\")
    out.append(r"trend $z$ & " + " & ".join(R[f"{TASK_KEY[t]}.vote.trendz"] for t in TASKS) + r" \\")
    return "\n".join(out) + "\n"


def tab_controls() -> str:
    out = []
    systems = [("C", r"C: roles, greedy"), ("H", r"H: one role, $T{=}0.7$")]
    if HAVE_R:
        systems.append(("R", r"R: roles, $T{=}0.7$"))
    for t in TASKS:
        for i, (code, name) in enumerate(systems):
            k = f"{TASK_KEY[t]}.{code}"
            head = rf"\multirow{{{len(systems)}}}{{*}}{{{TASK_TEX[t]}}}" if i == 0 else ""
            mark = ""
            if code != "C":
                mark = stars(CTRL_P[(t, "triage", code, "C")]["p_holm"])
            out.append(" & ".join([head, name, R[f"{TASK_KEY[t]}.{code}.triage.acc"] + mark,
                                   R[f"{k}.bestsingle"], R[f"{k}.oracle"], R[f"{k}.phi"],
                                   R[f"{k}.labeldis"]]) + r" \\")
        if not HAVE_R:
            out.append(r" & R: roles, $T{=}0.7$ & \multicolumn{5}{c}{\textit{run in progress}} \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_components() -> str:
    out = []
    for key, name in [("cnn", "CNN"), ("clip", "BiomedCLIP"), ("init", "MedGemma triage"),
                      ("final", r"Pipeline final (\systemA)")]:
        out.append(name + " & " + " & ".join(R[f"{TASK_KEY[t]}.comp.{key}"] for t in TASKS) + r" \\")
    return "\n".join(out) + "\n"


def tab_consultation() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        out.append(" & ".join([TASK_TEX[t], R[f"{k}.shift.n"],
                               f"{R[f'{k}.shift.changed']} ({R[f'{k}.shift.changedpct']}\\%)",
                               R[f"{k}.shift.recovered"], R[f"{k}.shift.degraded"],
                               R[f"{k}.shift.delta"] + " pp"]) + r" \\")
    return "\n".join(out) + "\n"


def tab_overhead() -> str:
    calls = {"A": "3", "C": "6", "B": "7--9", "H": "4"}
    names = {"A": r"A ($-$)", "C": r"C ($N{=}4$)", "B": r"B ($R{=}2$)", "H": r"H ($N{=}4$, triage only)"}
    out = []
    for code in "ACBH":
        ratio = (r"$1\times$" if code == "A"
                 else f"${R[code + '.ratio.lo']}$--${R[code + '.ratio.hi']}\\times$")
        out.append(f"{names[code]} & {calls[code]} & {R[code + '.latency.lo']}--{R[code + '.latency.hi']}\\,s & "
                   f"{ratio} \\\\")
    return "\n".join(out) + "\n"


# ── figure ────────────────────────────────────────────────────────────────────

def figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_A, C_B, C_C = "#2a78d6", "#eb6834", "#1baf7a"
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    labels = ["Binary", "Multi.", "MS", "Stroke"]
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "figure.dpi": 400, "savefig.dpi": 400,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.4, 2.85))
    for ax in (ax1, ax2, ax3):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # (a) end-to-end accuracy
    x, w = np.arange(4), 0.26
    chance = []
    for t in TASKS:
        tr = FRAMES[t]["true"]
        chance.append(100 * tr.value_counts(normalize=True).max())
    for i, (s, col, hatch) in enumerate([("A", C_A, ""), ("B", C_B, "///"), ("C", C_C, "...")]):
        vals = [100 * ACC[(t, s, "final")]["accuracy"] for t in TASKS]
        ax1.bar(x + (i - 1) * w, vals, w * 0.92, label=f"System {s}", color=col,
                edgecolor="white", linewidth=0.6, hatch=hatch)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax1.text(xi, v + 1.6, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2, color=MUTED, rotation=90)
    for xi, b in zip(x, chance):
        ax1.plot([xi - 1.5 * w, xi + 1.5 * w], [b, b], color=INK, lw=0.9, ls=(0, (2, 1.6)), zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("End-to-end accuracy (%)")
    ax1.set_ylim(0, 118)
    ax1.set_title("(a) Final accuracy", loc="left", fontweight="bold", pad=16)
    ax1.legend(frameon=False, ncol=3, columnspacing=0.8, handlelength=1.0, handletextpad=0.4,
               loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)

    # (b) debate round 1 vs round 2, against the baseline
    r1 = [100 * DEBATE[t]["r1"].mean() for t in TASKS]
    r2 = [100 * ACC[(t, "B", "final")]["accuracy"] for t in TASKS]
    base = [100 * ACC[(t, "A", "final")]["accuracy"] for t in TASKS]
    ax2.bar(x - 0.18, r1, 0.34, color="#f4a582", edgecolor="white", linewidth=0.6, label="B, round 1")
    ax2.bar(x + 0.18, r2, 0.34, color=C_B, edgecolor="white", linewidth=0.6, hatch="///",
            label="B, round 2 (final)")
    for xi, a, f in zip(x, r1, r2):
        ax2.text(xi - 0.18, a + 1.5, f"{a:.0f}", ha="center", fontsize=5.2, color=MUTED, rotation=90)
        ax2.text(xi + 0.18, f + 1.5, f"{f:.0f}", ha="center", fontsize=5.2, color=MUTED, rotation=90)
    for xi, b in zip(x, base):
        ax2.plot([xi - 0.4, xi + 0.4], [b, b], color=C_A, lw=1.1, zorder=5)
    ax2.plot([], [], color=C_A, lw=1.1, label="A (baseline)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("End-to-end accuracy (%)")
    ax2.set_ylim(0, 118)
    ax2.set_title("(b) Round 2 undoes round 1", loc="left", fontweight="bold", pad=16)
    ax2.legend(frameon=False, ncol=3, columnspacing=0.6, handlelength=1.0, handletextpad=0.4,
               loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)

    # (c) forest accuracy by vote shape
    shapes = [("40", "4–0"), ("31", "3–1"), ("22", "2–2")]
    for t, col, mk in zip(TASKS, [C_A, "#7b5ea7", C_C, C_B], ["o", "s", "^", "D"]):
        ys = [100 * RAW[f"{TASK_KEY[t]}.vote{n}.acc"] if f"{TASK_KEY[t]}.vote{n}.acc" in RAW else np.nan
              for n, _ in shapes]
        ax3.plot(range(3), ys, marker=mk, color=col, lw=1.1, ms=3.5, label=labels[TASKS.index(t)])
    ax3.set_xticks(range(3))
    ax3.set_xticklabels([s for _, s in shapes])
    ax3.set_xlabel("Forest vote shape (C)")
    ax3.set_ylabel("End-to-end accuracy (%)")
    ax3.set_ylim(0, 105)
    ax3.set_title("(c) Agreement is not reliability", loc="left", fontweight="bold", pad=16)
    ax3.legend(frameon=False, ncol=4, columnspacing=0.6, handlelength=1.2, handletextpad=0.3,
               loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(HERE / "main_result.png", bbox_inches="tight")


# ── write ─────────────────────────────────────────────────────────────────────

OUT.mkdir(exist_ok=True)
tables = {"tab_main": tab_main(), "tab_sensspec": tab_sensspec(), "tab_mcnemar": tab_mcnemar(),
          "tab_transition": tab_transition(), "tab_stability": tab_stability(), "tab_judge": tab_judge(),
          "tab_voteshape": tab_voteshape(), "tab_controls": tab_controls(),
          "tab_components": tab_components(), "tab_consultation": tab_consultation(),
          "tab_overhead": tab_overhead()}
for name, body in tables.items():
    (OUT / f"{name}.tex").write_text("% generated by eval/make_paper_results.py, do not edit\n" + body)

lines = ["% generated by eval/make_paper_results.py, do not edit",
         r"\makeatletter",
         r"\newcommand{\res}[1]{\@ifundefined{res@#1}{\textbf{??#1??}}{\@nameuse{res@#1}}}"]
for key, text in sorted(R.items()):
    lines.append(rf"\@namedef{{res@{key}}}{{{text}}}")
lines.append(r"\makeatother")
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
(OUT / "results.json").write_text(json.dumps(RAW, indent=1, default=float))
figure()
print(f"wrote {len(R)} numbers, {len(tables)} tables, main_result.png "
      f"(rolesamp {'included' if HAVE_R else 'pending'})")
