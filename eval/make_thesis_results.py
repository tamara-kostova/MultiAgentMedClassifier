"""Every v2 multi-agent number and table in the thesis (Master_Thesis/main.tex, Ch. 7-11).

    python eval/make_thesis_results.py                     (repo root, in .venv)

Runs eval/make_paper_results.py for the shared numbers (same scoring, same 16-test main
Holm family), then adds what only the thesis reports, and writes Master_Thesis/generated/:
    numbers.tex   \\res{key} macros (the paper's keys plus the thesis-only ones below)
    tab_*.tex     table bodies, \\input inside the table environments of main.tex
    results.json  the same numbers, for checking

Thesis-only:
    per-role accuracy / sensitivity / specificity for the greedy forest (C) and the
    role-sampled forest (R); pairwise phi ranges; unanimous-vs-split z-tests (Holm over 4);
    exact vote shapes (4-0, 3-1, 2-2, other: the paper's top-count bucket merges 2-2 with
    2-1-1); and a control family at triage: {H vs A, R vs A, H vs C, R vs C, R vs H} x 4
    tasks = 20 tests. A task whose rolesamp run is incomplete gets "pending" cells, and the
    control family is marked provisional until all 20 tests exist. Re-run when it finishes.
"""

from __future__ import annotations

import itertools
import json
import math
import runpy
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.ensemble_posthoc import correct, forest_analyses, forest_table, majority  # noqa: E402
from eval.forest_votes import ballots  # noqa: E402
from eval.jsonl_io import by_image_path, load_records  # noqa: E402
from eval.metrics import holm_bonferroni, mcnemar_exact, phi_coefficient, wilson_ci  # noqa: E402

THESIS = ROOT / "Master_Thesis"
OUT = THESIS / "generated"
PENDING = r"\textit{pending}"
ROLES = ["radiologist", "conservative", "emergency", "differential"]
ROLE_TEX = {"radiologist": "Radiologist", "conservative": "Conservative",
            "emergency": "Emergency", "differential": "Differential"}
TASK_NAME = {"binary_tumor": "Binary tumor", "multiclass_tumor": r"Multi.\ tumor", "ms": "MS",
             "stroke": "Stroke"}

with tempfile.TemporaryDirectory() as tmp:  # the paper's own tables and figure are not used here
    argv, sys.argv = sys.argv, [sys.argv[0], "--paper_dir", tmp, "--runs", "outputs/eval/v2"]
    P = runpy.run_path(str(ROOT / "eval" / "make_paper_results.py"))
    sys.argv = argv

R, RAW, put, p_tex, stars = P["R"], P["RAW"], P["put"], P["p_tex"], P["stars"]
TASKS, TASK_KEY, FRAMES, PATHS, MAIN_P = P["TASKS"], P["TASK_KEY"], P["FRAMES"], P["PATHS"], P["MAIN_P"]
HAVE_R = {t: "R" in PATHS[t] for t in TASKS}
put("thesis.rolesampTasks", ", ".join(TASK_NAME[t] for t in TASKS if HAVE_R[t]) or "none")


def records(t: str, code: str) -> list[dict]:
    return list(by_image_path(load_records(PATHS[t][code], verbose=False)[0]).values())


def z_two_prop(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided two-proportion z-test (pooled)."""
    n1, n2 = len(a), len(b)
    p = (a.sum() + b.sum()) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (a.mean() - b.mean()) / se if se else float("nan")
    return z, math.erfc(abs(z) / math.sqrt(2))


def ci_tex(hits: np.ndarray) -> str:
    lo, hi = wilson_ci(int(hits.sum()), len(hits))
    return rf"{{\scriptsize[{lo:.3f},{hi:.3f}]}}".replace("0.", ".")


# ── per-role behaviour and pairwise phi (C greedy roles, R sampled roles, H one role) ──

FT = {}
for t in TASKS:
    for code in "CHR":
        if code not in PATHS[t]:
            continue
        ft = forest_table(records(t, code), t)
        FT[(t, code)] = ft
        k = f"{TASK_KEY[t]}.{code}"
        truths = np.array(ft["truths"])
        pos = truths != "normal"
        best_acc, best_name = -1.0, ""
        for a in ft["agents"]:
            hits = np.array([correct(lbl, tr, t) for lbl, tr in zip(ft["labels"][a], ft["truths"])])
            name = ft["names"][a]
            role = ft["roles"][a]
            rk = f"{k}.role.{name.replace('#', '')}"
            put(f"{rk}.acc", hits.mean(), ".3f")
            R[f"{rk}.accci"] = ci_tex(hits)
            if t != "multiclass_tumor":
                put(f"{rk}.sens", hits[pos].mean(), ".3f")
                put(f"{rk}.spec", hits[~pos].mean(), ".3f")
                R[f"{rk}.sensci"] = ci_tex(hits[pos])
                R[f"{rk}.specci"] = ci_tex(hits[~pos])
            if hits.mean() > best_acc:
                best_acc, best_name = hits.mean(), role
        put(f"{k}.bestrole", ROLE_TEX.get(best_name, best_name))
        put(f"{k}.bestrole.short", ROLE_TEX.get(best_name, best_name)[:4].lower() + ".")
        phi = forest_analyses(ft, t)["forest_pairwise_phi"]
        pairs = phi[phi.agent_a != "MEAN"].phi_correctness
        put(f"{k}.phi.min", float(pairs.min()), ".2f")
        put(f"{k}.phi.max", float(pairs.max()), ".2f")
        # mean of the four members' accuracies: what a vote over independent errors should beat
        put(f"{k}.meanmember", float(np.mean([
            np.mean([correct(lbl, tr, t) for lbl, tr in zip(ft["labels"][a], ft["truths"])])
            for a in ft["agents"]])), ".3f")
        # phi within each true class: the pooled phi also counts the agreement that comes from
        # every member finding the same class easy or hard
        hits_by = {a: np.array([correct(lbl, tr, t) for lbl, tr in zip(ft["labels"][a], ft["truths"])])
                   for a in ft["agents"]}
        for cls in sorted(set(truths)):
            m = truths == cls
            vals = [phi_coefficient(hits_by[a][m], hits_by[b][m])
                    for a, b in itertools.combinations(ft["agents"], 2)]
            put(f"{k}.phi.class.{cls}", float(np.nanmean(vals)), ".2f")
            R[f"{k}.phi.class.{cls}"] = R[f"{k}.phi.class.{cls}"].replace("-", "$-$")
        # majority-vote accuracy if members erred independently given the true class: the product
        # of each member's empirical label distribution within the class, same vote and tie rule
        indep = 0.0
        for cls in sorted(set(truths)):
            idx = np.flatnonzero(truths == cls)
            dists = [list(Counter(ft["labels"][a][i] for i in idx).items()) for a in ft["agents"]]
            p_cls = sum(np.prod([c / len(idx) for _, c in combo]) * correct(majority([l for l, _ in combo])[0], cls, t)
                        for combo in itertools.product(*dists))
            indep += p_cls * len(idx) / len(truths)
        put(f"{k}.indep", indep, ".3f")

# Uncorrected two-proportion z-tests for the role contrasts the text discusses
# (C: does a role prompt move sensitivity/specificity the way it says it should?).
ROLE_TESTS = [("binary_tumor", "spec", "radiologist", "emergency"),
              ("binary_tumor", "spec", "radiologist", "conservative"),
              ("ms", "sens", "emergency", "radiologist"),
              ("stroke", "sens", "emergency", "radiologist")]
for t, metric, r1, r2 in ROLE_TESTS:
    ft = FT[(t, "C")]
    by_role = {ft["roles"][a]: a for a in ft["agents"]}
    truths = np.array(ft["truths"])
    mask = (truths != "normal") if metric == "sens" else (truths == "normal")
    hits = {r: np.array([correct(l, tr, t) for l, tr in zip(ft["labels"][by_role[r]], ft["truths"])])[mask]
            for r in (r1, r2)}
    z, p = z_two_prop(hits[r1], hits[r2])
    k = f"{TASK_KEY[t]}.C.z.{metric}.{r1}.{r2}"
    put(f"{k}.z", z, "+.2f")
    RAW[f"{k}.p"] = p
    R[f"{k}.ptex"] = p_tex(p)
    put(f"{TASK_KEY[t]}.n{metric}", int(mask.sum()))

phis = [RAW[f"{TASK_KEY[t]}.C.phi"] for t in TASKS]
put("thesis.C.phi.lo", min(phis), ".2f")
put("thesis.C.phi.hi", max(phis), ".2f")


# ── forest vote shapes and agreement as an escalation signal (C, against the final output) ──

def shape_of(votes, task: str) -> str:
    counts = sorted(__import__("collections").Counter(
        b["label"] for b in ballots(votes, task) if b["label"]).values(), reverse=True)
    if counts[:1] == [4]:
        return "40"
    if counts[:1] == [3]:
        return "31"
    if counts[:2] == [2, 2]:
        return "22"
    return "other"


UNSPLIT = []
for t in TASKS:
    recs = by_image_path(load_records(PATHS[t]["C"], verbose=False)[0])
    df = FRAMES[t]
    shapes, dissent = [], []
    for img in df["image_path"]:
        votes = recs[img].get("forest_votes") or []
        shapes.append(shape_of(votes, t) if votes else "other")
        labels = [b["label"] for b in ballots(votes, t)]
        top = max((labels.count(l) for l in set(labels) if l), default=0)
        dissent.append(1 - top / len(labels) if labels else 1.0)
    shapes = np.array(shapes)
    fin = df["C_final_correct"].values.astype(int)
    k = TASK_KEY[t]
    put(f"{k}.vote.dissent", float(np.mean(dissent)), ".3f")
    for name in ["40", "31", "22", "other"]:
        m = shapes == name
        put(f"{k}.shape{name}.pct", m.mean(), "pct")
        if m.any():
            put(f"{k}.shape{name}.acc", fin[m].mean(), ".3f")
        else:
            R[f"{k}.shape{name}.acc"] = "---"
    a31, a22 = fin[shapes == "31"], fin[shapes == "22"]
    if len(a31) and len(a22):
        _, p = z_two_prop(a31, a22)
        RAW[f"{k}.shape.p3122"] = p
        R[f"{k}.shape.p3122"] = p_tex(p)
    unan, split = fin[shapes == "40"], fin[shapes != "40"]
    z, p = z_two_prop(unan, split)
    UNSPLIT.append((t, z, p))
for (t, z, p), adj in zip(UNSPLIT, holm_bonferroni([u[2] for u in UNSPLIT])):
    k = TASK_KEY[t]
    put(f"{k}.vote.z", z, "+.2f")
    R[f"{k}.vote.ptex"] = p_tex(adj) + stars(adj)
    RAW[f"{k}.vote.p"] = adj


# ── control family at triage: sampled forests against the baseline, the greedy forest, each other ──

CTRL_TESTS = [("H", "A"), ("R", "A"), ("H", "C"), ("R", "C"), ("R", "H")]
ctrl = []
for t in TASKS:
    for a, b in CTRL_TESTS:
        if a not in PATHS[t] or b not in PATHS[t]:
            continue
        df = FRAMES[t]
        res = mcnemar_exact(df[f"{a}_triage_correct"].values, df[f"{b}_triage_correct"].values)
        ctrl.append({"task": t, "a": a, "b": b, **res})
for row, adj in zip(ctrl, holm_bonferroni([r["p_raw"] for r in ctrl])):
    k = f"{TASK_KEY[row['task']]}.ctrl.{row['a']}vs{row['b']}"
    put(f"{k}.aonly", row["a_only_correct"])
    put(f"{k}.bonly", row["b_only_correct"])
    R[f"{k}.ptex"] = p_tex(adj) + stars(adj)
    RAW[f"{k}.p"] = adj
CTRL_P = {(r["task"], r["a"], r["b"]): r for r in ctrl}
put("thesis.ctrl.size", len(ctrl))
put("thesis.ctrl.planned", len(TASKS) * len(CTRL_TESTS))
R["thesis.ctrl.status"] = ("" if len(ctrl) == len(TASKS) * len(CTRL_TESTS) else
                           rf" \emph{{Provisional:}} {len(ctrl)} of the {len(TASKS) * len(CTRL_TESTS)} "
                           r"planned tests are available while the tumor rolesamp runs finish, so these "
                           r"$p$-values will move slightly.")


# ── dedicated R=1 debate: confirms the post-hoc round-1 reading of the R=2 runs ──

def d1_sentence() -> str:
    """Plain-language summary of the confirmatory R=1 run; directions and significance come from the data."""
    keys = [TASK_KEY[t] for t in TASKS]
    agree = [RAW[f"{k}.D1.agree"] for k in keys]
    if all(a == 1.0 for a in agree):
        first = r"returns exactly the saved round-1 verdict on every image of all four tasks"
    else:
        first = (r"agrees with the saved round-1 verdict on " + ", ".join(R[f"{k}.D1.agree"] + r"\%" for k in keys)
                 + r" of images")
    names = {"bin": "binary tumor", "multi": "multiclass tumor", "ms": "MS", "stroke": "stroke"}

    def vs(k, other, label):
        d, o = RAW[f"{k}.D1.final.acc"], RAW[f"{k}.{other}.final.acc"]
        sig = RAW[f"{k}.final.D1vs{other}.p"] < 0.05
        word = ("significantly " if sig else "") + ("above" if d > o else "below")
        pt = R[f"{k}.final.D1vs{other}.ptexs"]
        return f"{word} {label}, " + (r"$p$\," if pt.startswith("$<") else r"$p={}$\,") + pt

    parts = [f"{names[k]} " + R[f"{k}.D1.final.acc"] + r"\% (" + vs(k, "A", r"\Base") + "; "
             + vs(k, "C", r"\Forest") + ")" for k in keys]
    return (r"The $R=1$ run, made after this analysis as its confirmatory test on the same images, " + first
            + r". With Holm correction over its 8-test family, its accuracy is " + ", ".join(parts[:-1])
            + ", and " + parts[-1] + ".")


HAVE_D1 = P["HAVE_D1"]   # the confirmatory R=1 family is computed in make_paper_results.py
if HAVE_D1:
    R["thesis.d1.sentence"] = d1_sentence()
else:
    R["thesis.d1.sentence"] = (r"A dedicated $R=1$ run on the same images, the confirmatory test, is in "
                               r"progress; on its first images it returns exactly the saved round-1 verdicts.")


# ── dedicated R=3 debate: does a third round undo round 2? ──

def d3_sentence() -> str:
    """Plain-language summary of the R=3 run; directions and significance come from the data."""
    keys = [TASK_KEY[t] for t in TASKS]
    names = {"bin": "binary tumor", "multi": "multiclass tumor", "ms": "MS", "stroke": "stroke"}
    if RAW["thesis.d3.allprefix"]:
        first = r"Its first two rounds reproduce the $R=2$ verdicts on every image of all four tasks."
    else:
        first = (r"Its first two rounds match the $R=2$ verdicts on "
                 + ", ".join(R[f"{k}.D3.prefix"] + r"\%" for k in keys) + r" of images.")

    def part(k):
        sig = RAW[f"{k}.final.D3vsB.p"] < 0.05
        d, b = RAW[f"{k}.D3.final.acc"], RAW[f"{k}.B.final.acc"]
        word = ("significantly " if sig else "") + ("above" if d > b else "below")
        pt = R[f"{k}.final.D3vsB.ptexs"]
        return (f"{names[k]} " + R[f"{k}.D3.final.acc"] + r"\% (" + f"{word} $R=2$, "
                + (r"$p$\," if pt.startswith("$<") else r"$p={}$\,") + pt + ")")

    parts = [part(k) for k in keys]
    below_a = all(RAW[f"{k}.D3.final.acc"] < RAW[f"{k}.A.final.acc"] and RAW[f"{k}.final.D3vsA.p"] < 0.05
                  for k in keys)
    back = sum(RAW[f"{k}.D3.r3back"] for k in keys)
    changed = sum(RAW[f"{k}.D3.r3changed"] for k in keys)
    return (r"The $R=3$ run tests whether a third round undoes the second. " + first
            + r" The third round changes " + ", ".join(R[f"{k}.D3.r3changed"] for k in keys[:-1])
            + " and " + R[f"{keys[-1]}.D3.r3changed"] + r" verdicts on the four tasks, and "
            + f"{back} of these {changed} changes return to the round-1 label. "
            + r"With Holm correction over its 8-test family, the $R=3$ accuracy is " + ", ".join(parts[:-1])
            + ", and " + parts[-1] + "."
            + (r" On every task it stays significantly below \Base." if below_a else ""))


HAVE_D3 = P["HAVE_D3"]   # the R=3 family is computed in make_paper_results.py
if HAVE_D3:
    R["thesis.d3.sentence"] = d3_sentence()
else:
    R["thesis.d3.sentence"] = r"The $R=3$ run is in progress."


# ── macro-F1 on multiclass tumor (over the three ground-truth classes; a prediction outside them
#    counts only as a miss) and the human-review flag rate, per system and stage ──

def macro_f1(preds, truths, classes) -> float:
    f1s = []
    for c in classes:
        tp = sum(1 for p, t in zip(preds, truths) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, truths) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, truths) if p != c and t == c)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


_df = FRAMES["multiclass_tumor"]
_classes = sorted(set(_df["true"]))
for code in "ABC":
    for stage in ("triage", "final"):
        put(f"multi.{code}.{stage}.macrof1", macro_f1(_df[f"{code}_{stage}_pred"].tolist(), _df["true"].tolist(), _classes), ".3f")

for t in TASKS:
    for code in "ABC":
        recs = records(t, code)
        put(f"{TASK_KEY[t]}.{code}.review.n", sum(bool(r.get("requires_human_review")) for r in recs))
        put(f"{TASK_KEY[t]}.{code}.review.pct", np.mean([bool(r.get("requires_human_review")) for r in recs]), "pct")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def r_(key: str) -> str:
    return R.get(key, PENDING)


def acc_ci(t, code, stage):
    k = f"{TASK_KEY[t]}.{code}.{stage}"
    if f"{k}.acc" not in RAW:
        return PENDING
    return (f"{RAW[k + '.acc']:.3f} {{\\scriptsize[{RAW[k + '.lo']:.3f},{RAW[k + '.hi']:.3f}]}}"
            .replace("[0.", "[.").replace(",0.", ",."))


def main_p(t, stage, a, b):
    row = MAIN_P[(t, stage, a, b)]
    return p_tex(row["p_holm"]) + stars(row["p_holm"])


def tab_comparison() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        best = f"{RAW[k + '.C.bestsingle']:.3f} ({R[k + '.C.bestrole.short']})"
        out.append(" & ".join([TASK_NAME[t], acc_ci(t, "A", "triage"), acc_ci(t, "C", "triage"), best,
                               main_p(t, "triage", "C", "A"), acc_ci(t, "A", "final"),
                               acc_ci(t, "C", "final"), main_p(t, "final", "C", "A")]) + r" \\")
    return "\n".join(out) + "\n"


def tab_phicond() -> str:
    """Pooled vs within-class phi, and the vote against its value under conditional independence."""
    sys_tex = {"C": r"\Forest", "H": r"\Homog", "R": r"\Rolesamp"}
    out = []
    for t in TASKS:
        rows = [c for c in "CHR" if f"{TASK_KEY[t]}.{c}.indep" in R]
        for j, code in enumerate(rows):
            k = f"{TASK_KEY[t]}.{code}"
            within = sorted(key for key in RAW if key.startswith(f"{k}.phi.class."))
            normal = [R[key] for key in within if key.endswith(".normal")]
            abnormal = [RAW[key] for key in within if not key.endswith(".normal")]
            ab = f"{abnormal[0]:.2f}" if len(abnormal) == 1 else f"{min(abnormal):.2f}--{max(abnormal):.2f}"
            name = rf"\multirow{{{len(rows)}}}{{*}}{{{TASK_NAME[t]}}}" if j == 0 else ""
            out.append(" & ".join([name, sys_tex[code], R[f"{k}.phi"], normal[0] if normal else "---", ab,
                                   R[f"{k}.meanmember"], R[f"{k}.majority"], R[f"{k}.indep"]]) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_phicorr() -> str:
    out = []
    for t in TASKS:
        cells = []
        for code in "CHR":
            k = f"{TASK_KEY[t]}.{code}"
            if f"{k}.phi" in R:
                cells.append(f"{R[k + '.phi']} & {R[k + '.phi.min']}--{R[k + '.phi.max']}")
            else:
                cells.append(rf"\multicolumn{{2}}{{c}}{{{PENDING}}}")
        out.append(TASK_NAME[t] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def tab_roles(code: str) -> str:
    out = []
    for t in ["binary_tumor", "ms", "stroke"]:
        k = f"{TASK_KEY[t]}.{code}"
        head = TASK_NAME[t].replace(" ", r"\\")
        for i, role in enumerate(ROLES):
            h = rf"\multirow{{4}}{{*}}{{\makecell[l]{{{head}}}}}" if i == 0 else ""
            rk = f"{k}.role.{role}"
            if f"{rk}.acc" not in R:
                out.append(f"{h} & {ROLE_TEX[role]} & \\multicolumn{{3}}{{c}}{{{PENDING}}} \\\\")
                continue
            out.append(" & ".join([h, ROLE_TEX[role], R[f"{rk}.acc"], f"{R[rk + '.sens']} {R[rk + '.sensci']}",
                                   f"{R[rk + '.spec']} {R[rk + '.specci']}"]) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_voting() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        out.append(" & ".join([TASK_NAME[t], R[f"{k}.vote.dissent"], R[f"{k}.shape40.pct"],
                               R[f"{k}.shape40.acc"], R[f"{k}.vote.acc.split"], R[f"{k}.vote.ptex"]]) + r" \\")
    return "\n".join(out) + "\n"


def tab_splittype() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        cells = []
        for s in ["31", "22", "other"]:
            cells += [R[f"{k}.shape{s}.pct"], R[f"{k}.shape{s}.acc"]]
        out.append(TASK_NAME[t] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def tab_debate_main() -> str:
    out = []
    for t in TASKS:
        for i, (code, name) in enumerate([("A", r"\Base"), ("B", r"\Debate"), ("C", r"\Forest")]):
            k = f"{TASK_KEY[t]}.{code}"
            h = rf"\multirow{{3}}{{*}}{{{TASK_NAME[t]}}}" if i == 0 else ""
            out.append(" & ".join([h, name, f"{RAW[k + '.triage.acc']:.3f}", f"{RAW[k + '.final.acc']:.3f}",
                                   R[k + ".triage.ece"], R[k + ".final.ece"], R[k + ".final.conf"],
                                   R[k + ".review.pct"]]) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_macrof1() -> str:
    out = []
    for code, name in [("A", r"\Base"), ("C", r"\Forest"), ("B", r"\Debate")]:
        out.append(" & ".join([name, f"{RAW['multi.' + code + '.triage.acc']:.3f}", R[f"multi.{code}.triage.macrof1"],
                               f"{RAW['multi.' + code + '.final.acc']:.3f}", R[f"multi.{code}.final.macrof1"]]) + r" \\")
    return "\n".join(out) + "\n"


def tab_debate_comparison() -> str:
    out = []
    for t in TASKS:
        cells = []
        for b in "AC":
            row = MAIN_P[(t, "final", "B", b)]
            cells.append(f"{row['a_only_correct']}/{row['b_only_correct']} & {main_p(t, 'final', 'B', b)}")
        row = MAIN_P[(t, "final", "C", "A")]
        cells.append(f"{row['a_only_correct']}/{row['b_only_correct']} & {main_p(t, 'final', 'C', 'A')}")
        out.append(TASK_NAME[t] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def tab_debate_rounds() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        out.append(" & ".join([TASK_NAME[t], R[f"{k}.deb.r1acc"], f"{RAW[k + '.B.final.acc'] * 100:.1f}",
                               R[f"{k}.deb.changedpct"], R[f"{k}.deb.fixed"], R[f"{k}.deb.broke"],
                               R[f"{k}.deb.ptex"], R[f"{k}.deb.acc.stable"], R[f"{k}.deb.acc.changed"],
                               R[f"{k}.deb.ece.stable"], R[f"{k}.deb.ece.changed"]]) + r" \\")
    return "\n".join(out) + "\n"


def tab_debate_r3() -> str:
    """Accuracy by number of rounds (R=1, R=2, R=3 runs on the same images) and round-3 transitions."""
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        if not HAVE_D3:
            out.append(rf"{TASK_NAME[t]} & \multicolumn{{8}}{{c}}{{{PENDING}}} \\")
            continue
        out.append(" & ".join([TASK_NAME[t], R[f"{k}.A.final.acc"], r_(f"{k}.D1.final.acc"), R[f"{k}.B.final.acc"],
                               R[f"{k}.D3.final.acc"], R[f"{k}.D3.r3changed"], R[f"{k}.D3.r3back"],
                               f"{R[k + '.D3.r3fixed']}/{R[k + '.D3.r3broke']}",
                               R[f"{k}.final.D3vsB.ptexs"]]) + r" \\")
    return "\n".join(out) + "\n"


def tab_judge() -> str:
    out = []
    for t in TASKS:
        k = TASK_KEY[t]
        for i, (key, name) in enumerate([("cnn", "CNN"), ("clip", "BiomedCLIP"), ("sam", "SAM3")]):
            h = rf"\multirow{{3}}{{*}}{{{TASK_NAME[t]}}}" if i == 0 else ""
            if f"{k}.{key}.follow" not in R:
                out.append(rf"{h} & {name} & \multicolumn{{5}}{{c}}{{\textit{{not an advocate (probe ineligible)}}}} \\")
                continue
            out.append(" & ".join([h, name, R[f"{k}.{key}.advacc"], R[f"{k}.{key}.claimrate"],
                                   R[f"{k}.{key}.follow"], R[f"{k}.{key}.accfollow"],
                                   R[f"{k}.{key}.accnotfollow"]]) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_controls() -> str:
    out = []
    names = [("A", r"\Base (1 agent, greedy)"), ("C", r"\Forest (4 roles, greedy)"),
             ("H", r"\Homog (1 role, $T{=}0.7$)"), ("R", r"\Rolesamp (4 roles, $T{=}0.7$)")]
    for t in TASKS:
        for i, (code, name) in enumerate(names):
            k = f"{TASK_KEY[t]}.{code}"
            h = rf"\multirow{{4}}{{*}}{{{TASK_NAME[t]}}}" if i == 0 else ""
            if code not in PATHS[t]:
                out.append(rf"{h} & {name} & \multicolumn{{6}}{{c}}{{{PENDING}}} \\")
                continue
            acc = f"{RAW[k + '.triage.acc']:.3f}"
            if code == "A":
                cells = [acc, r_(k + ".triage.sens"), r_(k + ".triage.spec"), "---", "---", "---"]
            else:
                cells = [acc, r_(k + ".triage.sens"), r_(k + ".triage.spec"), R[k + ".bestsingle"],
                         R[k + ".oracle"], R[k + ".phi"]]
            if t == "multiclass_tumor":
                cells[1] = cells[2] = "---"
            out.append(" & ".join([h, name] + cells) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_controls_tests() -> str:
    out = []
    for t in TASKS:
        cells = []
        for a, b in CTRL_TESTS:
            k = f"{TASK_KEY[t]}.ctrl.{a}vs{b}"
            cells.append(f"{R[k + '.aonly']}/{R[k + '.bonly']} & {R[k + '.ptex']}" if k + ".ptex" in R
                         else rf"\multicolumn{{2}}{{c}}{{{PENDING}}}")
        out.append(TASK_NAME[t] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def tab_synthesis() -> str:
    out = []
    for code, name in [("A", r"\Base"), ("C", r"\Forest"), ("B", r"\Debate")]:
        out.append(name + " & " + " & ".join(f"{RAW[f'{TASK_KEY[t]}.{code}.final.acc']:.3f}" for t in TASKS)
                   + r" \\")
    return "\n".join(out) + "\n"


# ── figure: accuracy by system, triage and end-to-end ─────────────────────────

def figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    color = {"A": "#2a78d6", "C": "#eb6834", "H": "#1baf7a", "R": "#eda100", "B": "#5f5e5a"}
    hatch = {"A": "", "C": "///", "H": "...", "R": "xx", "B": "\\\\"}
    label = {"A": "Baseline", "C": "Forest (roles, greedy)", "H": "Homog. (1 role, T=0.7)",
             "R": "Role-sampled (T=0.7)", "B": "Debate (R=2)"}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    x = np.arange(len(TASKS))
    for ax, stage, systems, title in [(axes[0], "triage", "ACHR", "(a) Triage stage"),
                                      (axes[1], "final", "ACB", "(b) End-to-end (final)")]:
        w = 0.8 / len(systems)
        for i, s in enumerate(systems):
            vals = [100 * RAW[f"{TASK_KEY[t]}.{s}.{stage}.acc"] if f"{TASK_KEY[t]}.{s}.{stage}.acc" in RAW
                    else np.nan for t in TASKS]
            xs = x + (i - (len(systems) - 1) / 2) * w
            ax.bar(xs, vals, w * 0.9, color=color[s], hatch=hatch[s], edgecolor="white", linewidth=0.6,
                   label=label[s])
            for xi, v in zip(xs, vals):
                ax.text(xi, (v if v == v else 0) + 1.5, f"{v:.0f}" if v == v else "pending", ha="center",
                        va="bottom", fontsize=5.5, color=MUTED, rotation=90)
        for xi, t in zip(x, TASKS):
            chance = 100 * RAW[f"{TASK_KEY[t]}.majority"]
            ax.plot([xi - 0.42, xi + 0.42], [chance, chance], color=INK, lw=0.8, ls=(0, (2, 1.5)))
        ax.set_xticks(x)
        ax.set_xticklabels(["Binary", "Multiclass", "MS", "Stroke"])
        ax.set_ylim(0, 112)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(THESIS / "v2_system_comparison.png", bbox_inches="tight")


# ── write ─────────────────────────────────────────────────────────────────────

figure()
OUT.mkdir(exist_ok=True)
tables = {"tab_comparison": tab_comparison(), "tab_phicorr": tab_phicorr(), "tab_phicond": tab_phicond(), "tab_roles": tab_roles("C"),
          "tab_roles_sampled": tab_roles("R"), "tab_voting": tab_voting(), "tab_splittype": tab_splittype(),
          "tab_debate_main": tab_debate_main(), "tab_macrof1": tab_macrof1(), "tab_debate_comparison": tab_debate_comparison(),
          "tab_debate_rounds": tab_debate_rounds(), "tab_debate_r3": tab_debate_r3(), "tab_judge": tab_judge(), "tab_controls": tab_controls(),
          "tab_controls_tests": tab_controls_tests(), "tab_synthesis": tab_synthesis()}
for name, body in tables.items():
    (OUT / f"{name}.tex").write_text("% generated by eval/make_thesis_results.py, do not edit\n" + body)

lines = ["% generated by eval/make_thesis_results.py, do not edit",
         r"\makeatletter",
         r"\providecommand{\res}[1]{\@ifundefined{res@#1}{\textit{pending}}{\@nameuse{res@#1}}}"]
lines += [rf"\@namedef{{res@{key}}}{{{text}}}" for key, text in sorted(R.items())]
lines.append(r"\makeatother")
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
(OUT / "results.json").write_text(json.dumps(RAW, indent=1, default=float))
print(f"wrote {len(R)} numbers, {len(tables)} tables to {OUT.relative_to(ROOT)} "
      f"(rolesamp complete for: {R['thesis.rolesampTasks']}; control family {len(ctrl)}/"
      f"{len(TASKS) * len(CTRL_TESTS)} tests)")
