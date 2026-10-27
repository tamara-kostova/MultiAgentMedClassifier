"""
Post-hoc ensemble analyses on logged Forest votes and Debate records (no inference).

Reviewer requests answered from the existing JSONLs:

Forest JSONLs (records carry `forest_votes`):
  (a) triage accuracy of the majority vote vs. a confidence-weighted vote vs. each
      single agent vs. the oracle (any agent correct);
  (b) majority-vote accuracy of every agent subset of size >= 2 (11 subsets for 4
      roles), with Wilson 95% CIs;
  (c) pairwise phi (Matthews) correlation of per-image correctness between agents,
      and its mean — the "are the agents diverse at all?" question;
  (d) when roles repeat (homogeneous / sampled forests whose votes carry
      agent_idx / role / temperature): within-role vs. between-role disagreement;
  (e) lenient-scoring sensitivity: strict vs. `abnormal` scoring on binary tasks;
      exact subtype vs. "any tumor label counts" on multiclass_tumor;
  (f) two or more forest JSONLs of the same task: McNemar exact on majority-vote
      triage correctness, restricted to their identical image_paths.

Debate JSONLs (records carry `debate_rounds_completed`):
  * per-advocate claim distribution. An advocate's position is recovered from the
    tool output it was told to defend (CNN: cnn_predicted_class; BiomedCLIP:
    biomedclip_top_label; SAM3: "lesion detected" iff not skipped and a bbox
    exists, "not applicable" when the probe is ineligible). Newer files that log
    `debate_arguments` additionally get a keyword reading of the argument text
    (heuristic; reported separately);
  * pairwise phi between advocates' positive-claim indicators and correctness;
  * judge-follows-advocate rates (final verdict == advocate's implied label), and
    the judge's own "most compelling" attributions (eval.judge_attribution).

Scoring: eval.labels.canonical_label + eval_analysis.prep (strict); abstentions and
error-only records count as wrong. Ballots follow eval.forest_votes (multiclass:
diagnosis_detailed, falling back to diagnosis_name; empty ballot = abstain vote;
ties by list order; agents keyed by agent_idx/position, never role).

Usage:
    python eval/ensemble_posthoc.py --jsonl outputs/eval/ms_forest_n4.jsonl
    python eval/ensemble_posthoc.py --jsonl outputs/eval/*_forest_n4.jsonl outputs/eval/*_debate_r2.jsonl
    python eval/ensemble_posthoc.py --jsonl a_forest.jsonl b_forest_homog.jsonl   # adds (f)

Outputs: <out_dir>/<stem>/*.csv (default out_dir: outputs/analysis/, i.e.
outputs/analysis/<stem>_posthoc/), and <out_dir>/posthoc_comparisons/ for (f).
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys as _sys
from pathlib import Path

if __package__ in (None, ""):  # run as `python eval/ensemble_posthoc.py`
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from eval.eval_analysis import ERROR_PRED, is_multiclass, pos_class, prep
from eval.forest_votes import ballots, majority, weighted_vote
from eval.jsonl_io import ERROR_FLAG, by_image_path, load_records
from eval.labels import TUMOR_SUBTYPES, canonical_label
from eval.metrics import mcnemar_exact, phi_coefficient, wilson_ci

_TUMORISH = set(TUMOR_SUBTYPES) | {"pituitary_tumor", "tumor"}
MAX_SUBSET_AGENTS = 10


# ── scoring helpers ────────────────────────────────────────────────────────────

def _true(rec: dict, task: str) -> str:
    v = rec.get("true_label_canonical") or rec.get("true_label_name") or rec.get("true_label") or ""
    return canonical_label(str(v), task)


def _scored(label: str, task: str, scoring: str = "strict") -> str:
    if label == ERROR_PRED:
        return ERROR_PRED
    return prep(label, task, scoring) if label else "unknown"


def correct(pred_label: str, true_label: str, task: str, scoring: str = "strict") -> int:
    """1/0 on canonical labels. scoring: strict | abnormal (binary) | any_tumor (multiclass)."""
    if pred_label in ("", ERROR_PRED):
        return 0
    if scoring == "any_tumor":
        p, t = canonical_label(pred_label, task), canonical_label(true_label, task)
        if t in _TUMORISH:
            return int(p in _TUMORISH)
        return int(p == t)
    p = _scored(pred_label, task, scoring)
    t = _scored(true_label, task, scoring)
    return int(p not in ("unknown", ERROR_PRED) and p == t)


def _acc_row(name: str, hits: list[int], **extra) -> dict:
    n, k = len(hits), int(sum(hits))
    lo, hi = wilson_ci(k, n)
    return {"method": name, "n": n, "correct": k,
            "accuracy": round(k / n, 4) if n else float("nan"),
            "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4), **extra}


def _infer_task(records: list[dict], cli_task: str | None) -> str:
    if cli_task:
        return cli_task
    tasks = {r.get("task") for r in records if r.get("task")}
    if len(tasks) != 1:
        raise SystemExit(f"cannot infer task (found {tasks}); pass --task")
    return tasks.pop()


def _kind(records: list[dict]) -> str:
    n = max(1, len(records))
    if sum(1 for r in records if isinstance(r.get("forest_votes"), list) and r["forest_votes"]) > n / 2:
        return "forest"
    if sum(1 for r in records if r.get("debate_rounds_completed") is not None) > n / 2:
        return "debate"
    return "other"


# ── forest ─────────────────────────────────────────────────────────────────────

def forest_table(records: list[dict], task: str) -> dict:
    """Per-image ballots in a common agent space."""
    imgs, truths, per_agent, conf_votes, roles, temps = [], [], {}, [], {}, {}
    for r in records:
        if r.get(ERROR_FLAG):
            bs = []
        else:
            bs = ballots(r.get("forest_votes"), task)
        imgs.append(r.get("image_path"))
        truths.append(_true(r, task))
        conf_votes.append(r.get("forest_votes") if not r.get(ERROR_FLAG) else [])
        for b in bs:
            roles.setdefault(b["agent"], b["role"])
            if b.get("temperature") is not None:
                temps.setdefault(b["agent"], b["temperature"])
        per_agent_img = {b["agent"]: b["label"] for b in bs}
        per_agent.setdefault("__rows__", []).append(per_agent_img)
        per_agent.setdefault("__confs__", []).append({b["agent"]: b["conf"] for b in bs})
    agents = sorted(roles)
    rows = per_agent.get("__rows__", [])
    labels = {a: [row.get(a, "") for row in rows] for a in agents}
    conf_rows = per_agent.get("__confs__", [])
    confs = {a: [row.get(a) for row in conf_rows] for a in agents}
    unique = len(set(roles.values())) == len(roles)
    names = {a: (roles[a] if unique else f"{roles[a]}#{a}") for a in agents}
    return {"images": imgs, "truths": truths, "agents": agents, "roles": roles, "names": names,
            "labels": labels, "confs": confs, "raw_votes": conf_votes, "temps": temps,
            "error": [bool(r.get(ERROR_FLAG)) for r in records],
            "final": [ERROR_PRED if r.get(ERROR_FLAG) else canonical_label(r.get("predicted_class") or "", task)
                      for r in records]}


def _majority_labels(ft: dict, agents: list) -> list[str]:
    """Majority over the given agents per image; ties by the agents' list order."""
    out = []
    for i in range(len(ft["images"])):
        if ft["error"][i]:
            out.append(ERROR_PRED)
            continue
        out.append(majority([ft["labels"][a][i] for a in agents])[0])
    return out


def _subset_weighted_and_ties(ft: dict, agents: list, default_conf: float = 0.5):
    """Confidence-weighted labels over a subset (ties by list order) and per-image tie flags.

    With an even subset a split vote is a tie, and the published rule breaks ties by
    list order, so a 2-agent majority is just its first agent. The weighted vote and
    the tie rate make the pair rows interpretable.
    """
    labels, ties = [], []
    for i in range(len(ft["images"])):
        if ft["error"][i]:
            labels.append(ERROR_PRED)
            ties.append(False)
            continue
        weights, order, counts = {}, [], {}
        for a in agents:
            l = ft["labels"][a][i]
            if not l:
                continue
            c = ft["confs"][a][i]
            weights[l] = weights.get(l, 0.0) + (c if c is not None else default_conf)
            counts[l] = counts.get(l, 0) + 1
            if l not in order:
                order.append(l)
        if not weights:
            labels.append("")
            ties.append(False)
            continue
        best = max(weights.values())
        labels.append(next(l for l in order if abs(weights[l] - best) < 1e-12))
        top = max(counts.values())
        ties.append(sum(1 for v in counts.values() if v == top) > 1)
    return labels, ties


def forest_analyses(ft: dict, task: str) -> dict[str, pd.DataFrame]:
    agents, names, truths = ft["agents"], ft["names"], ft["truths"]
    n_img = len(truths)
    maj = _majority_labels(ft, agents)
    wtd = [ERROR_PRED if ft["error"][i] else weighted_vote(ft["raw_votes"][i], task)
           for i in range(n_img)]
    agent_hits = {a: [correct(ft["labels"][a][i], truths[i], task) for i in range(n_img)] for a in agents}

    # (a) vote rules
    rows = [
        _acc_row("majority_vote", [correct(p, t, task) for p, t in zip(maj, truths)],
                 n_abstained=sum(1 for p in maj if not p)),
        _acc_row("confidence_weighted_vote", [correct(p, t, task) for p, t in zip(wtd, truths)],
                 n_abstained=sum(1 for p in wtd if not p)),
    ]
    for a in agents:
        rows.append(_acc_row(f"single:{names[a]}", agent_hits[a],
                             n_abstained=sum(1 for l in ft["labels"][a] if not l)))
    rows.append(_acc_row("oracle_any_agent",
                         [int(any(agent_hits[a][i] for a in agents)) for i in range(n_img)]))
    rows.append(_acc_row("pipeline_final", [correct(p, t, task) for p, t in zip(ft["final"], truths)]))
    vote_rules = pd.DataFrame(rows)

    # (b) agent subsets
    sub_rows = []
    if len(agents) <= MAX_SUBSET_AGENTS:
        for k in range(2, len(agents) + 1):
            for combo in itertools.combinations(agents, k):
                lbls = _majority_labels(ft, list(combo))
                wlbls, ties = _subset_weighted_and_ties(ft, list(combo))
                row = _acc_row(" + ".join(names[a] for a in combo),
                               [correct(p, t, task) for p, t in zip(lbls, truths)], size=k)
                row["accuracy_conf_weighted"] = round(
                    float(np.mean([correct(p, t, task) for p, t in zip(wlbls, truths)])), 4) if truths else None
                row["tie_rate"] = round(float(np.mean(ties)), 4) if ties else None
                sub_rows.append(row)
    subsets = pd.DataFrame(sub_rows)

    # (c) pairwise phi of correctness
    phi_rows = []
    for a, b in itertools.combinations(agents, 2):
        phi_rows.append({"agent_a": names[a], "agent_b": names[b],
                         "phi_correctness": round(phi_coefficient(agent_hits[a], agent_hits[b]), 4),
                         "label_disagreement": round(float(np.mean(
                             [ft["labels"][a][i] != ft["labels"][b][i] for i in range(n_img)])), 4)})
    phi = pd.DataFrame(phi_rows)
    if not phi.empty:
        phi = pd.concat([phi, pd.DataFrame([{
            "agent_a": "MEAN", "agent_b": "",
            "phi_correctness": round(float(np.nanmean(phi["phi_correctness"])), 4),
            "label_disagreement": round(float(phi["label_disagreement"].mean()), 4)}])],
            ignore_index=True)

    # (d) within- vs between-role disagreement
    roles = ft["roles"]
    within, between = [], []
    for a, b in itertools.combinations(agents, 2):
        diffs = [ft["labels"][a][i] != ft["labels"][b][i] for i in range(n_img)
                 if ft["labels"][a][i] and ft["labels"][b][i]]
        (within if roles[a] == roles[b] else between).extend(diffs)
    role_counts = pd.Series(list(roles.values())).value_counts().to_dict()
    disagreement = pd.DataFrame([{
        "n_agents": len(agents), "roles": "; ".join(f"{r}x{c}" for r, c in role_counts.items()),
        "temperatures": "; ".join(f"{names[a]}={t}" for a, t in ft["temps"].items()) or "not logged",
        "roles_repeat": any(c > 1 for c in role_counts.values()),
        "within_role_pairs": len(within),
        "within_role_disagreement": round(float(np.mean(within)), 4) if within else float("nan"),
        "between_role_pairs": len(between),
        "between_role_disagreement": round(float(np.mean(between)), 4) if between else float("nan"),
    }])

    # (e) lenient scoring
    lenient = "any_tumor" if is_multiclass(task) else "abnormal"
    len_rows = []
    for name, preds in [("majority_vote", maj), ("confidence_weighted_vote", wtd),
                        ("pipeline_final", ft["final"])] + \
                       [(f"single:{names[a]}", ft["labels"][a]) for a in agents]:
        s = [correct(p, t, task, "strict") for p, t in zip(preds, truths)]
        l = [correct(p, t, task, lenient) for p, t in zip(preds, truths)]
        len_rows.append({"method": name, "n": n_img,
                         "accuracy_strict" if not is_multiclass(task) else "accuracy_exact_subtype":
                             round(float(np.mean(s)), 4),
                         f"accuracy_{lenient}": round(float(np.mean(l)), 4),
                         "delta": round(float(np.mean(l) - np.mean(s)), 4)})
    lenient_df = pd.DataFrame(len_rows)

    return {"forest_vote_rules": vote_rules, "forest_agent_subsets": subsets,
            "forest_pairwise_phi": phi, "forest_role_disagreement": disagreement,
            "forest_lenient_scoring": lenient_df}


def compare_forests(a: tuple[str, list[dict]], b: tuple[str, list[dict]], task: str) -> dict:
    """(f) McNemar exact on majority-vote triage correctness over identical image_paths."""
    (na, ra), (nb, rb) = a, b
    da, db = by_image_path(ra), by_image_path(rb)
    common = sorted(set(da) & set(db))
    fa = forest_table([da[i] for i in common], task)
    fb = forest_table([db[i] for i in common], task)
    for i, img in enumerate(common):
        assert fa["truths"][i] == fb["truths"][i], f"label mismatch on {img}"
    ca = [correct(p, t, task) for p, t in zip(_majority_labels(fa, fa["agents"]), fa["truths"])]
    cb = [correct(p, t, task) for p, t in zip(_majority_labels(fb, fb["agents"]), fb["truths"])]
    res = mcnemar_exact(ca, cb)
    return {"task": task, "a": na, "b": nb, "n_a": len(da), "n_b": len(db), "n_common": len(common),
            "acc_a": round(float(np.mean(ca)), 4) if ca else float("nan"),
            "acc_b": round(float(np.mean(cb)), 4) if cb else float("nan"), **res}


# ── debate ─────────────────────────────────────────────────────────────────────

ADVOCATES = ("CNN", "BiomedCLIP", "SAM3")
_ARG_ROLE = {"cnn": "CNN", "clip": "BiomedCLIP", "sam": "SAM3"}
_NEG_PAT = re.compile(r"\b(no|absence of|without|not)\b[^.]{0,40}\b(lesion|tumou?r|abnormalit|patholog|mass)",
                      re.I)


def _sam_claim(rec: dict) -> str:
    if rec.get("sam3_skipped"):
        return "ineligible"
    bbox = rec.get("sam3_bbox")
    has = bool(bbox) and any(b is not None for b in bbox) and not rec.get("sam3_mask_empty")
    return "lesion" if has else "no_lesion"


def advocate_labels(rec: dict, task: str) -> dict[str, str]:
    """Canonical label each advocate was asked to defend (from the tool outputs)."""
    pos = pos_class(task)
    sam = _sam_claim(rec)
    sam_lbl = ("tumor" if is_multiclass(task) else pos) if sam == "lesion" else "normal"
    return {
        "CNN": canonical_label(rec.get("cnn_predicted_class") or "", task),
        "BiomedCLIP": canonical_label(rec.get("biomedclip_top_label") or "", task),
        "SAM3": sam_lbl,
    }


_TEXT_VOCAB = (
    [(rf"\b{t}", t) for t in TUMOR_SUBTYPES]
    + [(r"\bpituitary", "pituitary"), (r"\btumou?r", "tumor"), (r"\bneoplas", "tumor"),
       (r"\bmultiple sclerosis\b", "multiple sclerosis"), (r"\bms\b", "ms"),
       (r"\bdemyelinat", "demyelinating"), (r"\bwhite matter lesion", "white matter lesion"),
       (r"\bstroke", "stroke"), (r"\bischemi", "ischemic"), (r"\bhemorrhag|\bhaemorrhag", "hemorrhagic"),
       (r"\binfarct", "infarct"), (r"\bnormal\b", "normal")]
)


def text_implied_label(text: str, task: str) -> str:
    """Keyword reading of an advocate argument (heuristic, reported separately).

    An explicit negation of pathology ("no evidence of lesion", "absence of
    segmentable pathology") without a positive diagnosis reads as normal;
    otherwise the earliest label-like phrase in the text wins.
    """
    if not text:
        return ""
    low = text.lower()
    hits = []
    for pat, phrase in _TEXT_VOCAB:
        m = re.search(pat, low)
        if m:
            hits.append((m.start(), canonical_label(phrase, task)))
    positive = [h for h in hits if h[1] != "normal"]
    if _NEG_PAT.search(text) and not re.search(
            r"(consistent with|suggest\w*|indicat\w*|predict\w*|favou?r\w*)\s+(an?\s+)?"
            r"(glioma|meningioma|pituitary|tumou?r|stroke|ms\b|multiple|ischemi|hemorrhag)", low):
        return "normal"
    if not hits:
        return ""
    return min(positive or hits)[1]


def debate_analyses(records: list[dict], task: str) -> dict[str, pd.DataFrame]:
    from eval.judge_attribution import endorsed, judge_reason

    pos = pos_class(task)
    multi = is_multiclass(task)
    truths = [_true(r, task) for r in records]
    finals = [ERROR_PRED if r.get(ERROR_FLAG) else canonical_label(r.get("predicted_class") or "", task)
              for r in records]
    adv = [advocate_labels(r, task) for r in records]
    sam_claims = [_sam_claim(r) for r in records]

    def _pos_claim(lbl: str) -> int:
        if multi:
            return int(lbl in _TUMORISH)
        return int(_scored(lbl, task) == pos)

    # coarse comparison space for "follows": binary -> prep'd; multiclass SAM3 -> tumor/normal
    def _same(a: str, b: str, who: str) -> bool:
        if not a or not b or b == ERROR_PRED:
            return False
        if multi and who == "SAM3":
            return (a in _TUMORISH) == (b in _TUMORISH)
        return _scored(a, task) == _scored(b, task)

    claim_rows, follow_rows = [], []
    final_hits = [correct(f, t, task) for f, t in zip(finals, truths)]
    hits_by = {}
    for who in ADVOCATES:
        lbls = [a[who] for a in adv]
        if multi and who == "SAM3":
            hits = [int((l in _TUMORISH) == (t in _TUMORISH)) for l, t in zip(lbls, truths)]
            scoring = "coarse tumor/normal"
        else:
            hits = [correct(l, t, task) for l, t in zip(lbls, truths)]
            scoring = "strict"
        hits_by[who] = hits
        dist = pd.Series([_scored(l, task) if not (multi and who == "SAM3") else l or "unknown"
                          for l in lbls]).value_counts()
        row = {"advocate": who, "source": "tool output", "n": len(lbls),
               "positive_claim_rate": round(float(np.mean([_pos_claim(l) for l in lbls])), 4),
               "advocate_accuracy": round(float(np.mean(hits)), 4), "accuracy_scoring": scoring,
               "claim_distribution": "; ".join(f"{k}={v}" for k, v in dist.items())}
        if who == "SAM3":
            sc = pd.Series(sam_claims).value_counts()
            row["sam3_claims"] = "; ".join(f"{k}={v}" for k, v in sc.items())
        claim_rows.append(row)

        follows = [_same(l, f, who) for l, f in zip(lbls, finals)]
        fl = np.array(follows, dtype=bool)
        h = np.array(hits, dtype=bool)
        fh = np.array(final_hits, dtype=bool)
        follow_rows.append({
            "advocate": who, "n": len(fl),
            "judge_follows_rate": round(float(fl.mean()), 4),
            "follows_when_advocate_correct": round(float(fl[h].mean()), 4) if h.any() else float("nan"),
            "follows_when_advocate_wrong": round(float(fl[~h].mean()), 4) if (~h).any() else float("nan"),
            "final_acc_when_following": round(float(fh[fl].mean()), 4) if fl.any() else float("nan"),
            "final_acc_when_not_following": round(float(fh[~fl].mean()), 4) if (~fl).any() else float("nan"),
        })

    # judge's own attribution in its free-text reason (judge_attribution.endorsed)
    reasons = [judge_reason(r) for r in records]
    endorse = [endorsed(x) for x in reasons]
    name_map = {"CNN": "CNN", "BiomedCLIP": "BiomedCLIP", "SAM3": "SAM3"}
    for row in follow_rows:
        who = name_map[row["advocate"]]
        e = np.array([who in s for s in endorse], dtype=bool)
        row["judge_calls_decisive_rate"] = round(float(e.mean()), 4)
        row["final_acc_when_called_decisive"] = (round(float(np.array(final_hits)[e].mean()), 4)
                                                 if e.any() else float("nan"))

    # text-heuristic claims (only when arguments are logged)
    has_args = any(isinstance(r.get("debate_arguments"), list) and r["debate_arguments"] for r in records)
    if has_args:
        for who_key, who in _ARG_ROLE.items():
            lbls = []
            for r in records:
                args_ = [a for a in (r.get("debate_arguments") or []) if a.get("role") == who_key]
                last = max(args_, key=lambda a: a.get("round", 0)) if args_ else None
                lbls.append(text_implied_label((last or {}).get("argument", ""), task))
            n_read = sum(1 for l in lbls if l)
            claim_rows.append({"advocate": who, "source": "argument text (heuristic, last round)",
                               "n": n_read,
                               "positive_claim_rate": round(float(np.mean([_pos_claim(l) for l in lbls if l])), 4)
                               if n_read else float("nan"),
                               "advocate_accuracy": round(float(np.mean(
                                   [correct(l, t, task) for l, t in zip(lbls, truths) if l])), 4)
                               if n_read else float("nan"),
                               "accuracy_scoring": "strict",
                               "claim_distribution": "; ".join(
                                   f"{k}={v}" for k, v in pd.Series([l or "unread" for l in lbls]).value_counts().items())})

    phi_rows = []
    claims = {who: [_pos_claim(a[who]) for a in adv] for who in ADVOCATES}
    for a, b in itertools.combinations(ADVOCATES, 2):
        phi_rows.append({"advocate_a": a, "advocate_b": b,
                         "phi_positive_claim": round(phi_coefficient(claims[a], claims[b]), 4),
                         "phi_correctness": round(phi_coefficient(hits_by[a], hits_by[b]), 4),
                         "label_agreement": round(float(np.mean([_same(x[a], x[b], "SAM3" if "SAM3" in (a, b) else a)
                                                                 for x in adv])), 4)})
    return {"debate_advocate_claims": pd.DataFrame(claim_rows),
            "debate_advocate_phi": pd.DataFrame(phi_rows),
            "debate_judge_follows": pd.DataFrame(follow_rows),
            "debate_overall": pd.DataFrame([_acc_row("debate_final", final_hits,
                                                     n_abstained=sum(1 for f in finals if not f),
                                                     n_error=sum(1 for f in finals if f == ERROR_PRED))])}


# ── main ───────────────────────────────────────────────────────────────────────

def _print(title: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    print(f"\n── {title} ──")
    print(df.to_string(index=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--task", default=None, help="Override the task (default: inferred from records).")
    ap.add_argument("--out_dir", default=None,
                    help="Root for outputs: <out_dir>/<stem>_posthoc/ (default outputs/analysis).")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)

    root = Path(args.out_dir) if args.out_dir else Path("outputs/analysis")
    forests: dict[str, list[tuple[str, list[dict]]]] = {}
    for path in args.jsonl:
        recs, stats = load_records(path)
        if not recs:
            print(f"[skip] {path}: empty")
            continue
        task = _infer_task(recs, args.task)
        kind = _kind(recs)
        stem = Path(path).stem
        out = root / f"{stem}_posthoc"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 78}\n{stem}  [{kind}, task={task}]  {stats.summary()}")
        if kind == "forest":
            tables = forest_analyses(forest_table(recs, task), task)
            forests.setdefault(task, []).append((stem, recs))
        elif kind == "debate":
            tables = debate_analyses(recs, task)
        else:
            print("  neither a forest nor a debate run — nothing to do")
            continue
        for name, df in tables.items():
            _print(name, df)
            if df is not None and not df.empty:
                df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n  CSVs -> {out}/")

    comp_rows = []
    for task, runs in forests.items():
        for a, b in itertools.combinations(runs, 2):
            comp_rows.append(compare_forests(a, b, task))
    if comp_rows:
        comp = pd.DataFrame(comp_rows)
        cdir = root / "posthoc_comparisons"
        cdir.mkdir(parents=True, exist_ok=True)
        comp.to_csv(cdir / "forest_mcnemar.csv", index=False)
        _print("(f) forest vs forest — McNemar exact on majority-vote triage (identical images)", comp)
        print(f"\n  CSV -> {cdir}/forest_mcnemar.csv")


if __name__ == "__main__":
    main()
