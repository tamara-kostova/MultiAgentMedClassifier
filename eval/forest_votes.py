"""Recompute Agent Forest triage votes from the logged per-agent ballots.

The stored ``vote_fraction`` / ``medgemma_diagnosis`` of a forest record are not
trustworthy for scoring: older multiclass runs voted over the raw
``diagnosis_name`` (always "tumor"), and ``medgemma_diagnosis`` is a seeded object
that can disagree with the ballot. Every scorer recomputes the vote here.

Conventions:
  * ballot label: canonical_label of ``diagnosis_detailed`` falling back to
    ``diagnosis_name`` for multiclass_tumor; ``diagnosis_name`` (falling back to
    ``diagnosis_detailed``) otherwise;
  * an empty ballot is an abstain vote: it counts in n (the vote_fraction
    denominator) but is never a label;
  * ties are broken by the fixed order of the ballots in the list (the first
    agent, in list order, whose label is among the tied winners);
  * agents are keyed by ``agent_idx`` when present, else by list position. Roles
    can repeat (homogeneous / sampled forests), so never key by role.
"""

from __future__ import annotations

from collections import Counter

from eval.labels import canonical_label


def ballot_label(vote: dict, task: str) -> str:
    if not isinstance(vote, dict):
        return ""
    if task == "multiclass_tumor":
        fields = ("diagnosis_detailed", "diagnosis_name")
    else:
        fields = ("diagnosis_name", "diagnosis_detailed")
    for f in fields:
        lbl = canonical_label(vote.get(f) or "", task)
        if lbl:
            return lbl
    return ""


def ballots(votes, task: str) -> list[dict]:
    """[{agent, role, label, conf, temperature}] in list order."""
    out = []
    for pos, v in enumerate(votes or []):
        if not isinstance(v, dict):
            continue
        idx = v.get("agent_idx")
        conf = v.get("diagnosis_confidence")
        try:
            conf = None if conf is None else float(conf)
        except (TypeError, ValueError):
            conf = None
        out.append({
            "agent": int(idx) if isinstance(idx, (int, float)) and idx == idx else pos,
            "role": v.get("role") or f"agent{pos}",
            "label": ballot_label(v, task),
            "conf": conf,
            "temperature": v.get("temperature"),
        })
    return out


def majority(labels: list[str]) -> tuple[str, int]:
    """(winner, count) over non-empty labels, ties by list order; ('', 0) if none."""
    valid = [l for l in labels if l]
    if not valid:
        return "", 0
    counts = Counter(valid)
    best = max(counts.values())
    for l in valid:
        if counts[l] == best:
            return l, best
    return "", 0  # unreachable


def forest_vote(votes, task: str) -> dict:
    """Majority vote from ballots. Returns label, vote_fraction, n, n_abstain."""
    bs = ballots(votes, task)
    label, k = majority([b["label"] for b in bs])
    n = len(bs)
    return {
        "label": label,
        "vote_fraction": (k / n) if n else None,
        "n": n,
        "n_abstain": sum(1 for b in bs if not b["label"]),
    }


def weighted_vote(votes, task: str, default_conf: float = 0.5) -> str:
    """Confidence-weighted vote (sum of diagnosis_confidence per label; None -> default)."""
    bs = ballots(votes, task)
    weights: dict[str, float] = {}
    first: dict[str, int] = {}
    for i, b in enumerate(bs):
        if not b["label"]:
            continue
        w = b["conf"] if b["conf"] is not None else default_conf
        weights[b["label"]] = weights.get(b["label"], 0.0) + w
        first.setdefault(b["label"], i)
    if not weights:
        return ""
    best = max(weights.values())
    return min((l for l, w in weights.items() if abs(w - best) < 1e-12), key=lambda l: first[l])
