"""
Reprocess eval JSONL records written before the "null"-sentinel fix (commit a760f0a)
and before task-aware forest voting.

Two defects are repaired offline, with no model re-inference:

1. "null" sentinel leaking into the predicted label.
   The prompt schema uses the literal string "null" (not JSON null) for an
   indeterminable field. `report_node` preferred `diagnosis_detailed` whenever it was
   non-empty, so "null" was read as a class name. This only bit *normal* cases (the
   ones with no tumor subtype), so it selectively destroyed normal-class predictions —
   exactly the class whose specificity the paper reports. Those records then scored as
   "unknown" and were silently dropped from the metric denominator.

   Repair: normalize "null"/"none"/"nan" strings to None inside the stored diagnosis
   dicts, then recompute predicted_class / predicted_class_canonical using the same
   task-aware field precedence the fixed `report_node` now applies.

2. Forest voting on the wrong field for multiclass tumor.
   `AgentForest.vote` took the majority over `diagnosis_name`, which the schema pins to
   "tumor" for every multiclass tumor case — so the vote was degenerate and
   dissent_rate/vote_fraction measured nothing. The per-agent ballots are stored in
   `forest_votes`, so the vote is fully recoverable offline.

   Repair: re-run the vote over `diagnosis_detailed` and rewrite dissent_rate,
   vote_fraction, suspected_pathology and the predicted label.

Usage:
    python eval/backfill_labels.py --jsonl outputs/eval/binary_forest_n4.jsonl
    python eval/backfill_labels.py --jsonl outputs/eval/*.jsonl --dry_run

Writes <file>.bak once before modifying, unless --no_backup.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from eval.tumor_eval import canonical_label

_NULLISH = ("none", "null", "nan", "")

# Fields that may carry the "null" string sentinel.
_STR_FIELDS = (
    "modality", "specialized_sequence", "plane",
    "diagnosis_name", "diagnosis_detailed", "icd10_code",
)
_DIAG_KEYS = ("medgemma_diagnosis", "final_medgemma_diagnosis", "medgemma_bbox_diagnosis")


def _clean(value) -> object:
    """Normalize the "null" string sentinel to None."""
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def _clean_diagnosis(diag: object) -> tuple[object, bool]:
    if not isinstance(diag, dict):
        return diag, False
    changed = False
    for field in _STR_FIELDS:
        if field in diag:
            cleaned = _clean(diag[field])
            if cleaned != diag[field]:
                diag[field] = cleaned
                changed = True
    return diag, changed


def _dx_label(diag: object, task: str) -> str | None:
    """Task-aware final label from a diagnosis dict (mirrors the fixed report_node)."""
    if not isinstance(diag, dict):
        return None
    fields = (
        ("diagnosis_detailed", "diagnosis_name")
        if task == "multiclass_tumor"
        else ("diagnosis_name", "diagnosis_detailed")
    )
    for field in fields:
        value = _clean(diag.get(field))
        if value:
            return str(value)
    return None


def _revote(votes: list[dict], task: str) -> dict | None:
    """Re-run the forest majority vote over the field that discriminates `task`."""
    field = "diagnosis_detailed" if task == "multiclass_tumor" else "diagnosis_name"

    def ballot(v: dict) -> str | None:
        return _clean(v.get(field)) or _clean(v.get("diagnosis_name"))

    ballots = [ballot(v) for v in votes]
    if not any(ballots):
        return None
    labels = [b or "unknown" for b in ballots]
    counts = Counter(labels)
    winner, count = counts.most_common(1)[0]
    supporters = [v for v, b in zip(votes, labels) if b == winner]
    confs = [v.get("diagnosis_confidence") or 0.5 for v in supporters]
    return {
        "winner": winner,
        "vote_field": field,
        "vote_fraction": round(count / len(votes), 4),
        "dissent_rate": round((len(votes) - count) / len(votes), 4),
        "confidence_weighted_confidence": round(sum(confs) / len(confs), 4),
        "n_agents": len(votes),
    }


def backfill_record(record: dict) -> tuple[dict, set[str]]:
    """Repair one record in place. Returns (record, set of change tags)."""
    task = str(record.get("task") or "")
    changes: set[str] = set()

    for key in _DIAG_KEYS:
        if key in record:
            record[key], changed = _clean_diagnosis(record[key])
            if changed:
                changes.add("null_sentinel")

    votes = record.get("forest_votes")
    if isinstance(votes, list) and votes:
        for vote in votes:
            for field in ("diagnosis_name", "diagnosis_detailed"):
                if field in vote:
                    cleaned = _clean(vote[field])
                    if cleaned != vote[field]:
                        vote[field] = cleaned
                        changes.add("null_sentinel")
        consensus = _revote(votes, task)
        if consensus:
            if record.get("dissent_rate") != consensus["dissent_rate"]:
                changes.add("forest_revote")
            record["dissent_rate"] = consensus["dissent_rate"]
            record["vote_fraction"] = consensus["vote_fraction"]
            record["forest_consensus"] = {
                **(record.get("forest_consensus") or {}), **consensus
            }
            record["routing_reasoning"] = (
                f"Forest consensus ({consensus['n_agents']} agents): "
                f"{consensus['winner']} ({consensus['vote_fraction'] * 100:.0f}% "
                f"agreement, dissent={consensus['dissent_rate'] * 100:.0f}%)"
            )

    # Recompute the final label with the corrected field precedence. Fall back through
    # the same chain report_node uses so a record is never left without a prediction.
    label = _dx_label(record.get("final_medgemma_diagnosis"), task)
    if label is None:
        label = _clean(record.get("cnn_predicted_class"))
    if label is None:
        label = _clean(record.get("biomedclip_top_label"))
    if label is not None:
        label = str(label)
        if record.get("predicted_class") != label:
            changes.add("predicted_class")
        record["predicted_class"] = label
        record["predicted_class_canonical"] = canonical_label(label, task) or ""

    # Backfill canonical columns absent from older files.
    if not record.get("true_label_canonical"):
        raw = record.get("true_label_name") or record.get("true_label")
        record["true_label_canonical"] = canonical_label(raw, task) or ""
        changes.add("true_label_canonical")
    if record.get("cnn_predicted_class") and not record.get("cnn_predicted_class_canonical"):
        record["cnn_predicted_class_canonical"] = canonical_label(
            record["cnn_predicted_class"], task
        ) or ""
        changes.add("cnn_canonical")

    return record, changes


def backfill_file(path: Path, dry_run: bool = False, backup: bool = True) -> Counter:
    records = []
    tally: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record, changes = backfill_record(json.loads(line))
            records.append(record)
            tally.update(changes)
            tally["records"] += 1

    if dry_run:
        return tally

    bak = path.with_suffix(path.suffix + ".bak")
    if backup and not bak.exists():
        shutil.copy2(path, bak)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair 'null' sentinel labels and forest votes in eval JSONLs"
    )
    parser.add_argument("--jsonl", nargs="+", required=True, help="JSONL file(s) to repair")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--no_backup", action="store_true", help="Skip writing <file>.bak")
    args = parser.parse_args()

    for raw in args.jsonl:
        path = Path(raw)
        if not path.exists():
            print(f"[skip] {path} — not found")
            continue
        tally = backfill_file(path, dry_run=args.dry_run, backup=not args.no_backup)
        detail = ", ".join(
            f"{k}={tally[k]}" for k in
            ("null_sentinel", "forest_revote", "predicted_class",
             "true_label_canonical", "cnn_canonical")
            if tally[k]
        ) or "no changes"
        prefix = "[dry-run]" if args.dry_run else "[fixed]  "
        print(f"{prefix} {path.name}: n={tally['records']}  {detail}")


if __name__ == "__main__":
    main()
