"""
Export evaluation results from JSONL to TSV.

The pipeline writes one JSONL line per image (crash-safe, resumable, full detail).
The faculty server hand-off asked for CSV/TSV, so this flattens each JSONL into two
tab-separated tables:

    outputs/results_tsv/<stem>_per_image.tsv   one row per image, flat columns
    outputs/results_tsv/<stem>_summary.tsv     one row: n, accuracy, F1, latency, ...

Nested fields (class probability dicts, forest votes, ...) are JSON-encoded into a
single cell so the TSV stays rectangular and loads cleanly in pandas/Excel/R.

Usage:
    python server_bundle/scripts/export_results.py --jsonl outputs/eval/ms_forest_n4.jsonl

    # combined overview across several runs, no per-image tables
    python server_bundle/scripts/export_results.py --summary_only \
        --jsonl outputs/eval/ms_forest_n4.jsonl \
        --jsonl outputs/eval/stroke_forest_n4.jsonl \
        --combined_summary outputs/results_tsv/all_runs_summary.tsv

The richer metric tables (confusion matrices, calibration bins, forest voting
quality, debate round analysis) come from eval/eval_analysis.py, which writes CSVs
into outputs/analysis/<stem>/. Both are run automatically by the step scripts.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from eval.forest_votes import forest_vote
from eval.jsonl_io import ERROR_FLAG, load_records
from eval.labels import canonical_label
from eval.metrics import parse_bool


def _scored_label(value, task) -> str:
    """Paper scoring: canonical_label + eval_analysis.prep (strict binarization);
    "unknown" for an abstention. Imported lazily (eval_analysis pulls matplotlib)."""
    from eval.eval_analysis import prep

    lbl = canonical_label(value, task)
    return prep(lbl, task) if lbl else "unknown"

# Flat, human-readable column order for the per-image table. Any other key present
# in the JSONL is appended after these.
PREFERRED_COLUMNS = [
    "image_path",
    "task",
    "true_label",
    "true_label_name",
    "true_label_canonical",
    "predicted_class",
    "predicted_class_canonical",
    "correct",
    "final_confidence",
    "requires_human_review",
    "cnn_predicted_class",
    "cnn_predicted_class_canonical",
    "cnn_correct",
    "cnn_confidence",
    "biomedclip_top_label",
    "biomedclip_top_score",
    "biomedclip_mode",
    "routing_decision",
    "routing_confidence",
    "routing_path",
    "suspected_pathology",
    "sam3_skipped",
    "sam3_mask_empty",
    "sam3_bbox",
    "saliency_sam3_iou",
    "verification_agreement",
    "dissent_rate",
    "vote_fraction",
    "debate_rounds_completed",
    "debate_round_changed",
    "debate_winner",
    "latency_s",
    "error",
    "timestamp",
]

# Long free-text / deeply nested fields: kept, but pushed to the end of the table.
BULKY_COLUMNS = [
    "medgemma_diagnosis",
    "medgemma_bbox_diagnosis",
    "final_medgemma_diagnosis",
    "cnn_all_probs",
    "biomedclip_ranked_labels",
    "biomedclip_scores",
    "forest_votes",
    "forest_consensus",
    "routing_reasoning",
    "verification_reasoning",
    "verification_alternative_dx",
    "final_report",
    "sam3_mask_path",
    "sam3_guided_image_path",
    "gradcam_pp_path",
    "integrated_gradients_path",
    "fhir_bundle_id",
]


def load_jsonl(path: Path) -> list[dict]:
    """eval.jsonl_io: last non-error row per image_path; error-only images kept (flagged)."""
    rows, stats = load_records(path, verbose=False)
    if stats.n_unparsable or stats.n_duplicate_rows or stats.n_error_only:
        print(f"  [info] {stats.summary()}")
    return rows


def flatten_cell(value):
    """JSON-encode dicts/lists so every cell is a scalar; keep TSV rectangular."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        # Tabs and newlines inside free text (reports, reasoning) would break the TSV.
        return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return value


def build_per_image(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    task = df["task"].iloc[0] if "task" in df.columns and len(df) else None
    true_canon = df.get("true_label_canonical")
    if true_canon is None:
        true_canon = df.get("true_label_name", df.get("true_label")).map(
            lambda v: canonical_label(v, task)
        )
    pred_canon = df.get("predicted_class_canonical")
    if pred_canon is None:
        pred_canon = df.get("predicted_class").map(lambda v: canonical_label(v, task))

    # Same scoring as the paper path (strict binarization, abstain/error = wrong).
    t_s = true_canon.map(lambda v: _scored_label(v, task))
    p_s = pred_canon.map(lambda v: _scored_label(v, task))
    err = df[ERROR_FLAG].fillna(False).astype(bool) if ERROR_FLAG in df.columns else False
    df["correct"] = ((t_s == p_s) & (p_s != "unknown") & ~err).astype(int)
    if "cnn_predicted_class_canonical" in df.columns:
        c_s = df["cnn_predicted_class_canonical"].map(lambda v: _scored_label(v, task))
        df["cnn_correct"] = ((t_s == c_s) & (c_s != "unknown") & ~err).astype(int)

    ordered = [c for c in PREFERRED_COLUMNS if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered and c not in BULKY_COLUMNS]
    ordered += [c for c in BULKY_COLUMNS if c in df.columns]

    out = df[ordered].copy()
    for col in out.columns:
        out[col] = out[col].map(flatten_cell)
    return out


def build_summary(rows: list[dict], name: str) -> dict:
    ok_rows = [r for r in rows if r.get("error") is None]
    n_err = len(rows) - len(ok_rows)
    task = rows[0].get("task") if rows else None

    summary: dict = {
        "run": name,
        "task": task,
        "pipeline_mode": (
            "forest"
            if any(r.get("dissent_rate") is not None for r in rows)
            else "debate"
            if any(r.get("debate_rounds_completed") is not None for r in rows)
            else "standard"
        ),
        "n_images": len(rows),
        "n_ok": len(ok_rows),
        "n_errors": n_err,
    }

    if not ok_rows:
        return summary

    # Accuracy over every image with known truth: abstentions and error-only
    # images count as wrong (the paper convention). F1 over the same rows.
    def _t(r):
        return _scored_label(r.get("true_label_canonical")
                             or r.get("true_label_name") or r.get("true_label"), r.get("task"))

    def _p(r, key):
        if r.get(ERROR_FLAG):
            return "error"
        return _scored_label(r.get(f"{key}_canonical") or r.get(key), r.get("task"))

    scored = [r for r in rows if _t(r) != "unknown"]
    y_true = [_t(r) for r in scored]
    y_pred = [_p(r, "predicted_class") for r in scored]
    y_cnn = [_p(r, "cnn_predicted_class") for r in scored]
    labels = sorted(set(y_true) | (set(y_pred) - {"unknown", "error"}))

    latencies = sorted(float(r.get("latency_s") or 0.0) for r in ok_rows)
    n_abstained = sum(1 for p in y_pred if p == "unknown")

    summary.update(
        {
            "n_scored": len(scored),
            "accuracy_final": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro_final": round(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4
            ),
            "accuracy_cnn_only": round(accuracy_score(y_true, y_cnn), 4),
            "n_abstained": n_abstained,
            "human_review_rate": round(
                sum(1 for r in ok_rows if r.get("requires_human_review")) / len(ok_rows), 4
            ),
            "sam3_ran_rate": round(
                sum(1 for r in ok_rows if r.get("sam3_mask_path")) / len(ok_rows), 4
            ),
            "mean_latency_s": round(sum(latencies) / len(latencies), 2),
            "median_latency_s": round(latencies[len(latencies) // 2], 2),
            "total_gpu_hours": round(sum(latencies) / 3600, 2),
        }
    )

    dissent = [r["dissent_rate"] for r in ok_rows if r.get("dissent_rate") is not None]
    if dissent:
        summary["mean_dissent_rate"] = round(sum(dissent) / len(dissent), 4)
        # Recomputed from the canonical ballots (stored vote_fraction can be stale).
        votes = [forest_vote(r.get("forest_votes"), task)["vote_fraction"] for r in ok_rows
                 if isinstance(r.get("forest_votes"), list) and r["forest_votes"]]
        votes = [v for v in votes if v is not None]
        if votes:
            summary["mean_vote_fraction"] = round(sum(votes) / len(votes), 4)

    changed = [
        parse_bool(r["debate_round_changed"])
        for r in ok_rows
        if parse_bool(r.get("debate_round_changed")) is not None
    ]
    if changed:
        # Judge-reported (legacy) flag; eval_analysis prefers per-round verdicts.
        summary["debate_verdict_change_rate"] = round(sum(changed) / len(changed), 4)
        rounds = [
            r["debate_rounds_completed"]
            for r in ok_rows
            if r.get("debate_rounds_completed") is not None
        ]
        if rounds:
            summary["mean_debate_rounds"] = round(sum(rounds) / len(rounds), 2)

        parse_failed = [
            parse_bool(r.get("debate_judge_parse_failed")) is True
            for r in ok_rows
            if r.get("debate_judge_parse_failed") is not None
        ]
        if parse_failed:
            summary["judge_parse_failure_rate"] = round(
                sum(parse_failed) / len(parse_failed), 4
            )

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Export eval JSONL to TSV")
    ap.add_argument(
        "--jsonl",
        action="append",
        required=True,
        help="Path to an eval JSONL (repeatable)",
    )
    ap.add_argument(
        "--output_dir",
        default="outputs/results_tsv",
        help="Where the TSV files are written (default: outputs/results_tsv)",
    )
    ap.add_argument(
        "--summary_only",
        action="store_true",
        help="Skip the per-image tables; only compute summaries",
    )
    ap.add_argument(
        "--combined_summary",
        default=None,
        help="Also write one TSV with a summary row per input JSONL",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for jsonl_arg in args.jsonl:
        path = Path(jsonl_arg)
        if not path.is_file():
            print(f"[skip] not found: {path}")
            continue
        rows = load_jsonl(path)
        if not rows:
            print(f"[skip] empty: {path}")
            continue

        summary = build_summary(rows, path.stem)
        summaries.append(summary)

        if not args.summary_only:
            per_image = build_per_image(rows)
            per_image_path = out_dir / f"{path.stem}_per_image.tsv"
            per_image.to_csv(per_image_path, sep="\t", index=False)
            print(f"[write] {per_image_path}  ({len(per_image)} rows, {len(per_image.columns)} cols)")

            summary_path = out_dir / f"{path.stem}_summary.tsv"
            pd.DataFrame([summary]).to_csv(summary_path, sep="\t", index=False)
            print(f"[write] {summary_path}")

        print(
            f"         {path.stem}: n={summary['n_ok']}/{summary['n_images']} "
            f"acc={summary.get('accuracy_final')} f1={summary.get('f1_macro_final')} "
            f"median_latency={summary.get('median_latency_s')}s"
        )

    if args.combined_summary and summaries:
        combined = Path(args.combined_summary)
        combined.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(combined, sep="\t", index=False)
        print(f"[write] {combined}  ({len(summaries)} runs)")

    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())
