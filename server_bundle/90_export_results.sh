#!/usr/bin/env bash
# Step 90 — (re-)export every result JSONL to TSV/CSV and pack them for sending back.
# Safe to run at any time, including while a run is still in progress: it only reads
# the JSONL files. Each step script already exports automatically when it finishes,
# so this is mainly for collecting partial results early or re-packing.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

STEP=90_export_results
log_start "$STEP"

for jsonl in \
    outputs/eval/stroke_forest_n4.jsonl \
    outputs/eval/ms_forest_n4.jsonl \
    outputs/eval/binary_debate_r2.jsonl \
    outputs/eval/stroke_debate_r2.jsonl \
    outputs/eval/ms_debate_r2.jsonl \
    outputs/eval/multiclass_debate_r2.jsonl
do
    [ -f "$jsonl" ] || continue
    export_one "$jsonl"
done

# One combined overview across all runs.
pyrun server_bundle/scripts/export_results.py --summary_only \
    --jsonl outputs/eval/stroke_forest_n4.jsonl \
    --jsonl outputs/eval/ms_forest_n4.jsonl \
    --jsonl outputs/eval/binary_debate_r2.jsonl \
    --jsonl outputs/eval/stroke_debate_r2.jsonl \
    --jsonl outputs/eval/ms_debate_r2.jsonl \
    --jsonl outputs/eval/multiclass_debate_r2.jsonl \
    --combined_summary outputs/results_tsv/all_runs_summary.tsv \
    || echo "[export] WARNING: combined summary failed"

ARCHIVE="results_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz"
tar czf "$ARCHIVE" \
    outputs/results_tsv \
    outputs/eval \
    outputs/analysis \
    logs \
    2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────────────────"
echo " Results archive ready to send back:"
echo "   $(pwd)/$ARCHIVE"
echo " It contains:"
echo "   outputs/results_tsv/  — TSV tables (one row per image + summaries)"
echo "   outputs/analysis/     — CSV metric tables + plots"
echo "   outputs/eval/         — raw JSONL (full detail, source of truth)"
echo "   logs/                 — run logs"
echo "──────────────────────────────────────────────────────────────"

log_done "$STEP"
