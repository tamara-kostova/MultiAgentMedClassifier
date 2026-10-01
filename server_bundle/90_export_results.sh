#!/usr/bin/env bash
# Step 90 — (re-)export every v2 result JSONL to TSV/CSV and pack them for sending back.
# Safe to run at any time, including while runs are in progress: it only reads the
# JSONL files. Each step already exports when it finishes, so this is mainly for
# collecting partial results early or re-packing.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

STEP=90_export_results
log_start "$STEP"

shopt -s nullglob
JSONLS=("$V2_OUT"/*.jsonl)
shopt -u nullglob

if [ "${#JSONLS[@]}" -eq 0 ]; then
    echo "[export] no result files in $V2_OUT yet"
else
    for jsonl in "${JSONLS[@]}"; do
        export_one "$jsonl"
    done
    summary_args=()
    for jsonl in "${JSONLS[@]}"; do
        summary_args+=(--jsonl "$jsonl")
    done
    pyrun server_bundle/scripts/export_results.py --summary_only \
        "${summary_args[@]}" \
        --combined_summary outputs/results_tsv/all_runs_summary.tsv \
        || echo "[export] WARNING: combined summary failed"
fi

ARCHIVE="results_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz"
PARTS=()
for p in outputs/results_tsv "$V2_OUT" outputs/analysis logs CODE_VERSION; do
    [ -e "$p" ] && PARTS+=("$p")
done
if ! tar czf "$ARCHIVE" "${PARTS[@]}"; then
    echo "[export] ERROR: creating $ARCHIVE failed (disk full?). The JSONLs in $V2_OUT are intact."
    log_done "$STEP"
    exit 1
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo " Results archive ready to send back:"
echo "   $(pwd)/$ARCHIVE   ($(du -h "$ARCHIVE" | cut -f1))"
echo " It contains:"
echo "   $V2_OUT/          — raw JSONL (full detail, source of truth)"
echo "   outputs/results_tsv/  — TSV tables (one row per image + summaries)"
echo "   outputs/analysis/     — CSV metric tables, plots, cnn_diag/"
echo "   logs/                 — run logs"
echo "──────────────────────────────────────────────────────────────"

log_done "$STEP"
