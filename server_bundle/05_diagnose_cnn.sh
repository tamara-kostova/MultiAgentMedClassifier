#!/usr/bin/env bash
# Step 5 — diagnose the 12-class multiclass tumour CNN on figshare (~5 min, GPU).
# Done 2026-10-01; the label order it found is now in agents/cnn_tool.py, so run_all.sh /
# run_parallel.sh no longer run it. Kept for a manual re-check.
# It scores 0.14 on figshare although figshare was in its training data, so this
# checks for a class-index-order bug (best label permutation) or a preprocessing
# mismatch (min-max / percentile / inverted variants). Read-only: it changes nothing.
# Results: outputs/analysis/cnn_diag/summary.md — send it back with the other logs.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

STEP=05_diagnose_cnn
log_start "$STEP"
pyrun scripts/diagnose_multiclass_cnn.py \
    --dataset "figshare=$DATA_FIGSHARE:figshare3" \
    --max_per_class 200 \
    --out_dir outputs/analysis/cnn_diag \
    2>&1 | tee -a "logs/${STEP}.log"
status=$?
log_done "$STEP"
exit $status
