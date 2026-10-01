#!/usr/bin/env bash
# Step 10 — smoke test of every v2 mode on SMOKE_N images (default 6) of binary_tumor
# and stroke, then a field-by-field check of the output rows. ~45 min on one GPU.
# Run after 00_preflight and before any long run. It must print "SMOKE OK".
#
# binary_tumor exercises SAM3 (including empty masks on normal scans); stroke
# exercises the debate without SAM3 (it is ineligible there). Output goes to
# $V2_OUT/smoke/, which is wiped first so a re-run re-tests the current code.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

SMOKE_DIR="$V2_OUT/smoke"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

status=0
for step in \
    base_binary_tumor forest_binary_tumor debate_binary_tumor homog_binary_tumor \
    base_stroke debate_stroke homog_stroke
do
    STEP_OUT_DIR="$SMOKE_DIR" STEP_MAX_SAMPLES="$SMOKE_N" \
        bash "$BUNDLE_DIR/step.sh" "$step" || status=1
done

log_start 10_smoke_check
pyrun server_bundle/scripts/check_smoke.py --dir "$SMOKE_DIR" --n "$SMOKE_N" \
    --temperature "$FOREST_TEMPERATURE" 2>&1 | tee -a logs/10_smoke_check.log
check=$?
log_done 10_smoke_check

[ "$status" -eq 0 ] && [ "$check" -eq 0 ] && exit 0
echo "SMOKE FAILED — do not start the long runs. Send logs/ back."
exit 1
