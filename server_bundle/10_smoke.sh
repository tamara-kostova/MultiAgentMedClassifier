#!/usr/bin/env bash
# Step 10 — smoke test of every v2 step (steps.sh) on SMOKE_N images (default 2), then a
# field-by-field check of the output rows. ~1.5 h on one A100 at SMOKE_N=2.
# Run after 00_preflight and before any long run. It must print "SMOKE OK".
#
# All four tasks are covered: binary_tumor exercises SAM3 (including empty masks on
# normal scans), multiclass_tumor the fixed CNN label order and the debate's subtype
# verdict, ms/stroke the debate without SAM3 (it is ineligible there). Output goes to
# $V2_OUT/smoke/, which is wiped first so a re-run re-tests the current code.
#
# To smoke only some steps:  SMOKE_STEPS="debate_stroke homog_ms" bash server_bundle/10_smoke.sh
set -uo pipefail
source "$(dirname "$0")/_lib.sh"
# shellcheck source=steps.sh
source "$BUNDLE_DIR/steps.sh"

if [ -n "${SMOKE_STEPS:-}" ]; then
    read -ra SMOKE_LIST <<< "$SMOKE_STEPS"
else
    SMOKE_LIST=("${V2_STEPS[@]}")
fi

SMOKE_DIR="$V2_OUT/smoke"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

status=0
for step in "${SMOKE_LIST[@]}"; do
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
