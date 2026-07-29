#!/usr/bin/env bash
# Run everything, in order, on one GPU. Total ~3-4 days for all six runs.
#
#   bash server_bundle/run_all.sh
#
# Recommended: start it under nohup so it survives a closed SSH session:
#   nohup bash server_bundle/run_all.sh > logs/run_all.log 2>&1 &
#
# Every step is resumable and crash-safe. If a step fails, the script logs it and
# continues with the next one, so one broken run cannot waste the whole slot.
# Re-running this script continues each unfinished step from where it stopped.
#
# If the server has several free GPUs, use server_bundle/run_parallel.sh instead —
# the six runs are independent and each needs about 14 GB of VRAM.
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$BUNDLE_DIR/.." && pwd)"

STEPS=(
    00_preflight
    01_forest_stroke
    02_forest_ms
    03_debate_binary_tumor
    04_debate_stroke
    05_debate_ms
    06_debate_multiclass_tumor
    90_export_results
)

declare -A RESULT
START_ALL=$(date -Is)

for step in "${STEPS[@]}"; do
    echo ""
    echo "###########################################################"
    echo "#  $step   ($(date -Is))"
    echo "###########################################################"

    if bash "$BUNDLE_DIR/${step}.sh"; then
        RESULT[$step]="OK"
    else
        RESULT[$step]="FAILED"
        if [ "$step" = "00_preflight" ]; then
            echo ""
            echo "PREFLIGHT FAILED — stopping before the long runs."
            echo "Please send logs/00_preflight.log back; do not start the other steps."
            exit 1
        fi
    fi
done

echo ""
echo "==========================================================="
echo " ALL STEPS FINISHED     started $START_ALL   ended $(date -Is)"
echo "==========================================================="
for step in "${STEPS[@]}"; do
    printf '  %-28s %s\n' "$step" "${RESULT[$step]:-SKIPPED}"
done
echo ""
echo " Send back the results_*.tar.gz archive created by 90_export_results."
