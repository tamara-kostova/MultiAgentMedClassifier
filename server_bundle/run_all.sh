#!/usr/bin/env bash
# Run the whole v2 campaign, in order, on one GPU. ~130 GPU-hours (5-6 days).
#
#   bash server_bundle/run_all.sh
#
# Recommended: start it under nohup so it survives a closed SSH session:
#   nohup bash server_bundle/run_all.sh > logs/run_all.log 2>&1 &
#
# 00_preflight and 10_smoke must pass, or nothing long is started. After that every
# step is resumable and crash-safe: a failed step is logged and the next one starts,
# and re-running this script continues each unfinished step where it stopped.
#
# With several GPUs (or an 80 GB GPU that fits two runs), use run_parallel.sh.
# To run a subset, set STEPS, e.g.:
#   STEPS="debate_binary_tumor debate_stroke" bash server_bundle/run_all.sh
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$BUNDLE_DIR/.." && pwd)"
# shellcheck source=steps.sh
source "$BUNDLE_DIR/steps.sh"

if [ -n "${STEPS:-}" ]; then
    read -ra RUN_STEPS <<< "$STEPS"
else
    RUN_STEPS=("${V2_STEPS[@]}")
fi

declare -A RESULT
START_ALL=$(date -Is)

for gate in 00_preflight 05_diagnose_cnn 10_smoke; do
    echo ""
    echo "###########################################################"
    echo "#  $gate   ($(date -Is))"
    echo "###########################################################"
    if bash "$BUNDLE_DIR/${gate}.sh"; then
        RESULT[$gate]="OK"
    elif [ "$gate" = "05_diagnose_cnn" ]; then
        RESULT[$gate]="FAILED"   # diagnostic only: never blocks the runs
    else
        echo ""
        echo "$gate FAILED — stopping before the long runs."
        echo "Please send logs/ back; do not start the other steps."
        exit 1
    fi
done

for step in "${RUN_STEPS[@]}"; do
    echo ""
    echo "###########################################################"
    echo "#  $step   ($(date -Is))"
    echo "###########################################################"
    if bash "$BUNDLE_DIR/step.sh" "$step"; then
        RESULT[$step]="OK"
    else
        RESULT[$step]="FAILED"
    fi
done

bash "$BUNDLE_DIR/90_export_results.sh" && RESULT[90_export_results]="OK" || RESULT[90_export_results]="FAILED"

echo ""
echo "==========================================================="
echo " ALL STEPS FINISHED     started $START_ALL   ended $(date -Is)"
echo "==========================================================="
for step in 00_preflight 05_diagnose_cnn 10_smoke "${RUN_STEPS[@]}" 90_export_results; do
    printf '  %-28s %s\n' "$step" "${RESULT[$step]:-SKIPPED}"
done
echo ""
echo " Send back the results_*.tar.gz archive created by 90_export_results."
