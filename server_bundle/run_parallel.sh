#!/usr/bin/env bash
# Spread the v2 runs over several GPUs, or several slots on one GPU. The runs are
# independent processes writing to different output files, so this is safe.
#
#   bash server_bundle/run_parallel.sh 0 1 2        # GPUs 0, 1 and 2
#   bash server_bundle/run_parallel.sh 0 0          # two slots on one 80 GB GPU
#   nohup bash server_bundle/run_parallel.sh 0 0 1 1 > logs/run_parallel.log 2>&1 &
#
# Each run needs about 14 GB of VRAM in bfloat16 (MedGemma ~9 GB + SAM3 ~3.5 GB +
# CNN/BiomedCLIP ~1 GB): one slot per 16 GB GPU, two on 40 GB, up to five on 80 GB.
# With LOAD_4BIT=1 in config.env a run fits in about 7 GB.
#
# Preflight, the CNN diagnostic and the smoke test run once, first, on the first
# GPU; nothing is launched unless preflight and smoke pass. To run a subset:
#   STEPS="debate_binary_tumor debate_stroke" bash server_bundle/run_parallel.sh 0 1
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$BUNDLE_DIR/.." && pwd)"
# shellcheck source=steps.sh
source "$BUNDLE_DIR/steps.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash server_bundle/run_parallel.sh <gpu_id> [<gpu_id> ...]"
    echo "Example: bash server_bundle/run_parallel.sh 0 1 2"
    exit 2
fi

SLOTS=("$@")
if [ -n "${STEPS:-}" ]; then
    read -ra RUN_STEPS <<< "$STEPS"
else
    RUN_STEPS=("${V2_STEPS[@]}")
fi

# A step listed twice would put two processes on one output file.
dupes=$(printf '%s\n' "${RUN_STEPS[@]}" | sort | uniq -d)
if [ -n "$dupes" ]; then
    echo "ERROR: steps listed more than once: $dupes"
    exit 2
fi

mkdir -p logs

for gate in 00_preflight 05_diagnose_cnn 10_smoke; do
    echo "### $gate on GPU ${SLOTS[0]} ($(date -Is))"
    if ! CUDA_VISIBLE_DEVICES="${SLOTS[0]}" bash "$BUNDLE_DIR/${gate}.sh"; then
        if [ "$gate" = "05_diagnose_cnn" ]; then
            echo "### 05_diagnose_cnn failed — diagnostic only, continuing"
        else
            echo "$gate FAILED — nothing launched. Please send back logs/."
            exit 1
        fi
    fi
done

# Round-robin: each slot gets a queue of steps and works through it sequentially.
pids=()
for i in "${!SLOTS[@]}"; do
    gpu="${SLOTS[$i]}"
    queue=()
    for j in "${!RUN_STEPS[@]}"; do
        if [ $(( j % ${#SLOTS[@]} )) -eq "$i" ]; then
            queue+=("${RUN_STEPS[$j]}")
        fi
    done
    [ "${#queue[@]}" -eq 0 ] && continue

    echo "slot $i (GPU $gpu) queue: ${queue[*]}"
    (
        for step in "${queue[@]}"; do
            echo "### [slot $i GPU $gpu] $step start $(date -Is)"
            CUDA_VISIBLE_DEVICES="$gpu" bash "$BUNDLE_DIR/step.sh" "$step" \
                || echo "### [slot $i GPU $gpu] $step FAILED — continuing"
            echo "### [slot $i GPU $gpu] $step end   $(date -Is)"
        done
    ) > "logs/slot${i}_gpu${gpu}_queue.log" 2>&1 &
    pids+=("$!")
done

echo "Launched ${#pids[@]} queues. Follow progress with:  tail -f logs/slot*_queue.log"
wait "${pids[@]}"

echo ""
echo "All queues finished ($(date -Is)). Exporting results..."
bash "$BUNDLE_DIR/90_export_results.sh"
