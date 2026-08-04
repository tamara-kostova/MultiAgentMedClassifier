#!/usr/bin/env bash
# Optional: spread the six runs over several GPUs. The runs are fully independent
# processes writing to different output files, so this is safe.
#
#   bash server_bundle/run_parallel.sh 0 1 2        # use GPUs 0, 1 and 2
#   nohup bash server_bundle/run_parallel.sh 0 1 > logs/run_parallel.log 2>&1 &
#
# Each run needs roughly 14 GB of VRAM (MedGemma bfloat16 ~9 GB + SAM3 ~3.5 GB +
# CNN/BiomedCLIP ~1 GB), so use one run per GPU of 16 GB or more. On 40/80 GB cards
# a GPU id may be listed twice to get two concurrent runs on it, e.g. "0 0 1 1".
# With LOAD_4BIT=1 in config.env a run fits in about 7 GB.
#
# Preflight is run once, first, before anything is launched.
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$BUNDLE_DIR/.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash server_bundle/run_parallel.sh <gpu_id> [<gpu_id> ...]"
    echo "Example: bash server_bundle/run_parallel.sh 0 1 2"
    exit 2
fi

GPUS=("$@")
STEPS=(
    01_forest_stroke
    02_forest_ms
    03_debate_binary_tumor
    04_debate_stroke
    05_debate_ms
    06_debate_multiclass_tumor
)

mkdir -p logs

echo "### Preflight on GPU ${GPUS[0]} ($(date -Is))"
if ! CUDA_VISIBLE_DEVICES="${GPUS[0]}" bash "$BUNDLE_DIR/00_preflight.sh"; then
    echo "PREFLIGHT FAILED — nothing launched. Please send back logs/00_preflight.log."
    exit 1
fi

# Round-robin: each GPU gets a queue of steps and works through it sequentially.
pids=()
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    queue=()
    for j in "${!STEPS[@]}"; do
        if [ $(( j % ${#GPUS[@]} )) -eq "$i" ]; then
            queue+=("${STEPS[$j]}")
        fi
    done
    [ "${#queue[@]}" -eq 0 ] && continue

    echo "GPU $gpu queue: ${queue[*]}"
    (
        for step in "${queue[@]}"; do
            echo "### [GPU $gpu] $step start $(date -Is)"
            CUDA_VISIBLE_DEVICES="$gpu" bash "$BUNDLE_DIR/${step}.sh" \
                || echo "### [GPU $gpu] $step FAILED — continuing"
            echo "### [GPU $gpu] $step end   $(date -Is)"
        done
    ) > "logs/gpu${gpu}_queue.log" 2>&1 &
    pids+=("$!")
done

echo "Launched ${#pids[@]} GPU queues. Follow progress with:  tail -f logs/gpu*_queue.log"
wait "${pids[@]}"

echo ""
echo "All GPU queues finished ($(date -Is)). Exporting results..."
bash "$BUNDLE_DIR/90_export_results.sh"
