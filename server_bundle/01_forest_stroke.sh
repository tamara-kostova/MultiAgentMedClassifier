#!/usr/bin/env bash
# Step 1 — Agent Forest (N=4), stroke CT.  Est. ~8 h for 500 images.
# Resumable: if it stops for any reason, run this same script again.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

run_step 01_forest_stroke outputs/eval/stroke_forest_n4.jsonl \
    --dataset_eval \
    --task stroke \
    --label_map stroke_binary \
    --dataset_eval_dir "$DATA_STROKE" \
    --max_samples "$MAX_SAMPLES" \
    --pipeline_mode forest \
    --forest_n_agents 4 \
    --dataset_eval_output outputs/eval/stroke_forest_n4.jsonl
