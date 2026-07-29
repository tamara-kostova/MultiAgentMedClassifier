#!/usr/bin/env bash
# Step 2 — Agent Forest (N=4), multiple sclerosis FLAIR MRI.  Est. ~8 h for 500 images.
# Resumable: if it stops for any reason, run this same script again.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

run_step 02_forest_ms outputs/eval/ms_forest_n4.jsonl \
    --dataset_eval \
    --task ms \
    --label_map ms_binary \
    --dataset_eval_dir "$DATA_MS" \
    --max_samples "$MAX_SAMPLES" \
    --pipeline_mode forest \
    --forest_n_agents 4 \
    --dataset_eval_output outputs/eval/ms_forest_n4.jsonl
