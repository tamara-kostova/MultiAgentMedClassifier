#!/usr/bin/env bash
# Step 5 — Multi-Agent Debate (2 rounds), multiple sclerosis FLAIR MRI.
# Est. ~12-16 h for 500 images. Resumable: re-run this same script.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

run_step 05_debate_ms outputs/eval/ms_debate_r2.jsonl \
    --dataset_eval \
    --task ms \
    --label_map ms_binary \
    --dataset_eval_dir "$DATA_MS" \
    --max_samples "$MAX_SAMPLES" \
    --pipeline_mode debate \
    --debate_rounds 2 \
    --dataset_eval_output outputs/eval/ms_debate_r2.jsonl
