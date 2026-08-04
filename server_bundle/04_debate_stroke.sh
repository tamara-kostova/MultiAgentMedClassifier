#!/usr/bin/env bash
# Step 4 — Multi-Agent Debate (2 rounds), stroke CT.
# Est. ~12-16 h for 500 images. Resumable: re-run this same script.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

run_step 04_debate_stroke outputs/eval/stroke_debate_r2.jsonl \
    --dataset_eval \
    --task stroke \
    --label_map stroke_binary \
    --dataset_eval_dir "$DATA_STROKE" \
    --max_samples "$MAX_SAMPLES" \
    --pipeline_mode debate \
    --debate_rounds 2 \
    --dataset_eval_output outputs/eval/stroke_debate_r2.jsonl
