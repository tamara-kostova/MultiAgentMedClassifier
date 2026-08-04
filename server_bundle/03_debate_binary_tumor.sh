#!/usr/bin/env bash
# Step 3 — Multi-Agent Debate (2 rounds), binary tumour MRI (Br35H).
# Est. ~13-16 h for 500 images. Resumable: re-run this same script.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

run_step 03_debate_binary_tumor outputs/eval/binary_debate_r2.jsonl \
    --tumor_eval \
    --task binary_tumor \
    --label_map br35h \
    --tumor_eval_dir "$DATA_BR35H" \
    --max_samples "$MAX_SAMPLES" \
    --pipeline_mode debate \
    --debate_rounds 2 \
    --tumor_eval_output outputs/eval/binary_debate_r2.jsonl
