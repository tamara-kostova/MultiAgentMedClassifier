#!/usr/bin/env bash
# Step 6 — Multi-Agent Debate (2 rounds), 3-class tumour MRI (figshare).
# Est. ~14-16 h for 500 images. Resumable: re-run this same script.
#
# Deliberately last: on this task the CNN and BiomedCLIP advocates are at or below
# chance, so it is the least informative of the six runs. Skip this one first if
# time on the server runs short.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

run_step 06_debate_multiclass_tumor outputs/eval/multiclass_debate_r2.jsonl \
    --tumor_eval \
    --task multiclass_tumor \
    --label_map figshare3 \
    --tumor_eval_dir "$DATA_FIGSHARE" \
    --max_samples "$MAX_SAMPLES" \
    --pipeline_mode debate \
    --debate_rounds 2 \
    --tumor_eval_output outputs/eval/multiclass_debate_r2.jsonl
