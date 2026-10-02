#!/usr/bin/env bash
# Run one v2 evaluation step:  bash server_bundle/step.sh <system>_<task>
#
#   system  base | forest | debate | homog | rolesamp | nosam
#   task    binary_tumor | multiclass_tumor | ms | stroke
#
#   base      System A: standard pipeline
#   forest    System C: role-diverse Agent Forest, N=4, greedy
#   debate    System B: advocate debate, 2 rounds
#   homog     H1: homogeneous forest, 4 x radiologist sampled at FOREST_TEMPERATURE, triage only
#   rolesamp  H2: the 4 roles sampled at FOREST_TEMPERATURE, triage only
#   nosam     S1: debate without the SAM3 advocate (optional)
#
# base, forest and debate run with explainability on, so every system gets the same
# evidence. Every step runs exactly the images in server_bundle/image_lists/<task>.txt
# (the published Forest/Debate set) and writes $V2_OUT/<task>_<system>.jsonl.
# Resumable: if a step stops for any reason, run the same command again.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

STEP="${1:-}"
SYSTEM="${STEP%%_*}"
TASK="${STEP#*_}"

case "$TASK" in
    binary_tumor)     TASK_ARGS=(--tumor_eval   --label_map br35h         --tumor_eval_dir   "$DATA_BR35H");   OUT_FLAG=--tumor_eval_output ;;
    multiclass_tumor) TASK_ARGS=(--tumor_eval   --label_map figshare3     --tumor_eval_dir   "$DATA_FIGSHARE"); OUT_FLAG=--tumor_eval_output ;;
    ms)               TASK_ARGS=(--dataset_eval --label_map ms_binary     --dataset_eval_dir "$DATA_MS");       OUT_FLAG=--dataset_eval_output ;;
    stroke)           TASK_ARGS=(--dataset_eval --label_map stroke_binary --dataset_eval_dir "$DATA_STROKE");   OUT_FLAG=--dataset_eval_output ;;
    *) echo "Unknown task in step '$STEP'. Usage: bash server_bundle/step.sh <system>_<task>"; exit 2 ;;
esac

SAMPLED=(--forest_n_agents 4 --forest_temperature "$FOREST_TEMPERATURE" --forest_seed "$FOREST_SEED" --triage_only)
case "$SYSTEM" in
    base)     SYS_ARGS=(--generate_explainability) ;;
    forest)   SYS_ARGS=(--pipeline_mode forest --forest_n_agents 4 --generate_explainability) ;;
    debate)   SYS_ARGS=(--pipeline_mode debate --debate_rounds 2 --generate_explainability) ;;
    nosam)    SYS_ARGS=(--pipeline_mode debate --debate_rounds 2 --debate_advocates cnn,clip --generate_explainability) ;;
    homog)    SYS_ARGS=(--pipeline_mode forest --forest_roles radiologist "${SAMPLED[@]}") ;;
    rolesamp) SYS_ARGS=(--pipeline_mode forest "${SAMPLED[@]}") ;;
    *) echo "Unknown system in step '$STEP'. Usage: bash server_bundle/step.sh <system>_<task>"; exit 2 ;;
esac

# STEP_OUT_DIR / STEP_MAX_SAMPLES let 10_smoke reuse this table on a few images.
OUT_DIR="${STEP_OUT_DIR:-$V2_OUT}"
N="${STEP_MAX_SAMPLES:-$MAX_SAMPLES}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${TASK}_${SYSTEM}.jsonl"

run_step "$STEP" "$OUT" \
    "${TASK_ARGS[@]}" \
    --task "$TASK" \
    --image_list "server_bundle/image_lists/${TASK}.txt" \
    --max_samples "$N" \
    "${SYS_ARGS[@]}" \
    "$OUT_FLAG" "$OUT"
