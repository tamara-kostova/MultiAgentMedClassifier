# Shared helpers for the numbered step scripts. Sourced, not executed.
# shellcheck shell=bash

# Without pipefail, `python ... | tee` would report tee's exit status and a crashed
# run would look successful.
set -o pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BUNDLE_DIR/.." && pwd)"

# shellcheck source=config.env
source "$BUNDLE_DIR/config.env"

cd "$PROJECT_ROOT"

mkdir -p logs outputs/eval outputs/results_tsv

# Resolve the .sif (allow an absolute path in config.env or SIF= in the environment).
case "$SIF" in
    /*) SIF_PATH="$SIF" ;;
    *)  SIF_PATH="$PROJECT_ROOT/$SIF" ;;
esac

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: Singularity image not found: $SIF_PATH"
    echo "Build it first:  singularity build --remote container.sif server_bundle/container.def"
    exit 1
fi

SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [ -z "$SINGULARITY_BIN" ]; then
    if command -v singularity >/dev/null 2>&1; then
        SINGULARITY_BIN=singularity
    elif command -v apptainer >/dev/null 2>&1; then
        SINGULARITY_BIN=apptainer
    else
        echo "ERROR: neither 'singularity' nor 'apptainer' is on PATH."
        exit 1
    fi
fi

EXTRA_FLAGS=""
if [ "${LOAD_4BIT:-0}" = "1" ]; then
    EXTRA_FLAGS="--load_4bit"
fi

# Run a python script inside the container, from the project root, fully offline.
pyrun() {
    "$SINGULARITY_BIN" exec --nv \
        --pwd "$PROJECT_ROOT" \
        --bind "$PROJECT_ROOT":"$PROJECT_ROOT" \
        --env HF_HOME="$PROJECT_ROOT/$HF_CACHE_DIR" \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        --env HF_HUB_DISABLE_TELEMETRY=1 \
        --env CHECKPOINT_SOURCE=local \
        --env PYTHONPATH="$PROJECT_ROOT" \
        --env DATA_BR35H="$DATA_BR35H" \
        --env DATA_FIGSHARE="$DATA_FIGSHARE" \
        --env DATA_MS="$DATA_MS" \
        --env DATA_STROKE="$DATA_STROKE" \
        --env PREFLIGHT_MAX_SAMPLES="$MAX_SAMPLES" \
        --env MEDGEMMA_4BIT="${LOAD_4BIT:-0}" \
        --env PYTHONUNBUFFERED=1 \
        --env TOKENIZERS_PARALLELISM=false \
        --env PYTORCH_ALLOC_CONF=expandable_segments:True \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "$SIF_PATH" /opt/venv/bin/python "$@"
}

log_start() {
    STEP_NAME="$1"
    STEP_LOG="logs/${STEP_NAME}.log"
    {
        echo ""
        echo "=========================================================="
        echo "START  $STEP_NAME   $(date -Is)"
        echo "  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-0}  max_samples=$MAX_SAMPLES"
        echo "=========================================================="
    } | tee -a "$STEP_LOG"
}

log_done() {
    {
        echo "DONE   $1   $(date -Is)"
        echo ""
    } | tee -a "logs/${1}.log"
}

# JSONL -> per-image TSV + summary TSV, plus the CSV metric tables.
# Never fails the step: the JSONL is the primary artifact.
export_one() {
    local jsonl="$1"
    [ -f "$jsonl" ] || { echo "[export] no such file: $jsonl"; return 0; }
    echo "[export] $jsonl -> outputs/results_tsv/"
    pyrun server_bundle/scripts/export_results.py --jsonl "$jsonl" \
        || echo "[export] WARNING: TSV export failed for $jsonl (JSONL is intact)"
    pyrun eval/eval_analysis.py --jsonl "$jsonl" \
        || echo "[export] WARNING: eval_analysis failed for $jsonl (JSONL is intact)"
}

# Run one evaluation step: name, output jsonl, then the run_pipeline.py arguments.
run_step() {
    local step="$1"; shift
    local out="$1"; shift
    log_start "$step"
    if pyrun run_pipeline.py "$@" $EXTRA_FLAGS 2>&1 | tee -a "logs/${step}.log"; then
        export_one "$out"
        log_done "$step"
        return 0
    else
        echo "FAILED $step — see logs/${step}.log" | tee -a "logs/${step}.log"
        export_one "$out"
        return 1
    fi
}
