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

# Environment the container needs: offline HF cache, local checkpoints, dataset
# paths for preflight, and unbuffered output so the logs update live.
CONTAINER_ENV=(
    "HF_HOME=$PROJECT_ROOT/$HF_CACHE_DIR"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "HF_HUB_DISABLE_TELEMETRY=1"
    "CHECKPOINT_SOURCE=local"
    "PYTHONPATH=$PROJECT_ROOT"
    "DATA_BR35H=$DATA_BR35H"
    "DATA_FIGSHARE=$DATA_FIGSHARE"
    "DATA_MS=$DATA_MS"
    "DATA_STROKE=$DATA_STROKE"
    "PREFLIGHT_MAX_SAMPLES=$MAX_SAMPLES"
    "MEDGEMMA_4BIT=${LOAD_4BIT:-0}"
    "PYTHONUNBUFFERED=1"
    "TOKENIZERS_PARALLELISM=false"
    "PYTORCH_ALLOC_CONF=expandable_segments:True"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
)

# `exec --env` only exists in Singularity >= 3.6. On older installs the equivalent
# is the SINGULARITYENV_ / APPTAINERENV_ prefix on host variables, which every
# version understands — so detect once and use whichever this install supports.
if "$SINGULARITY_BIN" exec --help 2>&1 | grep -q -- '--env'; then
    USE_ENV_FLAG=1
else
    USE_ENV_FLAG=0
    echo "[note] $SINGULARITY_BIN has no 'exec --env' (pre-3.6); using SINGULARITYENV_* instead."
    for kv in "${CONTAINER_ENV[@]}"; do
        export "SINGULARITYENV_${kv%%=*}=${kv#*=}"
        export "APPTAINERENV_${kv%%=*}=${kv#*=}"
    done
fi

# Run a python script inside the container, from the project root, fully offline.
pyrun() {
    local env_args=()
    if [ "$USE_ENV_FLAG" = "1" ]; then
        local kv
        for kv in "${CONTAINER_ENV[@]}"; do
            env_args+=(--env "$kv")
        done
    fi
    # ${a[@]+"${a[@]}"} expands to nothing when the array is empty without tripping
    # `set -u` on bash < 4.4 (CentOS 7 ships 4.2).
    "$SINGULARITY_BIN" exec --nv \
        --pwd "$PROJECT_ROOT" \
        --bind "$PROJECT_ROOT":"$PROJECT_ROOT" \
        ${env_args[@]+"${env_args[@]}"} \
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
