#!/usr/bin/env bash
# Submit bundle steps as SLURM jobs from the cluster login node. One GPU per job.
# NOT YET TESTED on the OSC cluster: set the SLURM_* values in config.env from the output
# of scripts/cluster_info.sh first.
#
#   bash server_bundle/submit.sh preflight              # 00_preflight
#   bash server_bundle/submit.sh smoke                  # 10_smoke (after preflight passes)
#   bash server_bundle/submit.sh base_binary_tumor debate_ms ...   # one job per step
#   bash server_bundle/submit.sh all                    # every step in steps.sh, one job each
#   bash server_bundle/submit.sh pair debate_binary_tumor debate_multiclass_tumor
#                                                       # two steps on one GPU (needs >= 40 GB)
#
# Steps are resumable, so walltime is handled by chaining: CHAIN=3 queues each step three
# times, each copy starting after the previous one ends (afterany). A copy that finds its
# step already finished reloads the models, skips every image and exits.
#   CHAIN=3 bash server_bundle/submit.sh debate_stroke
# AFTER=<jobid> makes the first job wait for another one to succeed (e.g. the smoke test).
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$BUNDLE_DIR/.." && pwd)"
# shellcheck source=config.env
source "$BUNDLE_DIR/config.env"
# shellcheck source=steps.sh
source "$BUNDLE_DIR/steps.sh"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: no sbatch here. Run scripts/cluster_info.sh and adapt this script to the scheduler it finds."; exit 1; }
[ "$#" -ge 1 ] || { sed -n '2,18p' "$0"; exit 2; }

mkdir -p "$PROJECT_ROOT/logs"

SBATCH_ARGS=(
    --chdir="$PROJECT_ROOT"
    --gres="$SLURM_GRES"
    --time="$SLURM_TIME"
    --cpus-per-task="$SLURM_CPUS"
    --mem="$SLURM_MEM"
    --output="$PROJECT_ROOT/logs/slurm-%x-%j.out"
)
[ -n "$SLURM_PARTITION" ] && SBATCH_ARGS+=(--partition="$SLURM_PARTITION")
[ -n "$SLURM_ACCOUNT" ] && SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")
[ -n "$SLURM_EXTRA" ] && read -ra extra <<< "$SLURM_EXTRA" && SBATCH_ARGS+=("${extra[@]}")

# Submit one command CHAIN times, each copy after the previous; prints the last job id.
submit_chain() {
    local name="$1" cmd="$2" dep="${AFTER:+afterok:$AFTER}" id="" i
    for ((i = 1; i <= ${CHAIN:-1}; i++)); do
        id=$(sbatch --parsable --job-name="maclf-$name" \
            ${dep:+--dependency="$dep"} "${SBATCH_ARGS[@]}" \
            --wrap="$cmd")
        id="${id%%;*}"
        echo "submitted $name  job $id${dep:+  (dependency $dep)}" >&2
        dep="afterany:$id"
    done
    echo "$id"
}

# Steps after `pair` share the GPU SLURM gave the job, so pass its id, not a literal 0
# (without cgroup device isolation 0 would be someone else's GPU).
GPU='${CUDA_VISIBLE_DEVICES:-0}'

case "$1" in
    preflight) submit_chain preflight "bash server_bundle/00_preflight.sh" >/dev/null ;;
    smoke)     submit_chain smoke     "bash server_bundle/10_smoke.sh" >/dev/null ;;
    pair)
        shift
        [ "$#" -ge 2 ] || { echo "pair needs at least two steps"; exit 2; }
        slots=$(printf "$GPU %.0s" "$@")
        submit_chain "pair-$1" "SKIP_GATES=1 STEPS='$*' bash server_bundle/run_parallel.sh $slots" >/dev/null
        ;;
    *)
        if [ "$1" = all ]; then set -- "${V2_STEPS[@]}"; fi
        for step in "$@"; do
            case " ${V2_STEPS[*]} nosam_binary_tumor debate1_binary_tumor debate1_multiclass_tumor debate1_ms debate1_stroke " in
                *" $step "*) ;;
                *) echo "unknown step '$step' (see server_bundle/steps.sh)"; exit 2 ;;
            esac
        done
        for step in "$@"; do
            submit_chain "$step" "bash server_bundle/step.sh $step" >/dev/null
        done
        ;;
esac

echo "Watch: squeue -u \$USER    Logs: logs/<step>.log and logs/slurm-maclf-*.out"
