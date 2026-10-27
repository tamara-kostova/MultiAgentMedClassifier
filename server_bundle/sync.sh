#!/usr/bin/env bash
# Run on THIS machine (not the cluster). Moves the bundle to the OSC GPU cluster and the
# results back, over SSH. Needs the faculty network/VPN: 10.10.65.6 is a private address.
#
#   bash server_bundle/sync.sh check     # can we reach the login node, and as whom
#   bash server_bundle/sync.sh push      # upload ../bundle_out/* (resumable) + verify SHA256SUMS
#   bash server_bundle/sync.sh pull      # results -> outputs/eval/v2/, logs + TSVs -> outputs/cluster/
#   bash server_bundle/sync.sh status    # queue, recent jobs, images done per step
#   bash server_bundle/sync.sh ssh       # interactive shell in the project dir
#
# CLUSTER_HOST / CLUSTER_DIR / BUNDLE_OUT come from config.env and can be overridden:
#   CLUSTER_HOST=tamara.kostova@10.10.65.6 bash server_bundle/sync.sh check
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$BUNDLE_DIR/.." && pwd)"
# shellcheck source=config.env
source "$BUNDLE_DIR/config.env"

BUNDLE_OUT="${BUNDLE_OUT:-$PROJECT_ROOT/../bundle_out}"
REMOTE_PROJECT="$CLUSTER_DIR/MultiAgentMedClassifier"

remote() { ssh "$CLUSTER_HOST" "$@"; }

case "${1:-}" in
check)
    remote 'echo "logged in as $(id -un) on $(hostname -f 2>/dev/null || hostname); home $HOME"; df -h "$HOME" | tail -1'
    ;;
push)
    for f in maclf-code-data.tar.gz maclf-models.tar SHA256SUMS.txt; do
        [ -f "$BUNDLE_OUT/$f" ] || { echo "missing $BUNDLE_OUT/$f — run scripts/pack_bundle.sh first"; exit 1; }
    done
    remote "mkdir -p '$CLUSTER_DIR'"
    # --partial: an interrupted 12 GB upload resumes instead of restarting.
    # SKIP_MODELS=1: send only the code archive (the server already has hf_cache/).
    files=("$BUNDLE_OUT/maclf-code-data.tar.gz" "$BUNDLE_OUT/SHA256SUMS.txt")
    [ "${SKIP_MODELS:-0}" = "1" ] || files+=("$BUNDLE_OUT/maclf-models.tar")
    rsync -av --partial --progress "${files[@]}" "$CLUSTER_HOST:$CLUSTER_DIR/"
    echo "--- verifying checksums on the cluster ---"
    remote "cd '$CLUSTER_DIR' && sha256sum -c --ignore-missing SHA256SUMS.txt"
    echo
    echo "Next, on the cluster (see server_bundle/SELF_RUN.md §3):"
    echo "  cd $CLUSTER_DIR && tar xzf maclf-code-data.tar.gz && tar xf maclf-models.tar -C MultiAgentMedClassifier/"
    ;;
pull)
    mkdir -p "$PROJECT_ROOT/outputs/eval/v2" "$PROJECT_ROOT/outputs/cluster"
    # v2 JSONLs only ever go to outputs/eval/v2/, never over the published outputs/eval/*.jsonl.
    rsync -av "$CLUSTER_HOST:$REMOTE_PROJECT/outputs/eval/v2/" "$PROJECT_ROOT/outputs/eval/v2/"
    rsync -av --exclude='*.jsonl' \
        "$CLUSTER_HOST:$REMOTE_PROJECT/logs" \
        "$CLUSTER_HOST:$REMOTE_PROJECT/outputs/results_tsv" \
        "$PROJECT_ROOT/outputs/cluster/"
    echo "JSONLs in outputs/eval/v2/, logs and TSVs in outputs/cluster/."
    ;;
status)
    # Queue, recent job outcomes, and images done per step (500 = finished).
    remote "cd '$REMOTE_PROJECT' && squeue -u \$USER -o '%.6i %.34j %.8T %.10M %R' && echo &&
        sacct -X -S now-3days -o JobID,JobName%34,State,Elapsed -u \$USER | grep -v -- '-pair-.*PENDING' | tail -12 && echo &&
        for f in $V2_OUT/*.jsonl; do [ -f \"\$f\" ] && printf '%-44s %4s / 500\n' \"\$(basename \$f)\" \"\$(grep -c '\"error\": null' \$f)\"; done"
    ;;
ssh)
    exec ssh -t "$CLUSTER_HOST" "cd '$REMOTE_PROJECT' 2>/dev/null || cd '$CLUSTER_DIR'; exec \$SHELL -l"
    ;;
*)
    sed -n '2,14p' "$0"
    exit 2
    ;;
esac
