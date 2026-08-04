#!/usr/bin/env bash
# Step 0 — verify the environment and prove both pipelines run. Takes ~5-15 min.
# Run this first. If it prints "PREFLIGHT OK", the long runs can be started.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"

STEP=00_preflight
log_start "$STEP"
pyrun server_bundle/scripts/preflight.py 2>&1 | tee -a "logs/${STEP}.log"
status=$?
log_done "$STEP"
exit $status
