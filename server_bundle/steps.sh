# The v2 step list (RERUN_PLAN.md: Tier 2 + H1 + H2), in run order. Sourced by
# run_all.sh, run_parallel.sh and 10_smoke.sh. Each name is <system>_<task> for step.sh.
# shellcheck shell=bash
#
# The multiclass CNN label order was fixed in agents/cnn_tool.py (2026-10-01, from
# 05_diagnose_cnn), so multiclass runs alongside the other tasks. Order is by priority
# (RERUN_PLAN.md §3): Tier 2 first, then the triage-only H1/H2 controls.
#
# Optional extra, not in the default list (add it via STEPS=...):
#   nosam_binary_tumor  S1, debate without the SAM3 advocate (~10 h)
# shellcheck disable=SC2034
V2_STEPS=(
    base_binary_tumor     base_multiclass_tumor     base_ms     base_stroke
    forest_binary_tumor   forest_multiclass_tumor   forest_ms   forest_stroke
    debate_binary_tumor   debate_multiclass_tumor   debate_ms   debate_stroke
    homog_binary_tumor    homog_multiclass_tumor    homog_ms    homog_stroke
    rolesamp_binary_tumor rolesamp_multiclass_tumor rolesamp_ms rolesamp_stroke
)
