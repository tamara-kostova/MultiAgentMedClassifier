# The v2 step list (RERUN_PLAN.md: Tier 2 + H1), in run order. Sourced by run_all.sh
# and run_parallel.sh. Each name is <system>_<task> for step.sh.
# shellcheck shell=bash
#
# Multiclass tumour runs come last: they depend on the multiclass CNN, which
# 05_diagnose_cnn checks first. If that diagnosis finds a bug, stop before them.
#
# Optional extras, not in the default list (add them via STEPS=...):
#   rolesamp_<task>   H2, role-diverse forest sampled at FOREST_TEMPERATURE (~6 h each)
#   nosam_binary_tumor  S1, debate without the SAM3 advocate (~10 h)
# shellcheck disable=SC2034
V2_STEPS=(
    base_binary_tumor   base_ms   base_stroke
    forest_binary_tumor forest_ms forest_stroke
    debate_binary_tumor debate_ms debate_stroke
    homog_binary_tumor  homog_ms  homog_stroke
    base_multiclass_tumor forest_multiclass_tumor debate_multiclass_tumor homog_multiclass_tumor
)
