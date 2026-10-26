# Neuroimaging multi-agent classifier — evaluation results (v2 campaign, October 2026)

Summary tables of the final evaluation campaign, derived from `outputs/eval/v2/*.jsonl` (one row per
image with the output of every agent; 65 MB, kept out of git). Regenerate the `paired_*`, `mcnemar_tests`,
`holm_family` and `system_inputs` files with `eval/paired_system_comparison.py --system <name> <task> <jsonl>`
over the v2 files, and `thesis_numbers.csv` from `Master_Thesis/generated/results.json`
(`eval/make_thesis_results.py`).

## What was evaluated

Three complete end-to-end pipeline variants, each run on the **same 500 images per task**, four tasks
(binary_tumor = Br35H, multiclass_tumor = Figshare meningioma/glioma/pituitary, ms = MS FLAIR dataset,
stroke = Stroke CT dataset, clean PNGs only):

| system | pipeline |
|---|---|
| `base`   | MedGemma triage → CNN → SAM3 → BiomedCLIP → explainability → verification → MedGemma report → FHIR |
| `forest` | 4 role-specialised MedGemma agents + majority vote replace the single triage; rest as base |
| `debate` | tool advocates (CNN / BiomedCLIP / SAM3) + MedGemma judge replace verification + report; 2 rounds |
| `debate1`, `debate3` | the same debate with 1 and 3 rounds |

Two stages are scored per image: `triage` (the first MedGemma agent / the forest vote, before any tool)
and `final` (the end of the whole flow). All models: MedGemma 1.5 4B (bf16, greedy), task CNNs, SAM3 +
linear probe (tumor tasks only), BiomedCLIP linear probes. Commit 19717fa.

Scoring: strict canonical label match; an empty or unparseable prediction counts as **wrong** (never dropped);
forest triage is recomputed from the saved ballots. ECE with 10 bins (first bin [0, 0.1]). Wilson 95% CIs.
McNemar exact tests, Holm-Bonferroni corrected over the family listed in `holm_family.csv`.

## Files

- `paired_accuracy.csv` — one row per (task, system, stage): n, correct, accuracy, Wilson CI, ECE,
  mean confidence, sensitivity/specificity (binary-style tasks).
- `mcnemar_tests.csv`, `holm_family.csv` — paired tests debate vs base / debate vs forest.
- `paired_raw_scores.csv` — per-image correctness for every system and stage (for your own tests).
- `system_inputs.csv` — which JSONL each row came from and data-quality counters.
- `thesis_numbers.csv` — every number reported in the thesis as `key,value`, including the two
  control systems and the voting/latency analyses that are not in `paired_accuracy.csv`. Key prefix:
  `bin`/`multi`/`ms`/`stroke` = task; `A` = base, `B` = forest, `C` = debate R=2, `D1`/`D3` = debate
  R=1/R=3, `H` = homogeneous forest control, `R` = role-sampling control; then stage and metric
  (`acc`, `lo`/`hi` = Wilson CI, `ece`, `sens`, `spec`, `conf`).
