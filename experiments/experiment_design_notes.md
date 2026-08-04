# Experiment Design Notes

## Purpose

The `experiments/` module is the sweep orchestration layer on top of the diagnostic pipeline. It runs the pipeline at scale across many configurations, collects per-sample predictions and aggregate metrics, and produces thesis-ready DataFrames and Markdown reports.

It does not contain agents or pipeline logic — those live in `agents/` and `pipeline/`.

---

## Experiment Families

Six families are defined in `experiments.py`. Each is a list of `SweepPoint` objects (experiment_id, description, routing_overrides, pipeline_mode, pipeline_kwargs).

### Core: System A routing ablations

| Family | What varies | Points | Primary thesis question |
|---|---|---|---|
| `threshold_sweep` | `sam3_threshold` ∈ [0.50, 0.85] | 8 | Where on the ROC curve does confidence-gated SAM3 operate? |
| `human_review_sweep` | `human_review_threshold` ∈ [0.30, 0.55] | 6 | What is the accuracy–coverage trade-off for human deferral? |
| `ablation` | full / no_sam3 / always_sam3 / no_biomedclip | 4 | What does each component contribute independently? |
| `biomedclip_threshold` | `biomedclip_rerank_threshold` ∈ [0.50, 0.80] | 7 | How sensitive is multiclass accuracy to the BiomedCLIP reranking threshold? |

`threshold_sweep` and `ablation` are primary. `human_review_sweep` and `biomedclip_threshold` are secondary.

### System B and C comparisons

| Family | What varies | Points | Primary thesis question |
|---|---|---|---|
| `debate_rounds` | debate rounds R ∈ {1, 2, 3} | 3 | Does multi-round advocate debate outperform single-step verification? |
| `agent_forest` | N agents ∈ {1, 3, 4} | 3 | Does role-diverse ensemble triage outperform single-agent triage? |

**Total: 31 sweep points across 6 families.**

---

## Metrics collected

Per sweep point, `comparison_summary.csv` contains:

| Metric | Meaning |
|---|---|
| `accuracy`, `f1_macro`, `roc_auc` | Standard classification metrics |
| `normal_specificity` | Specificity on normal scans — key SAM3 trade-off metric |
| `sam3_invocation_rate` | Fraction of cases that triggered SAM3 |
| `human_review_rate` | Fraction deferred to human review |
| `mean_latency_s` | Mean per-sample wall-clock time |
| `ece` | Expected Calibration Error (post temperature scaling) |

Per-sample `all_predictions.csv` additionally carries `routing_path`, `confidence`, `correct`, and (where applicable) `dissent_rate`, `vote_fraction`, `debate_rounds_completed`, `debate_round_changed`.

---

## Analysis functions (`analysis.py`)

| Function | Output | Used for |
|---|---|---|
| `sensitivity_specificity_table()` | Pivot: threshold → specificity + sam3_rate + accuracy | Routing chapter |
| `ablation_summary()` | Component contribution per task | Routing chapter |
| `routing_distribution()` | Decision counts/% per task and config | System description |
| `calibration_by_routing_path()` | ECE per routing path per task | Calibration chapter |
| `forest_voting_analysis()` | dissent_rate, unanimous_pct, accuracy by agreement level | System C chapter |
| `debate_round_analysis()` | pct_verdict_changed, ECE/accuracy split by verdict stability | System B chapter |

All functions return DataFrames suitable for `.to_markdown()` or `.to_latex()`.

---

## Orchestration graph (`graph.py`)

4-node linear LangGraph: `plan_experiments → run_experiments → analyze_results → write_report → END`

- `plan_experiments` — validates family name, creates timestamped output directory, logs sweep plan.
- `run_experiments` — calls `run_experiment_family()` from `runner.py`; assembles a pipeline per sweep point (no model reloading), calls `compare_configurations()`.
- `analyze_results` — runs all applicable analysis functions on merged results.
- `write_report` — renders DataFrames to `analysis/report.md`.

---

## Output structure

```
outputs/research/{family}_{timestamp}/
├── sweep_manifest.json
├── results/{experiment_id}/
│   ├── comparison_summary.csv
│   └── all_predictions.csv
└── analysis/
    ├── sweep_summary.csv
    └── report.md
```

---

## Threshold choices

- `sam3_threshold` default 0.70 — chosen to balance sensitivity recovery vs specificity loss from unconditional SAM3. The `threshold_sweep` family quantifies where this sits on the ROC curve.
- `human_review_threshold` default 0.45 — cases below this confidence are deferred. The `human_review_sweep` shows the accuracy–coverage trade-off at each setting.
- `biomedclip_rerank_threshold` default 0.65 — minimum CLIP score margin to accept re-ranking. The `biomedclip_threshold` family shows sensitivity to this choice for multiclass tumour cases.

The ablation (`full / no_sam3 / always_sam3 / no_biomedclip`) answers the examiner question "how did you choose these thresholds?" — by showing what each component contributes when removed or forced on.
