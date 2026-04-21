# Deep Research for an Agentic Neuroimaging Classifier

## Executive Summary

Deep research in AI refers to multi-step, evidence-seeking workflows where an agent (or team of agents) decomposes a complex question, retrieves and evaluates diverse sources or experiments, and then synthesizes a structured answer with explicit uncertainty and citations. In contrast to single-pass inference, deep research agents repeatedly refine queries, adjust tools and data, and maintain a persistent research state. Applied to an agentic neuroimaging classifier, deep research means treating the entire MedGemma–CNN–SAM3–BiomedCLIP pipeline as an object of systematic investigation, not just as a fixed model.[^1][^2][^3][^4][^5]

This report first formalizes the concept of deep research agents and their typical architecture, then maps those principles onto a LangGraph-based neuroimaging pipeline for brain tumour, MS, and stroke classification. It then documents the implemented research orchestration layer and the experiment families available for thesis experiments.[^2][^6][^7][^1]

## 1. Concept of Deep Research in AI

### 1.1 From single-shot models to research workflows

Deep research emerged as a response to the limitations of single-shot LLM answers, especially for complex or open-ended questions where naïve retrieval or one-pass reasoning is unreliable. Instead of answering immediately, a deep research system:[^3][^4][^5][^2]

- Decomposes the original query into subquestions or tasks.
- Plans a sequence of retrieval and analysis steps.
- Executes those steps with tools (web search, code, databases, simulators, or experiments).
- Cross-checks and reconciles conflicting evidence.
- Produces a final synthesis with explicit structure and citations.

Industrial systems such as OpenAI's deep research mode and Perplexity's Deep Research perform multiple rounds of querying, iterative retrieval, and cross-source validation before generating a long-form, source-grounded report. Academic work on deep research agents generalizes this idea to architectures where LLM-based planners orchestrate tool calls, manage external memory, and refine workflows over multiple iterations.[^8][^9][^4][^10][^5][^11][^1][^2][^3]

### 1.2 Key properties of deep research agents

Across industry and research, deep research agents share several characteristics:[^4][^5][^1][^2][^3]

- **Query decomposition:** splitting a high-level goal into subproblems (e.g., literature review, experimental design, error analysis, robustness checks).
- **Multi-pass retrieval and reasoning:** running several rounds of search, experiment, or simulation, where later steps depend on earlier findings.
- **Tool orchestration:** invoking specialized tools (search, code execution, data loaders, evaluation scripts) as first-class actions.
- **Persistent research state:** maintaining notes, intermediate results, and hypotheses across steps.
- **Cross-validation and uncertainty handling:** explicitly checking for conflicting evidence and reporting limitations.
- **Structured synthesis:** generating reports with sections, tables, and citations that reflect the underlying workflow.

These properties align naturally with multi-agent frameworks in clinical AI, where ensembles of agents handle image analysis, report drafting, critique, and consensus formation.[^6][^12][^13][^7][^14]

## 2. Multi-Agent Systems in Medical Imaging as Deep Research Environments

### 2.1 Multi-agent radiology and neurological reasoning

Recent work in radiology and neurology has started to frame multi-agent systems as research environments rather than static pipelines. For example, multi-agent frameworks for radiology report generation use separate agents for detection, segmentation, description, critique, and consensus evaluation, allowing fine-grained measurement at both the component and system level.[^12][^13][^7][^6]

These systems emphasize:

- Agent-level metrics (e.g., segmentation Dice, detection sensitivity, calibration).
- Consensus-level metrics (e.g., report completeness, clinical correctness, agreement with radiologists).
- Modular experimentation, where individual agents or routing rules can be swapped and re-evaluated without rebuilding the entire system.[^7][^6][^12]

This perspective aligns well with the LangGraph neuroimaging pipeline, where MedGemma, CNNs, SAM3, and BiomedCLIP are already modular tools coordinated by routing logic.

### 2.2 Agentic maintenance and continual evaluation

Recent medical-agent frameworks such as ReclAIm demonstrate how agents can monitor model performance over time, detect degradation, and autonomously trigger fine-tuning or re-evaluation. They treat evaluation and maintenance as ongoing research tasks controlled by an agentic layer rather than as offline, one-shot experiments.[^7]

For this thesis, deep research is framed as an **agentic evaluation and analysis layer** on top of the classifier pipeline, responsible for planning experiments, running evaluation scripts, analyzing results (e.g., from `comparison_summary.csv`), and producing structured analyses (tables, routing distributions, calibration breakdowns) for thesis chapters.[^9][^10][^5][^1]

## 3. Mapping Deep Research Concepts onto the Pipeline

### 3.1 Treating the pipeline as a research subject

The current system has rich structure:

- **MedGemmaAgent:** triage routing, bbox-guided diagnosis, and final report.
- **CNNClassifier:** task-specific classification with VGG16, DenseNet169, and ResNet101.
- **SAM3Tool:** lesion segmentation with a BraTS-trained linear probe (Dice ≈ 0.836 on tumour).[^6][^7]
- **BiomedCLIPTool:** layer-6 feature extraction (ViT-B/16 middle layer) for zero-shot re-ranking of ambiguous multiclass tumour cases. Layer 6 was identified as optimal across all four neuroimaging tasks — see Section 4.2.
- **Calibration and explainability:** Grad-CAM/Grad-CAM++, Integrated Gradients, and temperature scaling with ECE.

The research layer built around these components can:

- Plan experiment sweeps over RoutingConfig parameters (thresholds, always_run_sam3, always_run_biomedclip).
- Run `eval/evaluate.py`'s `compare_configurations()` programmatically for many configurations, reusing loaded model agents to avoid repeated weight loading.
- Parse `comparison_summary.csv` and `all_predictions.csv`.
- Generate structured analyses (routing distributions, sensitivity–specificity tables, calibration breakdowns).

### 3.2 Implemented research graph

The research orchestration layer is a LangGraph `StateGraph` with four nodes in a linear chain:

```
plan_experiments → run_experiments → analyze_results → write_report → END
```

**`ResearchState` fields:**

| Field | Type | Set by |
|---|---|---|
| `experiment_family` | `str` | CLI input |
| `dataset_dirs` | `dict[str, str]` | CLI input |
| `output_base` | `str` | CLI input |
| `base_cfg` | `PipelineConfig` | CLI (DEFAULT_CONFIG) |
| `preloaded_agents` | `tuple` | CLI (load_agents()) |
| `results_dir` | `str` | plan_experiments |
| `sweep_summary` | `pd.DataFrame` | run_experiments |
| `analysis` | `dict` | analyze_results |
| `report_md` | `str` | write_report |

**Node responsibilities:**

- `plan_experiments`: validates the family name, creates a timestamped output directory, logs the full sweep plan.
- `run_experiments`: loads test samples via `load_test_split()`, calls `run_experiment_family()` from `research/runner.py`, which assembles a new LangGraph pipeline per sweep point (same agents, different `RoutingConfig`) and calls `compare_configurations()`.
- `analyze_results`: runs four analysis functions from `research/analysis.py` on the merged results.
- `write_report`: renders a Markdown report with all analysis tables to `analysis/report.md`.

**CLI:**

```bash
python run_research.py \
  --family ablation \
  --binary_tumor_dir data/test/binary_tumor \
  --multiclass_dir   data/test/multiclass_tumor \
  --ms_dir           data/test/ms \
  --stroke_dir       data/test/stroke
```

## 4. Experiment Families and Research Questions

### 4.1 Routing and specificity–sensitivity trade-offs

Prior work showed that always using SAM3 before MedGemma increased tumour detection from roughly 85% to 96% but reduced specificity from about 67% to 41%, illustrating the classic sensitivity–specificity trade-off when adding a sensitive segmentation step. The current agentic routing (MedGemma confidence thresholds plus optional SAM3/BiomedCLIP) is explicitly designed to recover specificity.[^6][^7]

**Implemented experiment families:**

| Family | What varies | Points | Core research question |
|---|---|---|---|
| `threshold_sweep` | `sam3_threshold` ∈ [0.50, 0.85] | 8 | Where on the ROC curve does confidence-gated SAM3 operate? |
| `human_review_sweep` | `human_review_threshold` ∈ [0.30, 0.55] | 6 | What is the accuracy–coverage trade-off for human deferral? |
| `ablation` | full / no_sam3 / always_sam3 / no_biomedclip | 4 | What does each component contribute independently? |
| `biomedclip_threshold` | `biomedclip_rerank_threshold` ∈ [0.50, 0.80] | 7 | How sensitive is multiclass accuracy to the BiomedCLIP reranking threshold? |

`threshold_sweep` and `ablation` are the primary experiments for the thesis routing chapter. `human_review_sweep` and `biomedclip_threshold` are secondary.

**Metrics collected per sweep point** (from `comparison_summary.csv`):
- `accuracy`, `f1_macro`, `roc_auc`
- `normal_specificity` — specificity on normal scans (key metric for SAM3 trade-off)
- `sam3_invocation_rate` — what fraction of cases trigger SAM3
- `human_review_rate`
- `mean_latency_s`, `ece`

**Analysis outputs** (from `research/analysis.py`):

| Function | Output | Thesis chapter |
|---|---|---|
| `sensitivity_specificity_table()` | Pivot: threshold → specificity + sam3_rate + accuracy | Routing chapter |
| `ablation_summary()` | Component contribution per task | Routing chapter |
| `routing_distribution()` | Decision counts/% per task and config | System description |
| `calibration_by_routing_path()` | ECE per path (sam3_then_cnn, cnn_direct, etc.) | Calibration chapter |


### 4.2 Calibration and uncertainty as first-class citizens

Temperature scaling and ECE are already implemented. The `calibration_by_routing_path()` analysis function answers:

- Does adding SAM3-guided routing improve or worsen calibration for tumour predictions?
- Are cases routed through `sam3_then_cnn` better calibrated than `cnn_direct` cases?
- Does disagreement between MedGemma and CNN (high routing uncertainty) correlate with miscalibration?

These can be investigated by running `threshold_sweep` or `ablation` and reading `analysis/report.md`.

### 4.3 Explainability as a research dimension

Grad-CAM++ and Integrated Gradients are already generated when `--generate_explainability` is passed. The existing IoU check between Grad-CAM++ maps and SAM3 masks (low IoU < 0.30 triggers confidence penalty and `requires_human_review=True`) provides a quantitative plausibility signal. Future work can:

- Stratify accuracy and calibration by saliency plausibility (IoU bins).
- Compare saliency plausibility across routing paths.

## 5. Repository Structure

```
MultiAgentMedClassifier/
├── pipeline/
│   └── graph.py           load_agents() + assemble_pipeline() + build_pipeline()
├── research/
│   ├── experiments.py     SweepPoint dataclass + EXPERIMENT_FAMILIES dict
│   ├── runner.py          run_experiment_family() — sweep loop + CSV output
│   ├── analysis.py        4 analysis functions returning thesis-ready DataFrames
│   └── graph.py           ResearchState + build_research_pipeline() (4-node LangGraph)
├── eval/
│   └── evaluate.py        compare_configurations(), PipelineEvaluator, compute_ece()
└── run_research.py        CLI entry point
```

**Output structure** (per family run):

```
outputs/research/{family}_{timestamp}/
├── sweep_manifest.json          metadata: configs, timings, paths
├── results/{experiment_id}/
│   ├── comparison_summary.csv   metrics per (config, task)
│   └── all_predictions.csv      per-sample: routing_path, confidence, correct
└── analysis/
    ├── sweep_summary.csv        merged across all sweep points
    └── report.md                Markdown tables for all 4 analyses
```

## 6. Thesis Positioning

### 6.1 Two-layer contribution

The thesis makes contributions at two levels:

1. **Diagnostic agent layer** — MedGemma + CNNs + SAM3 + BiomedCLIP with confidence-gated routing, post-hoc calibration, explainability, and FHIR output.
2. **Research agent layer** — LangGraph orchestrator that plans experiment sweeps, runs the diagnostic pipeline at scale, analyzes routing and calibration behaviour, and produces structured reports.

The second layer lets the thesis claim a contribution in **agentic neuroimaging evaluation**, not only in classification performance.

### 6.2 Results narrative

When presenting results, emphasize:

- How different routing policies trade sensitivity vs specificity and calibration — quantified by the `threshold_sweep` and `ablation` experiment families.
- Where BiomedCLIP layer-wise analysis reveals representation strengths (layer 6 dominant) and domain-specific limitations (CLIP fails on MS; BiomedCLIP resolves it).
- How calibration varies by routing path — are confidently-routed cases actually better calibrated?
- How the research agent layer itself demonstrates deep research: the system was used to **investigate** its own routing behaviour across 25 configurations, not just achieve high accuracy.

## 7. What Is and Isn't Built

### Implemented

- `research/` module with all four files
- `run_research.py` CLI
- `pipeline/graph.py` split into `load_agents()` + `assemble_pipeline()` (backward-compatible)
- All four experiment families (25 sweep points total)
- All four analysis functions in `research/analysis.py`
- BiomedCLIP layer correction: layer 6 (ViT-B/16, 0-indexed), block path `model.visual.trunk.blocks`, batch-first CLS extraction
- Probe head loader for MLP checkpoints from `18_layer_fusion_benchmark.py`

### Not built (future work)

- **Matplotlib visualization** — analysis functions return DataFrames; plotting can be added to `research/analysis.py` as needed.
- **LLM-based research planning** — experiment families are predefined Python lists, which is more reproducible for thesis evaluation. An LLM planner could be added as a `plan_experiments` node variant.
- **Parallel sweep execution** — sweep is sequential to avoid GPU OOM with MedGemma loaded. Could be parallelized on multi-GPU setup.
- **Weighted fusion probe support** — `BiomedCLIPTool` currently loads concat-fusion and single-layer probes; `FusionWeightedClassifier` checkpoints are not yet handled.
- **Radiologist-validated saliency quantification** — Grad-CAM++ maps are generated; quantitative IoU-vs-performance stratification is future work.

---

## References

1. [How to Build a Deep-Research Multi‑Agent System | Langflow](https://www.langflow.org/blog/how-to-build-a-deep-research-multi-agent-system) - Deep research isn't a single prompt—it's a workflow: break a question down, fetch sources, extract s...

2. [Perplexity AI Deep Research: How It Works, Limitations, and Use ...](https://www.datastudios.org/post/perplexity-ai-deep-research-how-it-works-limitations-and-use-cases-for-professionals) - Perplexity AI's Deep Research mode is the company's most advanced workflow for generating long-form,...

3. [Introducing deep research - OpenAI](https://openai.com/index/introducing-deep-research/) - Today we're launching deep research in ChatGPT, a new agentic capability that conducts multi-step re...

4. [Introducing Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) - Deep Research accelerates question answering by completing in 2-4 minutes what would take a human ex...

5. [Deep Research Agents: A Systematic Examination And Roadmap](https://arxiv.org/html/2506.18096v2)

6. [Medical AI Consensus: A Multi-Agent Framework for Radiology ...](https://arxiv.org/html/2509.17353v1) - We introduce a multi-agent reinforcement learning framework that serves as both a benchmark and eval...

7. [ReclAIm: A multi-agent framework for degradation-aware ...](https://www.arxiv.org/pdf/2510.17004.pdf)

8. [What is Perplexity AI's Deep Research mode? - First AI Movers](https://www.firstaimovers.com/p/what-is-perplexity-ai-s-deep-research-mode) - Perplexity's Research mode (formerly "Deep Research") is an advanced feature where the AI spends a f...

9. [Building a Deep Research Agent Using MCP-Agent | Blog - AI Alliance](https://thealliance.ai/blog/building-a-deep-research-agent-using-mcp-agent) - Learn how MCP-Agent powers Deep Research Agents with simple orchestration, external memory, and scal...

10. [Create an Open Deep Research Multi-Agent in Python (Step by Step)](https://www.youtube.com/watch?v=vHBRmXpDIFY) - Build a complete multi-agent deep research system with open-source models: use Hugging Face Inferenc...

11. [What is Perplexity Deep Research – A Detailed Overview](https://www.usaii.org/ai-insights/what-is-perplexity-deep-research-a-detailed-overview) - This Perplexity AI tool can generate work artifacts across various niches, including finance, market...

12. [A multi-agent approach to neurological clinical reasoning - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12677565/) - We evaluated ten LLMs of varying architectures and specializations using this benchmark, testing bas...

13. [Towards Robust Evaluation of Multi-Agent Systems in Clinical Settings](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/towards-robust-evaluation-of-multi-agent-systems-in-clinical-settings/4435119) - As multi-agent systems become more capable and autonomous, robust evaluation must evolve in parallel...

14. [Multi-agent systems for clinical decision support: A systematic review](https://www.sciencedirect.com/science/article/abs/pii/S1568494625017600) - The methodology was designed to comprehensively identify, evaluate, and synthesize research at the i...
