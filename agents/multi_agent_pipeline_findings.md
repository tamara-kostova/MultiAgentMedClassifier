# Multi-Agent Pipeline: Implementation and Findings

## Overview

This document records the design rationale and implementation of the two multi-agent system variants added to the pipeline, grounded in the literature review. It is distinct from the experiment orchestration layer (`experiments/`), which sweeps over these systems programmatically.

The baseline pipeline (System A) is a sequential specialist graph. Two additional systems were implemented:

- **System B — Multi-Agent Debate** (`agents/debate.py`): three MedGemma advocates + MedGemma judge, 1–3 rounds. Inspired by Du et al. and Liang et al. (MAD).
- **System C — Agent Forest** (`agents/forest.py`): N role-specialised MedGemma instances + majority vote. Inspired by Li et al.

---

## 1. Literature Grounding

### 1.1 Multi-Agent Debate — Du et al. and Liang et al.

Du et al. showed that having multiple LLM instances independently propose and then debate their answers — each reading and critiquing the others' responses before updating — consistently outperforms single-model self-reflection. The key mechanism is that disagreement between agents forces each one to engage with external evidence rather than reinforcing its own prior.

Liang et al. formalised this as the Multi-Agent Debate (MAD) framework and identified a concrete failure mode in single-agent reflection: once a model becomes confident in an answer, it cannot generate novel thoughts even if the answer is wrong (Degeneration-of-Thought / DoT). A judge-managed debate with advocates in "tit for tat" breaks this degeneracy. They also found that using the same model family for both advocates and judge avoids calibration biases introduced by mixing LLM families.

**Mapping to System B:** The verification node in the baseline pipeline is a single-round implicit debate (MedGemma checks CNN output against the saliency map). System B makes this explicit: three advocates argue from different tool outputs (CNN probabilities + GradCAM++, BiomedCLIP similarity scores, SAM3 morphological evidence), and a fourth MedGemma instance acts as judge. Round N>1 gives each advocate the prior verdict and all other arguments, allowing position revision — directly operationalising the MAD round structure.

### 1.2 Agent Forest — Li et al.

Li et al. demonstrated that sampling N independent instances of the same LLM with majority voting reliably improves accuracy without any architectural change. The gains are largest on harder problems where the single-instance answer is uncertain. Homogeneous ensemble size (N agents, same weights, varied temperature/prompt) is their baseline.

**Mapping to System C:** Rather than homogeneous instances, System C uses role-specialised prompts: radiologist (visual pattern recognition), conservative (specificity-focused), emergency (sensitivity-focused), differential (uncertainty-aware). This is a heterogeneous forest — each agent applies the same underlying MedGemma weights but with a different clinical reasoning frame, providing diversity through prompt rather than temperature. This is a deliberate contrast to Li et al.'s homogeneous baseline and is framed as such in the thesis.

### 1.3 Answer Verification — Agashe et al.

Agashe et al. found that adding an explicit answer-verification step significantly reduces fatal mistakes caused by LLM hallucinations. The baseline pipeline's `verification_node` independently operationalises this finding: MedGemma cross-checks the CNN prediction against the saliency map before writing the final report. System B subsumes and generalises this — the judge role acts as a structured verifier across three evidence streams simultaneously.

---

## 2. System B — Multi-Agent Debate

### 2.1 Architecture

```
triage → cnn_classify → sam3_segment → biomedclip → explainability
    → debate (DebateOrchestrator)
    → fhir_output → END
```

The debate node replaces `verification + report`. All upstream nodes (CNN, SAM3, BiomedCLIP, explainability) remain unchanged; their outputs become the inputs the advocates argue from.

### 2.2 Advocate roles

| Advocate | Input evidence | Argument frame |
|---|---|---|
| CNN | `classification_result` (predicted class, confidence, all_probs), GradCAM++ availability | Confidence margin over competing classes; saliency corroboration |
| BiomedCLIP | `biomedclip_result` (top_label, top_score, ranked_labels) | Similarity score margin between top and second prediction |
| SAM3 | `segmentation_result` (bbox, mask_area, lesion detected) | Spatial/morphological evidence; absence of segmentable pathology if no lesion |

### 2.3 Round structure

- **Round 1:** Each advocate receives the image + its tool output and generates an argument independently.
- **Round N>1:** Each advocate receives its tool output, the prior verdict (winner, confidence, reason), and all round-(N-1) arguments from the other two advocates. It may revise or reinforce its position.
- **Judge:** Receives all three arguments (and prior verdict section for N>1). Responds in structured JSON: `winner`, `winner_detailed`, `confidence`, `reason`, `round_changed`.

Rounds are clamped to 1–3 (`DebateOrchestrator.MAX_ROUNDS`). The same MedGemma weights are used for all four roles (three advocates + judge), consistent with Liang et al.'s finding that mixing LLM families introduces calibration bias.

### 2.4 Experiment family: `debate_rounds`

| Sweep point | Description |
|---|---|
| `debate_r1` | Single-round arbitration (baseline debate) |
| `debate_r2` | Advocates respond to round-1 verdict |
| `debate_r3` | Maximum rounds |

**Research questions:**
- Does adding rounds improve accuracy on ambiguous multiclass tumour cases?
- Does verdict instability (cases where `round_changed=True`) correlate with miscalibration (ECE)?
- Does multi-round debate recover specificity on MS/stroke where single-agent confidence is unreliable?

Answered by `debate_round_analysis()` in `experiments/analysis.py`: outputs per-(experiment_id, task) `pct_verdict_changed`, `ece_changed`, `ece_unchanged`, `accuracy_changed`, `accuracy_unchanged`.

---

## 3. System C — Agent Forest

### 3.1 Architecture

```
forest_triage (AgentForest)
    → cnn_classify → sam3_segment → biomedclip → explainability
    → verification → report → fhir_output → END
```

The forest replaces only the `triage` node. All downstream nodes run unchanged on the consensus routing decision derived from majority vote.

### 3.2 Roles

| Role | Prompt file | Clinical frame |
|---|---|---|
| `radiologist` | `prompts/forest_radiologist.txt` | Visual pattern recognition; specialist neuroradiologist |
| `conservative` | `prompts/forest_conservative.txt` | Specificity-focused; avoid false positives |
| `emergency` | `prompts/forest_emergency.txt` | Sensitivity-focused; avoid missing dangerous findings |
| `differential` | `prompts/forest_differential.txt` | Uncertainty-aware; considers multiple competing diagnoses |

Each role prepends its frame to the shared `system_prompt.txt` schema. The same MedGemma weights run four times; diversity comes from the prompt, not from weight variation or temperature sampling (contrast with Li et al.'s homogeneous approach).

### 3.3 Voting

Majority vote on `diagnosis_name` across N agents. Tiebreaking is confidence-weighted: the winning label's mean `diagnosis_confidence` across its voters determines the winning `MedicalDiagnosis` object passed downstream. `dissent_rate` (fraction of agents that did not vote for the winner) is recorded in state for analysis.

### 3.4 Experiment family: `agent_forest`

| Sweep point | N agents | Roles used |
|---|---|---|
| `forest_n1` | 1 | radiologist only (single-agent baseline) |
| `forest_n3` | 3 | radiologist, conservative, emergency |
| `forest_n4` | 4 | all four roles |

**Research questions:**
- Does N=3 or N=4 improve accuracy over N=1 (single MedGemma)?
- Does agent agreement (low `dissent_rate`) correlate with correct predictions — connecting to Li et al.'s uncertainty reduction claim?
- Does the heterogeneous role diversity provide gains beyond what a homogeneous forest would yield?

Answered by `forest_voting_analysis()` in `experiments/analysis.py`: outputs per-(experiment_id, task) `mean_dissent_rate`, `unanimous_pct`, `accuracy_unanimous`, `accuracy_split`.

---

## 4. Thesis Positioning

### 4.1 Three-system comparison

| System | Triage | Verification/Report | Research question |
|---|---|---|---|
| A — Baseline | Single MedGemma | Verification node + Report node | Routing threshold ablation, calibration |
| B — Debate | Single MedGemma | DebateOrchestrator (R=1–3 rounds) | Does multi-round advocate debate outperform single-step verification? |
| C — Forest | AgentForest (N=1–4) | Verification node + Report node | Does role-diverse ensemble triage outperform single-agent triage? |

### 4.2 Key contrasts for the thesis

- **System B vs. Du et al. / MAD:** Your debate is domain-grounded — advocates argue from tool outputs (CNN probabilities, CLIP scores, SAM3 morphology), not from free-form LLM generation. The judge arbitrates over structured evidence streams. This is closer to structured multi-expert consensus than free debate.
- **System C vs. Li et al.:** Li et al.'s homogeneous forest samples the same model N times. Your heterogeneous forest uses role-specialised prompts, making model diversity a design choice rather than a byproduct of sampling. Frame this as the heterogeneous extension.
- **System A verification node vs. Agashe et al.:** The baseline already operationalises their finding — cite it explicitly and use `human_review_rate` as supporting evidence.

### 4.3 What the experiment families measure

Running `python run_research.py --family debate_rounds` and `--family agent_forest` produces:

- `analysis/report.md` with `debate_round_analysis` and `forest_voting_analysis` tables
- Per-sample `all_predictions.csv` with `debate_rounds_completed`, `debate_round_changed`, `dissent_rate`, `vote_fraction` columns for fine-grained analysis

These directly answer whether multi-agent deliberation — in either form — improves accuracy, calibration, or specificity over the single-agent baseline on the same four neuroimaging tasks.
