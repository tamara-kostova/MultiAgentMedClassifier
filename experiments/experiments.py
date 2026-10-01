"""
Experiment family definitions for the research orchestration layer.

Each family is a list of SweepPoints — RoutingConfig override dicts paired
with a stable experiment_id string used for output directory names.

Families:
  threshold_sweep     — vary sam3_threshold (8 points, 0.50–0.85)
  human_review_sweep  — vary human_review_threshold (6 points, 0.30–0.55)
  ablation            — 4 structural variants (full / no_sam3 / always_sam3 / no_biomedclip)
  biomedclip_threshold — vary biomedclip_rerank_threshold (7 points, 0.50–0.80)

WARNING — routing-only sweep points. The LangGraph pipeline is linear: every node
(SAM3, BiomedCLIP, ...) runs for every image regardless of the routing decision.
sam3_threshold, biomedclip_rerank_threshold, always_run_sam3 and
always_run_biomedclip only change the *recorded* routing_decision label
(agents/medgemma_agent.py:diagnosis_to_routing), never a prediction or confidence.
So the whole threshold_sweep and biomedclip_threshold families and the ablation
points no_sam3 / always_sam3 / no_biomedclip measure nothing about the pipeline's
output: their accuracy/ECE equal full_pipeline's up to MedGemma sampling noise.
They are kept (routing_distribution still reflects them) and flagged with
``routing_only=True``; run_research.py warns when one is selected. Do not report
them as an ablation. human_review_threshold is NOT routing-only: report_node uses
it to set requires_human_review.
"""

from dataclasses import dataclass, field


@dataclass
class SweepPoint:
    experiment_id: str       # used as subdirectory name and CSV label
    description: str
    routing_overrides: dict  # fields to override on RoutingConfig
    pipeline_mode: str = "standard"        # "standard" | "debate" | "forest"
    pipeline_kwargs: dict = field(default_factory=dict)  # extra args for assembler
    # True when the override only changes the recorded routing label, not the output
    # (see module docstring). Such points are not a real ablation.
    routing_only: bool = False


_RO = "[ROUTING LABEL ONLY — pipeline output unchanged] "


EXPERIMENT_FAMILIES: dict[str, list[SweepPoint]] = {
    # ── Core: sensitivity–specificity trade-off for SAM3 routing threshold ──
    "threshold_sweep": [
        SweepPoint(
            experiment_id=f"sam3_{t:.2f}",
            description=f"{_RO}SAM3 threshold = {t:.2f} (default 0.70)",
            routing_overrides={"sam3_threshold": t},
            routing_only=True,
        )
        for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    ],

    # ── Human review rate vs coverage trade-off ──────────────────────────────
    "human_review_sweep": [
        SweepPoint(
            experiment_id=f"human_{t:.2f}",
            description=f"Human review threshold = {t:.2f} (default 0.45)",
            routing_overrides={"human_review_threshold": t},
        )
        for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    ],

    # ── Core: component contribution (ablation study) ────────────────────────
    "ablation": [
        SweepPoint(
            experiment_id="full_pipeline",
            description="Full pipeline — all components active (baseline for ablation)",
            routing_overrides={},
        ),
        SweepPoint(
            experiment_id="no_sam3",
            description=f"{_RO}No SAM3 — sam3_threshold=1.0 (SAM3 still runs on every image; only the label changes)",
            routing_overrides={"sam3_threshold": 1.0},
            routing_only=True,
        ),
        SweepPoint(
            experiment_id="always_sam3",
            description=f"{_RO}Always SAM3 — always_run_sam3=True (SAM3 already runs on every image)",
            routing_overrides={"always_run_sam3": True},
            routing_only=True,
        ),
        SweepPoint(
            experiment_id="no_biomedclip",
            description=f"{_RO}No BiomedCLIP — rerank threshold=0.0 (BiomedCLIP still runs on every image)",
            routing_overrides={"biomedclip_rerank_threshold": 0.0},
            routing_only=True,
        ),
    ],

    # ── BiomedCLIP reranking sensitivity sweep ───────────────────────────────
    "biomedclip_threshold": [
        SweepPoint(
            experiment_id=f"biomedclip_{t:.2f}",
            description=f"{_RO}BiomedCLIP rerank threshold = {t:.2f} (default 0.65)",
            routing_overrides={"biomedclip_rerank_threshold": t},
            routing_only=True,
        )
        for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    ],

    # ── System C: Agent Forest — vary ensemble size ───────────────────────────
    "agent_forest": [
        SweepPoint(
            experiment_id="forest_n1",
            description="Agent Forest N=1 (equivalent to single-agent baseline)",
            routing_overrides={},
            pipeline_mode="forest",
            pipeline_kwargs={"n_agents": 1},
        ),
        SweepPoint(
            experiment_id="forest_n3",
            description="Agent Forest N=3 (radiologist, conservative, emergency)",
            routing_overrides={},
            pipeline_mode="forest",
            pipeline_kwargs={"n_agents": 3},
        ),
        SweepPoint(
            experiment_id="forest_n4",
            description="Agent Forest N=4 (all four roles)",
            routing_overrides={},
            pipeline_mode="forest",
            pipeline_kwargs={"n_agents": 4},
        ),
    ],

    # ── System B: Multi-Agent Debate — vary debate rounds ────────────────────
    "debate_rounds": [
        SweepPoint(
            experiment_id="debate_r1",
            description="Multi-Agent Debate R=1 (single-round arbitration)",
            routing_overrides={},
            pipeline_mode="debate",
            pipeline_kwargs={"rounds": 1},
        ),
        SweepPoint(
            experiment_id="debate_r2",
            description="Multi-Agent Debate R=2 (advocates respond to round-1 verdict)",
            routing_overrides={},
            pipeline_mode="debate",
            pipeline_kwargs={"rounds": 2},
        ),
        SweepPoint(
            experiment_id="debate_r3",
            description="Multi-Agent Debate R=3 (maximum rounds)",
            routing_overrides={},
            pipeline_mode="debate",
            pipeline_kwargs={"rounds": 3},
        ),
    ],
}
