"""
Experiment family definitions for the research orchestration layer.

Each family is a list of SweepPoints — RoutingConfig override dicts paired
with a stable experiment_id string used for output directory names.

Families:
  threshold_sweep     — vary sam3_threshold (8 points, 0.50–0.85)
  human_review_sweep  — vary human_review_threshold (6 points, 0.30–0.55)
  ablation            — 4 structural variants (full / no_sam3 / always_sam3 / no_biomedclip)
  biomedclip_threshold — vary biomedclip_rerank_threshold (7 points, 0.50–0.80)
"""

from dataclasses import dataclass


@dataclass
class SweepPoint:
    experiment_id: str      # used as subdirectory name and CSV label
    description: str
    routing_overrides: dict  # fields to override on RoutingConfig


EXPERIMENT_FAMILIES: dict[str, list[SweepPoint]] = {
    # ── Core: sensitivity–specificity trade-off for SAM3 routing threshold ──
    "threshold_sweep": [
        SweepPoint(
            experiment_id=f"sam3_{t:.2f}",
            description=f"SAM3 threshold = {t:.2f} (default 0.70)",
            routing_overrides={"sam3_threshold": t},
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
            description="No SAM3 — sam3_threshold=1.0 so SAM3 path is never triggered",
            routing_overrides={"sam3_threshold": 1.0},
        ),
        SweepPoint(
            experiment_id="always_sam3",
            description="Always SAM3 — every non-normal case routed through segmentation",
            routing_overrides={"always_run_sam3": True},
        ),
        SweepPoint(
            experiment_id="no_biomedclip",
            description="No BiomedCLIP — rerank threshold=0.0 so BiomedCLIP path is never triggered",
            routing_overrides={"biomedclip_rerank_threshold": 0.0},
        ),
    ],

    # ── BiomedCLIP reranking sensitivity sweep ───────────────────────────────
    "biomedclip_threshold": [
        SweepPoint(
            experiment_id=f"biomedclip_{t:.2f}",
            description=f"BiomedCLIP rerank threshold = {t:.2f} (default 0.65)",
            routing_overrides={"biomedclip_rerank_threshold": t},
        )
        for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    ],
}
