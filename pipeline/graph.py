"""
LangGraph state machine assembly for the neuroimaging multi-agent pipeline.

Graph structure:
                         ┌─────────────────────────────────────────────┐
                         │                  triage                      │
                         │  (MedGemma routes based on visual assessment)│
                         └──────┬──────────┬──────────┬────────────────┘
                                │          │          │           │
                         cnn_direct  sam3_then_cnn  biomedclip  human_review
                                │          │          │           │
                         cnn_classify  sam3_segment  biomedclip  │
                                │          │          │           │
                                │    cnn_with_mask   │           │
                                │          │          │           │
                         [explainability?] ┘──────────┘           │
                                │                                  │
                           verification                            │
                                │                                  │
                             report                                │
                                └──────────────────────────────────┘
                                              │
                                          fhir_output
                                              │
                                             END
"""

from langgraph.graph import END, StateGraph

from agents.biomedclip_tool import BiomedCLIPTool
from agents.cnn_tool import CNNClassifier
from agents.medgemma_agent import MedGemmaAgent
from agents.sam3_tool import SAM3Tool
from config import DEFAULT_CONFIG, PipelineConfig
from pipeline.nodes import (
    human_review_node,
    make_biomedclip_node,
    make_cnn_node,
    make_cnn_with_mask_node,
    make_explainability_node,
    make_fhir_node,
    make_report_node,
    make_sam3_node,
    make_triage_node,
    make_verification_node,
    route_from_triage,
)
from pipeline.state import NeuroimagingState


def build_pipeline(cfg: PipelineConfig = None):
    """
    Instantiate all agents/tools and assemble the LangGraph pipeline.

    Returns a compiled LangGraph app ready for .invoke() or .stream().

    Example:
        app = build_pipeline()
        result = app.invoke(initial_state("scan.png", "binary_tumor"))
    """
    cfg = cfg or DEFAULT_CONFIG

    # ── Load models (once, at graph-build time) ───────────────────────────────
    print("=== Building multi-agent neuroimaging pipeline ===")
    medgemma = MedGemmaAgent(cfg.model, cfg.routing)
    cnn = CNNClassifier(cfg.model, cfg.preprocess)
    sam3 = SAM3Tool(cfg.model, output_dir=f"{cfg.output_dir}/segmentation")
    clip = BiomedCLIPTool(cfg.model, cfg.preprocess)

    # ── Create node functions via factories ───────────────────────────────────
    triage_fn = make_triage_node(medgemma, cfg.routing)
    cnn_fn = make_cnn_node(cnn)
    sam3_fn = make_sam3_node(sam3)
    cnn_with_mask_fn = make_cnn_with_mask_node(cnn, agent=medgemma)
    biomedclip_fn = make_biomedclip_node(clip, cfg.routing)
    report_fn = make_report_node(medgemma, cfg.routing)
    verification_fn = make_verification_node(medgemma)
    fhir_fn = make_fhir_node(cfg.output_dir)

    # ── Assemble graph ────────────────────────────────────────────────────────
    workflow = StateGraph(NeuroimagingState)

    workflow.add_node("triage", triage_fn)
    workflow.add_node("cnn_classify", cnn_fn)
    workflow.add_node("sam3_segment", sam3_fn)
    workflow.add_node("cnn_with_mask", cnn_with_mask_fn)
    workflow.add_node("biomedclip", biomedclip_fn)
    workflow.add_node("verification", verification_fn)
    workflow.add_node("report", report_fn)
    workflow.add_node("fhir_output", fhir_fn)
    workflow.add_node("human_review", human_review_node)

    # Entry point
    workflow.set_entry_point("triage")

    # Conditional routing from triage
    workflow.add_conditional_edges(
        "triage",
        route_from_triage,
        {
            "cnn_direct": "cnn_classify",
            "sam3_then_cnn": "sam3_segment",
            "biomedclip": "biomedclip",
            "human_review": "human_review",
        },
    )

    # SAM3 always feeds into CNN-with-mask (MedGemma gets overlay; CNN gets original)
    workflow.add_edge("sam3_segment", "cnn_with_mask")

    # Optional explainability node between CNN classification and verification
    if cfg.generate_explainability:
        explainability_fn = make_explainability_node(
            cnn, output_dir=f"{cfg.output_dir}/explainability"
        )
        workflow.add_node("explainability", explainability_fn)
        workflow.add_edge("cnn_classify", "explainability")
        workflow.add_edge("cnn_with_mask", "explainability")
        workflow.add_edge("explainability", "verification")
    else:
        workflow.add_edge("cnn_classify", "verification")
        workflow.add_edge("cnn_with_mask", "verification")

    workflow.add_edge("biomedclip", "verification")
    workflow.add_edge("verification", "report")

    # Terminal edges
    workflow.add_edge("report", "fhir_output")
    workflow.add_edge("human_review", "fhir_output")
    workflow.add_edge("fhir_output", END)

    app = workflow.compile()
    print("=== Pipeline ready ===")
    return app
