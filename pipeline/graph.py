"""
LangGraph state machine assembly for the neuroimaging multi-agent pipeline.

Graph structure:
                         triage
                           │
                     cnn_classify
                           │
                      sam3_segment
                           |
                    atlas_enrichment
                           │
                       biomedclip
                           │
                    explainability
                           │
                      verification
                           │
                         report
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
from agents.sibra_tool import SiibraAtlasTool
from config import DEFAULT_CONFIG, PipelineConfig
from pipeline.nodes import (
    human_review_node,
    make_atlas_enrichment_node,
    make_biomedclip_node,
    make_cnn_node,
    make_explainability_node,
    make_fhir_node,
    make_report_node,
    make_sam3_node,
    make_skip_explainability_node,
    make_triage_node,
    make_verification_node,
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
    siibra = SiibraAtlasTool()

    # ── Create node functions via factories ───────────────────────────────────
    triage_fn = make_triage_node(medgemma, cfg.routing)
    cnn_fn = make_cnn_node(cnn)
    sam3_fn = make_sam3_node(sam3)
    atlas_fn = make_atlas_enrichment_node(siibra)
    biomedclip_fn = make_biomedclip_node(clip, cfg.routing)
    report_fn = make_report_node(medgemma, cfg.routing, skip_report=cfg.skip_report)
    verification_fn = make_verification_node(medgemma)
    fhir_fn = make_fhir_node(cfg.output_dir)

    # ── Assemble graph ────────────────────────────────────────────────────────
    workflow = StateGraph(NeuroimagingState)

    workflow.add_node("triage", triage_fn)
    workflow.add_node("cnn_classify", cnn_fn)
    workflow.add_node("sam3_segment", sam3_fn)
    workflow.add_node("atlas_enrichment", atlas_fn)
    workflow.add_node("biomedclip", biomedclip_fn)
    workflow.add_node("verification", verification_fn)
    workflow.add_node("report", report_fn)
    workflow.add_node("fhir_output", fhir_fn)
    workflow.set_entry_point("triage")

    explainability_fn = (
        make_explainability_node(cnn, output_dir=f"{cfg.output_dir}/explainability")
        if cfg.generate_explainability
        else make_skip_explainability_node()
    )
    workflow.add_node("explainability", explainability_fn)

    workflow.add_edge("triage", "cnn_classify")
    workflow.add_edge("cnn_classify", "sam3_segment")
    workflow.add_edge("sam3_segment", "atlas_enrichment")
    workflow.add_edge("atlas_enrichment", "biomedclip")
    workflow.add_edge("biomedclip", "explainability")
    workflow.add_edge("explainability", "verification")
    workflow.add_edge("verification", "report")
    workflow.add_edge("report", "fhir_output")
    workflow.add_edge("fhir_output", END)

    app = workflow.compile()
    print("=== Pipeline ready ===")
    return app
