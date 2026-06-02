"""
LangGraph state machine assembly for the neuroimaging multi-agent pipeline.

Graph structure:
                         triage
                           │
                     cnn_classify
                           │
                      sam3_segment
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
from config import DEFAULT_CONFIG, PipelineConfig
from pipeline.nodes import (
    make_biomedclip_node,
    make_cnn_node,
    make_debate_node,
    make_explainability_node,
    make_fhir_node,
    make_forest_triage_node,
    make_report_node,
    make_sam3_node,
    make_skip_explainability_node,
    make_triage_node,
    make_verification_node,
)
from pipeline.state import NeuroimagingState


def load_agents(cfg: PipelineConfig) -> tuple:
    """
    Load all model agents from cfg. Returns (medgemma, cnn, sam3, clip).

    Separating loading from assembly lets the research sweep reuse agents
    across many RoutingConfig variants without reloading model weights.
    """
    print("=== Loading pipeline agents ===")
    medgemma = MedGemmaAgent(cfg.model, cfg.routing)
    cnn = CNNClassifier(cfg.model, cfg.preprocess)
    sam3 = SAM3Tool(cfg.model, output_dir=f"{cfg.output_dir}/segmentation")
    clip = BiomedCLIPTool(cfg.model, cfg.preprocess)
    return medgemma, cnn, sam3, clip


def assemble_pipeline(
    medgemma: MedGemmaAgent,
    cnn: CNNClassifier,
    sam3: SAM3Tool,
    clip: BiomedCLIPTool,
    cfg: PipelineConfig,
):
    """
    Assemble and compile the LangGraph pipeline from pre-loaded agents.

    Called by build_pipeline() and by the research sweep runner to rebuild
    the graph with a different RoutingConfig without reloading model weights.
    """
    # ── Create node functions via factories ───────────────────────────────────
    triage_fn = make_triage_node(medgemma, cfg.routing)
    cnn_fn = make_cnn_node(cnn)
    sam3_fn = make_sam3_node(sam3, cfg.routing)
    biomedclip_fn = make_biomedclip_node(clip, cfg.routing)
    report_fn = make_report_node(medgemma, cfg.routing, skip_report=cfg.skip_report)
    verification_fn = make_verification_node(medgemma)
    fhir_fn = make_fhir_node(cfg.output_dir)

    # ── Assemble graph ────────────────────────────────────────────────────────
    workflow = StateGraph(NeuroimagingState)

    workflow.add_node("triage", triage_fn)
    workflow.add_node("cnn_classify", cnn_fn)
    workflow.add_node("sam3_segment", sam3_fn)
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
    workflow.add_edge("sam3_segment", "biomedclip")
    workflow.add_edge("biomedclip", "explainability")
    workflow.add_edge("explainability", "verification")
    workflow.add_edge("verification", "report")
    workflow.add_edge("report", "fhir_output")
    workflow.add_edge("fhir_output", END)

    return workflow.compile()


def assemble_debate_pipeline(
    medgemma: MedGemmaAgent,
    cnn: CNNClassifier,
    sam3: SAM3Tool,
    clip: BiomedCLIPTool,
    cfg: PipelineConfig,
    rounds: int = 1,
):
    """
    System B — Multi-Agent Debate pipeline.
    Replaces verification + report with a structured advocate-judge debate node.
    Advocates (CNN, BiomedCLIP, SAM3) are represented by MedGemma instances that
    argue on behalf of each tool's output. MedGemma judges the final verdict.
    """
    from agents.debate import DebateOrchestrator

    orchestrator = DebateOrchestrator(medgemma)

    triage_fn      = make_triage_node(medgemma, cfg.routing)
    cnn_fn         = make_cnn_node(cnn)
    sam3_fn        = make_sam3_node(sam3, cfg.routing)
    biomedclip_fn  = make_biomedclip_node(clip, cfg.routing)
    debate_fn      = make_debate_node(orchestrator, rounds=rounds, routing_cfg=cfg.routing)
    fhir_fn        = make_fhir_node(cfg.output_dir)

    explainability_fn = (
        make_explainability_node(cnn, output_dir=f"{cfg.output_dir}/explainability")
        if cfg.generate_explainability
        else make_skip_explainability_node()
    )

    workflow = StateGraph(NeuroimagingState)
    workflow.add_node("triage",        triage_fn)
    workflow.add_node("cnn_classify",  cnn_fn)
    workflow.add_node("sam3_segment",  sam3_fn)
    workflow.add_node("biomedclip",    biomedclip_fn)
    workflow.add_node("explainability", explainability_fn)
    workflow.add_node("debate",        debate_fn)
    workflow.add_node("fhir_output",   fhir_fn)
    workflow.set_entry_point("triage")

    workflow.add_edge("triage",        "cnn_classify")
    workflow.add_edge("cnn_classify",  "sam3_segment")
    workflow.add_edge("sam3_segment",  "biomedclip")
    workflow.add_edge("biomedclip",    "explainability")
    workflow.add_edge("explainability", "debate")
    workflow.add_edge("debate",        "fhir_output")
    workflow.add_edge("fhir_output",   END)

    return workflow.compile()


def assemble_forest_pipeline(
    medgemma: MedGemmaAgent,
    cnn: CNNClassifier,
    sam3: SAM3Tool,
    clip: BiomedCLIPTool,
    cfg: PipelineConfig,
    n_agents: int = 3,
):
    """
    System C — Agent Forest pipeline.
    Replaces the single triage node with N role-specialized MedGemma agents and
    a majority vote consensus. All downstream nodes run unchanged on the consensus.
    """
    from agents.forest import AgentForest

    forest = AgentForest(medgemma)

    forest_triage_fn = make_forest_triage_node(forest, n_agents=n_agents)
    cnn_fn           = make_cnn_node(cnn)
    sam3_fn          = make_sam3_node(sam3, cfg.routing)
    biomedclip_fn    = make_biomedclip_node(clip, cfg.routing)
    verification_fn  = make_verification_node(medgemma)
    report_fn        = make_report_node(medgemma, cfg.routing, skip_report=cfg.skip_report)
    fhir_fn          = make_fhir_node(cfg.output_dir)

    explainability_fn = (
        make_explainability_node(cnn, output_dir=f"{cfg.output_dir}/explainability")
        if cfg.generate_explainability
        else make_skip_explainability_node()
    )

    workflow = StateGraph(NeuroimagingState)
    workflow.add_node("forest_triage",  forest_triage_fn)
    workflow.add_node("cnn_classify",   cnn_fn)
    workflow.add_node("sam3_segment",   sam3_fn)
    workflow.add_node("biomedclip",     biomedclip_fn)
    workflow.add_node("explainability", explainability_fn)
    workflow.add_node("verification",   verification_fn)
    workflow.add_node("report",         report_fn)
    workflow.add_node("fhir_output",    fhir_fn)
    workflow.set_entry_point("forest_triage")

    workflow.add_edge("forest_triage",  "cnn_classify")
    workflow.add_edge("cnn_classify",   "sam3_segment")
    workflow.add_edge("sam3_segment",   "biomedclip")
    workflow.add_edge("biomedclip",     "explainability")
    workflow.add_edge("explainability", "verification")
    workflow.add_edge("verification",   "report")
    workflow.add_edge("report",         "fhir_output")
    workflow.add_edge("fhir_output",    END)

    return workflow.compile()


def build_debate_pipeline(cfg: PipelineConfig = None, rounds: int = 1):
    """Convenience wrapper: load agents and assemble the debate pipeline."""
    cfg = cfg or DEFAULT_CONFIG
    print("=== Building Multi-Agent Debate pipeline ===")
    agents = load_agents(cfg)
    app = assemble_debate_pipeline(*agents, cfg, rounds=rounds)
    print("=== Debate pipeline ready ===")
    return app


def build_forest_pipeline(cfg: PipelineConfig = None, n_agents: int = 3):
    """Convenience wrapper: load agents and assemble the Agent Forest pipeline."""
    cfg = cfg or DEFAULT_CONFIG
    print(f"=== Building Agent Forest pipeline (n_agents={n_agents}) ===")
    agents = load_agents(cfg)
    app = assemble_forest_pipeline(*agents, cfg, n_agents=n_agents)
    print("=== Forest pipeline ready ===")
    return app


def build_pipeline(cfg: PipelineConfig = None):
    """
    Instantiate all agents/tools and assemble the LangGraph pipeline.

    Returns a compiled LangGraph app ready for .invoke() or .stream().

    Example:
        app = build_pipeline()
        result = app.invoke(initial_state("scan.png", "binary_tumor"))
    """
    cfg = cfg or DEFAULT_CONFIG
    print("=== Building multi-agent neuroimaging pipeline ===")
    agents = load_agents(cfg)
    app = assemble_pipeline(*agents, cfg)
    print("=== Pipeline ready ===")
    return app
