"""
Every LangGraph node of pipeline/graph.py as a registered stage.

Each stage builds its node with the unchanged factory from pipeline/nodes.py
(same arguments as assemble_*_pipeline) and applies it to every selected slice
state, merging the returned update — exactly what LangGraph does for these
linear graphs. Stage configs that list the nodes in graph.py's order therefore
reproduce the standard / forest / debate pipelines slice for slice.

pipeline.nodes imports torch/cv2, so it is imported in the factories only
(registry.discover() must stay cheap).
"""

from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.base import Stage


def _triage(res):
    from pipeline.nodes import make_triage_node

    return make_triage_node(res.medgemma, res.cfg.routing)


def _forest_triage(res, n_agents: int | None = None):
    from pipeline.nodes import make_forest_triage_node

    n_agents = res.cfg.forest_n_agents if n_agents is None else n_agents
    return make_forest_triage_node(res.forest, n_agents=n_agents)


def _cnn_classify(res):
    from pipeline.nodes import make_cnn_node

    return make_cnn_node(res.cnn)


def _sam3_segment(res):
    from pipeline.nodes import make_sam3_node

    return make_sam3_node(res.sam3, res.cfg.routing)


def _biomedclip(res):
    from pipeline.nodes import make_biomedclip_node

    return make_biomedclip_node(res.clip, res.cfg.routing)


def _explainability(res, enabled: bool | None = None):
    from pipeline.nodes import make_explainability_node, make_skip_explainability_node

    enabled = res.cfg.generate_explainability if enabled is None else enabled
    if enabled:
        return make_explainability_node(res.cnn, output_dir=f"{res.cfg.output_dir}/explainability")
    return make_skip_explainability_node()


def _verification(res):
    from pipeline.nodes import make_verification_node

    return make_verification_node(res.medgemma)


def _debate(res, rounds: int | None = None, advocates=None):
    from pipeline.nodes import make_debate_node

    rounds = res.cfg.debate_rounds if rounds is None else rounds
    advocates = res.cfg.debate_advocates if advocates is None else advocates
    return make_debate_node(
        res.debate, rounds=rounds, routing_cfg=res.cfg.routing, advocates=advocates
    )


def _report(res):
    from pipeline.nodes import make_report_node

    return make_report_node(res.medgemma, res.cfg.routing, skip_report=res.cfg.skip_report)


def _triage_final(res):
    from pipeline.nodes import make_triage_final_node

    return make_triage_final_node(res.cfg.routing)


def _fhir_output(res):
    from pipeline.nodes import make_fhir_node

    return make_fhir_node(res.cfg.output_dir)


# stage suffix → node factory. Names match the LangGraph node names in graph.py.
NODE_FACTORIES = {
    "triage": _triage,
    "forest_triage": _forest_triage,
    "cnn_classify": _cnn_classify,
    "sam3_segment": _sam3_segment,
    "biomedclip": _biomedclip,
    "explainability": _explainability,
    "verification": _verification,
    "debate": _debate,
    "report": _report,
    "triage_final": _triage_final,
    "fhir_output": _fhir_output,
}


class NodeStage(Stage):
    """Runs one LangGraph node on each selected slice state."""

    node_name: str = ""
    factory = None

    def __init__(self, resources=None, **node_kwargs):
        if resources is None:
            raise ValueError(f"{self.name} needs shared resources")
        self.resources = resources
        self.node = type(self).factory(resources, **node_kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        for state in context.selected_states():
            state.update(self.node(state) or {})
        return context


def _make_stage_class(node_name: str, factory) -> type:
    cls = type(
        f"{''.join(part.title() for part in node_name.split('_'))}Stage",
        (NodeStage,),
        {"node_name": node_name, "factory": staticmethod(factory), "__module__": __name__},
    )
    return register(f"inference.agents.{node_name}")(cls)


for _name, _factory in NODE_FACTORIES.items():
    _make_stage_class(_name, _factory)
