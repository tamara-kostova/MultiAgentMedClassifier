from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.base import Stage


def abnormal_score(all_probs: dict, task: str) -> float:
    """1 − p(normal class). Normal class resolved with eval's canonical_label rule."""
    from eval.tumor_eval import canonical_label

    p_normal = sum(p for cls, p in all_probs.items() if canonical_label(cls, task) == "normal")
    return float(1.0 - p_normal)


@register("inference.screening.cnn_topk")
class CNNTopKScreen(Stage):
    """Cheap CNN pass over every slice; keep the top_k most abnormal for the full agents.

    Forest/Debate cost several MedGemma calls per slice, so they only see the
    shortlisted slices. Slice states are left untouched (scores go to
    context.screening), so the later agent stages see exactly the state the
    LangGraph pipeline would.
    """

    def __init__(self, top_k: int = 3, resources=None):
        if resources is None:
            raise ValueError("cnn_topk needs shared resources")
        self.top_k = int(top_k)
        self.resources = resources

    def run(self, context: PipelineContext) -> PipelineContext:
        cnn = self.resources.cnn
        screening = []
        for pos, state in enumerate(context.slice_states):
            result = cnn.classify(state["image_path"], state["task"])
            screening.append(
                {
                    "position": pos,
                    "slice_index": state["metadata"].get("slice_index", pos),
                    "cnn_predicted_class": result["predicted_class"],
                    "cnn_confidence": float(result["confidence"]),
                    "abnormal_score": abnormal_score(result["all_probs"], state["task"]),
                }
            )

        ranked = sorted(screening, key=lambda row: row["abnormal_score"], reverse=True)
        context.screening = screening
        # Keep volume order among the chosen slices (stable, readable reports).
        context.selected = sorted(row["position"] for row in ranked[: self.top_k])
        return context
