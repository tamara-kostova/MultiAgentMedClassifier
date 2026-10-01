from collections import Counter

from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.base import Stage


def slice_prediction(state: dict) -> tuple[str, float]:
    """Final (class, confidence) of one slice, whichever stage produced it last."""
    if state.get("final_predicted_class"):
        return str(state["final_predicted_class"]), float(state.get("final_confidence") or 0.0)
    verdict = state.get("debate_verdict") or {}
    if verdict.get("winner"):
        return str(verdict.get("winner_detailed") or verdict["winner"]), float(verdict.get("confidence") or 0.0)
    cnn = state.get("classification_result") or {}
    if cnn.get("predicted_class"):
        return str(cnn["predicted_class"]), float(cnn.get("confidence") or 0.0)
    return "", 0.0


@register("inference.aggregation.study")
class StudyAggregation(Stage):
    """Combine the selected slices into one study-level result.

    max_abnormal: any abnormal slice makes the study abnormal; the most
                  confident abnormal slice decides the class.
    majority:     most common canonical class across slices; ties go to the
                  class with the highest summed confidence.
    """

    STRATEGIES = ("max_abnormal", "majority")

    def __init__(self, strategy: str = "max_abnormal"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy

    def run(self, context: PipelineContext) -> PipelineContext:
        from eval.labels import canonical_label

        rows = []
        for pos in context.selected:
            state = context.slice_states[pos]
            label, conf = slice_prediction(state)
            fhir = state.get("fhir_report") or {}
            rows.append(
                {
                    "slice_index": state["metadata"].get("slice_index", pos),
                    "image_path": state["image_path"],
                    "predicted_class": label,
                    "canonical_class": canonical_label(label, context.task),
                    "confidence": conf,
                    "requires_human_review": bool(state.get("requires_human_review")),
                    "fhir_bundle_id": fhir.get("id"),
                    "routing_path": list(state.get("routing_path") or []),
                }
            )

        if not rows:
            context.study_result = {"predicted_class": "", "confidence": 0.0, "slices": []}
            return context

        if self.strategy == "max_abnormal":
            abnormal = [r for r in rows if r["canonical_class"] not in ("normal", "")]
            if abnormal:
                best = max(abnormal, key=lambda r: r["confidence"])
                label, conf = best["predicted_class"], best["confidence"]
            else:
                label = next((r["predicted_class"] for r in rows if r["predicted_class"]), "normal")
                conf = sum(r["confidence"] for r in rows) / len(rows)
        else:
            counts = Counter(r["canonical_class"] for r in rows)
            conf_sum = Counter()
            for r in rows:
                conf_sum[r["canonical_class"]] += r["confidence"]
            winner = max(counts, key=lambda c: (counts[c], conf_sum[c]))
            members = [r for r in rows if r["canonical_class"] == winner]
            label = max(members, key=lambda r: r["confidence"])["predicted_class"]
            conf = conf_sum[winner] / len(members)

        context.study_result = {
            "task": context.task,
            "predicted_class": label,
            "canonical_class": canonical_label(label, context.task),
            "confidence": float(conf),
            "requires_human_review": any(r["requires_human_review"] for r in rows),
            "strategy": self.strategy,
            "n_slices_extracted": len(context.slice_states),
            "slices": rows,
            "screening": list(context.screening),
        }
        return context
