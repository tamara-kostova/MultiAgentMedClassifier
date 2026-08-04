"""
Multi-Agent Debate for neuroimaging diagnosis.

Three MedGemma advocate agents argue from the perspective of different specialist
tool outputs (CNN classifier, BiomedCLIP, SAM3 segmentation). A MedGemma judge
arbitrates and produces a structured verdict. Supports 1–3 debate rounds.

Round 1: Each advocate receives the image + its tool output and generates an argument.
Round N>1: Each advocate receives the image + tool output + prior verdict + all
           round-(N-1) arguments, allowing it to revise or reinforce its position.

In the LangGraph pipeline this replaces the verification + report tail:
    ... → biomedclip → explainability → debate → fhir_output
"""

import json
from typing import Optional

from agents.medgemma_agent import MedGemmaAgent
from pipeline.state import NeuroimagingState

# ── Advocate prompt templates ─────────────────────────────────────────────────

_CNN_ADVOCATE_R1 = """You are a CNN classifier advocate in a multi-agent neuroimaging debate.
Your role is to argue in favor of the CNN classifier's prediction.

Task: {task}
CNN Predicted Class: {predicted_class}
CNN Confidence: {confidence:.1%}
Class Probabilities: {all_probs}
Saliency Evidence: {gradcam_info}

Make a concise one-paragraph argument for why the CNN prediction is correct.
Reference the confidence margin over competing classes and the saliency evidence if available.
Output ONLY the argument paragraph."""

_CNN_ADVOCATE_RN = """You are a CNN classifier advocate in round {round_num} of a neuroimaging debate.

Task: {task}
CNN Predicted Class: {predicted_class}
CNN Confidence: {confidence:.1%}
Class Probabilities: {all_probs}

The judge's previous verdict: "{prior_winner}" (confidence {prior_confidence:.1%})
Reason given: {prior_reason}

Other advocates argued:
- BiomedCLIP advocate: {clip_argument}
- SAM3 advocate: {sam_argument}

Respond to the judge's verdict and the other advocates' arguments.
If you agree with the verdict, say so briefly. If you disagree, argue your case with specific evidence.
Output ONLY your response paragraph."""

_CLIP_ADVOCATE_R1 = """You are a BiomedCLIP visual-language model advocate in a multi-agent neuroimaging debate.
Your role is to argue in favor of BiomedCLIP's prediction.

Task: {task}
BiomedCLIP Top Prediction: {top_label} (score: {top_score:.3f})
All Ranked Predictions: {ranked}

Make a concise one-paragraph argument for why the BiomedCLIP prediction is correct.
Reference the similarity score margin between the top and second predictions.
Output ONLY the argument paragraph."""

_CLIP_ADVOCATE_RN = """You are a BiomedCLIP advocate in round {round_num} of a neuroimaging debate.

Task: {task}
BiomedCLIP Top Prediction: {top_label} (score: {top_score:.3f})
Ranked Predictions: {ranked}

The judge's previous verdict: "{prior_winner}" (confidence {prior_confidence:.1%})
Reason given: {prior_reason}

Other advocates argued:
- CNN advocate: {cnn_argument}
- SAM3 advocate: {sam_argument}

Respond to the judge's verdict and the other advocates' arguments.
If you agree, say so briefly. If you disagree, argue your case.
Output ONLY your response paragraph."""

_SAM_ADVOCATE_R1 = """You are a SAM3 segmentation specialist advocate in a multi-agent neuroimaging debate.
Your role is to argue based on the spatial and morphological evidence from lesion segmentation.

Task: {task}
Lesion Detected: {lesion_detected}
Bounding Box: {bbox}
Mask Coverage: {mask_area}

Make a concise one-paragraph argument about what the segmentation evidence suggests about the diagnosis.
If no lesion was detected, argue what the absence of segmentable pathology implies.
Output ONLY the argument paragraph."""

_SAM_ADVOCATE_RN = """You are a SAM3 segmentation advocate in round {round_num} of a neuroimaging debate.

Task: {task}
Lesion Detected: {lesion_detected}
Bounding Box: {bbox}
Mask Coverage: {mask_area}

The judge's previous verdict: "{prior_winner}" (confidence {prior_confidence:.1%})
Reason given: {prior_reason}

Other advocates argued:
- CNN advocate: {cnn_argument}
- BiomedCLIP advocate: {clip_argument}

Respond to the judge's verdict and the other advocates' arguments.
Output ONLY your response paragraph."""

_JUDGE_TEMPLATE = """You are a senior neuroradiologist judging a multi-agent debate about a brain scan.

Task: {task}
{prior_verdict_section}
CNN Classifier Advocate:
{cnn_argument}

BiomedCLIP Visual-Language Advocate:
{clip_argument}

SAM3 Segmentation Advocate:
{sam_argument}

Weigh all three arguments carefully against what you observe in the scan.
Identify which evidence is most compelling and internally consistent.

Respond in JSON only:
{{
  "winner": "<tumor|stroke|multiple sclerosis|normal|other abnormalities>",
  "winner_detailed": "<glioma|meningioma|pituitary_tumor|ischemic|hemorrhagic|null>",
  "confidence": <0.0-1.0>,
  "reason": "<one sentence: which evidence was most persuasive and why>",
  "round_changed": <true|false>
}}"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_cnn_prompt(state: NeuroimagingState, round_num: int,
                      prior: Optional[dict], others: Optional[dict]) -> str:
    cnn = state.get("classification_result") or {}
    expl = state.get("explainability_result") or {}
    gradcam = expl.get("gradcam_pp")
    gradcam_info = f"GradCAM++ available" if gradcam else "GradCAM++ not available"
    all_probs_str = json.dumps(cnn.get("all_probs", {}))

    if round_num == 1:
        return _CNN_ADVOCATE_R1.format(
            task=state["task"],
            predicted_class=cnn.get("predicted_class", "unknown"),
            confidence=cnn.get("confidence", 0.0),
            all_probs=all_probs_str,
            gradcam_info=gradcam_info,
        )
    return _CNN_ADVOCATE_RN.format(
        round_num=round_num,
        task=state["task"],
        predicted_class=cnn.get("predicted_class", "unknown"),
        confidence=cnn.get("confidence", 0.0),
        all_probs=all_probs_str,
        prior_winner=prior.get("winner", "unknown"),
        prior_confidence=prior.get("confidence", 0.0),
        prior_reason=prior.get("reason", ""),
        clip_argument=others.get("clip", "[not available]"),
        sam_argument=others.get("sam", "[not available]"),
    )


def _build_clip_prompt(state: NeuroimagingState, round_num: int,
                       prior: Optional[dict], others: Optional[dict]) -> str:
    clip = state.get("biomedclip_result") or {}
    ranked = list(zip(clip.get("ranked_labels", []), clip.get("scores", [])))
    ranked_str = ", ".join(f"{lbl} ({sc:.3f})" for lbl, sc in ranked) or "not available"

    if round_num == 1:
        return _CLIP_ADVOCATE_R1.format(
            task=state["task"],
            top_label=clip.get("top_label", "unknown"),
            top_score=clip.get("top_score", 0.0),
            ranked=ranked_str,
        )
    return _CLIP_ADVOCATE_RN.format(
        round_num=round_num,
        task=state["task"],
        top_label=clip.get("top_label", "unknown"),
        top_score=clip.get("top_score", 0.0),
        ranked=ranked_str,
        prior_winner=prior.get("winner", "unknown"),
        prior_confidence=prior.get("confidence", 0.0),
        prior_reason=prior.get("reason", ""),
        cnn_argument=others.get("cnn", "[not available]"),
        sam_argument=others.get("sam", "[not available]"),
    )


def _build_sam_prompt(state: NeuroimagingState, round_num: int,
                      prior: Optional[dict], others: Optional[dict]) -> str:
    seg = state.get("segmentation_result") or {}
    if seg.get("skipped"):
        lesion_detected = "No (SAM3 not applicable for this task)"
        bbox = "N/A"
        mask_area = "N/A"
    else:
        bbox = seg.get("bbox")
        has_lesion = bbox and any(b is not None for b in bbox)
        lesion_detected = "Yes" if has_lesion else "No (no lesion segmented)"
        mask_area = "available" if has_lesion else "empty mask"

    if round_num == 1:
        return _SAM_ADVOCATE_R1.format(
            task=state["task"],
            lesion_detected=lesion_detected,
            bbox=bbox or "N/A",
            mask_area=mask_area,
        )
    return _SAM_ADVOCATE_RN.format(
        round_num=round_num,
        task=state["task"],
        lesion_detected=lesion_detected,
        bbox=bbox or "N/A",
        mask_area=mask_area,
        prior_winner=prior.get("winner", "unknown"),
        prior_confidence=prior.get("confidence", 0.0),
        prior_reason=prior.get("reason", ""),
        cnn_argument=others.get("cnn", "[not available]"),
        clip_argument=others.get("clip", "[not available]"),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DebateOrchestrator:
    """
    Orchestrates a multi-agent debate between CNN, BiomedCLIP, and SAM3 advocates
    with MedGemma as the judge. Supports 1–3 rounds.

    Usage:
        orchestrator = DebateOrchestrator(medgemma_agent)
        result = orchestrator.run(state, rounds=2)
    """

    MAX_ROUNDS = 3

    def __init__(self, medgemma: MedGemmaAgent):
        self.medgemma = medgemma

    def run(self, state: NeuroimagingState, rounds: int = 1) -> dict:
        """
        Run the debate for `rounds` rounds (clamped to 1–3).

        Returns dict with:
            winner, winner_detailed, confidence, reason,
            rounds_completed, round_changed, arguments
        """
        rounds = max(1, min(rounds, self.MAX_ROUNDS))
        image_path = state["image_path"]
        prior_verdict: Optional[dict] = None
        all_arguments: list[dict] = []
        current_args: dict[str, str] = {}

        for round_num in range(1, rounds + 1):
            print(f"[debate] Round {round_num}/{rounds}", flush=True)

            cnn_prompt = _build_cnn_prompt(state, round_num, prior_verdict, current_args)
            clip_prompt = _build_clip_prompt(state, round_num, prior_verdict, current_args)
            sam_prompt = _build_sam_prompt(state, round_num, prior_verdict, current_args)

            cnn_arg = self.medgemma.generate_for_prompt(image_path, cnn_prompt)
            clip_arg = self.medgemma.generate_for_prompt(image_path, clip_prompt)
            sam_arg = self.medgemma.generate_for_prompt(image_path, sam_prompt)

            current_args = {"cnn": cnn_arg, "clip": clip_arg, "sam": sam_arg}
            all_arguments.extend([
                {"round": round_num, "role": "cnn",  "argument": cnn_arg},
                {"round": round_num, "role": "clip", "argument": clip_arg},
                {"round": round_num, "role": "sam",  "argument": sam_arg},
            ])

            prior_section = ""
            if prior_verdict:
                prior_section = (
                    f"Previous round verdict: {prior_verdict['winner']} "
                    f"(confidence {prior_verdict.get('confidence', 0):.1%}). "
                    "Re-evaluate whether the new arguments change your verdict.\n\n"
                )

            judge_prompt = _JUDGE_TEMPLATE.format(
                task=state["task"],
                prior_verdict_section=prior_section,
                cnn_argument=cnn_arg,
                clip_argument=clip_arg,
                sam_argument=sam_arg,
            )
            verdict_raw = self.medgemma.generate_for_prompt(
                image_path, judge_prompt, max_new_tokens=200
            )

            try:
                verdict = MedGemmaAgent._extract_json_object(verdict_raw)
            except (json.JSONDecodeError, ValueError):
                verdict = {
                    "winner": state.get("suspected_pathology", "unknown"),
                    "winner_detailed": None,
                    "confidence": 0.5,
                    "reason": "Judge parse failed — defaulting to suspected pathology.",
                    "round_changed": False,
                }

            verdict.setdefault("winner_detailed", None)
            verdict.setdefault("round_changed", prior_verdict is not None and
                               verdict.get("winner") != (prior_verdict or {}).get("winner"))
            prior_verdict = verdict
            print(
                f"[debate] Round {round_num} verdict: {verdict.get('winner')} "
                f"conf={verdict.get('confidence', 0):.2f} "
                f"changed={verdict.get('round_changed')}"
            )

        return {
            "winner": prior_verdict.get("winner", "unknown"),
            "winner_detailed": prior_verdict.get("winner_detailed"),
            "confidence": float(prior_verdict.get("confidence", 0.5)),
            "reason": prior_verdict.get("reason", ""),
            "round_changed": prior_verdict.get("round_changed", False),
            "rounds_completed": rounds,
            "arguments": all_arguments,
        }
