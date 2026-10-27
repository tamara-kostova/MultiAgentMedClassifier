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
import re
from typing import Optional

from agents.medgemma_agent import NO_THINK_PREFILL, MedGemmaAgent
from pipeline.state import NeuroimagingState

# The judge only has to emit a small JSON object, so a short budget is enough
# *provided* the hidden thought block is skipped (NO_THINK_PREFILL). If a call
# still fails to parse we retry once without the prefill and with room for a
# full thought block plus the JSON after it.
JUDGE_MAX_NEW_TOKENS = 400
JUDGE_RETRY_MAX_NEW_TOKENS = 1500
ADVOCATE_MAX_NEW_TOKENS = 300
# If the prefill itself is the problem (e.g. a model revision that tokenizes it
# differently), stop paying for a doomed first attempt on every later image.
PREFILL_DISABLE_AFTER = 3

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

The judge's previous verdict: "{prior_winner}" (confidence {prior_confidence})
Reason given: {prior_reason}

{others_section}
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

The judge's previous verdict: "{prior_winner}" (confidence {prior_confidence})
Reason given: {prior_reason}

{others_section}
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

The judge's previous verdict: "{prior_winner}" (confidence {prior_confidence})
Reason given: {prior_reason}

{others_section}
Respond to the judge's verdict and the other advocates' arguments.
Output ONLY your response paragraph."""

_JUDGE_TEMPLATE = """You are a senior neuroradiologist judging a multi-agent debate about a brain scan.

Task: {task}
{prior_verdict_section}
{advocate_sections}

Weigh {weigh_phrase} carefully against what you observe in the scan.
Identify which evidence is most compelling and internally consistent.

Output ONLY the JSON object below. Do not include any reasoning, analysis,
or text before or after it — your entire response must be the JSON object.
{{
  "winner": "<tumor|stroke|multiple sclerosis|normal|other abnormalities>",
  "winner_detailed": "<glioma|meningioma|pituitary_tumor|ischemic|hemorrhagic|null>",
  "confidence": <0.0-1.0>,
  "reason": "<one sentence: which evidence was most persuasive and why>",
  "round_changed": <true|false>
}}"""


ADVOCATES = ("cnn", "clip", "sam")
# Titles used in the judge prompt and in the round-N "other advocates" lists.
_JUDGE_TITLES = {
    "cnn": "CNN Classifier Advocate",
    "clip": "BiomedCLIP Visual-Language Advocate",
    "sam": "SAM3 Segmentation Advocate",
}
_PEER_TITLES = {"cnn": "CNN advocate", "clip": "BiomedCLIP advocate", "sam": "SAM3 advocate"}
_WEIGH_PHRASES = {1: "the argument", 2: "both arguments", 3: "all three arguments"}


def parse_advocates(value) -> tuple:
    """Normalize an advocate list ("cnn,clip" or an iterable) to ADVOCATES order."""
    if value is None:
        return ADVOCATES
    items = value.split(",") if isinstance(value, str) else list(value)
    chosen = {str(a).strip().lower() for a in items if str(a).strip()}
    unknown = chosen - set(ADVOCATES)
    if unknown or not chosen:
        raise ValueError(
            f"debate advocates must be a non-empty subset of {list(ADVOCATES)}, got {items!r}"
        )
    return tuple(a for a in ADVOCATES if a in chosen)


def parse_confidence(value) -> Optional[float]:
    """Judge confidence → [0, 1], or None when it cannot be read.

    Accepts floats, numeric strings and percentages ("85%", 85 → 0.85).
    """
    if isinstance(value, bool) or value is None:
        return None
    pct = False
    if isinstance(value, str):
        text = value.strip()
        pct = text.endswith("%")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        value = match.group(0)
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf != conf:  # NaN
        return None
    if pct or 1.0 < conf <= 100.0:
        conf /= 100.0
    return max(0.0, min(1.0, conf))


def verdict_label(verdict: dict, task: Optional[str]) -> str:
    """Canonical class of a judge verdict (winner_detailed preferred on multiclass)."""
    from eval.labels import canonical_label

    detailed = verdict.get("winner_detailed")
    if task == "multiclass_tumor" and canonical_label(detailed, task):
        return canonical_label(detailed, task)
    return canonical_label(verdict.get("winner"), task)


def _fmt_conf(conf: Optional[float]) -> str:
    return "unknown" if conf is None else f"{conf:.1%}"


def _others_section(own: str, others: dict, advocates: tuple) -> str:
    peers = [a for a in advocates if a != own]
    if not peers:
        return ""
    lines = "\n".join(
        f"- {_PEER_TITLES[a]}: {others.get(a, '[not available]')}" for a in peers
    )
    return f"Other advocates argued:\n{lines}\n"


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_cnn_prompt(state: NeuroimagingState, round_num: int,
                      prior: Optional[dict], others: Optional[dict],
                      advocates: tuple = ADVOCATES) -> str:
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
        prior_confidence=_fmt_conf(prior.get("confidence")),
        prior_reason=prior.get("reason", ""),
        others_section=_others_section("cnn", others, advocates),
    )


def _build_clip_prompt(state: NeuroimagingState, round_num: int,
                       prior: Optional[dict], others: Optional[dict],
                       advocates: tuple = ADVOCATES) -> str:
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
        prior_confidence=_fmt_conf(prior.get("confidence")),
        prior_reason=prior.get("reason", ""),
        others_section=_others_section("clip", others, advocates),
    )


def _build_sam_prompt(state: NeuroimagingState, round_num: int,
                      prior: Optional[dict], others: Optional[dict],
                      advocates: tuple = ADVOCATES) -> str:
    seg = state.get("segmentation_result") or {}
    if seg.get("skipped"):
        lesion_detected = "No (SAM3 not applicable for this task)"
        bbox = "N/A"
        mask_area = "N/A"
    else:
        bbox = seg.get("bbox")
        has_lesion = (
            not seg.get("mask_empty")
            and bool(bbox)
            and all(b is not None for b in bbox)
        )
        if not has_lesion:
            bbox = None
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
        prior_confidence=_fmt_conf(prior.get("confidence")),
        prior_reason=prior.get("reason", ""),
        others_section=_others_section("sam", others, advocates),
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
        self._prefill_enabled = True
        self._prefill_failures = 0

    def _prefill(self) -> Optional[str]:
        return NO_THINK_PREFILL if self._prefill_enabled else None

    def _note_prefill_failure(self, why: str) -> None:
        self._prefill_failures += 1
        if self._prefill_enabled and self._prefill_failures >= PREFILL_DISABLE_AFTER:
            self._prefill_enabled = False
            print(
                f"[debate] disabling the no-think prefill for the rest of this run "
                f"after {self._prefill_failures} failures ({why})",
                flush=True,
            )

    def _judge(self, image_path: str, judge_prompt: str, round_num: int) -> Optional[dict]:
        """
        Ask the judge for a verdict. Returns the parsed verdict, or None if every
        attempt failed to produce a JSON object with a `winner` field.
        """
        attempts: list[tuple[str, Optional[str], int]] = []
        if self._prefill_enabled:
            attempts.append(("prefill", NO_THINK_PREFILL, JUDGE_MAX_NEW_TOKENS))
        attempts.append(("retry-long", None, JUDGE_RETRY_MAX_NEW_TOKENS))

        for kind, prefill, budget in attempts:
            try:
                raw = self.medgemma.generate_for_prompt(
                    image_path, judge_prompt, max_new_tokens=budget, prefill=prefill
                )
            except Exception as exc:  # a broken prefill must not kill a long run
                print(
                    f"[debate] WARNING: judge generation ({kind}) raised "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if kind == "prefill":
                    self._note_prefill_failure(f"{type(exc).__name__}")
                continue

            try:
                verdict = MedGemmaAgent._extract_json_object(raw)
            except (json.JSONDecodeError, ValueError):
                verdict = None
            if verdict is None or "winner" not in verdict:
                reason = "no JSON object" if verdict is None else "JSON without a 'winner' field"
                print(
                    f"[debate] WARNING: judge parse failed on round {round_num} "
                    f"({kind}, {reason}), raw output: {raw[:300]!r}",
                    flush=True,
                )
                if kind == "prefill":
                    self._note_prefill_failure(reason)
                continue

            if kind == "prefill":
                self._prefill_failures = 0
            return verdict

        return None

    def run(
        self,
        state: NeuroimagingState,
        rounds: int = 1,
        advocates=ADVOCATES,
    ) -> dict:
        """
        Run the debate for `rounds` rounds (1–3; anything else raises ValueError)
        with the given advocate subset of ADVOCATES.

        Returns dict with:
            winner, winner_detailed, confidence (None if unreadable), reason,
            rounds_completed, round_changed (computed from the verdict labels),
            judge_claimed_round_changed, round_verdicts, advocates,
            judge_parse_failed, judge_parse_failed_rounds, arguments
        """
        if not isinstance(rounds, int) or not 1 <= rounds <= self.MAX_ROUNDS:
            raise ValueError(f"debate rounds must be in 1..{self.MAX_ROUNDS}, got {rounds!r}")
        advocates = parse_advocates(advocates)
        # SAM3 is not run for ineligible tasks (ms, stroke). An advocate without
        # evidence still argues (in practice "normal" on every image) and the judge
        # follows it, so it does not take part when segmentation was skipped.
        if "sam" in advocates and (state.get("segmentation_result") or {}).get("skipped"):
            advocates = tuple(a for a in advocates if a != "sam")
            if not advocates:
                raise ValueError("debate needs at least one advocate besides SAM3 when SAM3 is skipped")
        task = state.get("task")
        image_path = state["image_path"]
        prior_verdict: Optional[dict] = None
        parse_failures = 0
        all_arguments: list[dict] = []
        round_verdicts: list[dict] = []
        current_args: dict[str, str] = {}
        builders = {"cnn": _build_cnn_prompt, "clip": _build_clip_prompt, "sam": _build_sam_prompt}

        for round_num in range(1, rounds + 1):
            print(f"[debate] Round {round_num}/{rounds}", flush=True)

            prompts = {
                a: builders[a](state, round_num, prior_verdict, current_args, advocates)
                for a in advocates
            }
            # Advocates are asked for a single paragraph, so they too are better
            # off skipping the hidden thought block — otherwise a truncated
            # thought is what the judge ends up reading.
            advocate_kwargs = {
                "max_new_tokens": ADVOCATE_MAX_NEW_TOKENS,
                "prefill": self._prefill(),
            }
            current_args = {
                a: self.medgemma.generate_for_prompt(image_path, prompts[a], **advocate_kwargs)
                for a in advocates
            }
            all_arguments.extend(
                {"round": round_num, "role": a, "argument": current_args[a]} for a in advocates
            )

            prior_section = ""
            if prior_verdict:
                prior_section = (
                    f"Previous round verdict: {prior_verdict['winner']} "
                    f"(confidence {_fmt_conf(prior_verdict.get('confidence'))}). "
                    "Re-evaluate whether the new arguments change your verdict.\n\n"
                )

            judge_prompt = _JUDGE_TEMPLATE.format(
                task=state["task"],
                prior_verdict_section=prior_section,
                advocate_sections="\n\n".join(
                    f"{_JUDGE_TITLES[a]}:\n{current_args[a]}" for a in advocates
                ),
                weigh_phrase=_WEIGH_PHRASES[len(advocates)],
            )
            verdict = self._judge(image_path, judge_prompt, round_num)
            round_failed = verdict is None
            if round_failed:
                verdict = {
                    "winner": state.get("suspected_pathology") or "unknown",
                    "winner_detailed": None,
                    "confidence": None,
                    "reason": "Judge parse failed — defaulting to suspected pathology.",
                }
            raw_conf = verdict.get("confidence")
            verdict["confidence"] = parse_confidence(raw_conf)
            if verdict["confidence"] is None:
                round_failed = True
                if raw_conf is not None:
                    verdict["confidence_raw"] = raw_conf
            if round_failed:
                parse_failures += 1
            verdict.setdefault("winner_detailed", None)
            verdict["judge_claimed_round_changed"] = verdict.pop("round_changed", None)
            label = verdict_label(verdict, task)
            verdict["round_changed"] = bool(
                round_verdicts and label != round_verdicts[-1]["label_canonical"]
            )
            verdict["judge_parse_failed"] = round_failed
            round_verdicts.append({"round": round_num, **verdict, "label_canonical": label})
            prior_verdict = verdict
            print(
                f"[debate] Round {round_num} verdict: {verdict.get('winner')} "
                f"conf={_fmt_conf(verdict.get('confidence'))} "
                f"changed={verdict['round_changed']}"
            )

        return {
            "winner": prior_verdict.get("winner", "unknown"),
            "winner_detailed": prior_verdict.get("winner_detailed"),
            "confidence": prior_verdict.get("confidence"),
            "reason": prior_verdict.get("reason", ""),
            # True when any round's verdict label differs from the previous round's.
            "round_changed": any(v["round_changed"] for v in round_verdicts),
            "judge_claimed_round_changed": prior_verdict.get("judge_claimed_round_changed"),
            "round_verdicts": round_verdicts,
            "advocates": list(advocates),
            "rounds_completed": rounds,
            # Surfaced all the way to the eval JSONL: a run where this is True on
            # any image did not fully debate (fallback verdict or unreadable confidence).
            "judge_parse_failed": parse_failures > 0,
            "judge_parse_failed_rounds": parse_failures,
            "arguments": all_arguments,
        }
