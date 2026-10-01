"""
LangGraph node functions for the neuroimaging pipeline.

Each node receives the full NeuroimagingState, performs its action,
and returns a (partial) state dict with the fields it modifies.
"""

import uuid
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import BEST_CNN_PER_TASK, DEFAULT_CONFIG, RoutingConfig
from explainability.saliency import (
    GRADCAM_TARGET_LAYERS,
    GradCAMPlusPlus,
    IntegratedGradientsForCNN,
)
from pipeline.state import NeuroimagingState

VERIFICATION_DISAGREEMENT_CONFIDENCE_CAP = 0.55
# ── Tool instances are passed in at graph-build time via closure ──────────────
# (avoids re-loading multi-GB models on every invocation)


def _case_name(state: NeuroimagingState) -> str:
    return Path(state["image_path"]).name


def _log_node_start(node: str, state: NeuroimagingState) -> float:
    print(f"[{node}] start {_case_name(state)}", flush=True)
    return time.perf_counter()


def _log_node_done(node: str, state: NeuroimagingState, t0: float) -> None:
    print(f"[{node}] done {_case_name(state)} in {time.perf_counter() - t0:.1f}s", flush=True)


def _usable(value) -> bool:
    return bool(value) and str(value).strip().lower() not in ("none", "null", "nan")


def _dx_label(dx, task: str):
    """Class label carried by a MedGemma diagnosis (object or dict) for `task`.

    Only multiclass tumor subtyping carries its label in diagnosis_detailed; for
    every other task diagnosis_name is the class and diagnosis_detailed is a
    free-text elaboration. Mirrors eval_analysis.mg_diag_pred().
    """
    if dx is None:
        return None
    fields = (
        ("diagnosis_detailed", "diagnosis_name")
        if task == "multiclass_tumor"
        else ("diagnosis_name", "diagnosis_detailed")
    )
    get = dx.get if isinstance(dx, dict) else (lambda f: getattr(dx, f, None))
    return next((get(f) for f in fields if _usable(get(f))), None)


def apply_iou_penalty(conf, saliency_iou, cfg: RoutingConfig):
    """Low GradCAM++/SAM3 IoU → conf * max(iou / threshold, floor).

    Returns (conf, factor); factor is None when no penalty applies.
    """
    if saliency_iou is None or saliency_iou >= cfg.low_iou_penalty_threshold:
        return conf, None
    factor = max(saliency_iou / cfg.low_iou_penalty_threshold, cfg.low_iou_penalty_floor)
    penalised = None if conf is None else conf * factor
    if conf is not None:
        print(
            f"[report] Low GradCAM++/SAM3 IoU={saliency_iou:.3f} → "
            f"confidence penalised {conf:.3f}→{penalised:.3f} (factor {factor:.3f})"
        )
    return penalised, factor


def make_triage_node(agent, routing_cfg: RoutingConfig = None):
    """
    Factory: returns an initial MedGemma triage node.
    Runs system_prompt.txt for a first-pass impression that is later fused with
    downstream specialist outputs.
    """
    def triage_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("triage", state)
        dx, routing = agent.diagnose(state["image_path"], task=state["task"])
        _log_node_done("triage", state, t0)

        return {
            "routing_decision": "full_workup",
            "routing_confidence": routing.confidence,
            "routing_reasoning": (
                "Initial MedGemma triage completed; proceeding to full specialist workup."
            ),
            "suspected_pathology": routing.suspected_pathology,
            "medgemma_diagnosis": dx.model_dump(),
            "triage_parse_failed": bool(getattr(dx, "parse_failed", False)),
            "routing_path": state["routing_path"] + ["triage"],
        }

    return triage_node


def make_triage_final_node(routing_cfg: RoutingConfig = None):
    """
    Terminal node of the --triage_only graphs: the triage (single or forest
    consensus) becomes the final prediction so eval rows score the triage stage.
    """
    cfg = routing_cfg or DEFAULT_CONFIG.routing

    def triage_final_node(state: NeuroimagingState) -> dict:
        task = state["task"]
        consensus = state.get("forest_consensus")
        if consensus is not None:
            fields = (
                ("winner_detailed", "winner") if task == "multiclass_tumor"
                else ("winner", "winner_detailed")
            )
            final_class = next((consensus.get(f) for f in fields if _usable(consensus.get(f))), None)
            if not consensus.get("winner_canonical"):
                final_class = None
            final_conf = consensus.get("confidence_weighted_confidence")
        else:
            final_class = _dx_label(state.get("medgemma_diagnosis"), task)
            final_conf = (state.get("medgemma_diagnosis") or {}).get("diagnosis_confidence")
        parse_failed = bool(state.get("triage_parse_failed"))
        if parse_failed:
            final_class = None
        final_conf = float(final_conf) if final_conf is not None else 0.0
        return {
            "final_predicted_class": final_class or "unknown",
            "final_confidence": final_conf,
            "final_medgemma_diagnosis": None,
            "final_report": "[triage only — no specialist workup]",
            "requires_human_review": (
                parse_failed or not final_class or final_conf < cfg.human_review_threshold
            ),
            "routing_path": state["routing_path"] + ["triage_final"],
        }

    return triage_final_node


def make_cnn_node(cnn_tool):
    """CNN classifier node. Uses best-performing model per task (from cnns.tex)."""

    def cnn_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("cnn_classify", state)
        result = cnn_tool.classify(state["image_path"], state["task"])
        _log_node_done("cnn_classify", state, t0)
        return {
            "classification_result": result,
            "routing_path": state["routing_path"] + ["cnn_classify"],
        }

    return cnn_node


def make_sam3_node(sam3_tool, routing_cfg: RoutingConfig = None):
    """SAM3 segmentation node. Skips automatically for ineligible tasks (ms, stroke)."""
    cfg = routing_cfg or DEFAULT_CONFIG.routing

    def sam3_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("sam3_segment", state)
        if state["task"] not in cfg.sam3_eligible_tasks:
            print(f"[sam3_segment] task={state['task']} not eligible — skipping")
            _log_node_done("sam3_segment", state, t0)
            return {
                "segmentation_result": {"skipped": True, "mask_path": None, "bbox": None, "guided_image_path": None},
                "routing_path": state["routing_path"] + ["sam3_segment"],
            }
        pathology = state.get("suspected_pathology") or "brain lesion"
        result = sam3_tool.segment(state["image_path"], text_prompt=pathology)
        _log_node_done("sam3_segment", state, t0)
        return {
            "segmentation_result": result,
            "sam3_mask_empty": bool((result or {}).get("mask_empty")),
            "routing_path": state["routing_path"] + ["sam3_segment"],
        }

    return sam3_node


def make_biomedclip_node(biomedclip_tool, routing_cfg: RoutingConfig = None):
    """
    BiomedCLIP node using layer-18 features (from clip.tex).
    Also used as a re-ranking step if CNN confidence is below threshold.
    """

    def biomedclip_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("biomedclip", state)
        result = biomedclip_tool.classify(state["image_path"], state["task"])
        _log_node_done("biomedclip", state, t0)
        return {
            "biomedclip_result": result,
            "routing_path": state["routing_path"] + ["biomedclip"],
        }

    return biomedclip_node


def make_report_node(agent, routing_cfg: RoutingConfig = None, skip_report: bool = False):
    """
    MedGemma report generation node.
    Synthesizes all tool outputs into a structured triage report.
    Determines final predicted class and whether human review is needed.
    Applies a proportional confidence penalty when GradCAM++/SAM3 IoU is low.
    """
    cfg = routing_cfg or DEFAULT_CONFIG.routing

    def report_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("report", state)
        saliency_iou = state.get("saliency_sam3_iou")
        final_dx = None
        if skip_report:
            report = "[report skipped in eval mode]"
        else:
            report, final_dx = agent.generate_report(
                image_path=state["image_path"],
                task=state["task"],
                routing_path=state["routing_path"],
                initial_medgemma_dx=state.get("medgemma_diagnosis"),
                sam3_medgemma_dx=state.get("medgemma_bbox_diagnosis"),
                cnn_result=state.get("classification_result"),
                sam3_result=state.get("segmentation_result"),
                biomedclip_result=state.get("biomedclip_result"),
                explainability_result=state.get("explainability_result"),
                verification_result=state.get("verification_result"),
                saliency_iou=saliency_iou,
            )

        # Determine final prediction from MedGemma's fused diagnosis when available.
        cnn = state.get("classification_result")
        clip = state.get("biomedclip_result")
        report_parse_failed = (not skip_report) and final_dx is None
        dx_label = _dx_label(final_dx, state["task"])
        if dx_label:
            final_class = dx_label
            final_conf = final_dx.diagnosis_confidence
        elif cnn:
            final_class = cnn["predicted_class"]
            final_conf = cnn["confidence"]
        elif clip:
            final_class = clip["top_label"]
            final_conf = clip["top_score"]
        else:
            final_class = state.get("suspected_pathology", "unknown")
            final_conf = state.get("routing_confidence", 0.0)

        requires_review = final_conf < cfg.human_review_threshold
        # Penalty for poor spatial alignment
        final_conf, penalty_factor = apply_iou_penalty(final_conf, saliency_iou, cfg)
        if penalty_factor is not None:
            requires_review = True
        # verification_node's verdict: an explicit disagreement caps the confidence.
        verification = state.get("verification_result") or {}
        if verification.get("agreement") is False:
            capped = min(final_conf, VERIFICATION_DISAGREEMENT_CONFIDENCE_CAP)
            if capped < final_conf:
                print(f"[report] verification disagreement → confidence capped {final_conf:.3f}→{capped:.3f}")
            final_conf = capped
            requires_review = True
        if report_parse_failed:
            requires_review = True
        requires_review = bool(requires_review or state.get("requires_human_review", False))

        updates = {
            "final_report": report,
            "final_predicted_class": final_class,
            "final_confidence": final_conf,
            "final_medgemma_diagnosis": (
                final_dx.model_dump() if final_dx is not None else None
            ),
            "requires_human_review": requires_review,
            "confidence_penalty_factor": penalty_factor,
            "report_parse_failed": report_parse_failed,
            "routing_path": state["routing_path"] + ["report"],
        }
        _log_node_done("report", state, t0)
        return updates

    return report_node


def make_explainability_node(cnn_tool, output_dir: str = "outputs/explainability"):
    """
    Optional post-classification node: generates Grad-CAM++ and Integrated Gradients
    for the CNN's prediction on the current image.

    Outputs are saved as PNG files; paths are stored in state["explainability_result"].
    Only runs if a classification_result exists (i.e., a CNN was used).

    Controlled by PipelineConfig.generate_explainability (default False).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def explainability_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("explainability", state)
        cnn_result = state.get("classification_result")
        if cnn_result is None:
            # No CNN ran — nothing to explain
            _log_node_done("explainability", state, t0)
            return {"explainability_result": None}

        task = state["task"]
        image_path = state["image_path"]

        # Resolve the model and its class mapping
        model, class_names = cnn_tool.get_model_and_classes(task)
        if model is None:
            _log_node_done("explainability", state, t0)
            return {"explainability_result": None}

        predicted_class = cnn_result["predicted_class"]
        target_idx = next(
            (i for i, c in enumerate(class_names) if c == predicted_class),
            0,
        )

        # Use the same transform as cnn_tool.classify (grayscale → 3-channel)
        pil_img = Image.open(image_path).convert("L").convert("RGB")
        img_tensor = cnn_tool._transform(pil_img).unsqueeze(0).to(cnn_tool.device)

        uid = uuid.uuid4().hex[:8]
        paths = {}
        orig_np = np.array(pil_img.resize((224, 224)))

        # ── Grad-CAM++ ────────────────────────────────────────────────────────
        # The architecture actually loaded (cnn_tool may switch it to match the
        # checkpoint), not the configured one.
        get_arch = getattr(cnn_tool, "get_arch", None)
        model_name = (get_arch(task) if get_arch else None) or BEST_CNN_PER_TASK.get(task, "")
        target_layer_fn = GRADCAM_TARGET_LAYERS.get(model_name)
        _cam_r = None  # retained for IoU computation below
        if target_layer_fn is not None:
            target_layer = target_layer_fn(model)
            gc_pp = GradCAMPlusPlus(model, target_layer)
            try:
                cam = gc_pp.generate_cam(img_tensor, target_idx)
            finally:
                gc_pp.cleanup()

            _cam_r = cv2.resize(cam, (224, 224))
            hm = cv2.applyColorMap(np.uint8(_cam_r * 255), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            overlay = (0.5 * orig_np + 0.5 * hm).astype(np.uint8)
            gc_path = str(out_path / f"gradcam_pp_{uid}.png")
            Image.fromarray(overlay).save(gc_path)
            paths["gradcam_pp"] = gc_path

        # ── Integrated Gradients ──────────────────────────────────────────────
        ig = IntegratedGradientsForCNN(
            model, steps=30
        )  # 30 steps is fast enough for inference
        attr_map = ig.attribute(img_tensor, target_idx)
        attr_r = cv2.resize(attr_map, (224, 224))
        hm_ig = cv2.applyColorMap(np.uint8(attr_r * 255), cv2.COLORMAP_INFERNO)
        hm_ig = cv2.cvtColor(hm_ig, cv2.COLOR_BGR2RGB)
        overlay_ig = (0.5 * orig_np + 0.5 * hm_ig).astype(np.uint8)
        ig_path = str(out_path / f"ig_{uid}.png")
        Image.fromarray(overlay_ig).save(ig_path)
        paths["integrated_gradients"] = ig_path

        # ── GradCAM++ / SAM3 mask IoU ─────────────────────────────────────────
        # IoU is only meaningful when SAM3 actually found something (non-empty
        # mask).  An all-zero mask means SAM3 predicted no lesion — that is
        # correct for normal scans and should NOT trigger a confidence penalty.
        # We record None in that case so report_node skips the penalty.
        saliency_sam3_iou = None
        seg_result = state.get("segmentation_result")
        if _cam_r is not None and seg_result and seg_result.get("mask_path"):
            try:
                sam_mask = np.array(
                    Image.open(seg_result["mask_path"]).convert("L").resize((224, 224))
                )
                sam_binary = (sam_mask > 127).astype(np.uint8)
                if sam_binary.sum() == 0:
                    # SAM3 found no lesion pixels — IoU undefined, skip penalty
                    print("[explainability] SAM3 mask empty (no lesion predicted) — IoU skipped")
                else:
                    cam_binary = (_cam_r >= 0.5).astype(np.uint8)
                    intersection = int((cam_binary & sam_binary).sum())
                    union = int((cam_binary | sam_binary).sum())
                    saliency_sam3_iou = intersection / union if union > 0 else 0.0
                    print(f"[explainability] GradCAM++/SAM3 IoU = {saliency_sam3_iou:.3f}")
            except Exception as e:
                print(f"[explainability] IoU computation failed: {e}")

        updates = {
            "explainability_result": paths,
            "saliency_sam3_iou": saliency_sam3_iou,
            "sam3_mask_empty": bool(
                state.get("sam3_mask_empty")
                or (saliency_sam3_iou is None and
                    seg_result is not None and
                    seg_result.get("mask_path") is not None)
            ),
            "routing_path": state["routing_path"] + ["explainability"],
        }
        _log_node_done("explainability", state, t0)
        return updates

    return explainability_node


def make_skip_explainability_node():
    """Fast no-op explainability node for evaluation and constrained devices."""

    def skip_explainability_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("explainability", state)
        _log_node_done("explainability", state, t0)
        return {
            "explainability_result": None,
            "saliency_sam3_iou": None,
            "routing_path": state["routing_path"] + ["explainability_skipped"],
        }

    return skip_explainability_node


def human_review_node(state: NeuroimagingState) -> dict:
    """Terminal node: flags the case for radiologist review."""
    print(
        f"[HumanReview] Case flagged: {state['image_path']} "
        f"(confidence={state['routing_confidence']:.2f}, "
        f"pathology={state.get('suspected_pathology', 'unknown')})"
    )
    return {
        "requires_human_review": True,
        "final_report": (
            f"FLAGGED FOR HUMAN REVIEW\n"
            f"Reason: Low routing confidence ({state['routing_confidence']:.2f})\n"
            f"Suspected: {state.get('suspected_pathology', 'unknown')}\n"
            f"Routing reasoning: {state.get('routing_reasoning', '')}"
        ),
        "routing_path": state["routing_path"] + ["human_review"],
    }

def make_verification_node(agent):
    """
    Factory: returns a verification node that runs MedGemma post-hoc against the
    Grad-CAM++ saliency map. No-ops gracefully when explainability was not run.
    """

    def verification_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("verification", state)
        saliency_paths = state.get("explainability_result") or {}
        gradcam_path = saliency_paths.get("gradcam_pp")

        if not gradcam_path:
            _log_node_done("verification", state, t0)
            return {"verification_result": None}

        cnn_result = state.get("classification_result") or {}
        verification = agent.verify_cnn_prediction(
            original_image_path=state["image_path"],
            saliency_image_path=gradcam_path,
            cnn_result=cnn_result,
        )

        current_conf = state.get("final_confidence") or cnn_result.get("confidence", 0.5)
        # Only an explicit disagreement caps; a parse failure (agreement=None) is no verdict.
        if verification.agreement is False:
            adjusted_conf = min(current_conf, VERIFICATION_DISAGREEMENT_CONFIDENCE_CAP)
            human_review = True
            print(
                f"[verification] MedGemma disagrees with CNN — "
                f"confidence capped {current_conf:.3f}→{adjusted_conf:.3f}, "
                f"flagging for human review. "
                f"Alternative: {verification.alternative_diagnosis}"
            )
        else:
            adjusted_conf = current_conf
            human_review = state.get("requires_human_review", False)

        updates = {
            "final_confidence": adjusted_conf,
            "requires_human_review": human_review,
            "verification_result": verification.model_dump(),
            "verification_parse_failed": bool(verification.parse_failed),
        }
        _log_node_done("verification", state, t0)
        return updates

    return verification_node


def make_forest_triage_node(forest, n_agents: int = 3):
    """
    Factory: returns a forest triage node that replaces the single MedGemma triage.
    Runs n_agents role-specialized MedGemma instances, votes, and writes consensus
    to state. Downstream nodes (CNN, SAM3, BiomedCLIP, report) run unchanged.
    """
    forest.roles_for(n_agents)  # fail at build time on an invalid forest

    def forest_triage_node(state: NeuroimagingState) -> dict:
        from agents.medgemma_agent import diagnosis_to_routing
        t0 = _log_node_start("forest_triage", state)
        votes = forest.run(state["image_path"], n_agents=n_agents)
        consensus, winner_dx = forest.vote(votes, task=state.get("task"))
        routing = diagnosis_to_routing(winner_dx, forest.medgemma.routing_cfg)

        # Strip internal _dx objects before writing to state
        serializable_votes = []
        for vote in votes:
            row = {k: v for k, v in vote.items() if not k.startswith("_")}
            row["ballot_canonical"] = forest.ballot(vote, state.get("task"))
            serializable_votes.append(row)
        winner_text = consensus["winner"] or "abstain"
        _log_node_done("forest_triage", state, t0)
        return {
            "routing_decision": "full_workup",
            "routing_confidence": consensus["confidence_weighted_confidence"],
            "routing_reasoning": (
                f"Forest consensus ({n_agents} agents): {winner_text} "
                f"({consensus['vote_fraction'] * 100:.0f}% agreement, "
                f"dissent={consensus['dissent_rate'] * 100:.0f}%)"
            ),
            "suspected_pathology": consensus.get("winner_detailed") or consensus["winner"] or "",
            "medgemma_diagnosis": winner_dx.model_dump(),
            "triage_parse_failed": consensus["n_abstain"] == consensus["n_agents"],
            "forest_votes": serializable_votes,
            "forest_consensus": consensus,
            "routing_path": state["routing_path"] + ["forest_triage"],
        }

    return forest_triage_node


def make_debate_node(orchestrator, rounds: int = 1, routing_cfg=None, advocates=None):
    """
    Factory: returns a debate node that replaces verification + report.
    Runs the DebateOrchestrator and resolves final_predicted_class / final_confidence
    from the judge's verdict.
    """
    from agents.debate import ADVOCATES, parse_advocates

    cfg = routing_cfg or DEFAULT_CONFIG.routing
    advocates = parse_advocates(advocates or ADVOCATES)
    if not isinstance(rounds, int) or not 1 <= rounds <= orchestrator.MAX_ROUNDS:
        raise ValueError(f"debate rounds must be in 1..{orchestrator.MAX_ROUNDS}, got {rounds!r}")

    def debate_node(state: NeuroimagingState) -> dict:
        t0 = _log_node_start("debate", state)
        verdict = orchestrator.run(state, rounds=rounds, advocates=advocates)

        # The judge's "winner" is coarse (tumor / stroke / multiple sclerosis /
        # normal) and carries the subtype in "winner_detailed". On multiclass_tumor
        # the label space *is* the subtype, so scoring "winner" there would mark
        # every prediction wrong — prefer the detailed field when it is populated.
        detailed = verdict.get("winner_detailed")
        if state.get("task") == "multiclass_tumor" and _usable(detailed):
            final_class = detailed
        else:
            final_class = verdict.get("winner") or state.get(
                "suspected_pathology", "unknown"
            )
        # None when the judge gave no readable confidence (flagged, never 0.5).
        final_conf = verdict.get("confidence")
        judge_parse_failed = bool(verdict.get("judge_parse_failed"))

        saliency_iou = state.get("saliency_sam3_iou")
        requires_review = final_conf is None or final_conf < cfg.human_review_threshold
        final_conf, penalty_factor = apply_iou_penalty(final_conf, saliency_iou, cfg)
        if penalty_factor is not None:
            requires_review = True
        if judge_parse_failed:
            requires_review = True
        requires_review = bool(requires_review or state.get("requires_human_review", False))

        conf_text = "unknown" if final_conf is None else f"{final_conf:.2f}"
        report_text = (
            f"Debate verdict ({verdict.get('rounds_completed', 1)} round(s)): "
            f"{final_class} (confidence {conf_text}). "
            f"Reason: {verdict.get('reason', 'N/A')}"
        )

        _log_node_done("debate", state, t0)
        return {
            "debate_arguments": verdict.get("arguments", []),
            "debate_verdict": {
                k: v for k, v in verdict.items() if k not in ("arguments", "round_verdicts")
            },
            "debate_rounds_completed": verdict.get("rounds_completed", 1),
            "debate_round_verdicts": verdict.get("round_verdicts", []),
            "debate_round_changed": bool(verdict.get("round_changed")),
            "debate_judge_claimed_round_changed": verdict.get("judge_claimed_round_changed"),
            "judge_parse_failed": judge_parse_failed,
            "final_predicted_class": final_class,
            "final_confidence": final_conf,
            "final_medgemma_diagnosis": None,
            "final_report": report_text,
            "requires_human_review": requires_review,
            "confidence_penalty_factor": penalty_factor,
            "routing_path": state["routing_path"] + ["debate"],
        }

    return debate_node


def make_fhir_node(output_dir: str):
    """Factory: returns a FHIR serialisation node that writes to output_dir/fhir/."""

    def fhir_node(state: NeuroimagingState) -> dict:
        from pipeline.fhir_output import build_diagnostic_report

        report = build_diagnostic_report(
            dict(state),
            output_dir=f"{output_dir}/fhir",
        )
        return {"fhir_report": report}

    return fhir_node


# ── Edge condition ─────────────────────────────────────────────────────────────


def route_from_triage(state: NeuroimagingState) -> str:
    """Legacy helper retained for compatibility with older experiments.

    NOT WIRED INTO ANY GRAPH. This is the only reader of `routing_decision`, and no
    pipeline in graph.py calls add_conditional_edges — all three graphs (standard,
    debate, forest) are linear, so every node runs for every image.

    Consequently `routing_decision` gates nothing: both make_triage_node and
    make_forest_triage_node hardcode it to "full_workup" and discard the
    diagnosis_to_routing() result. The routing thresholds in RoutingConfig
    (sam3_threshold, human_review_threshold, biomedclip_rerank_threshold) influence
    *flags and confidence penalties* — segmentation_result["skipped"],
    requires_human_review, the BiomedCLIP override — never which nodes execute.
    Keep this in mind when interpreting the threshold_sweep / human_review_sweep
    experiment families: they move flags and scores, not execution paths.
    """
    return state["routing_decision"]
