"""
LangGraph state definition for the neuroimaging multi-agent pipeline.
"""

from typing import Optional, TypedDict


class SegmentationResult(TypedDict):
    mask_path: str  # Path to saved binary mask PNG
    bbox: list  # [x1, y1, x2, y2] bounding box
    guided_image_path: str  # Original image with red bbox overlay (for MedGemma only)


class ClassificationResult(TypedDict):
    predicted_class: str
    confidence: float
    all_probs: dict  # class_name → probability


class BiomedCLIPResult(TypedDict):
    ranked_labels: list  # Sorted highest → lowest similarity
    scores: list  # Corresponding cosine similarities
    top_label: str
    top_score: float


class NeuroimagingState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    image_path: str
    task: str  # "binary_tumor" | "multiclass_tumor" | "ms" | "stroke"
    metadata: dict  # Optional clinical context (modality, patient info, etc.)

    # ── Routing ───────────────────────────────────────────────────────────────
    routing_decision: Optional[
        str
    ]  # "cnn_direct" | "sam3_then_cnn" | "biomedclip" | "human_review"
    routing_confidence: float  # MedGemma's confidence in its routing decision
    routing_reasoning: str  # Short explanation from MedGemma
    suspected_pathology: str  # MedGemma's initial assessment

    # ── MedGemma structured diagnosis outputs ────────────────────────────────
    medgemma_diagnosis: Optional[dict]  # from system_prompt.txt (raw image)
    medgemma_bbox_diagnosis: Optional[
        dict
    ]  # from system_prompt_bbox.txt (SAM3 overlay)
    final_medgemma_diagnosis: Optional[dict]  # final evidence-fusion diagnosis

    # ── Tool outputs ──────────────────────────────────────────────────────────
    segmentation_result: Optional[SegmentationResult]
    classification_result: Optional[ClassificationResult]
    biomedclip_result: Optional[BiomedCLIPResult]
    explainability_result: Optional[dict]  # saliency map paths keyed by method name
    saliency_sam3_iou: Optional[float]    # IoU between GradCAM++ heatmap and SAM3 mask (None when SAM3 mask is empty)
    sam3_mask_empty: bool                 # True when SAM3 predicted no lesion pixels

    # ── Final output ──────────────────────────────────────────────────────────
    final_report: Optional[str]
    final_predicted_class: Optional[str]
    final_confidence: float
    requires_human_review: bool
    verification_result: Optional[dict]  # MedGemma vs CNN agreement check
    fhir_report: Optional[dict]          # FHIR R4 DiagnosticReport resource

    # ── Diagnostics ───────────────────────────────────────────────────────────
    routing_path: list  # Ordered list of nodes visited, for evaluation


def initial_state(
    image_path: str, task: str, metadata: dict = None
) -> NeuroimagingState:
    """Create a blank state for a new image."""
    return NeuroimagingState(
        image_path=image_path,
        task=task,
        metadata=metadata or {},
        routing_decision=None,
        routing_confidence=0.0,
        routing_reasoning="",
        suspected_pathology="",
        medgemma_diagnosis=None,
        medgemma_bbox_diagnosis=None,
        final_medgemma_diagnosis=None,
        segmentation_result=None,
        classification_result=None,
        biomedclip_result=None,
        explainability_result=None,
        saliency_sam3_iou=None,
        sam3_mask_empty=False,
        final_report=None,
        final_predicted_class=None,
        final_confidence=0.0,
        requires_human_review=False,
        verification_result=None,
        fhir_report=None,
        routing_path=[],
    )
