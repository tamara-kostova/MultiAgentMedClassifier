"""
LangGraph state definition for the neuroimaging multi-agent pipeline.
"""

from pathlib import Path
from typing import Optional, TypedDict

from agents.dicom_tool import DICOMPreprocessor


_DICOM_PREPROCESSOR: Optional[DICOMPreprocessor] = None


class SegmentationResult(TypedDict):
    mask_path: str  # Path to saved binary mask PNG
    bbox: list  # [x1, y1, x2, y2] bounding box
    guided_image_path: str  # Original image with red bbox overlay (for MedGemma only)
    dice_estimate: float  # Model's self-estimated Dice (or 0.0 if unavailable)


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
    saliency_sam3_iou: Optional[float]    # IoU between GradCAM++ heatmap and SAM3 mask

    # ── Atlas enrichment (siibra) ─────────────────────────────────────────────
    atlas_enrichment: Optional[dict]  # assigned_region, mni_coords, hemisphere, scores

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
    """
    Create a blank state for a new image.

    Raw DICOM inputs are converted to a PNG in `outputs/preprocessed/` so the
    existing PNG-based agents can run unchanged. Relevant DICOM header fields
    are merged into `metadata`, and the original DICOM slice path is preserved
    as `metadata["dicom_path"]` for atlas coordinate mapping.
    """
    prepared = _prepare_input_image(image_path)
    prepared_metadata = dict(prepared.get("dicom_metadata") or {})
    if prepared.get("dicom_path"):
        prepared_metadata["dicom_path"] = prepared["dicom_path"]
    if prepared.get("nifti_path"):
        prepared_metadata["nifti_path"] = prepared["nifti_path"]
    prepared_metadata["source_image_path"] = str(Path(image_path))

    merged_metadata = dict(prepared_metadata)
    if metadata:
        merged_metadata.update(metadata)

    return NeuroimagingState(
        image_path=prepared["image_path"],
        task=task,
        metadata=merged_metadata,
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
        final_report=None,
        final_predicted_class=None,
        final_confidence=0.0,
        requires_human_review=False,
        verification_result=None,
        atlas_enrichment=None,
        fhir_report=None,
        routing_path=[],
    )


def _prepare_input_image(image_path: str) -> dict:
    global _DICOM_PREPROCESSOR
    if _DICOM_PREPROCESSOR is None:
        _DICOM_PREPROCESSOR = DICOMPreprocessor()
    return _DICOM_PREPROCESSOR.prepare(image_path)
