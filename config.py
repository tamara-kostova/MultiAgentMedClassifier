"""
Central configuration for the multi-agent neuroimaging pipeline.
Adjust model checkpoint paths and thresholds before running.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

_DEFAULT_SAM3_PROBE = "checkpoints/sam3_probe.pth"
_DEFAULT_SAM3_BPE_PATH = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


def cuda_is_usable() -> tuple[bool, str | None]:
    try:
        if not torch.cuda.is_available():
            return False, None
    except (AssertionError, RuntimeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"

    try:
        torch.empty(1, device="cuda")
    except (AssertionError, RuntimeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _mps_is_usable() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.empty(1, device="mps")
    except RuntimeError:
        return False
    return True


def resolve_torch_device(device_name: str, caller: str = "config") -> torch.device:
    device = torch.device(device_name)
    if device.type == "mps" and not _mps_is_usable():
        print(f"[{caller}] MPS requested but unusable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "cuda":
        usable, reason = cuda_is_usable()
        if not usable:
            detail = f" ({reason})" if reason else ""
            print(f"[{caller}] CUDA requested but unusable{detail}; falling back to CPU.")
            return torch.device("cpu")
    return device


def resolve_vision_device(
    device_name: str,
    caller: str = "config",
    prefer_cuda: bool = True,
) -> torch.device:
    """Resolve the device for CNN/BiomedCLIP vision models.

    Vision-side inference is small enough to share the GPU with the larger
    agents, and keeping it on CUDA avoids slow CPU fallbacks during evaluation.
    """
    if prefer_cuda and cuda_is_usable()[0]:
        return torch.device("cuda")
    return resolve_torch_device(device_name, caller=caller)


def _default_torch_device() -> str:
    if cuda_is_usable()[0]:
        return "cuda"
    if _mps_is_usable():
        return "mps"
    return "cpu"

# ── Task identifiers ──────────────────────────────────────────────────────────
TASKS = ["binary_tumor", "multiclass_tumor", "ms", "stroke"]

# Best CNN per task
BEST_CNN_PER_TASK = {
    "binary_tumor": "vgg16",
    "multiclass_tumor": "densenet169",
    "ms": "resnet101",
    "stroke": "densenet169",
}

# 12-class tumor labels
TUMOR_12_CLASSES = [
    "meningioma",
    "glioma",
    "neurocytoma",
    "pituitary",
    "schwannoma",
    "carcinoma",
    "granuloma",
    "medulloblastoma",
    "papilloma",
    "tuberculoma",
    "germinoma",
    "normal",
]

# Binary task labels
BINARY_LABELS = {
    "binary_tumor": ["normal brain MRI", "brain tumor MRI"],
    "ms": ["normal brain FLAIR MRI", "multiple sclerosis brain FLAIR MRI"],
    "stroke": ["normal brain CT", "ischemic stroke brain CT"],
}


@dataclass
class ModelConfig:
    # ── CNN checkpoints (None → ImageNet pretrained weights) ──────────────────
    cnn_checkpoints: dict = field(
        default_factory=lambda: {
            "binary_tumor": "checkpoints/vgg16_MRI_tumor_binary_norm_final.pt",
            "multiclass_tumor": "checkpoints/densenet169_MRI_tumor_multiclass_norm_final.pt",
            "ms": "checkpoints/resnet101_MRI_ms_norm_final.pt",
            "stroke": "checkpoints/densenet169_CT_stroke_binary_norm_final.pt",
        }
    )

    # ── BiomedCLIP linear probe checkpoints (None → zero-shot mode) ──────────
    # Probe heads from 18_layer_fusion_benchmark.py (layer-6 or concat fusion of layers 2,6,11)
    biomedclip_probe_checkpoints: dict = field(
        default_factory=lambda: {
            "binary_tumor": None,
            "multiclass_tumor": None,
            "ms": None,
            "stroke": None,
        }
    )

    # ── SAM3 ──────────────────────────────────────────────────────────────────
    # Linear probe trained on BraTS 2021; Dice = 0.836 pixel-level.
    sam3_linear_probe_checkpoint: str | None = _DEFAULT_SAM3_PROBE
    # BPE vocabulary file required by SAM3's text encoder.
    # Path: sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz (inside the sam3 repo)
    sam3_bpe_path: str | None = _DEFAULT_SAM3_BPE_PATH

    # ── MedGemma model ID ─────────────────────────────────────────────────────
    # Requires HuggingFace login and accepted terms of use for this gated model.
    medgemma_model_id: str = "google/medgemma-1.5-4b-it"
    # Set to True on GPUs with <12 GB VRAM (4-bit NF4 via bitsandbytes).
    # Set to False on GPUs with >=12 GB VRAM (bfloat16, faster inference).
    use_4bit_quantization: bool = False

    # ── Few-shot examples ─────────────────────────────────────────────────────
    # When True, MedGemmaAgent prepends one real image + expected JSON per class
    # as prior conversation turns before the triage query image.
    use_few_shot: bool = False
    # Root directory for resolving relative paths in few_shot_examples.csv.
    few_shot_data_dir: Optional[str] = None

    # ── BiomedCLIP model ID ───────────────────────────────────────────────────
    biomedclip_model_id: str = (
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # (sam3_checkpoint kept for backwards-compat but no longer used — SAM3's
    # backbone is loaded by build_sam3_image_model, not a raw checkpoint file)
    sam3_checkpoint: str | None = None

    # ── Post-hoc temperature scaling (Guo et al., 2017) ─────────────────────
    # Fitted via TemperatureScaler.fit() on a held-out validation set.
    # T > 1.0 → model was over-confident; T < 1.0 → under-confident.
    # Default 1.0 = no calibration (raw softmax).
    cnn_temperatures: dict = field(
        default_factory=lambda: {
            "binary_tumor": 1.0,
            "multiclass_tumor": 1.0,
            "ms": 1.0,
            "stroke": 1.0,
        }
    )

    # ── Device ────────────────────────────────────────────────────────────────
    # Auto-selects "cuda" on NVIDIA/Linux, "mps" on Apple Silicon when available,
    # otherwise "cpu". Override from the CLI with --device {cuda,mps,cpu}.
    device: str = field(default_factory=_default_torch_device)
    # Keep CNN and BiomedCLIP on CUDA whenever available, unless the CLI/user
    # explicitly asks for cpu or mps.
    prefer_cuda_for_vision: bool = True


@dataclass
class RoutingConfig:
    # Tasks for which SAM3 segmentation is valid.
    # The linear probe was trained on BraTS 2021 (tumor only) — MS and stroke probes
    # performed poorly and should not be used.
    sam3_eligible_tasks: tuple = ("binary_tumor", "multiclass_tumor")

    # Force SAM3 segmentation on every non-normal case, regardless of confidence.
    # Useful when MedGemma is overconfident and never falls below sam3_threshold.
    always_run_sam3: bool = False
    # Force BiomedCLIP on every case, regardless of confidence or diagnosis_name.
    # Useful when MedGemma is overconfident and never falls below biomedclip_rerank_threshold.
    always_run_biomedclip: bool = False
    # Confidence below this → route to SAM3 segmentation-guided path
    sam3_threshold: float = 0.70
    # Confidence below this → flag for human review
    human_review_threshold: float = 0.45
    # After CNN classification, re-rank with BiomedCLIP for multiclass if below this
    biomedclip_rerank_threshold: float = 0.65
    # Max JSON parse retries for MedGemma output
    max_parse_retries: int = 3
    # IoU between GradCAM++ heatmap and SAM3 mask below this → confidence penalty
    low_iou_penalty_threshold: float = 0.3


@dataclass
class PreprocessConfig:
    image_size: int = 224
    # ImageNet normalization (used for all pretrained models)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    output_dir: str = "outputs"
    # Set True to generate Grad-CAM++ and Integrated Gradients after CNN classification.
    # Adds ~1-2s per image but produces saliency PNGs in outputs/explainability/.
    generate_explainability: bool = False
    # Set True to skip MedGemma report generation during evaluation.
    # Saves ~5–9 s/image with no effect on accuracy/F1/ECE metrics.
    skip_report: bool = False


# Module-level default (import and mutate as needed)
DEFAULT_CONFIG = PipelineConfig()
