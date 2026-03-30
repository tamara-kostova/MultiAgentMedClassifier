"""
Central configuration for the multi-agent neuroimaging pipeline.
Adjust model checkpoint paths and thresholds before running.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

_DEFAULT_SAM3_PROBE = "checkpoints/sam3_probe.pth"
_DEFAULT_SAM3_BPE_PATH = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

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
    # Probe heads trained on layer-18 features
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

    # ── BiomedCLIP model ID ───────────────────────────────────────────────────
    biomedclip_model_id: str = (
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # (sam3_checkpoint kept for backwards-compat but no longer used — SAM3's
    # backbone is loaded by build_sam3_image_model, not a raw checkpoint file)
    sam3_checkpoint: str | None = None

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class RoutingConfig:
    # Confidence below this → route to SAM3 segmentation-guided path
    sam3_threshold: float = 0.70
    # Confidence below this → flag for human review
    human_review_threshold: float = 0.45
    # After CNN classification, re-rank with BiomedCLIP for multiclass if below this
    biomedclip_rerank_threshold: float = 0.65
    # Max JSON parse retries for MedGemma output
    max_parse_retries: int = 3


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


# Module-level default (import and mutate as needed)
DEFAULT_CONFIG = PipelineConfig()
