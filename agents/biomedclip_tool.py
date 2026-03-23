"""
BiomedCLIP tool using layer-18 feature extraction.

Key finding from clip.tex:
  - Layer 18 (middle) outperforms penultimate (layer 23) for medical imaging
  - Stroke: layer 18 → 93.5% acc vs 77.8% for full CLIP linear probe
  - Binary tumor: layer 18 → 98.2% acc
  - Concat fusion (layers 3+18+23) gives best overall performance

This tool supports two modes:
  1. Zero-shot: cosine similarity between layer-18 image features and text embeddings.
  2. Linear probe: pass image features through a trained classification head.
"""

from typing import Optional

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image

from config import (
    BINARY_LABELS,
    DEFAULT_CONFIG,
    TUMOR_12_CLASSES,
    ModelConfig,
    PreprocessConfig,
)

# Layer indices to extract (0-indexed from the ViT transformer blocks)
SHALLOW_LAYER = 2  # layer 3 in 1-indexed
MIDDLE_LAYER = 17  # layer 18 in 1-indexed  ← primary finding from clip.tex
DEEP_LAYER = 22  # layer 23 in 1-indexed (penultimate)

CANDIDATE_LABELS = {
    "binary_tumor": BINARY_LABELS["binary_tumor"],
    "ms": BINARY_LABELS["ms"],
    "stroke": BINARY_LABELS["stroke"],
    "multiclass_tumor": [
        f"brain MRI showing {cls} tumor" if cls != "normal" else "normal brain MRI"
        for cls in TUMOR_12_CLASSES
    ],
}


class _FeatureExtractor:
    """Registers forward hooks on ViT transformer blocks to capture intermediate features."""

    def __init__(self, model):
        self.features = {}
        self._hooks = []
        blocks = model.visual.transformer.resblocks
        for idx in [SHALLOW_LAYER, MIDDLE_LAYER, DEEP_LAYER]:
            hook = blocks[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            # output shape: (seq_len, batch, embed_dim) — take CLS token
            self.features[idx] = output[0].detach()  # CLS token

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


class BiomedCLIPTool:
    """
    Wraps BiomedCLIP for layer-18 feature extraction and classification.

    Usage:
        tool = BiomedCLIPTool()
        result = tool.classify(image_path, task="binary_tumor")
    """

    def __init__(
        self,
        model_cfg: ModelConfig = None,
        preprocess_cfg: PreprocessConfig = None,
    ):
        self.model_cfg = model_cfg or DEFAULT_CONFIG.model
        self.preprocess_cfg = preprocess_cfg or DEFAULT_CONFIG.preprocess
        self.device = torch.device(self.model_cfg.device)

        # BiomedCLIP runs on CPU to leave GPU VRAM for MedGemma
        self.clip_device = torch.device("cpu")
        print("[BiomedCLIPTool] Loading BiomedCLIP model (CPU)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_cfg.biomedclip_model_id
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_cfg.biomedclip_model_id)
        self.model = self.model.to(self.clip_device).eval()

        # Per-task linear probe heads (None → zero-shot mode)
        self._probe_heads: dict[str, nn.Linear] = {}
        for task, ckpt in self.model_cfg.biomedclip_probe_checkpoints.items():
            if ckpt is not None:
                head = self._load_probe_head(ckpt, task)
                self._probe_heads[task] = head
                print(f"[BiomedCLIPTool] Loaded linear probe for '{task}': {ckpt}")

    def _load_probe_head(self, checkpoint_path: str, task: str) -> nn.Linear:
        """Load a saved linear classification head for layer-18 concat fusion features."""
        state = torch.load(checkpoint_path, map_location=self.device)
        # Concat fusion: 3 × embed_dim features → num_classes
        in_features = state["weight"].shape[1]
        out_features = state["weight"].shape[0]
        head = nn.Linear(in_features, out_features)
        head.load_state_dict(state)
        return head.to(self.device).eval()

    def _extract_layer_features(
        self, image_tensor: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Extract CLS token features from shallow, middle, and deep layers."""
        extractor = _FeatureExtractor(self.model)
        with torch.no_grad():
            self.model.encode_image(image_tensor)
        features = {k: v.clone() for k, v in extractor.features.items()}
        extractor.remove()
        return features

    @torch.no_grad()
    def classify(self, image_path: str, task: str) -> dict:
        """
        Classify using BiomedCLIP with layer-18 features.

        Returns:
            {
                "ranked_labels": [str, ...],
                "scores": [float, ...],
                "top_label": str,
                "top_score": float,
                "mode": "linear_probe" | "zero_shot"
            }
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.clip_device)
        labels = CANDIDATE_LABELS.get(task, CANDIDATE_LABELS["binary_tumor"])

        if task in self._probe_heads:
            return self._linear_probe_classify(image_tensor, labels, task)
        else:
            return self._zero_shot_classify(image_tensor, labels)

    def _zero_shot_classify(
        self, image_tensor: torch.Tensor, labels: list[str]
    ) -> dict:
        """Cosine similarity between layer-18 image features and text embeddings."""
        features = self._extract_layer_features(image_tensor)
        image_feat = features[MIDDLE_LAYER]  # layer 18 CLS token

        # Normalize image features
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # Encode text labels
        tokens = self.tokenizer(labels).to(self.clip_device)
        text_feats = self.model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        logits = (image_feat @ text_feats.T).squeeze(0)
        scores = torch.softmax(logits * 100, dim=0).cpu().numpy()

        ranked_idx = np.argsort(scores)[::-1]
        return {
            "ranked_labels": [labels[i] for i in ranked_idx],
            "scores": [float(scores[i]) for i in ranked_idx],
            "top_label": labels[int(ranked_idx[0])],
            "top_score": float(scores[ranked_idx[0]]),
            "mode": "zero_shot",
        }

    def _linear_probe_classify(
        self, image_tensor: torch.Tensor, labels: list[str], task: str
    ) -> dict:
        """Concat-fusion linear probe: concatenate features from layers 3, 18, 23."""
        features = self._extract_layer_features(image_tensor)
        # Concatenate shallow + middle + deep CLS tokens
        fused = torch.cat(
            [features[SHALLOW_LAYER], features[MIDDLE_LAYER], features[DEEP_LAYER]],
            dim=-1,
        )
        head = self._probe_heads[task]
        logits = head(fused).squeeze(0)
        scores = torch.softmax(logits, dim=0).cpu().numpy()

        ranked_idx = np.argsort(scores)[::-1]
        return {
            "ranked_labels": [labels[i] for i in ranked_idx],
            "scores": [float(scores[i]) for i in ranked_idx],
            "top_label": labels[int(ranked_idx[0])],
            "top_score": float(scores[ranked_idx[0]]),
            "mode": "linear_probe",
        }
