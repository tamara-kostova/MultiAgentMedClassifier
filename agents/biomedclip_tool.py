"""
BiomedCLIP tool using layer-6 feature extraction (ViT-B/16, 12 blocks).

Key findings (Table III, BiomedCLIP ViT-B/16):
  - Layer 6 (middle, ≈50% depth) dominates consistently across all four tasks
  - Concat fusion of layers 2, 6, 11 gives best overall performance
  - Layers 2/17/22 from the prior CLIP ViT-L/14 analysis do not apply here

Two modes:
  1. Zero-shot: cosine similarity between layer-6 CLS features and text embeddings.
  2. Linear probe: MLP head on layer-6 features or concat fusion of layers 2, 6, 11.
     Set config.biomedclip_probe_checkpoints[task] to a checkpoint from
     18_layer_fusion_benchmark.py to enable probe mode for that task.
"""

from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import (
    BINARY_LABELS,
    CHECKPOINT_SOURCE,
    DEFAULT_CONFIG,
    HF_CHECKPOINT_REPOS,
    TUMOR_MULTICLASS_CLASSES,
    ModelConfig,
    PreprocessConfig,
    download_hf_checkpoint,
    resolve_torch_device,
)

# Layer indices (0-indexed into BiomedCLIP ViT-B/16's 12 transformer blocks).
# Matches 18_layer_fusion_benchmark.py: SHALLOW_LAYER=2, MIDDLE_LAYER=6, DEEP_LAYER=11.
SHALLOW_LAYER = 2   # ≈25% depth
MIDDLE_LAYER  = 6   # ≈50% depth — strongest single layer per paper (Table III)
DEEP_LAYER    = 11  # ≈92% depth

CANDIDATE_LABELS = {
    "binary_tumor": BINARY_LABELS["binary_tumor"],
    "ms": BINARY_LABELS["ms"],
    "stroke": BINARY_LABELS["stroke"],
    "multiclass_tumor": [
        f"brain MRI showing {cls} tumor" if cls != "normal" else "normal brain MRI"
        for cls in TUMOR_MULTICLASS_CLASSES
    ],
}
PROBE_CANDIDATE_LABELS: dict[str, list[str]] = {
    "multiclass_tumor": [
        "brain MRI showing carcinoma",       # [0]
        "brain MRI showing germinoma",       # [1]
        "brain MRI showing glioma",          # [2]
        "brain MRI showing granuloma",       # [3]
        "brain MRI showing medulloblastoma", # [4]
        "brain MRI showing meningioma",      # [5]
        "brain MRI showing neurocytoma",     # [6]
        "normal brain MRI",                  # [7]
        "brain MRI showing papilloma",       # [8]
        "brain MRI showing pituitary tumor", # [9]
        "brain MRI showing schwannoma",      # [10]
        "brain MRI showing tuberculoma",     # [11]
    ],
}


class _FeatureExtractor:
    """Forward hooks on ViT blocks to capture CLS-token features at three depths."""

    def __init__(self, model):
        self.features = {}
        self._hooks = []
        # BiomedCLIP uses timm ViT (model.visual.trunk.blocks, batch-first tensors);
        # standard CLIP uses model.visual.transformer.resblocks (seq-first tensors).
        if hasattr(model.visual, "trunk"):
            blocks = model.visual.trunk.blocks
            batch_first = True
        else:
            blocks = model.visual.transformer.resblocks
            batch_first = False
        for idx in [SHALLOW_LAYER, MIDDLE_LAYER, DEEP_LAYER]:
            self._hooks.append(
                blocks[idx].register_forward_hook(self._make_hook(idx, batch_first))
            )

    def _make_hook(self, idx: int, batch_first: bool):
        def hook(module, input, output):
            # timm: (batch, seq_len, embed) → CLS at [:, 0, :]
            # CLIP: (seq_len, batch, embed) → CLS at [0, :, :]
            self.features[idx] = (output[:, 0, :] if batch_first else output[0]).detach()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


class BiomedCLIPTool:
    """
    Wraps BiomedCLIP for layer-6 feature extraction and classification.

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
        self.device = resolve_torch_device(self.model_cfg.device, caller="BiomedCLIPTool")

        self.clip_device = self.device
        print(f"[BiomedCLIPTool] Loading BiomedCLIP model ({self.clip_device})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_cfg.biomedclip_model_id
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_cfg.biomedclip_model_id)
        self.model = self.model.to(self.clip_device).eval()

        # Per-task probe heads: dict[task] = (head_module, is_concat_fusion)
        self._probe_heads: dict[str, tuple[nn.Module, bool]] = {}
        for task, ckpt in self.model_cfg.biomedclip_probe_checkpoints.items():
            if ckpt is None:
                continue
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                if (
                    CHECKPOINT_SOURCE != "local"
                    and task in HF_CHECKPOINT_REPOS
                    and "biomedclip" in HF_CHECKPOINT_REPOS[task]
                ):
                    try:
                        ckpt_path = download_hf_checkpoint(
                            task, "biomedclip", ckpt_path, caller="BiomedCLIPTool"
                        )
                        ckpt = str(ckpt_path)
                    except Exception as exc:
                        print(
                            f"[BiomedCLIPTool] HF download failed for '{task}' probe "
                            f"({exc}); running zero-shot for this task."
                        )
                        continue
                else:
                    print(
                        f"[BiomedCLIPTool] Probe checkpoint not found: {ckpt}; "
                        "running zero-shot for this task."
                    )
                    continue
            head, is_fusion = self._load_probe_head(ckpt)
            self._probe_heads[task] = (head, is_fusion)
            mode = "concat-fusion" if is_fusion else "single-layer"
            print(f"[BiomedCLIPTool] Loaded {mode} probe for '{task}': {ckpt}")

    def _load_probe_head(self, checkpoint_path: str) -> tuple[nn.Module, bool]:
        """
        Load a probe head from an 18_layer_fusion_benchmark.py checkpoint.

        Supports:
          - MLP concat-fusion head  (in_dim=768*3, CLIP-Fusion-Concat checkpoints)
          - MLP single-layer head   (in_dim=768,   CLIP-Layer* checkpoints)
          - Legacy nn.Linear        (backward compat, various key conventions)
          - Wrapped checkpoints     (model_state_dict / state_dict wrappers)
        """
        raw = torch.load(checkpoint_path, map_location=self.device)

        # Unwrap common training-checkpoint wrappers
        if isinstance(raw, dict):
            state = (
                raw.get("model_state_dict")
                or raw.get("state_dict")
                or raw
            )
        else:
            state = raw

        if "head.1.weight" in state:
            # MLP head: LayerNorm → Linear(in→512) → ReLU → Dropout → Linear(512→out)
            in_dim  = state["head.1.weight"].shape[1]
            out_dim = state["head.4.weight"].shape[0]
            mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, out_dim),
            )
            mlp.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")})
            return mlp.to(self.device).eval(), (in_dim == 768 * 3)

        # Normalise alternative key prefixes to bare "weight" / "bias"
        for prefix in ("", "fc.", "linear.", "classifier.", "0."):
            w_key = f"{prefix}weight"
            b_key = f"{prefix}bias"
            if w_key in state:
                linear_state = {"weight": state[w_key], "bias": state[b_key]}
                in_dim  = linear_state["weight"].shape[1]
                out_dim = linear_state["weight"].shape[0]
                head = nn.Linear(in_dim, out_dim)
                head.load_state_dict(linear_state)
                return head.to(self.device).eval(), (in_dim == 768 * 3)

        raise ValueError(
            f"Unrecognised probe checkpoint format in {checkpoint_path}. "
            f"Top-level keys: {list(state.keys())[:10]}"
        )

    def _extract_layer_features(
        self, image_tensor: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Extract L2-normalised CLS-token features from shallow, middle, and deep layers."""
        extractor = _FeatureExtractor(self.model)
        with torch.no_grad():
            self.model.encode_image(image_tensor)
        features = {
            k: F.normalize(v.clone().float(), dim=-1)
            for k, v in extractor.features.items()
        }
        extractor.remove()
        return features

    @torch.no_grad()
    def classify(self, image_path: str, task: str) -> dict:
        """
        Classify using BiomedCLIP with layer-6 features.

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

        if task in self._probe_heads:
            labels = PROBE_CANDIDATE_LABELS.get(
                task, CANDIDATE_LABELS.get(task, CANDIDATE_LABELS["binary_tumor"])
            )
            return self._linear_probe_classify(image_tensor, labels, task)

        labels = CANDIDATE_LABELS.get(task, CANDIDATE_LABELS["binary_tumor"])
        return self._zero_shot_classify(image_tensor, labels)

    def _zero_shot_classify(
        self, image_tensor: torch.Tensor, labels: list[str]
    ) -> dict:
        """
        Standard CLIP zero-shot: encode_image() into the joint embedding space, then
        cosine similarity against text embeddings using the model's learned logit_scale.

        Previously used visual.head.proj applied to layer-6 intermediate features, which
        produced embeddings outside the joint space and caused label collapse (every image
        predicted as the same class). Layer-6 features are only valid for the linear probe
        path where the probe head is trained on those features directly.
        """
        image_feat = F.normalize(self.model.encode_image(image_tensor).float(), dim=-1)

        tokens = self.tokenizer(labels).to(self.clip_device)
        text_feats = F.normalize(self.model.encode_text(tokens).float(), dim=-1)

        logit_scale = self.model.logit_scale.exp()
        logits = (image_feat @ text_feats.T).squeeze(0) * logit_scale
        scores = torch.softmax(logits, dim=0).cpu().numpy()

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
        """MLP probe on layer-6 features or concat fusion of layers 2, 6, 11.

        If the probe head expects 512-dim input it was trained on encode_image()
        projected embeddings; use those instead of raw layer-6 CLS tokens.
        """
        head, is_fusion = self._probe_heads[task]

        # Determine the head's expected input dimension
        if isinstance(head, nn.Sequential):
            in_dim = head[1].in_features  # index 1 is the first Linear (after LayerNorm)
        else:
            in_dim = head.in_features

        if in_dim == 512:
            # Probe trained on the 512-dim projected CLIP embedding space
            feat = F.normalize(self.model.encode_image(image_tensor).float(), dim=-1)
        elif is_fusion:
            features = self._extract_layer_features(image_tensor)
            feat = torch.cat(
                [features[SHALLOW_LAYER], features[MIDDLE_LAYER], features[DEEP_LAYER]],
                dim=-1,
            )
        else:
            features = self._extract_layer_features(image_tensor)
            feat = features[MIDDLE_LAYER]

        logits = head(feat.to(self.device)).squeeze(0)
        scores = torch.softmax(logits, dim=0).cpu().numpy()

        ranked_idx = np.argsort(scores)[::-1]
        return {
            "ranked_labels": [labels[i] for i in ranked_idx],
            "scores": [float(scores[i]) for i in ranked_idx],
            "top_label": labels[int(ranked_idx[0])],
            "top_score": float(scores[ranked_idx[0]]),
            "mode": "linear_probe",
        }
