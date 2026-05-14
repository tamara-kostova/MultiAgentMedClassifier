"""
CNN classifier tool.
Wraps the best-performing CNN per task from cnns.tex:
  - binary_tumor    → VGG16     (100.0% acc)
  - multiclass_tumor → DenseNet169 (99.0% acc, F1=0.82)
  - ms              → ResNet101  (59.7% acc)
  - stroke          → DenseNet169 (97.7% acc)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import (
    BEST_CNN_PER_TASK,
    CHECKPOINT_SOURCE,
    DEFAULT_CONFIG,
    HF_CHECKPOINT_REPOS,
    ModelConfig,
    PreprocessConfig,
    download_hf_checkpoint,
    resolve_torch_device,
)

NUM_CLASSES = {
    "binary_tumor": 2,
    "multiclass_tumor": 12,
    "ms": 2,
    "stroke": 2,
}

_CNN_MULTICLASS_CLASSES = [
    "carcinoma", "germinoma", "glioma", "granuloma", "medulloblastoma",
    "meningioma", "neurocytoma", "normal", "papilloma", "pituitary_tumor",
    "schwannoma", "tuberculoma",
]

CLASS_NAMES = {
    "binary_tumor": ["normal", "tumor"],
    "multiclass_tumor": _CNN_MULTICLASS_CLASSES,
    "ms": ["normal", "ms"],
    "stroke": ["normal", "stroke"],
}


def _unwrap_state_dict(state: object) -> dict:
    """Extract a model state dict from common checkpoint wrappers."""
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state).__name__}")
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def _infer_resnet_variant(state_dict: dict) -> str:
    block_ids = set()
    for key in state_dict:
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "layer3" and parts[1].isdigit():
            block_ids.add(int(parts[1]))
    return "resnet101" if block_ids and max(block_ids) >= 22 else "resnet50"


def _infer_arch_from_state_dict(state_dict: dict) -> str | None:
    """Infer the torchvision architecture from checkpoint key patterns."""
    if "features.conv0.weight" in state_dict:
        classifier = state_dict.get("classifier.weight")
        if classifier is not None and getattr(classifier, "ndim", 0) == 2:
            in_features = int(classifier.shape[1])
            if in_features == 1664:
                return "densenet169"
            if in_features == 1024:
                return "densenet121"
        return "densenet169"

    if "classifier.6.weight" in state_dict:
        return "vgg16"

    if "fc.weight" in state_dict:
        return _infer_resnet_variant(state_dict)

    return None


def _build_model(
    arch: str, num_classes: int, use_imagenet_weights: bool = True
) -> nn.Module:
    """Instantiate an architecture with a replaced classification head."""
    if arch == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        model = models.vgg16(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "densenet169":
        weights = (
            models.DenseNet169_Weights.IMAGENET1K_V1
            if use_imagenet_weights
            else None
        )
        model = models.densenet169(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == "densenet121":
        weights = (
            models.DenseNet121_Weights.IMAGENET1K_V1
            if use_imagenet_weights
            else None
        )
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == "resnet101":
        weights = (
            models.ResNet101_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        )
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        )
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def _build_model_with_fallback(
    arch: str, num_classes: int, use_imagenet_weights: bool = True
) -> nn.Module:
    """Build a model, falling back to random init if pretrained weights are unavailable."""
    try:
        return _build_model(arch, num_classes, use_imagenet_weights=use_imagenet_weights)
    except Exception as exc:
        if not use_imagenet_weights:
            raise
        print(
            f"[CNNClassifier] Could not load ImageNet weights for '{arch}' "
            f"({type(exc).__name__}: {exc}). Falling back to random initialization."
        )
        return _build_model(arch, num_classes, use_imagenet_weights=False)


def _get_transform(cfg: PreprocessConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Grayscale(num_output_channels=3),  # replicate to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
        ]
    )


class CNNClassifier:
    """
    Loads the task-appropriate CNN and runs inference.
    If no checkpoint is provided, uses ImageNet pretrained weights (for testing).
    """

    def __init__(
        self,
        model_cfg: ModelConfig = None,
        preprocess_cfg: PreprocessConfig = None,
    ):
        self.model_cfg = model_cfg or DEFAULT_CONFIG.model
        self.preprocess_cfg = preprocess_cfg or DEFAULT_CONFIG.preprocess
        self.device = resolve_torch_device(self.model_cfg.device, caller="CNNClassifier")
        self._models: dict[str, nn.Module] = {}
        self._transform = _get_transform(self.preprocess_cfg)

    def _load_model(self, task: str) -> nn.Module:
        if task in self._models:
            return self._models[task]

        arch = BEST_CNN_PER_TASK[task]
        n_cls = NUM_CLASSES[task]
        checkpoint_path = self.model_cfg.cnn_checkpoints.get(task)
        model = _build_model_with_fallback(
            arch, n_cls, use_imagenet_weights=checkpoint_path is None
        )

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                if (
                    CHECKPOINT_SOURCE != "local"
                    and task in HF_CHECKPOINT_REPOS
                    and "cnn" in HF_CHECKPOINT_REPOS[task]
                ):
                    try:
                        checkpoint_path = download_hf_checkpoint(
                            task, "cnn", checkpoint_path, caller="CNNClassifier"
                        )
                    except Exception as exc:
                        print(
                            f"[CNNClassifier] HF download failed ({exc}); "
                            "falling back to ImageNet pretrained weights."
                        )
                        checkpoint_path = None
                else:
                    checkpoint_path = None
            if checkpoint_path is None or not checkpoint_path.exists():
                model = _build_model_with_fallback(
                    arch, n_cls, use_imagenet_weights=True
                )
                print(
                    f"[CNNClassifier] Checkpoint not found: {checkpoint_path}. "
                    f"Using ImageNet pretrained weights for '{arch}'."
                )
            else:
                state = torch.load(checkpoint_path, map_location=self.device)
                state_dict = _unwrap_state_dict(state)
                inferred_arch = _infer_arch_from_state_dict(state_dict)
                if inferred_arch and inferred_arch != arch:
                    print(
                        f"[CNNClassifier] Checkpoint architecture '{inferred_arch}' "
                        f"overrides configured '{arch}' for task '{task}'."
                    )
                    arch = inferred_arch
                    model = _build_model_with_fallback(
                        arch, n_cls, use_imagenet_weights=False
                    )
                model.load_state_dict(state_dict)
                print(f"[CNNClassifier] Loaded checkpoint: {checkpoint_path}")
        else:
            print(
                f"[CNNClassifier] No checkpoint for '{task}', using ImageNet pretrained weights."
            )

        model = model.to(self.device).eval()
        self._models[task] = model
        return model

    @torch.no_grad()
    def classify(self, image_path: str, task: str) -> dict:
        """
        Args:
            image_path: Path to image file.
            task: One of "binary_tumor", "multiclass_tumor", "ms", "stroke".
        Returns:
            {
                "predicted_class": str,
                "confidence": float,
                "all_probs": {class_name: probability, ...}
            }
        """
        if task not in NUM_CLASSES:
            raise ValueError(
                f"Unknown task '{task}'. Choose from {set(NUM_CLASSES.keys())}"
            )

        model = self._load_model(task)
        image = Image.open(image_path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        logits = model(tensor)
        temperature = float(self.model_cfg.cnn_temperatures.get(task, 1.0))
        calibrated_logits = logits / temperature
        probs = torch.softmax(calibrated_logits, dim=1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        class_names = CLASS_NAMES[task]

        return {
            "predicted_class": class_names[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "all_probs": {name: float(p) for name, p in zip(class_names, probs)},
            "temperature": temperature,
            "task": task,
        }

    def get_model_and_classes(self, task: str):
        """Return the loaded model and class name list for a task.

        Used by the explainability node to run Grad-CAM++ / Integrated Gradients
        without duplicating the model-loading logic.

        Returns:
            (model, class_names) or (None, []) if task is unknown.
        """
        if task not in NUM_CLASSES:
            return None, []
        return self._load_model(task), CLASS_NAMES[task]
