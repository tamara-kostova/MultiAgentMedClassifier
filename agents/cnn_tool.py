"""
CNN classifier tool.
Wraps the best-performing CNN per task from cnns.tex:
  - binary_tumor    → VGG16     (100.0% acc)
  - multiclass_tumor → DenseNet169 (99.0% acc, F1=0.82)
  - ms              → ResNet101  (59.7% acc)
  - stroke          → DenseNet169 (97.7% acc)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import (
    BEST_CNN_PER_TASK,
    DEFAULT_CONFIG,
    TUMOR_12_CLASSES,
    ModelConfig,
    PreprocessConfig,
)

NUM_CLASSES = {
    "binary_tumor": 2,
    "multiclass_tumor": 12,
    "ms": 2,
    "stroke": 2,
}

CLASS_NAMES = {
    "binary_tumor": ["normal", "tumor"],
    "multiclass_tumor": TUMOR_12_CLASSES,
    "ms": ["normal", "ms"],
    "stroke": ["normal", "stroke"],
}


def _build_model(arch: str, num_classes: int) -> nn.Module:
    """Instantiate an architecture with a replaced classification head."""
    if arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


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
        self.device = torch.device(self.model_cfg.device)
        self._models: dict[str, nn.Module] = {}
        self._transform = _get_transform(self.preprocess_cfg)

    def _load_model(self, task: str) -> nn.Module:
        if task in self._models:
            return self._models[task]

        arch = BEST_CNN_PER_TASK[task]
        n_cls = NUM_CLASSES[task]
        model = _build_model(arch, n_cls)

        checkpoint_path = self.model_cfg.cnn_checkpoints.get(task)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device)
            # Handle both raw state_dict and checkpoint dicts
            state_dict = state.get("model_state_dict", state)
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
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        class_names = CLASS_NAMES[task]

        return {
            "predicted_class": class_names[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "all_probs": {name: float(p) for name, p in zip(class_names, probs)},
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
