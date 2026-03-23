"""
Lightweight saliency methods for CNN classifiers.

Imported by the pipeline (pipeline/nodes.py) — intentionally kept minimal so
the pipeline does not drag in matplotlib, seaborn, sklearn, tqdm, or the heavy
standalone experiment scaffolding in explainability/cnns.py.

Classes:
    GradCAMForCNN         — standard Grad-CAM (Selvaraju et al., ICCV 2017)
    GradCAMPlusPlus       — per-pixel gradient weighting (Chattopadhyay et al., WACV 2018)
    IntegratedGradientsForCNN — path-integral attribution (Sundararajan et al., ICML 2017)

GRADCAM_TARGET_LAYERS — registry mapping architecture name → target layer accessor.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Registry: architecture name → callable(model) → target conv layer
GRADCAM_TARGET_LAYERS = {
    "resnet50": lambda m: m.layer4[-1],
    "resnet101": lambda m: m.layer4[-1],
    "vgg16": lambda m: m.features[-1],
    "densenet121": lambda m: m.features.denseblock4,
    "densenet169": lambda m: m.features.denseblock4,
    "mobilenet_v2": lambda m: m.features[-1],
    "efficientnet_b0": lambda m: m.features[-1],
    "efficientnet_b4": lambda m: m.features[-1],
}


class GradCAMForCNN:
    """Standard Grad-CAM for convolutional neural networks.

    Targets a specific convolutional layer per architecture (see
    GRADCAM_TARGET_LAYERS registry). Works with 4-D [B, C, H, W]
    activations — NOT the ViT variant.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """Return a [H, W] heatmap in [0, 1] for *target_class*."""
        self.model.eval()
        output = self.model(image)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


class GradCAMPlusPlus:
    """Grad-CAM++ for convolutional neural networks.

    Addresses the main criticism of Grad-CAM: uniform channel weights.
    Uses per-pixel α weights derived from 2nd- and 3rd-order gradient terms,
    producing sharper, more localised saliency maps.

    Reference: Chattopadhyay et al., WACV 2018.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """Return a [H, W] heatmap in [0, 1] using Grad-CAM++ pixel-wise weights."""
        self.model.eval()
        output = self.model(image)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        grads = self.gradients  # [1, C, h, w]
        acts = self.activations  # [1, C, h, w]

        grads_sq = grads**2
        grads_cu = grads**3
        denom = 2.0 * grads_sq + acts * grads_cu.sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        alpha = grads_sq / denom

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


class IntegratedGradientsForCNN:
    """Integrated Gradients attribution for CNN classifiers.

    Model-agnostic — no target layer needed. Satisfies the Completeness axiom:
    the sum of attributions equals the output difference between input and baseline.
    Baseline: black image (zeros).

    Reference: Sundararajan et al., ICML 2017.
    """

    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps

    def attribute(
        self,
        image: torch.Tensor,  # [1, C, H, W]
        target_class: int,
        baseline: torch.Tensor = None,  # defaults to zeros
    ) -> np.ndarray:
        """Return a [H, W] attribution map in [0, 1], abs-summed over channels."""
        if baseline is None:
            baseline = torch.zeros_like(image)

        self.model.eval()
        alphas = torch.linspace(0, 1, self.steps, device=image.device)
        integrated_grads = torch.zeros_like(image)

        for alpha in alphas:
            interp = (baseline + alpha * (image - baseline)).requires_grad_(True)
            output = self.model(interp)
            self.model.zero_grad()
            output[0, target_class].backward()
            integrated_grads += interp.grad.detach()

        avg_grads = integrated_grads / self.steps
        attribution = avg_grads * (image - baseline)  # [1, C, H, W]

        attr_map = attribution.squeeze(0).abs().sum(dim=0).cpu().numpy()
        attr_map = (attr_map - attr_map.min()) / (
            attr_map.max() - attr_map.min() + 1e-8
        )
        return attr_map
