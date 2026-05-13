"""
SAM3 segmentation tool.

Runtime prerequisites from upstream SAM3:
  - Python 3.12+
  - PyTorch 2.7+
  - CUDA GPU with CUDA 12.6+

The main pipeline can run in a Python 3.9 / macOS MPS environment, but SAM3
cannot be installed there cleanly. In that case this tool returns a skipped
segmentation result and the rest of the pipeline continues without SAM3.

- Zero-shot SAM3: Dice=0.189, IoU=0.124, Sensitivity=0.397 (insufficient)
- Linear probe on frozen SAM3 encoder: Dice=0.836 pixel-level, 0.801 per-case mean
- SAM3→MedGemma pipeline: tumor detection 85.1%→96.3% but specificity 67.1%→41.3%

This tool uses SAM3's frozen backbone as a feature extractor with a linear
1×1 conv probe head for segmentation. It also returns a bounding-box overlay
image for use by the MedGemma report agent (CNN always receives the original image).
"""

import uuid
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from config import CHECKPOINT_SOURCE, DEFAULT_CONFIG, HF_CHECKPOINT_REPOS, ModelConfig, download_hf_checkpoint, resolve_torch_device

# ── Try to import SAM3 ────────────────────────────────────────────────────────
_SAM_IMPORT_ERROR = None
_SAM3_REPO = Path(__file__).resolve().parents[1] / "sam3"
if _SAM3_REPO.exists():
    # Support a vendored checkout at ./sam3 without requiring editable install.
    sys.path.insert(0, str(_SAM3_REPO))


def _resolve_sam3_builder() -> Callable | None:
    try:
        from sam3.model_builder import build_sam3_image_model

        return build_sam3_image_model
    except Exception as exc:
        global _SAM_IMPORT_ERROR
        _SAM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None


build_sam3_image_model = _resolve_sam3_builder()
_SAM_AVAILABLE = build_sam3_image_model is not None

# SAM3 backbone input resolution
_SAM3_INPUT_SIZE = 1008

# Feature key candidates returned by SAM3 backbone
_FEATURE_KEYS = ["vision_features", "image_features", "features"]


def _patch_sam3_position_encoding_precompute() -> None:
    """
    Avoid SAM3's eager CUDA-only position-encoding precompute without editing
    the vendored `sam3/` sources.
    """
    try:
        import sam3.model_builder as sam3_model_builder
    except Exception:
        return

    if getattr(sam3_model_builder, "_codex_precompute_patch", False):
        return

    original_create_position_encoding = sam3_model_builder._create_position_encoding

    def patched_create_position_encoding(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["precompute_resolution"] = None
        return original_create_position_encoding(*args, **kwargs)

    sam3_model_builder._create_position_encoding = patched_create_position_encoding
    sam3_model_builder._codex_precompute_patch = True


def _extract_features(backbone_output) -> torch.Tensor:
    """Pull the spatial feature tensor out of the SAM3 backbone output dict."""
    if isinstance(backbone_output, torch.Tensor):
        return backbone_output
    for key in _FEATURE_KEYS:
        if key in backbone_output:
            return backbone_output[key]
    raise KeyError(
        f"Cannot find feature tensor in SAM3 backbone output. "
        f"Keys found: {list(backbone_output.keys())}"
    )


def _normalize_probe_state_dict(ckpt: dict) -> dict:
    """Handle the various checkpoint formats"""
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format for linear probe")

    if "weight" in state and "bias" in state:
        return {"weight": state["weight"], "bias": state["bias"]}
    if "classifier.weight" in state:
        return {
            "weight": state["classifier.weight"],
            "bias": state["classifier.bias"],
        }
    if "module.classifier.weight" in state:
        return {
            "weight": state["module.classifier.weight"],
            "bias": state["module.classifier.bias"],
        }
    raise ValueError(f"Unsupported probe checkpoint keys: {list(state.keys())[:8]}")


class SAM3Tool:
    """
    Wraps SAM3 as a segmentation tool using a frozen backbone + linear probe head.

    The backbone is called with a text prompt ("brain tumor" / whatever
    suspected_pathology MedGemma provides) so features are text-conditioned.

    If SAM3 is not installed or no checkpoint is configured, returns a null
    result so the pipeline continues without segmentation.
    """

    def __init__(
        self,
        model_cfg: ModelConfig = None,
        output_dir: str = "outputs/segmentation",
    ):
        self.model_cfg = model_cfg or DEFAULT_CONFIG.model
        self.device = resolve_torch_device(self.model_cfg.device, caller="SAM3Tool")
        self.sam3_dtype = (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sam3 = None
        self.probe_head = None
        self._load_models()

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        """Return the dtype of the first floating-point parameter/buffer."""
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_floating_point():
                return tensor.dtype
        return torch.float32

    def _load_models(self):
        if not _SAM_AVAILABLE:
            print(
                "[SAM3Tool] Unable to import SAM3. "
                f"Reason: {_SAM_IMPORT_ERROR}. "
                "Install SAM3 deps and/or run: pip install -e ./sam3"
            )
            return

        probe_ckpt = self.model_cfg.sam3_linear_probe_checkpoint
        bpe_path = self.model_cfg.sam3_bpe_path

        if probe_ckpt is None:
            print(
                "[SAM3Tool] No linear probe checkpoint configured — segmentation skipped."
            )
            return
        if bpe_path is None:
            print(
                "[SAM3Tool] No BPE path configured (sam3_bpe_path) — segmentation skipped."
            )
            return
        
        probe_ckpt_path = Path(probe_ckpt)

        if not probe_ckpt_path.exists():
            if (
                CHECKPOINT_SOURCE != "local"
                and "tumor_segmentation" in HF_CHECKPOINT_REPOS
                and "sam3" in HF_CHECKPOINT_REPOS["tumor_segmentation"]
            ):
                try:
                    probe_ckpt_path = download_hf_checkpoint(
                        "tumor_segmentation", "sam3", probe_ckpt_path, caller="SAM3Tool"
                    )
                    probe_ckpt = str(probe_ckpt_path)
                except Exception as exc:
                    print(
                        f"[SAM3Tool] HF download failed for SAM3 probe "
                        f"({exc}); segmentation skipped."
                    )
                    return
            else:
                print(
                    f"[SAM3Tool] Probe checkpoint not found: {probe_ckpt}; "
                    "segmentation skipped."
                )
                return

        print(f"[SAM3Tool] Loading SAM3 backbone (bpe={bpe_path})")
        _patch_sam3_position_encoding_precompute()
        self.sam3 = build_sam3_image_model(bpe_path=bpe_path, device=str(self.device))
        self.sam3 = self.sam3.to(dtype=self.sam3_dtype)
        self.sam3.eval()
        for param in self.sam3.parameters():
            param.requires_grad = False

        print(f"[SAM3Tool] Loading linear probe: {probe_ckpt}")
        ckpt = torch.load(probe_ckpt, map_location=self.device)
        feature_dim = ckpt.get("feature_dim", 256) if isinstance(ckpt, dict) else 256
        self.probe_head = nn.Conv2d(feature_dim, 2, kernel_size=1)
        self.probe_head.load_state_dict(_normalize_probe_state_dict(ckpt))
        self.probe_head.to(self.device)
        self.probe_head = self.probe_head.float().eval()
        print("[SAM3Tool] Ready.")

    @torch.no_grad()
    def segment(self, image_path: str, text_prompt: str = "brain tumor") -> dict:
        """
        Run SAM3 segmentation on a single image.

        Returns:
            {
                "mask_path": str,
                "bbox": [x1, y1, x2, y2],
                "guided_image_path": str,   # original image with red bbox overlay (for MedGemma)

                "skipped": bool
            }
        """
        if self.sam3 is None or self.probe_head is None:
            return self._null_result(image_path)

        pil_image = Image.open(image_path).convert("RGB")
        if max(pil_image.size) > 512:
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)

        image = np.array(pil_image)
        orig_h, orig_w = image.shape[:2]

        # ── Preprocess: [0,1] float, CHW tensor, resize to SAM3 input size ────
        img_f = image.astype(np.float32) / 255.0
        img_tensor = (
            torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )  # (1, 3, H, W)
        img_tensor = F.interpolate(
            img_tensor,
            size=(_SAM3_INPUT_SIZE, _SAM3_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        img_tensor = img_tensor.to(dtype=self.sam3_dtype)

        # ── Feature extraction (text-conditioned) ─────────────────────────────
        autocast_ctx = (
            torch.amp.autocast(device_type=self.device.type, enabled=False)
            if self.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_ctx:
            backbone_output = self.sam3.backbone(img_tensor, [text_prompt])
        features = _extract_features(backbone_output)  # (1, C, h, w)
        features = features.to(dtype=self._module_dtype(self.probe_head))

        # ── Linear probe → logits → upsample → mask ───────────────────────────
        logits = self.probe_head(features)  # (1, 2, h, w)
        logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        return self._save_results(image, mask, image_path)

    def _save_results(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_path: str,
    ) -> dict:
        """Save binary mask and bbox overlay image; compute bounding box."""
        uid = uuid.uuid4().hex[:8]

        # Save binary mask
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_path = str(self.output_dir / f"mask_{uid}.png")
        mask_pil.save(mask_path)

        # Bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = [int(cmin), int(rmin), int(cmax), int(rmax)]
        else:
            bbox = [0, 0, image.shape[1], image.shape[0]]

        # Bbox overlay for MedGemma
        overlay = Image.fromarray(image)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(bbox, outline="red", width=3)
        guided_path = str(self.output_dir / f"guided_{uid}.png")
        overlay.save(guided_path)

        return {
            "mask_path": mask_path,
            "bbox": bbox,
            "guided_image_path": guided_path,

            "skipped": False,
        }

    def _null_result(self, image_path: str) -> dict:
        """Return a no-op result when SAM3 is unavailable."""
        return {
            "mask_path": None,
            "bbox": None,
            "guided_image_path": image_path,
            "skipped": True,
        }
