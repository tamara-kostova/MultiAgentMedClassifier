"""
Upload the SAM3 linear probe segmentation checkpoint to Hugging Face Hub.

Examples:
    python checkpoints/upload_sam3_to_huggingface.py --repo-id tamara-kostova/multiagentmed-tumor-segmentation
    python checkpoints/upload_sam3_to_huggingface.py --repo-id tamara-kostova/multiagentmed-tumor-segmentation --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAM3_CHECKPOINTS = {
    "sam3_probe": {
        "local_path": ROOT / "checkpoints/sam3_probe.pth",
        "repo_path": "tumor_segmentation/sam3/sam3_linear_probe_tumor_segmentation_best.pt",
        "description": "SAM3 frozen backbone + 1x1 Conv2d linear probe for brain tumor segmentation.",
    },
}


def _model_card(repo_id: str) -> str:
    return f"""---
license: mit
language:
- en
library_name: pytorch
base_model:
- facebook/sam3
datasets:
- BraTS2021
tags:
- medical-imaging
- brain-mri
- tumor-segmentation
- sam3
- linear-probe
- pytorch
---

# Brain Tumor Segmentation — SAM3 Linear Probe

PyTorch checkpoint artifact for the MultiAgentMedClassifier tumor segmentation task.
Contains a linear 1×1 Conv2d probe head trained on top of a **frozen SAM3 backbone**
for pixel-level brain tumor segmentation.

These are checkpoint files for the accompanying project loaders, not standalone
Transformers models.

## Model Description

- Task: brain tumor MRI segmentation (binary mask: tumor / background)
- Architecture: frozen SAM3 image encoder + linear 1×1 Conv2d probe head
- Backbone input resolution: 1008 × 1008
- Probe head: `nn.Conv2d(feature_dim=256, out_channels=2, kernel_size=1)`
- Framework: PyTorch

## Performance

| Method | Dice | IoU | Sensitivity |
|--------|------|-----|-------------|
| Zero-shot SAM3 | 0.189 | 0.124 | 0.397 |
| **Linear probe (frozen encoder)** | **0.836** (pixel) / **0.801** (per-case mean) | — | — |

## Files

- `tumor_segmentation/sam3/sam3_linear_probe_tumor_segmentation_best.pt`:
  SAM3 frozen backbone + 1×1 Conv2d linear probe for brain tumor segmentation.

## Checkpoint Format

The checkpoint is a dict with:
```python
{{
    "model_state_dict": {{"weight": ..., "bias": ...}},  # Conv2d probe weights
    "feature_dim": 256,                                  # SAM3 feature channels
}}
```

Alternate supported key formats: `classifier.weight/bias`, `module.classifier.weight/bias`, or flat `weight/bias`.

## Runtime Requirements

SAM3 has strict runtime prerequisites:
- Python 3.12+
- PyTorch 2.7+
- CUDA GPU with CUDA 12.6+

The probe checkpoint alone can be loaded without SAM3 installed, but inference
requires the full SAM3 backbone.

## Inference Example

```python
from huggingface_hub import hf_hub_download
from agents.sam3_tool import SAM3Tool
from config import DEFAULT_CONFIG

probe_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="tumor_segmentation/sam3/sam3_linear_probe_tumor_segmentation_best.pt",
)

DEFAULT_CONFIG.model.sam3_linear_probe_checkpoint = probe_path

tool = SAM3Tool(DEFAULT_CONFIG.model)
result = tool.segment("path/to/brain_mri.png", text_prompt="brain tumor")

print(result["mask_path"])        # binary segmentation mask
print(result["bbox"])             # [x1, y1, x2, y2]
print(result["guided_image_path"]) # original image with red bbox overlay
```

## Output Format

```python
{{
    "mask_path": "outputs/segmentation/mask_<uid>.png",   # binary mask (0/255)
    "bbox": [x1, y1, x2, y2],                            # bounding box of mask
    "guided_image_path": "outputs/segmentation/guided_<uid>.png",  # bbox overlay for MedGemma
    "skipped": False
}}
```

## Intended Use

Research and experimentation only. Not a medical device. Always validate on your
own held-out test set before using in any pipeline.
"""


def upload_sam3_checkpoint(
    repo_id: str,
    private: bool = False,
    revision: str | None = None,
) -> None:
    from huggingface_hub import HfApi, ModelCard

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    for name, meta in SAM3_CHECKPOINTS.items():
        if not meta["local_path"].exists():
            raise FileNotFoundError(f"Missing checkpoint: {meta['local_path']}")
        print(f"Uploading {name}: {meta['local_path']} -> {meta['repo_path']}")
        api.upload_file(
            path_or_fileobj=str(meta["local_path"]),
            path_in_repo=meta["repo_path"],
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )

    card = ModelCard(_model_card(repo_id))
    card.push_to_hub(repo_id, repo_type="model", revision=revision)
    print(f"Done: https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_sam3_checkpoint(
        repo_id=args.repo_id,
        private=args.private,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()