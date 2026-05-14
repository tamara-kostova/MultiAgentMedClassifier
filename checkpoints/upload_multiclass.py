"""
Upload the trained multiclass tumor checkpoints to Hugging Face Hub.

Examples:
    python checkpoints/upload_multiclass.py --repo-id tamara-kostova/multiagentmed-multiclass-tumor
    python checkpoints/upload_multiclass.py --repo-id tamara-kostova/multiagentmed-multiclass-tumor --only cnn
    python checkpoints/upload_multiclass.py --repo-id tamara-kostova/multiagentmed-multiclass-tumor --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MULTICLASS_TUMOR_CHECKPOINTS = {
    "cnn": {
        "local_path": ROOT / "checkpoints/densenet169_MRI_tumor_multiclass_norm_final.pt",
        "repo_path": "multiclass_tumor/cnn/densenet169_MRI_tumor_multiclass_norm_final.pt",
        "description": (
            "DenseNet169 CNN checkpoint for multiclass brain tumor MRI classification."
        ),
    },
    "biomedclip": {
        "local_path": ROOT / "checkpoints/linear_probe_BiomedCLIP_MRI_tumor_multiclass_norm_best.pt",
        "repo_path": (
            "multiclass_tumor/biomedclip/"
            "linear_probe_BiomedCLIP_MRI_tumor_multiclass_norm_best.pt"
        ),
        "description": (
            "BiomedCLIP linear-probe checkpoint for multiclass brain tumor MRI classification."
        ),
    },
}


def _model_card(repo_id: str, selected: list[str]) -> str:
    rows = "\n".join(
        f"- `{meta['repo_path']}`: {meta['description']}"
        for name, meta in MULTICLASS_TUMOR_CHECKPOINTS.items()
        if name in selected
    )
    return f"""---
license: mit
language:
- en
library_name: pytorch
base_model:
- microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- torchvision/densenet169
datasets:
- figshare-brain-tumor-dataset
- kaggle-brain-tumor-17-classes
- kaggle-brain-tumor-44-classes
tags:
- medical-imaging
- brain-mri
- tumor-classification
- multiclass-classification
- pytorch
---

    # Brain Tumor Multiclass Classifier

    PyTorch checkpoint artifacts for the MultiAgentMedClassifier multiclass brain tumor
    MRI task. Contains a DenseNet169 CNN classifier and a BiomedCLIP linear-probe
    checkpoint for classifying brain MRI images into 12 categories (11 tumor subtypes
    + normal).

    These are checkpoint files for the accompanying project loaders, not standalone
    Transformers models.

    ## Model Description

    - Task: multiclass brain tumor MRI classification (12 classes)
    - CNN architecture: DenseNet169
    - Vision-language backbone for probe: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
    - Framework: PyTorch

    ## Classes

    | Index | Label |
    |-------|-------|
    | 0 | `glioma` |
    | 1 | `meningioma` |
    | 2 | `pituitary_tumor` |
    | 3 | `carcinoma` |
    | 4 | `germinoma` |
    | 5 | `granuloma` |
    | 6 | `medulloblastoma` |
    | 7 | `neurocytoma` |
    | 8 | `papilloma` |
    | 9 | `schwannoma` |
    | 10 | `tuberculoma` |
    | 11 | `normal` |

    ## Files

    {rows}

    ## Dataset

    Trained on a combination of three publicly available brain MRI datasets:

    - **Brain Tumor MRI Images — 17 Classes** (Kaggle):
    https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes
    - **Brain Tumor MRI Images — 44 Classes** (Kaggle):
    https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c
    - **Brain Tumor Dataset** (Figshare):
    https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

    ## Training Details

    - Input size: 224 x 224 RGB
    - Normalization: ImageNet mean/std
    - CNN checkpoint: DenseNet169 fine-tuned for the `multiclass_tumor` task
    - BiomedCLIP probe: linear/MLP probe over frozen BiomedCLIP image features

    ## Inference Example

    ```python
    from huggingface_hub import hf_hub_download
    from agents.cnn_tool import CNNClassifier
    from config import DEFAULT_CONFIG

    checkpoint_path = hf_hub_download(
        repo_id="{repo_id}",
        filename="multiclass_tumor/cnn/densenet169_MRI_tumor_multiclass_norm_final.pt",
    )
    DEFAULT_CONFIG.model.cnn_checkpoints["multiclass_tumor"] = checkpoint_path
    classifier = CNNClassifier(DEFAULT_CONFIG.model, DEFAULT_CONFIG.preprocess)
    result = classifier.classify("path/to/brain_mri.png", task="multiclass_tumor")
    print(result)
    ```

    ## Intended Use

    Research and experimentation only. Not a medical device. Always validate on your
    own held-out test set before using in any pipeline.
    """


def upload_multiclass_tumor_checkpoints(
    repo_id: str,
    selected: list[str],
    private: bool = False,
    revision: str | None = None,
) -> None:
    from huggingface_hub import HfApi, ModelCard

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    missing = [
        str(MULTICLASS_TUMOR_CHECKPOINTS[name]["local_path"])
        for name in selected
        if not MULTICLASS_TUMOR_CHECKPOINTS[name]["local_path"].exists()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    for name in selected:
        meta = MULTICLASS_TUMOR_CHECKPOINTS[name]
        print(f"Uploading {name}: {meta['local_path']} -> {meta['repo_path']}")
        api.upload_file(
            path_or_fileobj=str(meta["local_path"]),
            path_in_repo=meta["repo_path"],
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )

    card = ModelCard(_model_card(repo_id, selected))
    card.push_to_hub(repo_id, repo_type="model", revision=revision)
    print(f"Done: https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--only", choices=["all", "cnn", "biomedclip"], default="all")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(MULTICLASS_TUMOR_CHECKPOINTS) if args.only == "all" else [args.only]
    upload_multiclass_tumor_checkpoints(
        repo_id=args.repo_id,
        selected=selected,
        private=args.private,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()