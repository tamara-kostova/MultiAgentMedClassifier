"""
Upload the trained binary tumor checkpoints to Hugging Face Hub.

Prerequisite:
    huggingface-cli login

Examples:
    python checkpoints/upload_binary.py --repo-id USER/multiagentmed-binary-tumor
    python checkpoints/upload_binary.py --repo-id USER/multiagentmed-binary-tumor --private
    python checkpoints/upload_binary.py --repo-id USER/multiagentmed-binary-tumor --only cnn
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BINARY_TUMOR_CHECKPOINTS = {
    "cnn": {
        "local_path": ROOT / "checkpoints/vgg16_MRI_tumor_binary_norm_final.pt",
        "repo_path": "binary_tumor/cnn/vgg16_MRI_tumor_binary_norm_final.pt",
        "description": "VGG16 CNN checkpoint for binary brain tumor MRI classification.",
    },
    "biomedclip": {
        "local_path": ROOT
        / "checkpoints/linear_probe_BiomedCLIP_MRI_tumor_binary_norm_best.pt",
        "repo_path": (
            "binary_tumor/biomedclip/"
            "linear_probe_BiomedCLIP_MRI_tumor_binary_norm_best.pt"
        ),
        "description": "BiomedCLIP linear-probe checkpoint for binary brain tumor MRI classification.",
    },
}


def _model_card(repo_id: str, selected: list[str]) -> str:
    rows = "\n".join(
        f"- `{meta['repo_path']}`: {meta['description']}"
        for name, meta in BINARY_TUMOR_CHECKPOINTS.items()
        if name in selected
    )
    return f"""---
license: mit
language:
- en
library_name: pytorch
base_model:
- microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- torchvision/vgg16
datasets:
- Br35H
tags:
- medical-imaging
- brain-mri
- tumor-classification
- binary-classification
- pytorch
---

    # Brain Tumor Binary Classifier

    PyTorch checkpoint artifacts for the MultiAgentMedClassifier binary brain tumor
    MRI task. The repository contains a VGG16 CNN classifier checkpoint and,
    optionally, a BiomedCLIP linear-probe checkpoint for classifying brain MRI
    images as normal or tumor.

    These are checkpoint files for the accompanying project loaders, not standalone
    Transformers models.

    ## Model Description

    - Task: binary brain tumor MRI classification
    - CNN architecture: VGG16
    - Vision-language backbone for probe: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
    - Framework: PyTorch

    ## Classes

    - `normal`
    - `tumor`

    The project-level BiomedCLIP labels are:

    - `normal brain MRI`
    - `brain tumor MRI`

    ## Files

    {rows}

    ## Dataset

    Trained/evaluated for the binary tumor task using brain MRI tumor/normal data.
    The local evaluation script supports the Br35H binary layout:

    - `data/Br35H/yes`: brain tumor MRI
    - `data/Br35H/no`: normal brain MRI

    Update this section if you publish a model trained on a different dataset split
    or source.

    ## Training Details

    - Input size: 224 x 224 RGB
    - Normalization: ImageNet mean/std
    - CNN checkpoint: VGG16 fine-tuned for the `binary_tumor` task
    - BiomedCLIP probe: linear/MLP probe over frozen BiomedCLIP image features

    ## Metrics

    Evaluation is intended for the `binary_tumor` task on brain MRI tumor/normal
    datasets such as the Br35H binary layout described above. Recompute metrics on
    your held-out test set before using this model in a new domain or workflow.

    ## Inference Example

    Download the checkpoint from Hugging Face and point the local project config at
    it:

    ```python
    from huggingface_hub import hf_hub_download

    from agents.cnn_tool import CNNClassifier
    from config import DEFAULT_CONFIG

    checkpoint_path = hf_hub_download(
        repo_id="{repo_id}",
        filename="binary_tumor/cnn/vgg16_MRI_tumor_binary_norm_final.pt",
    )

    DEFAULT_CONFIG.model.cnn_checkpoints["binary_tumor"] = checkpoint_path

    classifier = CNNClassifier(DEFAULT_CONFIG.model, DEFAULT_CONFIG.preprocess)
    result = classifier.classify("path/to/brain_mri.png", task="binary_tumor")
    print(result)
    ```

    For the BiomedCLIP probe:

    ```python
    from huggingface_hub import hf_hub_download

    from agents.biomedclip_tool import BiomedCLIPTool
    from config import DEFAULT_CONFIG

    probe_path = hf_hub_download(
        repo_id="{repo_id}",
        filename=(
            "binary_tumor/biomedclip/"
            "linear_probe_BiomedCLIP_MRI_tumor_binary_norm_best.pt"
        ),
    )

    DEFAULT_CONFIG.model.biomedclip_probe_checkpoints["binary_tumor"] = probe_path

    tool = BiomedCLIPTool(DEFAULT_CONFIG.model, DEFAULT_CONFIG.preprocess)
    result = tool.classify("path/to/brain_mri.png", task="binary_tumor")
    print(result)
    ```

    ## Intended Use

    This model is intended for research and experimentation in automated
    neuroimaging pipelines. It may be useful for prototype triage, benchmarking,
    and comparison against other image classifiers.

    It is not a medical device and should not be used as the sole basis for
    diagnosis, treatment decisions, or patient management.

    ## Loading In This Repository

    Use these files with this repository's local loaders:

    - CNN: `config.ModelConfig.cnn_checkpoints["binary_tumor"]`
    - BiomedCLIP probe: `config.ModelConfig.biomedclip_probe_checkpoints["binary_tumor"]`
    """


def upload_binary_tumor_checkpoints(
    repo_id: str,
    selected: list[str],
    private: bool = False,
    revision: str | None = None,
) -> None:
    from huggingface_hub import HfApi, ModelCard

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    missing = [
        str(BINARY_TUMOR_CHECKPOINTS[name]["local_path"])
        for name in selected
        if not BINARY_TUMOR_CHECKPOINTS[name]["local_path"].exists()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    for name in selected:
        meta = BINARY_TUMOR_CHECKPOINTS[name]
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
    parser = argparse.ArgumentParser(
        description="Upload binary tumor CNN and BiomedCLIP checkpoints to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination Hub model repo, e.g. USER/multiagentmed-binary-tumor.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "cnn", "biomedclip"],
        default="all",
        help="Choose which checkpoint artifact(s) to upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or revision to upload to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = (
        list(BINARY_TUMOR_CHECKPOINTS)
        if args.only == "all"
        else [args.only]
    )
    upload_binary_tumor_checkpoints(
        repo_id=args.repo_id,
        selected=selected,
        private=args.private,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
