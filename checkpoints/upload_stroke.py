"""
Upload the trained stroke checkpoints to Hugging Face Hub.

Prerequisite:
    huggingface-cli login

Examples:
    python checkpoints/upload_stroke.py --repo-id tamara-kostova/multiagentmed-stroke
    python checkpoints/upload_stroke.py --repo-id tamara-kostova/multiagentmed-stroke --private
    python checkpoints/upload_stroke.py --repo-id tamara-kostova/multiagentmed-stroke --only cnn
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STROKE_CHECKPOINTS = {
    "cnn": {
        "local_path": ROOT / "checkpoints/densenet169_CT_stroke_binary_norm_final.pt",
        "repo_path": "stroke/cnn/densenet169_CT_stroke_binary_norm_final.pt",
        "description": "DenseNet169 CNN checkpoint for binary stroke CT classification.",
    },
    "biomedclip": {
        "local_path": ROOT / "checkpoints/linear_probe_BiomedCLIP_CT_stroke_binary_norm_best.pt",
        "repo_path": (
            "stroke/biomedclip/"
            "linear_probe_BiomedCLIP_CT_stroke_binary_norm_best.pt"
        ),
        "description": "BiomedCLIP linear-probe checkpoint for binary stroke CT classification.",
    },
}


def _model_card(repo_id: str, selected: list[str]) -> str:
    rows = "\n".join(
        f"- `{meta['repo_path']}`: {meta['description']}"
        for name, meta in STROKE_CHECKPOINTS.items()
        if name in selected
    )
    return f"""---
    license: mit
    library_name: pytorch
    tags:
    - medical-imaging
    - brain-ct
    - stroke-classification
    - binary-classification
    - pytorch
    ---

    # Ischemic Stroke Binary Classifier

    PyTorch checkpoint artifacts for the MultiAgentMedClassifier stroke task.
    Contains a DenseNet169 CNN checkpoint and a BiomedCLIP linear-probe checkpoint
    for classifying brain CT images as normal or ischemic stroke.

    These are checkpoint files for the accompanying project loaders, not standalone
    Transformers models.

    ## Model Description

    - Task: binary stroke CT classification
    - CNN architecture: DenseNet169
    - Vision-language backbone for probe: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
    - Framework: PyTorch

    ## Classes

    - `normal`
    - `stroke`

    The project-level BiomedCLIP labels are:

    - `normal brain CT`
    - `ischemic stroke brain CT`

    ## Files

    {rows}

    ## Training Details

    - Input size: 224 x 224 RGB
    - Normalization: ImageNet mean/std
    - CNN checkpoint: DenseNet169 fine-tuned for the `stroke` task
    - BiomedCLIP probe: linear/MLP probe over frozen BiomedCLIP image features (layer 6)

    ## Metrics

    | Model | Accuracy |
    |-------|----------|
    | DenseNet169 CNN | 97.7% |

    ## Inference Example

    ```python
    from huggingface_hub import hf_hub_download
    from agents.cnn_tool import CNNClassifier
    from config import DEFAULT_CONFIG

    checkpoint_path = hf_hub_download(
        repo_id="{repo_id}",
        filename="stroke/cnn/densenet169_CT_stroke_binary_norm_final.pt",
    )
    DEFAULT_CONFIG.model.cnn_checkpoints["stroke"] = checkpoint_path
    classifier = CNNClassifier(DEFAULT_CONFIG.model, DEFAULT_CONFIG.preprocess)
    result = classifier.classify("path/to/brain_ct.png", task="stroke")
    print(result)
    ```

    ## Intended Use

    Research and experimentation only. Not a medical device. Always validate on your
    own held-out test set before using in any pipeline.
    """


def upload_stroke_checkpoints(
    repo_id: str,
    selected: list[str],
    private: bool = False,
    revision: str | None = None,
) -> None:
    from huggingface_hub import HfApi, ModelCard

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    missing = [
        str(STROKE_CHECKPOINTS[name]["local_path"])
        for name in selected
        if not STROKE_CHECKPOINTS[name]["local_path"].exists()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    for name in selected:
        meta = STROKE_CHECKPOINTS[name]
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
        description="Upload stroke CNN and BiomedCLIP checkpoints to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination Hub model repo, e.g. tamara-kostova/multiagentmed-stroke.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "cnn", "biomedclip"],
        default="all",
        help="Choose which checkpoint artifact(s) to upload.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(STROKE_CHECKPOINTS) if args.only == "all" else [args.only]
    upload_stroke_checkpoints(
        repo_id=args.repo_id,
        selected=selected,
        private=args.private,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
