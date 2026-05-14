"""
Quick test for the atlas enrichment node on a single image.

Usage:
    python tests/test_atlas_enrichment.py --image path/to/scan.png
    python tests/test_atlas_enrichment.py --image scan.png --mask mask.png
    python tests/test_atlas_enrichment.py --image scan.png --dicom scan.dcm
    python tests/test_atlas_enrichment.py --image scan.png --nifti scan.nii.gz

    # For data NOT already in MNI152 space (BraTS/SRI24, raw scanner DICOMs):
    python tests/test_atlas_enrichment.py --image scan.png --nifti scan.nii.gz --register
    python tests/test_atlas_enrichment.py --image scan.png --nifti scan.nii.gz --register --registration_type SyN

Results are saved to outputs/siibra_test/ by default.

If --mask is omitted, a synthetic centre-blob mask is generated from the image
dimensions (simulates a lesion in the left-centre of the brain).

Requires:
    pip install siibra nibabel pydicom antspyx nilearn
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# ── resolve project root so imports work from any cwd ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_synthetic_mask(image_path: str) -> str:
    """
    Generate a binary mask PNG with an ellipse blob in the left-centre of the
    image (rough approximation of a left-hemisphere tumour location).
    Saved to a temp file; caller is responsible for cleanup.
    """
    img = Image.open(image_path).convert("L")
    w, h = img.size

    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 3          # left-centre
    ry, rx = h // 6, w // 8          # ellipse radii

    Y, X = np.ogrid[:h, :w]
    blob = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    mask[blob] = 255

    tmp = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
    Image.fromarray(mask).save(tmp.name)
    return tmp.name


def run(
    image_path: str,
    mask_path: str | None,
    nifti_path: str | None,
    dicom_path: str | None,
    output_path: str | None = None,
    auto_register: bool = False,
    registration_type: str = "Affine",
):
    from agents.sibra_tool import SiibraAtlasTool
    from pipeline.nodes import make_atlas_enrichment_node
    from pipeline.state import initial_state

    # ── Build a minimal fake state ────────────────────────────────────────────
    synthetic_mask = False
    if mask_path is None:
        print("[test] No mask supplied — generating synthetic centre-blob mask.")
        mask_path = make_synthetic_mask(image_path)
        synthetic_mask = True

    state = initial_state(image_path=image_path, task="binary_tumor", metadata={})
    state["segmentation_result"] = {
        "mask_path":         mask_path,
        "bbox":              [0, 0, 224, 224],
        "guided_image_path": image_path,
        "dice_estimate":     0.0,
    }
    if nifti_path:
        state["metadata"]["nifti_path"] = nifti_path
    if dicom_path:
        state["metadata"]["dicom_path"] = dicom_path
    state["routing_path"] = ["triage", "sam3_segment"]

    # ── Run tool directly ─────────────────────────────────────────────────────
    print("\n─── SiibraAtlasTool.assign_lesion ───────────────────────────────────")
    tool = SiibraAtlasTool(
        fetch_features=False,
        auto_register=auto_register,
        registration_type=registration_type,
    )

    mask_arr = np.array(Image.open(mask_path).convert("L")) > 127
    atlas_result = tool.assign_lesion(
        mask=mask_arr.astype(np.uint8),
        nifti_path=nifti_path,
        dicom_path=dicom_path,
    )
    print(json.dumps(atlas_result, indent=2, default=str))

    # ── Run through the node ──────────────────────────────────────────────────
    print("\n─── make_atlas_enrichment_node (node output) ────────────────────────")
    node = make_atlas_enrichment_node(tool)
    node_output = node(state)
    print(json.dumps(node_output, indent=2, default=str))

    # ── Summary ──────────────────────────────────────────────────────────────
    enrichment = node_output.get("atlas_enrichment") or {}
    print("\n─── Summary ─────────────────────────────────────────────────────────")
    print(f"  Image        : {image_path}")
    print(f"  Mask         : {mask_path}{'  (synthetic)' if synthetic_mask else ''}")
    print(f"  MNI coords   : {enrichment.get('mni_coords', 'n/a')}")
    print(f"  Region       : {enrichment.get('assigned_region', 'n/a')}")
    print(f"  Hemisphere   : {enrichment.get('hemisphere', 'n/a')}")
    top = enrichment.get("assignment_scores", [])[:3]
    if top:
        print("  Top regions  :")
        for c in top:
            print(f"    {c['score']:.4f}  {c['region']}")
    print(f"  Routing path : {node_output.get('routing_path')}")

    if output_path:
        record = {
            "image":            image_path,
            "mask":             mask_path,
            "mask_synthetic":   synthetic_mask,
            "nifti":            nifti_path,
            "dicom":            dicom_path,
            "auto_register":    auto_register,
            "registration_type": registration_type,
            "atlas_enrichment": enrichment,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
        print(f"\n  Saved to     : {output_path}")

    if synthetic_mask:
        Path(mask_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Test atlas enrichment node on a single image.")
    parser.add_argument("--image",  required=True, help="Path to input brain scan PNG")
    parser.add_argument("--mask",   default=None,  help="Binary mask PNG (white = lesion). Auto-generated if omitted.")
    parser.add_argument("--nifti",  default=None,  help="NIfTI file for accurate MNI affine")
    parser.add_argument("--dicom",  default=None,  help="DICOM file for scanner-space coords")
    parser.add_argument("--output", default=None,  help="Save result JSON to this path (default: outputs/siibra_test/<image_stem>_result.json)")
    parser.add_argument("--register", action="store_true",
                        help="Register NIfTI to MNI152 via ANTsPy before siibra lookup. "
                             "Required for non-MNI152 data (BraTS/SRI24, raw scanner space).")
    parser.add_argument("--registration_type", default="Affine",
                        choices=["Affine", "SyN"],
                        help="ANTsPy registration type: Affine (~5-15s) or SyN (~90s, more accurate). Default: Affine")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    stem = Path(args.image).stem
    output_path = args.output or f"outputs/siibra_test/{stem}/{stem}_result.json"

    try:
        run(args.image, args.mask, args.nifti, args.dicom, output_path,
            auto_register=args.register, registration_type=args.registration_type)
    except ModuleNotFoundError as e:
        print(f"\nERROR: missing dependency — {e}")
        print("Run:  pip install siibra nibabel pydicom antspyx nilearn")
        sys.exit(1)


if __name__ == "__main__":
    main()
