"""
Run the SAM3 probe (and optionally the EBRAINS atlas) over BraTS20 slices prepared by
scripts/brats20_prepare.py, and compare SAM3's mask with the ground-truth tumour mask.

For every <case>_slice_<z>_mask.png in --out_dir:
    SAM3   : lesion pixels predicted, Dice against the ground truth
    atlas  : Julich-Brain region of the ground-truth mask centroid (and of SAM3's mask when
             it is non-empty), through ANTs registration of the native NIfTI to MNI152

Usage:
    python scripts/brats20_sam3_atlas_check.py                 # SAM3 only
    python scripts/brats20_sam3_atlas_check.py --atlas         # + siibra (needs antspyx)
Writes --out_dir/sam3_atlas_check.csv.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for config/agents

_MASK_RE = re.compile(r"^(?P<case>BraTS20_\w+_\d{3})_slice_(?P<z>\d+)_mask\.png$")


def find_cases(out_dir: Path, channel: str) -> list[dict]:
    cases = []
    for mask in sorted(out_dir.glob("*_slice_*_mask.png")):
        m = _MASK_RE.match(mask.name)
        if not m:
            continue
        case, z = m["case"], int(m["z"])
        image = out_dir / f"{case}_slice_{z}_{channel}.png"
        nifti = out_dir / f"{case}_native.nii.gz"
        if image.exists():
            cases.append({"case": case, "slice": z, "image": image, "mask": mask,
                          "nifti": nifti if nifti.exists() else None})
    return cases


def load_mask(path: Path, shape: tuple | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if shape is not None and img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(img) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="outputs/brats_test")
    p.add_argument("--channel", default="t1ce")
    p.add_argument("--atlas", action="store_true", help="also run SiibraAtlasTool (auto_register=True)")
    p.add_argument("--registration_type", default="Affine", choices=["Affine", "SyN"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    cases = find_cases(out_dir, args.channel)
    if not cases:
        raise SystemExit(f"no prepared cases in {out_dir}; run scripts/brats20_prepare.py first")

    from config import DEFAULT_CONFIG
    from agents.sam3_tool import SAM3Tool

    sam3 = SAM3Tool(DEFAULT_CONFIG.model, output_dir=str(out_dir / "sam3"))
    atlas = None
    if args.atlas:
        from agents.sibra_tool import SiibraAtlasTool

        atlas = SiibraAtlasTool(auto_register=True, registration_type=args.registration_type)

    def assign(mask: np.ndarray, c: dict) -> str:
        if atlas is None or c["nifti"] is None or mask.sum() == 0:
            return ""
        r = atlas.assign_lesion(mask=mask.astype(np.uint8), nifti_path=str(c["nifti"]),
                                slice_index=c["slice"])
        score = r["assignment_scores"][0]["score"] if r.get("assignment_scores") else float("nan")
        return f"{r['assigned_region']} (MNI {r['mni_coords']}, score {score:.3f})"

    rows = []
    for c in cases:
        gt = load_mask(c["mask"])
        seg = sam3.segment(str(c["image"]), "brain tumor")
        pred = load_mask(Path(seg["mask_path"]), gt.shape) if seg.get("mask_path") else np.zeros_like(gt)
        row = {
            "case": c["case"], "slice": c["slice"],
            "gt_px": int(gt.sum()), "sam3_px": int(pred.sum()), "dice": round(dice(gt, pred), 3),
            "sam3_skipped": bool(seg.get("skipped")),
            "atlas_gt": assign(gt, c), "atlas_sam3": assign(pred, c),
        }
        rows.append(row)
        print(f"{row['case']} z={row['slice']:3d}  GT {row['gt_px']:5d} px  SAM3 {row['sam3_px']:5d} px  "
              f"Dice {row['dice']:.3f}" + (f"\n    GT   → {row['atlas_gt']}\n    SAM3 → {row['atlas_sam3'] or '-'}"
                                          if atlas else ""))

    csv_path = out_dir / "sam3_atlas_check.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} cases → {csv_path}")


if __name__ == "__main__":
    main()
