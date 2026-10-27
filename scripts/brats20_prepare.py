"""
Prepare Kaggle BraTS20 cases (NIfTI) for the atlas / SAM3 tests.

The Kaggle mirror awsaf49/brats20-dataset-training-validation holds the original
BraTS 2020 NIfTIs (240x240x155, 1 mm, SRI24 space), one folder per case:
    BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_<id>/
        BraTS20_Training_<id>_{t1,t1ce,t2,flair,seg}.nii
(utils/convert_h5.py is for the per-slice .h5 variant of the same data.)

For each case this writes, into --out_dir:
    BraTS20_Training_<id>_slice_<z>_<ch>.png   axial slice with the most tumour voxels
    BraTS20_Training_<id>_slice_<z>_mask.png   ground-truth tumour mask (seg > 0), white = tumour
    BraTS20_Training_<id>_native.nii.gz        the channel volume with the file's own affine
and prints the atlas test command. PNG orientation is row = j, col = i (the convention
SiibraAtlasTool._pixel_to_mni assumes: voxel = [col, row, z]).

Usage:
    # download the two files per case with kagglehub (needs ~/.kaggle/access_token or
    # KAGGLE_API_TOKEN, and the dataset licence accepted on kaggle.com once)
    python scripts/brats20_prepare.py --cases 1 2 3 --download

    # already-downloaded cases (a directory holding BraTS20_Training_<id>/ folders)
    python scripts/brats20_prepare.py --cases 1 --data_dir /path/to/MICCAI_BraTS2020_TrainingData

Then:
    python tests/test_atlas_enrichment.py --image outputs/brats_test/BraTS20_Training_001_slice_67_t1ce.png \
        --mask outputs/brats_test/BraTS20_Training_001_slice_67_mask.png \
        --nifti outputs/brats_test/BraTS20_Training_001_native.nii.gz --slice 67 --register
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

KAGGLE_DATASET = "awsaf49/brats20-dataset-training-validation"
KAGGLE_PREFIX = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
CHANNELS = ("t1", "t1ce", "t2", "flair")


def case_name(case_id: int) -> str:
    return f"BraTS20_Training_{case_id:03d}"


def download_case(case_id: int, channel: str) -> Path:
    """Fetch <case>_<channel>.nii and <case>_seg.nii; returns the case directory."""
    import kagglehub

    name = case_name(case_id)
    paths = []
    for suffix in (channel, "seg"):
        rel = f"{KAGGLE_PREFIX}/{name}/{name}_{suffix}.nii"
        paths.append(Path(kagglehub.dataset_download(KAGGLE_DATASET, path=rel)))
    return paths[0].parent


def best_slice(seg: np.ndarray) -> tuple[int, np.ndarray]:
    """Axial index with the most tumour voxels and the per-slice counts."""
    counts = (seg > 0).reshape(-1, seg.shape[2]).sum(axis=0)
    return int(counts.argmax()), counts


def to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    fg = slice_2d[slice_2d > 0]
    lo, hi = (np.percentile(fg, (1, 99)) if fg.size else (0.0, 1.0))
    if hi <= lo:
        return np.zeros(slice_2d.shape, dtype=np.uint8)
    return (np.clip((slice_2d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def prepare_case(case_dir: Path, out_dir: Path, channel: str) -> dict:
    name = case_dir.name
    img = nib.load(str(case_dir / f"{name}_{channel}.nii"))
    seg = nib.load(str(case_dir / f"{name}_seg.nii"))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    lab = np.asarray(seg.dataobj)
    if vol.shape != lab.shape:
        raise ValueError(f"{name}: image {vol.shape} and seg {lab.shape} differ")

    z, counts = best_slice(lab)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}_slice_{z}_{channel}.png"
    mask_path = out_dir / f"{name}_slice_{z}_mask.png"
    nifti_path = out_dir / f"{name}_native.nii.gz"

    Image.fromarray(to_uint8(vol[:, :, z].T)).save(png_path)
    Image.fromarray(((lab[:, :, z].T > 0) * 255).astype(np.uint8)).save(mask_path)
    nib.save(nib.Nifti1Image(vol, img.affine), str(nifti_path))

    print(f"{name}: shape {vol.shape}, tumour voxels {(lab > 0).sum()}, "
          f"slice z={z} ({counts[z]} px; middle slice would be {vol.shape[2] // 2})")
    print(f"  {png_path}\n  {mask_path}\n  {nifti_path}")
    return {"case": name, "slice": z, "image": str(png_path), "mask": str(mask_path),
            "nifti": str(nifti_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=int, nargs="+", required=True, help="case ids, e.g. 1 2 100")
    p.add_argument("--download", action="store_true", help="fetch the files from Kaggle with kagglehub")
    p.add_argument("--data_dir", default=None,
                   help="directory holding BraTS20_Training_<id>/ folders (when not downloading)")
    p.add_argument("--channel", default="t1ce", choices=CHANNELS)
    p.add_argument("--out_dir", default="outputs/brats_test")
    args = p.parse_args()
    if not args.download and not args.data_dir:
        p.error("give --download or --data_dir")

    out_dir = Path(args.out_dir)
    for case_id in args.cases:
        case_dir = download_case(case_id, args.channel) if args.download \
            else Path(args.data_dir) / case_name(case_id)
        info = prepare_case(case_dir, out_dir, args.channel)
        print(f"  python tests/test_atlas_enrichment.py --image {info['image']} --mask {info['mask']} "
              f"--nifti {info['nifti']} --slice {info['slice']} --register\n")


if __name__ == "__main__":
    main()
