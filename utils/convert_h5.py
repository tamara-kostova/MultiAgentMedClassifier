"""
Convert BraTS2020 per-slice h5 files to PNG + NIfTI for pipeline testing.

The BraTS2020 training h5 files have the structure:
    image: (240, 240, 4)  float64  — channels: T1, T1ce, T2, FLAIR
    mask:  (240, 240, 3)  uint8    — channels: NCR/NET, ED, ET (tumour sub-regions)

Usage:
    # Convert volume 100, auto-pick best slice, build NIfTI
    python utils/convert_h5.py --volume 100

    # Specific slice
    python utils/convert_h5.py --volume 100 --slice 80

    # Custom data dir and output dir
    python utils/convert_h5.py --volume 100 --data_dir data/BraTS2020/BraTS2020_training_data/data --out_dir outputs/brats_test

Output files:
    volume_{id}_slice_{n}_t1ce.png   — T1ce channel greyscale PNG (pipeline input)
    volume_{id}_slice_{n}_mask.png   — binary tumour mask PNG (white = tumour)
    volume_{id}.nii.gz               — full 3D T1ce volume in MNI152 space (accurate siibra coords)

Then test the atlas tool:
    python tests/test_atlas_enrichment.py \\
        --image outputs/brats_test/volume_100_slice_80_t1ce.png \\
        --mask  outputs/brats_test/volume_100_slice_80_mask.png \\
        --nifti outputs/brats_test/volume_100.nii.gz
"""

import argparse
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from PIL import Image

# BraTS2020 is pre-registered to MNI152 1mm isotropic space.
# Standard affine maps voxel (i,j,k) → MNI152 mm.
BRATS_MNI152_AFFINE = np.array([
    [-1,  0,  0,  90],
    [ 0, -1,  0, 126],
    [ 0,  0,  1, -72],
    [ 0,  0,  0,   1],
], dtype=np.float32)

CHANNEL_NAMES = ["t1", "t1ce", "t2", "flair"]


def load_slice(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        image = f["image"][()]   # (240, 240, 4) float64
        mask  = f["mask"][()]    # (240, 240, 3) uint8
    return image, mask


def find_slices(data_dir: Path, volume_id: int) -> list[Path]:
    prefix = f"volume_{volume_id}_slice_"
    slices = sorted(
        data_dir.glob(f"{prefix}*.h5"),
        key=lambda p: int(p.stem.split("_slice_")[-1]),
    )
    if not slices:
        raise FileNotFoundError(f"No h5 files found for volume {volume_id} in {data_dir}")
    return slices


def best_slice(slice_paths: list[Path]) -> tuple[int, Path]:
    """Return (slice_idx, path) for the slice with the most tumour voxels."""
    best_idx, best_path, best_count = 0, slice_paths[0], 0
    for path in slice_paths:
        _, mask = load_slice(path)
        count = int((mask > 0).sum())
        if count > best_count:
            best_count = count
            best_path = path
            best_idx = int(path.stem.split("_slice_")[-1])
    return best_idx, best_path


def normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def save_png(arr_2d: np.ndarray, path: Path) -> None:
    Image.fromarray(arr_2d).save(path)


def build_nifti(slice_paths: list[Path], channel: int = 1) -> nib.Nifti1Image:
    """Stack all slices into a 3D volume and return a NIfTI with BraTS MNI152 affine."""
    slices = []
    for p in slice_paths:
        img, _ = load_slice(p)
        slices.append(img[:, :, channel])   # (240, 240)

    volume = np.stack(slices, axis=2).astype(np.float32)  # (240, 240, N_slices)
    return nib.Nifti1Image(volume, affine=BRATS_MNI152_AFFINE)


def convert(volume_id: int, data_dir: Path, out_dir: Path, slice_idx: int | None = None, channel: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_paths = find_slices(data_dir, volume_id)
    print(f"Found {len(slice_paths)} slices for volume {volume_id}")

    if slice_idx is None:
        slice_idx, chosen_path = best_slice(slice_paths)
        print(f"Auto-selected slice {slice_idx} (most tumour voxels)")
    else:
        chosen_path = data_dir / f"volume_{volume_id}_slice_{slice_idx}.h5"
        if not chosen_path.exists():
            raise FileNotFoundError(f"Slice not found: {chosen_path}")

    image, mask = load_slice(chosen_path)
    ch_name = CHANNEL_NAMES[channel]

    # PNG: selected MRI channel
    png_path = out_dir / f"volume_{volume_id}_slice_{slice_idx}_{ch_name}.png"
    save_png(normalise_to_uint8(image[:, :, channel]), png_path)
    print(f"Saved image PNG : {png_path}")

    # Binary mask PNG: union of all 3 tumour sub-regions
    binary_mask = (mask.sum(axis=2) > 0).astype(np.uint8) * 255
    mask_path = out_dir / f"volume_{volume_id}_slice_{slice_idx}_mask.png"
    save_png(binary_mask, mask_path)
    print(f"Saved mask  PNG : {mask_path}")

    # NIfTI: full 3D volume with MNI152 affine
    nifti_path = out_dir / f"volume_{volume_id}.nii.gz"
    nii = build_nifti(slice_paths, channel=channel)
    nib.save(nii, nifti_path)
    print(f"Saved NIfTI     : {nifti_path}")

    print(f"\nRun atlas test with:")
    print(f"  python tests/test_atlas_enrichment.py \\")
    print(f"    --image {png_path} \\")
    print(f"    --mask  {mask_path} \\")
    print(f"    --nifti {nifti_path}")

    return png_path, mask_path, nifti_path


def main():
    parser = argparse.ArgumentParser(description="Convert BraTS2020 h5 slices to PNG + NIfTI")
    parser.add_argument("--volume",   type=int, required=True, help="Volume ID (e.g. 100)")
    parser.add_argument("--slice",    type=int, default=None,  help="Slice index (default: auto, picks slice with most tumour)")
    parser.add_argument("--channel",  type=int, default=1,     help="MRI channel: 0=T1 1=T1ce 2=T2 3=FLAIR (default: 1=T1ce)")
    parser.add_argument("--data_dir", default="data/BraTS2020/BraTS2020_training_data/data")
    parser.add_argument("--out_dir",  default="outputs/brats_test")
    args = parser.parse_args()

    convert(
        volume_id=args.volume,
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        slice_idx=args.slice,
        channel=args.channel,
    )


if __name__ == "__main__":
    main()
