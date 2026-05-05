"""
Convert matlab .mat files to JPEG images and PNG masks, organized by label.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
import mat73

root = Path("data/1512427")
output_root = Path("data/processed")
output_root.mkdir(parents=True, exist_ok=True)

mat_files = [f for f in root.rglob("*.mat") if f.name != "cvind.mat"]
print(f"Found {len(mat_files)} .mat files")

def load_cjdata(mat_file: Path):
    # Try classic MAT first
    try:
        data = loadmat(mat_file)
        cjdata = data["cjdata"]

        return {
            "label": int(cjdata["label"][0, 0][0, 0]),
            "image": cjdata["image"][0, 0],
            "tumorMask": cjdata["tumorMask"][0, 0] if "tumorMask" in cjdata.dtype.names else None,
        }
    except NotImplementedError:
        # MATLAB v7.3 / HDF5
        data = mat73.loadmat(str(mat_file))
        cjdata = data["cjdata"]

        return {
            "label": int(cjdata["label"]),
            "image": np.array(cjdata["image"]),
            "tumorMask": np.array(cjdata["tumorMask"]) if "tumorMask" in cjdata else None,
        }

for mat_file in mat_files:
    try:
        cj = load_cjdata(mat_file)

        image = cj["image"].astype(np.float32)
        min_val, max_val = image.min(), image.max()

        if max_val > min_val:
            image = 255.0 * (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        image = image.astype(np.uint8)

        label = cj["label"]
        label_dir = output_root / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(image).save(label_dir / f"{mat_file.stem}.jpg", quality=95)

        mask = cj["tumorMask"]
        if mask is not None:
            mask = (mask > 0).astype(np.uint8) * 255
            mask_dir = output_root / f"{label}_mask"
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mask_dir / f"{mat_file.stem}_mask.png")

    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

print("Done.")