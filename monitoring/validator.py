"""
Input image validator for MRI/CT scans.

Runs 9 checks before pipeline inference to catch corrupt files,
blank images, and dimension outliers early — without loading any model.
Pure PIL + stdlib implementation; no Great Expectations dependency.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_VALID_MODES = {"L", "RGB", "RGBA", "I", "F"}


@dataclass
class ValidationResult:
    passed: bool
    image_path: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    image_shape: Optional[tuple] = None   # (width, height)
    pixel_min: Optional[float] = None
    pixel_max: Optional[float] = None
    pixel_mean: Optional[float] = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.image_path}"]
        for f in self.failures:
            lines.append(f"  FAIL: {f}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        if self.image_shape:
            lines.append(f"  shape={self.image_shape}  mean={self.pixel_mean:.2f}")
        return "\n".join(lines)


class MRIImageValidator:
    """
    Validates a single MRI/CT image file before pipeline inference.

    Checks performed (in order):
      1. File exists at the given path
      2. File size > 0 bytes (not empty)
      3. File size <= max_file_size_mb (warn only)
      4. PIL can open the file without raising an exception
      5. Image mode is one of: L, RGB, RGBA, I, F
      6. Both dimensions are within [min_dim, max_dim] pixels
      7. Aspect ratio is between 0.5 and 2.0 (warn only — unusual but not fatal)
      8. Pixel values are in a valid range (uint8: [0,255]; float: [0.0,1.0])
      9. Image is not all-black (mean > 1.0)
     10. Image is not all-white / saturated (mean < 254.0)
    """

    def __init__(
        self,
        min_dim: int = 64,
        max_dim: int = 4096,
        max_file_size_mb: float = 100.0,
    ):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def validate(self, image_path: str) -> ValidationResult:
        """
        Run all checks. Returns a ValidationResult — never raises.
        """
        result = ValidationResult(passed=True, image_path=image_path)
        path = Path(image_path)

        # 1. File exists
        if not path.exists():
            result.failures.append(f"File not found: {image_path}")
            result.passed = False
            return result

        # 2. Non-empty
        size = path.stat().st_size
        if size == 0:
            result.failures.append("File is empty (0 bytes)")
            result.passed = False
            return result

        # 3. File size warning
        if size > self.max_file_size_bytes:
            result.warnings.append(
                f"File size {size / 1e6:.1f} MB exceeds {self.max_file_size_bytes / 1e6:.0f} MB"
            )

        # 4. PIL loadable
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            result.failures.append(f"Cannot open image: {e}")
            result.passed = False
            return result

        result.image_shape = img.size  # (width, height)

        # 5. Valid mode
        if img.mode not in _VALID_MODES:
            result.failures.append(
                f"Unsupported image mode '{img.mode}' (expected one of {sorted(_VALID_MODES)})"
            )
            result.passed = False

        # 6. Dimension range
        w, h = img.size
        if w < self.min_dim or h < self.min_dim:
            result.failures.append(
                f"Image too small: {w}x{h} px (minimum {self.min_dim}px per side)"
            )
            result.passed = False
        if w > self.max_dim or h > self.max_dim:
            result.failures.append(
                f"Image too large: {w}x{h} px (maximum {self.max_dim}px per side)"
            )
            result.passed = False

        # 7. Aspect ratio (warn only)
        if h > 0:
            ratio = w / h
            if ratio < 0.5 or ratio > 2.0:
                result.warnings.append(
                    f"Unusual aspect ratio {ratio:.2f} (width/height). "
                    "Expected between 0.5 and 2.0 for brain MRI/CT."
                )

        # 8–10. Pixel stats
        try:
            arr = np.array(img.convert("L"), dtype=np.float32)
            pmin = float(arr.min())
            pmax = float(arr.max())
            pmean = float(arr.mean())
            result.pixel_min = pmin
            result.pixel_max = pmax
            result.pixel_mean = pmean

            # 8. Value range
            if pmin < 0 or pmax > 255:
                result.warnings.append(
                    f"Pixel values outside [0, 255] range: min={pmin:.1f}, max={pmax:.1f}"
                )

            # 9. Not all black
            if pmean <= 1.0:
                result.failures.append(
                    f"Image appears all-black (mean pixel = {pmean:.3f}). "
                    "Possible blank export or corrupt file."
                )
                result.passed = False

            # 10. Not all white / saturated
            if pmean >= 254.0:
                result.failures.append(
                    f"Image appears all-white / saturated (mean pixel = {pmean:.3f}). "
                    "Possible corrupt export."
                )
                result.passed = False

        except Exception as e:
            result.warnings.append(f"Could not compute pixel statistics: {e}")

        return result

    def validate_batch(self, image_paths: list[str]) -> list[ValidationResult]:
        """Validate a list of images. Returns one ValidationResult per image."""
        return [self.validate(p) for p in image_paths]
