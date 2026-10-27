# agents/dicom_tool.py
"""
DICOM ingestion and preprocessing for the neuroimaging pipeline.

Handles:
  - Single .dcm file (one slice)
  - DICOM series directory (multi-slice → selects representative slice)
  - Passthrough for PNG/JPEG (no-op)

Output:
  - PNG saved to outputs/preprocessed/<stem>_<hash>.png (input to existing pipeline)
  - Metadata dict injected into NeuroimagingState
"""

import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
import hashlib

from pydicom.misc import is_dicom


class DICOMPreprocessor:

    def __init__(self, output_dir: str = "outputs/preprocessed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, input_path: str) -> dict:
        """
        Main entry point. Accepts a .dcm file, a directory of .dcm files,
        or a PNG/JPEG (passthrough).

        Returns:
            {
              "image_path":   str,   # path to PNG ready for CNN/MedGemma
              "nifti_path":   None,  # populated if you add NIfTI conversion (see below)
              "dicom_path":   str | None,
              "dicom_metadata": dict | None
            }
        """
        p = Path(input_path)

        if p.is_dir():
            return self._process_series(p)
        elif self._is_dicom_file(p):
            return self._process_single(p)
        else:
            # Already a PNG/JPEG — passthrough, no metadata
            return {
                "image_path": str(p),
                "nifti_path": None,
                "dicom_path": None,
                "dicom_metadata": None,
            }

    # ── Single slice ──────────────────────────────────────────────────────────

    def _process_single(self, dcm_path: Path) -> dict:
        ds = pydicom.dcmread(str(dcm_path))
        metadata = self._extract_metadata(ds)
        png_path = self._pixel_array_to_png(ds, self._output_stem(dcm_path))
        return {
            "image_path":    str(png_path),
            "nifti_path":    None,
            "dicom_path":    str(dcm_path),
            "dicom_metadata": metadata,
        }

    # ── Series (folder of slices) ─────────────────────────────────────────────

    def _process_series(self, series_dir: Path) -> dict:
        """
        Load a DICOM series, sort slices by InstanceNumber,
        select the middle slice as the representative 2D image,
        and extract metadata from that selected slice.
        """
        dcm_files = sorted(f for f in series_dir.iterdir() if self._is_dicom_file(f))
        if not dcm_files:
            raise ValueError(f"No DICOM files found in {series_dir}")

        # Sort by InstanceNumber (slice position)
        slices = []
        for f in dcm_files:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            instance = int(getattr(ds, "InstanceNumber", 0))
            slices.append((instance, f, ds))
        slices.sort(key=lambda x: x[0])

        # Representative slice: middle of the series
        # For tumour tasks the middle axial slice is usually most informative
        # You could also select by maximum lesion area if a rough threshold is known
        mid_idx  = len(slices) // 2
        _, mid_path, mid_ds = slices[mid_idx]

        metadata = self._extract_metadata(mid_ds)
        metadata["total_slices"]      = len(slices)
        metadata["selected_slice_idx"] = mid_idx

        png_path = self._pixel_array_to_png(mid_ds, self._output_stem(series_dir))

        return {
            "image_path":    str(png_path),
            "nifti_path":    None,           # see _series_to_nifti() below if needed
            "dicom_path":    str(mid_path),
            "dicom_metadata": metadata,
        }

    # ── Pixel array → normalised PNG ─────────────────────────────────────────

    def _pixel_array_to_png(self, ds: "pydicom.Dataset", stem: str) -> Path:
        """
        Convert DICOM pixel array to a normalised 8-bit grayscale PNG.

        Handles:
          - Modality LUT (rescale slope/intercept for CT Hounsfield units)
          - Window/level for soft tissue vs bone CT windows
          - Inversion for some MRI sequences
        """
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply RescaleSlope / RescaleIntercept (standard for CT)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel_array = pixel_array * slope + intercept

        # Apply window/level if present (typical for CT brain window)
        window_center = getattr(ds, "WindowCenter", None)
        window_width  = getattr(ds, "WindowWidth",  None)
        if window_center and window_width:
            wc = float(window_center[0] if hasattr(window_center, "__iter__") else window_center)
            ww = float(window_width[0]  if hasattr(window_width,  "__iter__") else window_width)
            lo = wc - ww / 2
            hi = wc + ww / 2
            pixel_array = np.clip(pixel_array, lo, hi)

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

        # Min-max normalise to 0–255
        lo, hi = pixel_array.min(), pixel_array.max()
        if hi > lo:
            pixel_array = (pixel_array - lo) / (hi - lo) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

        # Convert to RGB (your CNNs and MedGemma expect 3-channel input)
        img = Image.fromarray(pixel_array, mode="L").convert("RGB")

        out_path = self.output_dir / f"{stem}.png"
        img.save(str(out_path))
        return out_path

    @staticmethod
    def _is_dicom_file(path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() in {".dcm", ".dicom", ".ima"}:
            return True
        try:
            return bool(is_dicom(str(path)))
        except Exception:
            return False

    @staticmethod
    def _output_stem(path: Path) -> str:
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        return f"{path.stem}_{digest}" if path.is_file() else f"{path.name}_{digest}"

    # ── Metadata extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_metadata(ds: "pydicom.Dataset") -> dict:
        """
        Pull the clinically and technically relevant tags from the DICOM header.
        These are injected into NeuroimagingState and passed to MedGemma's prompt.
        """
        def safe(tag):
            val = getattr(ds, tag, None)
            if val is None:
                return None
            # pydicom sequences are not JSON-serialisable — convert to str
            return str(val) if not isinstance(val, (int, float, str)) else val

        return {
            # Patient context (anonymised in real deployments)
            "patient_age":          safe("PatientAge"),       # e.g. "068Y"
            "patient_sex":          safe("PatientSex"),       # "M" | "F" | "O"

            # Acquisition parameters — critical for modality routing
            "modality":             safe("Modality"),         # "MR" | "CT"
            "series_description":   safe("SeriesDescription"),# "T2 FLAIR AX"
            "sequence_name":        safe("SequenceVariant"),
            "scanning_sequence":    safe("ScanningSequence"), # "EP" | "GR" | "SE"
            "field_strength":       safe("MagneticFieldStrength"), # 1.5 | 3.0

            # Scanner provenance
            "manufacturer":         safe("Manufacturer"),
            "manufacturer_model":   safe("ManufacturerModelName"),
            "institution":          safe("InstitutionName"),

            # Geometric parameters (needed for siibra coordinate mapping)
            "slice_thickness_mm":   safe("SliceThickness"),
            "pixel_spacing":        safe("PixelSpacing"),     # [row_mm, col_mm]
            "image_orientation":    safe("ImageOrientationPatient"),
            "image_position":       safe("ImagePositionPatient"), # MNI registration anchor

            # Study context
            "study_description":    safe("StudyDescription"),
            "body_part":            safe("BodyPartExamined"),
            "accession_number":     safe("AccessionNumber"),  # links to RIS/PACS
        }
