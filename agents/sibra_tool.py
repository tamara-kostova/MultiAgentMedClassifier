# agents/siibra_tool.py
"""
siibra-python integration for anatomical region assignment.

Requires:
    pip install siibra
    EBRAINS account for Knowledge Graph queries (optional — atlas queries work without auth,
    KG feature queries require a token: https://ebrains.eu/register)

siibra queries the EBRAINS Human Brain Atlas (Julich-Brain parcellation by default)
to assign a lesion centroid to a named brain region and retrieve multimodal features.
"""

import tempfile

import siibra
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional


class SiibraAtlasTool:
    """
    Wraps siibra-python for anatomical assignment from lesion centroids.

    Usage in the pipeline:
        1. SAM3 outputs a binary mask and bbox in pixel space
        2. NIfTI affine converts pixel centroid → MNI152 mm coordinates
        3. siibra assigns those coordinates to a named parcellation region
        4. Optionally fetches receptor density / connectivity features for that region

    siibra 1.x API note:
        Assignment requires a Map object (siibra.get_map), not parcellation.assign().
        Julich Brain uses STATISTICAL (probability) maps — LABELLED maps are not available
        for the top-level parcellation and will raise NoMapAvailableError.
    """

    def __init__(
        self,
        parcellation: str = "julich 2.9",        # v2.9 has full STATISTICAL MNI152 coverage
        space: str = "mni152",
        fetch_features: bool = False,             # KG feature queries — requires EBRAINS token
        auto_register: bool = False,              # register NIfTI → MNI152 before siibra lookup
        registration_type: str = "Affine",        # "Affine" (fast, ~5-15s) or "SyN" (accurate, ~90s)
    ):
        self.parcellation = siibra.parcellations[parcellation]
        self.space = siibra.spaces[space]
        self.fetch_features = fetch_features
        self.auto_register = auto_register
        self.registration_type = registration_type
        # siibra 1.x: assignment is done on a Map, not the parcellation directly
        self._pmap = siibra.get_map(
            parcellation=self.parcellation,
            space=self.space,
            maptype=siibra.MapType.STATISTICAL,
        )
        # Lazily populated by _get_mni152_template()
        self._mni152_template_path: Optional[str] = None
        # Cache: nifti_path → invtransforms list (avoids re-registering same scan)
        self._registration_cache: dict[str, list[str]] = {}
        print(f"[SiibraAtlasTool] parcellation={self.parcellation.name}, map={self._pmap}, auto_register={auto_register}")

    # ── Main entry point ──────────────────────────────────────────────────────

    def assign_lesion(
        self,
        mask: np.ndarray,                   # binary mask from SAM3, shape (H, W)
        nifti_path: Optional[str] = None,   # NIfTI for affine (most accurate)
        dicom_path: Optional[str] = None,   # DICOM for scanner-space coords
        voxel_size_mm: float = 1.0,
    ) -> dict:
        """
        Assign a SAM3 lesion mask centroid to a brain atlas region.

        Args:
            mask:         binary 2D mask (from SAM3Tool output)
            nifti_path:   path to the NIfTI scan the mask was derived from
            voxel_size_mm: fallback voxel size if no NIfTI affine is available

        Returns dict with:
            mni_coords:          [x, y, z] in MNI152 mm
            assigned_region:     name of the Julich-Brain region
            region_description:  brief text description (if available)
            hemisphere:          "left" | "right" | "bilateral"
            features:            dict of multimodal features (if fetch_features=True)
            assignment_scores:   probability scores per candidate region
        """
        # Step 1: pixel centroid from SAM3 mask
        pixel_centroid = self._mask_centroid(mask)

        # Step 2: pixel → MNI mm via affine (NIfTI > DICOM > pixel fallback)
        mni_coords = self._pixel_to_mni(
            pixel_centroid, nifti_path, voxel_size_mm,
            image_size=mask.shape[:2],
            dicom_path=dicom_path,
        )

        # Step 3: siibra anatomical assignment via Map (siibra 1.x API)
        # Map.assign() returns a DataFrame with columns:
        #   'input structure', 'centroid', 'fragment', 'map value', 'region'
        point = siibra.Point(tuple(float(v) for v in mni_coords), space=self.space)
        try:
            assignments = self._pmap.assign(point)
        except (IndexError, ValueError) as e:
            # Coordinates outside atlas bounds — most often caused by a NIfTI that is
            # in native/SRI24 scanner space rather than MNI152. Set auto_register=True
            # to run ANTsPy registration before siibra lookup.
            print(f"[SiibraAtlasTool] coords {[round(float(v),1) for v in mni_coords]} "
                  f"outside atlas bounds ({e}). Use auto_register=True for non-MNI152 data.")
            return {
                "mni_coords":        [round(float(v), 2) for v in mni_coords],
                "assigned_region":   "unassigned",
                "hemisphere":        "unknown",
                "assignment_scores": [],
                "error":             "coordinates_out_of_bounds",
            }

        empty = assignments is None or (hasattr(assignments, "empty") and assignments.empty)
        if empty:
            return {
                "mni_coords":        [round(float(v), 2) for v in mni_coords],
                "assigned_region":   "unassigned",
                "hemisphere":        "unknown",
                "assignment_scores": [],
            }

        # Sort by probability score descending
        top5 = assignments.nlargest(5, "map value")
        best  = top5.iloc[0]
        region_name = str(best["region"])

        result = {
            "mni_coords":       [round(float(v), 2) for v in mni_coords],
            "assigned_region":  region_name,
            "hemisphere":       self._hemisphere(region_name),
            "assignment_scores": [
                {
                    "region": str(row["region"]),
                    "score":  round(float(row["map value"]), 4),
                }
                for _, row in top5.iterrows()
            ],
        }

        # Step 4: optional KG feature queries
        if self.fetch_features:
            result["features"] = self._fetch_regional_features(best["region"])

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
        """Compute (row, col) centroid of a binary mask."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            h, w = mask.shape
            return (h / 2, w / 2)
        return tuple(coords.mean(axis=0))

    def _pixel_to_mni(
        self,
        pixel_centroid: tuple,
        nifti_path: Optional[str],
        voxel_size_mm: float,
        image_size: tuple = (224, 224),  # (H, W) of the mask
        dicom_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convert 2D pixel centroid to 3D MNI mm coordinates.

        Priority:
          1. NIfTI affine → native-space coords; if auto_register=True, ANTsPy
             then maps native → MNI152 via a cached registration transform.
          2. DICOM ImagePositionPatient + PixelSpacing (scanner space, ≈MNI for
             pre-registered datasets such as BraTS)
          3. Normalised pixel fallback (approximate — axial centre slice only)

        For the DICOM path, pass the original .dcm file path via
        state["metadata"]["dicom_path"]; the main pipeline scan can remain a PNG.
        """
        row, col = pixel_centroid
        h, w = image_size

        if nifti_path and Path(nifti_path).exists():
            img    = nib.load(nifti_path)
            affine = img.affine
            z_slice = img.shape[2] // 2
            voxel   = np.array([col, row, z_slice, 1.0])
            native_coords = (affine @ voxel)[:3]
            if self.auto_register:
                return self._transform_point_to_mni152(native_coords, nifti_path)
            return native_coords

        if dicom_path and Path(dicom_path).exists():
            try:
                import pydicom
                dcm     = pydicom.dcmread(dicom_path)
                ipp     = np.array(dcm.ImagePositionPatient, dtype=float)   # [x,y,z] mm
                iop     = np.array(dcm.ImageOrientationPatient, dtype=float)
                spacing = np.array(dcm.PixelSpacing, dtype=float)           # [row_sp, col_sp]
                F       = iop.reshape(2, 3).T   # 3×2 direction-cosine matrix
                # Scanner-space coords (≈ MNI for pre-registered BraTS/ADNI scans)
                return ipp + F[:, 0] * col * spacing[1] + F[:, 1] * row * spacing[0]
            except Exception as e:
                print(f"[SiibraAtlasTool] DICOM affine failed ({e}), using pixel fallback")

        # Fallback: normalise pixel position into MNI152 1mm range
        # Maps (0,0)→(−91,−109) and (W,H)→(+91,+109) regardless of image resolution
        mni_x = (col / w * 182 - 91) * voxel_size_mm
        mni_y = (row / h * 218 - 109) * voxel_size_mm
        return np.array([mni_x, mni_y, 0.0])

    def _get_mni152_template(self) -> str:
        """
        Return a path to the MNI152 1mm template NIfTI, downloading via nilearn
        on first call and caching for the lifetime of this object.
        """
        if self._mni152_template_path and Path(self._mni152_template_path).exists():
            return self._mni152_template_path

        from nilearn import datasets as nilearn_datasets
        mni_img = nilearn_datasets.load_mni152_template(resolution=1)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".nii.gz", delete=False, prefix="mni152_template_"
        )
        tmp.close()
        nib.save(mni_img, tmp.name)
        self._mni152_template_path = tmp.name
        print(f"[SiibraAtlasTool] MNI152 template saved to {tmp.name}")
        return self._mni152_template_path

    def _transform_point_to_mni152(
        self,
        native_coords: np.ndarray,
        nifti_path: str,
    ) -> np.ndarray:
        """
        Register a NIfTI scan to MNI152 and map a single native-space point
        through the resulting transform.

        Uses an image-based approach to avoid apply_transforms_to_points
        convention ambiguity: creates a single-voxel indicator volume at the
        native centroid, warps it forward to MNI152 via apply_transforms, then
        reads the peak voxel location in MNI152 mm coordinates.

        Registration (fwdtransforms) is cached per nifti_path.
        """
        import ants

        if nifti_path not in self._registration_cache:
            print(f"[SiibraAtlasTool] Registering {Path(nifti_path).name} → MNI152 "
                  f"({self.registration_type}) …")
            template_path = self._get_mni152_template()
            fixed  = ants.image_read(template_path)
            moving = ants.image_read(nifti_path)
            reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform=self.registration_type,
            )
            self._registration_cache[nifti_path] = reg["fwdtransforms"]
            print(f"[SiibraAtlasTool] Registration complete.")

        fwd_transforms = self._registration_cache[nifti_path]
        template_path  = self._get_mni152_template()
        fixed  = ants.image_read(template_path)
        moving = ants.image_read(nifti_path)

        # Convert native mm coords → voxel indices in moving space
        img        = nib.load(nifti_path)
        inv_affine = np.linalg.inv(img.affine)
        vox        = np.round((inv_affine @ np.append(native_coords, 1))[:3]).astype(int)
        vox        = np.clip(vox, 0, np.array(img.shape) - 1)

        # Build a single-voxel indicator volume in moving space and warp it forward
        indicator = np.zeros(img.shape, dtype=np.float32)
        indicator[vox[0], vox[1], vox[2]] = 1.0
        indicator_ants = moving.new_image_like(indicator)
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=indicator_ants,
            transformlist=fwd_transforms,
            interpolator="linear",
        )

        # Peak voxel in MNI152 space → MNI152 mm via template affine
        warped_data = warped.numpy()
        if warped_data.max() < 1e-6:
            print(f"[SiibraAtlasTool] Warped indicator is empty, returning native coords")
            return native_coords
        peak_vox  = np.unravel_index(warped_data.argmax(), warped_data.shape)
        mni_affine = nib.load(template_path).affine
        return (mni_affine @ np.array([*peak_vox, 1]))[:3]

    @staticmethod
    def _hemisphere(region_name: str) -> str:
        name = region_name.lower()
        if "_l" in name or "left" in name:
            return "left"
        elif "_r" in name or "right" in name:
            return "right"
        return "bilateral"

    def _fetch_regional_features(self, region) -> dict:
        """
        Fetch multimodal features linked to this region from EBRAINS KG.
        Requires EBRAINS authentication token.
        """
        features = {}
        try:
            receptor_features = siibra.get_features(
                region, siibra.features.molecular.ReceptorDensityFingerprint
            )
            if receptor_features:
                features["receptor_densities"] = {
                    (f.receptors[0] if f.receptors else "unknown"):
                    round(float(np.mean(list(f.data.values()))), 4)
                    for f in receptor_features[:3]
                }
        except Exception:
            pass

        try:
            conn_features = siibra.get_features(
                region, siibra.features.connectivity.StreamlineCounts
            )
            if conn_features:
                features["structural_connectivity_available"] = True
        except Exception:
            pass

        return features