import numpy as np
from PIL import Image

from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.inference.slice_extraction import BaseSliceExtractor


def _window_ct(volume: np.ndarray, metadata: dict, level: float, width: float) -> np.ndarray:
    """HU → [0, 1] under a (level, width) window. Rescale tags applied when present."""
    slope = float(metadata.get("rescale_slope", 1.0))
    intercept = float(metadata.get("rescale_intercept", 0.0))
    hu = volume * slope + intercept
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def _normalize_mr(volume: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Percentile clip over non-background voxels of the whole volume (consistent across slices)."""
    body = volume[volume > 0]
    if body.size == 0:
        return np.zeros_like(volume)
    lo, hi = np.percentile(body, [low_pct, high_pct])
    return np.clip((volume - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _to_rgb(slice_2d: np.ndarray, pad_square: bool) -> Image.Image:
    s = (255 * slice_2d).astype(np.uint8)
    if pad_square:
        h, w = s.shape
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.uint8)
        top, left = (side - h) // 2, (side - w) // 2
        canvas[top:top + h, left:left + w] = s
        s = canvas
    return Image.fromarray(np.stack([s, s, s], axis=-1))


@register("inference.slice_extraction.neuro_axial")
class NeuroAxialSliceExtractor(BaseSliceExtractor):
    """Axial slices prepared the way the classifier's training PNGs were.

    Unlike axial_sampler: windows CT in HU (brain window) and percentile-clips MR
    instead of volume min-max, drops slices with little tissue (skull cap, neck,
    empty), keeps native resolution (agents resize themselves) and records which
    volume index each slice came from.

    TODO: take the slicing axis from context.affine once the host context's
    fields are known; for now `axis` assumes an axial-first volume.
    """

    def __init__(
        self,
        n_slices: int = 32,
        min_foreground_fraction: float = 0.15,
        foreground_threshold: float = 0.05,
        ct_window: tuple = (40.0, 80.0),  # (level, width) in HU
        mr_percentiles: tuple = (1.0, 99.0),
        axis: int = 0,
        pad_square: bool = True,
        default_modality: str = "MR",
    ):
        self.n_slices = n_slices
        self.min_foreground_fraction = min_foreground_fraction
        self.foreground_threshold = foreground_threshold
        self.ct_window = tuple(ct_window)
        self.mr_percentiles = tuple(mr_percentiles)
        self.axis = axis
        self.pad_square = pad_square
        self.default_modality = default_modality

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.volume is None:
            raise ValueError("neuro_axial needs context.volume")
        volume = np.moveaxis(np.asarray(context.volume, dtype=np.float32), self.axis, 0)
        metadata = getattr(context, "metadata", None) or {}
        modality = (getattr(context, "modality", None) or self.default_modality).upper()

        if modality == "CT":
            norm = _window_ct(volume, metadata, *self.ct_window)
            # Bone saturates at 1.0 under a brain window; it is not brain tissue.
            tissue = (norm > self.foreground_threshold) & (norm < 1.0)
        else:
            norm = _normalize_mr(volume, *self.mr_percentiles)
            tissue = norm > self.foreground_threshold

        fractions = tissue.reshape(tissue.shape[0], -1).mean(axis=1)
        keep = np.flatnonzero(fractions >= self.min_foreground_fraction)
        if keep.size == 0:
            keep = np.array([int(np.argmax(fractions))])

        n = min(self.n_slices, keep.size)
        picked = keep[np.linspace(0, keep.size - 1, n).round().astype(int)]
        picked = np.unique(picked)

        context.slices = [_to_rgb(norm[i], self.pad_square) for i in picked]
        context.slice_indices = [int(i) for i in picked]
        return context
