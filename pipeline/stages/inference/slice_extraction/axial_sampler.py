import numpy as np
from PIL import Image

from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.inference.slice_extraction import BaseSliceExtractor


def _normalize(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    volume -= volume.min()
    volume /= volume.max() + 1e-8
    return volume


def _sample_slices(volume: np.ndarray, n_slices: int) -> np.ndarray:
    total = volume.shape[0]
    indices = np.linspace(0, total - 1, n_slices, dtype=int)
    return volume[indices]


def _slices_to_pil(slices: np.ndarray) -> list[Image.Image]:
    images = []

    for s in slices:
        s = (255 * s).astype(np.uint8)
        rgb = np.stack([s, s, s], axis=-1)
        images.append(Image.fromarray(rgb).resize((224, 224)))

    return images


@register("inference.slice_extraction.axial_sampler")
class AxialSliceExtractor(BaseSliceExtractor):
    """Normalizes context.volume and samples n equidistant axial slices as RGB PIL images."""

    def __init__(self, n_slices: int = 32):
        self.n_slices = n_slices

    def run(self, context: PipelineContext) -> PipelineContext:
        volume = _normalize(context.volume)
        sampled = _sample_slices(volume, self.n_slices)
        context.slices = _slices_to_pil(sampled)
        return context