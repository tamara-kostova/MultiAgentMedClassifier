from PIL import Image

from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.inference.slice_extraction import BaseSliceExtractor


@register("inference.slice_extraction.from_files")
class FileSliceExtractor(BaseSliceExtractor):
    """Already-prepared 2D slices (PNG/JPG) → context.slices.

    Paths come from the `paths` argument or context.metadata["image_paths"].
    The original paths are kept in metadata["source_paths"] so FHIR / reports
    can point at the caller's files.
    """

    def __init__(self, paths: list[str] | None = None):
        self.paths = list(paths or [])

    def run(self, context: PipelineContext) -> PipelineContext:
        paths = self.paths or list(context.metadata.get("image_paths") or [])
        if not paths:
            raise ValueError("from_files needs `paths` or context.metadata['image_paths']")
        context.slices = [Image.open(p).convert("RGB") for p in paths]
        context.slice_indices = list(range(len(paths)))
        context.metadata["source_paths"] = [str(p) for p in paths]
        return context
