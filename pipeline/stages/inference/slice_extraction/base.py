from abc import abstractmethod

from pipeline.context import PipelineContext
from pipeline.stages.base import Stage


class BaseSliceExtractor(Stage):
    """Samples axial slices per sequence into an array for the VLM branch."""

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        ...