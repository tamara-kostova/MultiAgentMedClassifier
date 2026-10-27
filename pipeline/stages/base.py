from abc import ABC, abstractmethod

from pipeline.context import PipelineContext


class Stage(ABC):
    """One step of a stage pipeline: takes the context, returns it (mutated or replaced)."""

    #: Registry name, set by @register.
    name: str = ""

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        ...
