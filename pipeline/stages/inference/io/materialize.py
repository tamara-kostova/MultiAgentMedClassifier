import tempfile
import uuid
from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.registry import register
from pipeline.stages.base import Stage
from pipeline.state import initial_state


@register("inference.io.materialize")
class MaterializeSlices(Stage):
    """Write context.slices to PNGs and open one NeuroimagingState per slice.

    The agents (CNN, SAM3, BiomedCLIP, MedGemma) read images by path, so every
    slice needs a file. Selects every slice; a screening stage can narrow it.
    """

    def __init__(self, workdir: str | None = None, resources=None):
        self.workdir = workdir
        self.resources = resources

    def _make_workdir(self) -> Path:
        if self.workdir:
            root = Path(self.workdir)
        elif self.resources is not None:
            root = Path(self.resources.cfg.output_dir) / "stage_slices"
        else:
            root = Path(tempfile.gettempdir()) / "neuro_stage_slices"
        path = root / uuid.uuid4().hex[:12]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.slices:
            raise ValueError("materialize needs context.slices (run a slice_extraction stage first)")
        indices = context.slice_indices or list(range(len(context.slices)))
        workdir = self._make_workdir()

        states = []
        for idx, image in zip(indices, context.slices):
            path = workdir / f"slice_{idx:04d}.png"
            image.save(path)
            states.append(
                dict(initial_state(str(path), context.task, metadata={**context.metadata, "slice_index": idx}))
            )

        context.workdir = str(workdir)
        context.slice_states = states
        context.selected = list(range(len(states)))
        return context
