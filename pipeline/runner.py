"""
Build and run a stage pipeline from a YAML (or dict) config.

Config format (see configs/pipelines/):

    name: forest
    defaults:            # optional; applied to the context when unset
      task: multiclass_tumor
    stages:
      - inference.slice_extraction.from_files
      - inference.screening.cnn_topk: {top_k: 3}
      - ...

Example:
    from pipeline.runner import Pipeline
    from pipeline.context import PipelineContext

    pipe = Pipeline.from_config("configs/pipelines/standard.yaml")
    ctx = pipe.run(PipelineContext(task="binary_tumor", metadata={"image_paths": ["scan.png"]}))
    print(ctx.study_result)
"""

import time
from pathlib import Path
from typing import Callable, Optional, Union

import yaml

from pipeline import registry
from pipeline.context import PipelineContext

StageCallback = Callable[[str, PipelineContext], None]


def load_config(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        cfg = dict(source)
    else:
        cfg = yaml.safe_load(Path(source).read_text()) or {}
    if not cfg.get("stages"):
        raise ValueError(f"Pipeline config {source!r} has no 'stages'")
    return cfg


class Pipeline:
    def __init__(self, stages: list, name: str = "pipeline", defaults: Optional[dict] = None):
        self.stages = stages
        self.name = name
        self.defaults = defaults or {}

    @classmethod
    def from_config(cls, source: Union[str, Path, dict], resources=None) -> "Pipeline":
        cfg = load_config(source)
        if resources is None:
            from pipeline.resources import Resources

            resources = Resources()
        stages = [registry.build(spec, resources) for spec in cfg["stages"]]
        return cls(stages, name=cfg.get("name", "pipeline"), defaults=cfg.get("defaults"))

    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    def run(self, context: PipelineContext, on_stage: Optional[StageCallback] = None) -> PipelineContext:
        for key, value in self.defaults.items():
            if getattr(context, key, None) in (None, "", [], {}):
                setattr(context, key, value)

        for stage in self.stages:
            t0 = time.perf_counter()
            context = stage.run(context)
            context.timings[stage.name] = context.timings.get(stage.name, 0.0) + time.perf_counter() - t0
            if on_stage is not None:
                on_stage(stage.name, context)
        return context
