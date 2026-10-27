"""
PipelineContext — the object every stage receives and returns.

Mirrors the context of the host multi-agent system this classifier plugs into
(its stages import `pipeline.context.PipelineContext`). PROVISIONAL: the field
set is a superset guess until the host system's real context is known. Stages
here read the optional fields with getattr defaults so a foreign context works.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image


@dataclass
class PipelineContext:
    # ── Input (set by the caller / host system) ───────────────────────────────
    task: str = "binary_tumor"  # "binary_tumor" | "multiclass_tumor" | "ms" | "stroke"
    volume: Optional[np.ndarray] = None  # (slices, H, W), stored axial-first
    modality: Optional[str] = None       # "CT" | "MR"
    spacing: Optional[tuple] = None      # voxel spacing (z, y, x) in mm
    affine: Optional[np.ndarray] = None  # 4x4 voxel → patient transform
    metadata: dict = field(default_factory=dict)  # DICOM UIDs, rescale tags, image_paths, ...

    # ── Written by slice extraction ───────────────────────────────────────────
    slices: list[Image.Image] = field(default_factory=list)
    slice_indices: list[int] = field(default_factory=list)  # position of each slice in `volume`

    # ── Written by materialize / screening / agent stages ─────────────────────
    workdir: Optional[str] = None
    slice_states: list[dict] = field(default_factory=list)  # one NeuroimagingState per slice
    screening: list[dict] = field(default_factory=list)     # CNN screen score per slice
    selected: list[int] = field(default_factory=list)       # positions in slice_states that get the full agents

    # ── Written by aggregation ────────────────────────────────────────────────
    study_result: Optional[dict] = None

    # ── Written by the runner ─────────────────────────────────────────────────
    timings: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)  # free slot for custom stages

    def selected_states(self) -> list[dict]:
        return [self.slice_states[i] for i in self.selected]
