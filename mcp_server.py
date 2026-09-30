"""
MCP endpoint for the neuroimaging classifier (stage pipeline, pipeline/runner.py).

    # network endpoint (Streamable HTTP) — token required unless bound to loopback
    NEURO_MCP_TOKEN=... python mcp_server.py --host 0.0.0.0 --port 8765 --load_4bit
    # local subprocess for an MCP client on the same machine
    python mcp_server.py --transport stdio --load_4bit

Tools:
    classify           base64 image or volume + tumor|ms|stroke → flat MedGemma-schema diagnosis
    list_capabilities  tasks, pipelines, registered stages, input limits
    classify_slices    base64 PNG/JPEG slices → study-level result
    classify_volume    base64 (or server-side) .npy/.npz volume → study-level result

Models load once at startup and are shared by every request; GPU work is
serialised behind one lock (a second process would load MedGemma again), so
run a single worker. Unknown flags go to run_pipeline.build_config
(--load_4bit, --calibration_file, --generate_explainability, --output_dir, ...).

Research prototype — not a medical device. Every result carries
requires_human_review and a FHIR status of "preliminary" when review is needed.
"""

import argparse
import base64
import binascii
import contextlib
import hmac
import io
import json
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, get_args

import anyio.from_thread
import numpy as np
from mcp.server.mcpserver import Context, MCPServer
from pydantic import BaseModel, Field
from mcp.server.mcpserver.exceptions import ToolError

from pipeline.context import PipelineContext
from pipeline.registry import available, parse_spec
from pipeline.runner import Pipeline, load_config

CONFIG_DIR = Path(__file__).parent / "configs" / "pipelines"
TOKEN_ENV = "NEURO_MCP_TOKEN"

Task = Literal["binary_tumor", "multiclass_tumor", "ms", "stroke"]
PipelineName = Literal["standard", "forest", "debate"]
Aggregation = Literal["max_abnormal", "majority"]

MAX_IMAGES = 64
MAX_TOP_K = 32
MAX_VOLUME_SLICES = 128
MAX_VOLUME_VOXELS = 256 * 1024 * 1024  # ~1 GB as float32 once normalised
MAX_FOREST_AGENTS = 8  # AgentForest cycles its 4 roles beyond 4
MAX_DEBATE_ROUNDS = 3  # DebateOrchestrator.MAX_ROUNDS; larger values are silently clamped

DISCLAIMER = (
    "Research prototype, not a medical device. Slice-level accuracy was measured on curated "
    "public datasets; study-level accuracy on clinical DICOM is unvalidated. Treat results with "
    "requires_human_review=true as needing radiologist review."
)

# Per-slice state fields returned to the caller (the rest is internal plumbing).
SLICE_FIELDS = [
    "routing_path",
    "suspected_pathology",
    "medgemma_diagnosis",
    "classification_result",
    "segmentation_result",
    "biomedclip_result",
    "saliency_sam3_iou",
    "verification_result",
    "forest_votes",
    "forest_consensus",
    "debate_verdict",
    "debate_rounds_completed",
    "final_predicted_class",
    "final_confidence",
    "final_medgemma_diagnosis",
    "requires_human_review",
    "final_report",
]


# Allowed values of the flat classify output; mirrors prompts/system_prompt.txt.
DiagnosisName = Literal["tumor", "stroke", "multiple sclerosis", "normal", "other abnormalities"]
DetailedDiagnosis = Literal[
    "glioma", "meningioma", "pituitary_tumor", "carcinoma", "germinoma", "granuloma", "medulloblastoma",
    "neurocytoma", "papilloma", "schwannoma", "tuberculoma", "ischemic", "hemorrhagic",
]
Sequence = Literal["FLAIR", "T1", "T2", "T1C+"]
Plane = Literal["axial", "sagittal", "coronal"]
Modality = Literal["MRI", "CT"]


class Diagnosis(BaseModel):
    """classify result: the MedGemma output schema of prompts/system_prompt.txt, field for field.

    Nullable fields are null when they could not be determined.
    """

    modality: Optional[Modality]
    specialized_sequence: Optional[Sequence] = Field(description="MRI sequence; null for CT")
    plane: Optional[Plane]
    diagnosis_name: Optional[DiagnosisName]
    diagnosis_detailed: Optional[DetailedDiagnosis] = Field(description="Tumor or stroke subtype; null otherwise")
    icd10_code: Optional[str]
    severity_score: Optional[float] = Field(ge=0, le=1, description="Extent of abnormality")
    diagnosis_confidence: float = Field(ge=0, le=1)
    severity_confidence: float = Field(ge=0, le=1)


# classify's task names → the pipeline's tasks. multiclass_tumor, not binary_tumor, because
# its 12 CNN classes (normal included) are exactly the tumor values of diagnosis_detailed.
SimpleTask = Literal["tumor", "ms", "stroke"]
SIMPLE_TASKS = {"tumor": "multiclass_tumor", "ms": "ms", "stroke": "stroke"}
TASK_MODALITY = {"tumor": "MR", "ms": "MR", "stroke": "CT"}


def _pick(value: Any, allowed) -> Optional[str]:
    """Case-insensitive match against a Literal's values; anything else becomes null."""
    if value is None:
        return None
    lookup = {a.lower(): a for a in get_args(allowed)}
    return lookup.get(str(value).strip().lower())


def _unit(value: Any) -> Optional[float]:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _flat_diagnosis(slice_result: dict) -> Diagnosis:
    """Flatten one slice of _run's output into the classify contract.

    Uses the report's fused diagnosis (MedGemma over every tool output), falling
    back to the triage diagnosis if the report did not parse. diagnosis_confidence
    is the pipeline's final confidence, which includes the low GradCAM++/SAM3 IoU
    penalty.
    """
    final = slice_result.get("final_medgemma_diagnosis")
    dx = final or slice_result.get("medgemma_diagnosis") or {}
    confidence = slice_result.get("final_confidence") if final else dx.get("diagnosis_confidence")
    return Diagnosis(
        modality=_pick(dx.get("modality"), Modality),
        specialized_sequence=_pick(dx.get("specialized_sequence"), Sequence),
        plane=_pick(dx.get("plane"), Plane),
        diagnosis_name=_pick(dx.get("diagnosis_name"), DiagnosisName),
        diagnosis_detailed=_pick(dx.get("diagnosis_detailed"), DetailedDiagnosis),
        icd10_code=dx.get("icd10_code") or None,
        severity_score=_unit(dx.get("severity_score")),
        diagnosis_confidence=_unit(confidence) or 0.0,
        severity_confidence=_unit(dx.get("severity_confidence")) or 0.0,
    )


def _most_abnormal(diagnoses: list[Diagnosis]) -> Diagnosis:
    """Study-level pick over slices, like aggregation.study's max_abnormal: the most
    confident abnormal slice, else the most confident one."""
    abnormal = [d for d in diagnoses if d.diagnosis_name not in (None, "normal")]
    return max(abnormal or diagnoses, key=lambda d: d.diagnosis_confidence)


class _Server:
    """Process-wide state: shared models, the GPU lock and server options."""

    resources = None
    gpu_lock = threading.Lock()
    input_root: Path = Path("outputs/mcp_inputs")
    volume_dir: Optional[Path] = None  # server-side volume_path reads are disabled unless set
    keep_files = False


# ── Helpers ────────────────────────────────────────────────────────────────────


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _decode_b64(data: str, what: str) -> bytes:
    if data.startswith("data:"):  # data URL: data:image/png;base64,....
        data = data.split(",", 1)[-1]
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ToolError(f"{what} is not valid base64: {exc}") from None


def _load_volume(raw: bytes) -> np.ndarray:
    try:
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise ToolError(f"volume is not a readable .npy/.npz file: {exc}") from None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        key = "volume" if "volume" in loaded.files else loaded.files[0]
        loaded = loaded[key]
    volume = np.asarray(loaded)
    if volume.ndim != 3:
        raise ToolError(f"volume must be 3D (slices, H, W), got shape {volume.shape}")
    if not (np.issubdtype(volume.dtype, np.integer) or np.issubdtype(volume.dtype, np.floating)):
        raise ToolError(f"volume must be an integer or float array, got dtype {volume.dtype}")
    if volume.size > MAX_VOLUME_VOXELS:
        raise ToolError(f"volume has {volume.size} voxels, limit is {MAX_VOLUME_VOXELS}")
    return volume


def _pipeline_config(name: str, *, volume: bool, overrides: dict[str, dict]) -> dict[str, Any]:
    """Load configs/pipelines/<name>.yaml and apply per-request stage kwargs."""
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    stages = []
    for spec in config["stages"]:
        stage, kwargs = parse_spec(spec)
        if volume and stage == "inference.slice_extraction.from_files":
            stage = "inference.slice_extraction.neuro_axial"
        kwargs.update(overrides.get(stage, {}))
        stages.append({stage: kwargs})
    config["stages"] = stages
    return config


def _overrides(top_k, aggregation, forest_n_agents, debate_rounds, n_slices=None) -> dict[str, Any]:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ToolError(f"top_k must be in 1..{MAX_TOP_K}")
    if not 1 <= forest_n_agents <= MAX_FOREST_AGENTS:
        raise ToolError(f"forest_n_agents must be in 1..{MAX_FOREST_AGENTS}")
    if not 1 <= debate_rounds <= MAX_DEBATE_ROUNDS:
        raise ToolError(f"debate_rounds must be in 1..{MAX_DEBATE_ROUNDS}")
    overrides = {
        "inference.screening.cnn_topk": {"top_k": top_k},
        "inference.aggregation.study": {"strategy": aggregation},
        "inference.agents.forest_triage": {"n_agents": forest_n_agents},
        "inference.agents.debate": {"rounds": debate_rounds},
    }
    if n_slices is not None:
        if not 1 <= n_slices <= MAX_VOLUME_SLICES:
            raise ToolError(f"n_slices must be in 1..{MAX_VOLUME_SLICES}")
        overrides["inference.slice_extraction.neuro_axial"] = {"n_slices": n_slices}
    return overrides


def _report(ctx: Optional[Context], done: int, total: int, message: str) -> None:
    """Progress notification from the worker thread; best effort."""
    if ctx is None:
        return
    with contextlib.suppress(Exception):
        anyio.from_thread.run(ctx.report_progress, done, total, message)


def _run(config: dict, context: PipelineContext, ctx: Optional[Context], include_fhir: bool, cleanup: list) -> dict[str, Any]:
    """Build the pipeline, run it under the GPU lock, shape the response."""
    progress = {"done": 0}

    try:
        pipe = Pipeline.from_config(config, resources=_Server.resources)
        names = pipe.stage_names()

        def on_stage(name: str, _context: PipelineContext) -> None:
            progress["done"] += 1
            _report(ctx, progress["done"], len(names), f"{name} done")

        _report(ctx, 0, len(names), "waiting for GPU" if _Server.gpu_lock.locked() else "starting")
        with _Server.gpu_lock:
            context = pipe.run(context, on_stage=on_stage)

        slices = []
        for pos in context.selected:
            state = context.slice_states[pos]
            detail = {"slice_index": state["metadata"].get("slice_index", pos)}
            detail.update({field: state.get(field) for field in SLICE_FIELDS})
            if include_fhir:
                detail["fhir_bundle"] = state.get("fhir_report")
            slices.append(detail)

        study = dict(context.study_result or {})
        # Paths under the per-request workdir are deleted below; keep only indices.
        study["slices"] = [{k: v for k, v in row.items() if k != "image_path"} for row in study.get("slices", [])]
        return _jsonable(
            {
                "study": study,
                "slices": slices,
                "pipeline": pipe.name,
                "stages": names,
                "timings_s": {k: round(v, 2) for k, v in context.timings.items()},
                "disclaimer": DISCLAIMER,
            }
        )
    finally:
        if not _Server.keep_files:
            for path in cleanup + [context.workdir]:
                if path:
                    shutil.rmtree(path, ignore_errors=True)


def _request_dir() -> Path:
    path = _Server.input_root / uuid.uuid4().hex[:12]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_images(images: list[str], what: str) -> tuple[Path, list[str]]:
    """Decode base64 images into a fresh request dir as PNGs; the dir is removed on failure."""
    from PIL import Image

    workdir = _request_dir()
    paths = []
    try:
        for i, data in enumerate(images):
            raw = _decode_b64(data, f"{what}[{i}]" if len(images) > 1 else what)
            try:
                image = Image.open(io.BytesIO(raw))
                image.load()
            except Exception as exc:
                raise ToolError(f"{what} is not a readable image: {exc}") from None
            path = workdir / f"input_{i:04d}.png"
            image.save(path)
            paths.append(str(path))
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return workdir, paths


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = MCPServer(
    name="neuro-classifier",
    title="Neuroimaging multi-agent classifier",
    instructions=(
        "Classifies brain MRI/CT for four tasks: binary_tumor, multiclass_tumor, ms, stroke. "
        "For one image or volume use classify, which returns a flat diagnosis. "
        "Send prepared 2D slices with classify_slices, or a whole volume with classify_volume. "
        "Every slice is screened with a CNN and only the top_k most abnormal slices go through the full "
        "multi-agent pipeline ('standard'; 'forest' = N role-specialised MedGemma agents + vote; "
        "'debate' = tool advocates + judge), so requests take tens of seconds to minutes; expect "
        "progress notifications and use a generous client timeout. Results are for research use "
        "and must be reviewed when requires_human_review is true."
    ),
)


@mcp.tool()
def classify(
    task: SimpleTask,
    image: Optional[str] = None,
    volume: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Diagnosis:
    """Diagnose one brain scan with the standard multi-agent pipeline.

    task: tumor | ms | stroke. stroke expects CT; tumor and ms expect MRI.
    Give exactly one of:
      image: base64 PNG or JPEG (raw base64 or a data: URL), one 2D slice
          windowed like a viewer shows it;
      volume: base64 of a .npy/.npz 3D array (slices, H, W), axial-first
          (npz: key "volume" or the first array). CT in HU, MR in raw intensities.
          32 slices are sampled, the 3 most abnormal (CNN screening) are analysed
          and the most abnormal diagnosis is returned.

    Returns the fused MedGemma diagnosis; fields are null when undetermined.
    """
    if (image is None) == (volume is None):
        raise ToolError("give exactly one of image or volume")
    pipeline_task = SIMPLE_TASKS[task]

    if image is not None:
        config = _pipeline_config("standard", volume=False, overrides=_overrides(1, "max_abnormal", 3, 1))
        workdir, paths = _save_images([image], "image")
        context = PipelineContext(task=pipeline_task, metadata={"image_paths": paths})
        result = _run(config, context, ctx, include_fhir=False, cleanup=[str(workdir)])
    else:
        array = _load_volume(_decode_b64(volume, "volume"))
        config = _pipeline_config("standard", volume=True, overrides=_overrides(3, "max_abnormal", 3, 1, n_slices=32))
        context = PipelineContext(task=pipeline_task, volume=array, modality=TASK_MODALITY[task], metadata={})
        result = _run(config, context, ctx, include_fhir=False, cleanup=[])

    if not result["slices"]:
        raise ToolError("no slice could be analysed")
    return _most_abnormal([_flat_diagnosis(s) for s in result["slices"]])


@mcp.tool()
def list_capabilities() -> dict[str, Any]:
    """Tasks, pipeline modes, registered stages and input limits of this server."""
    return {
        "tasks": list(Task.__args__),
        "pipelines": sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
        "aggregation_strategies": list(Aggregation.__args__),
        "stages": available(),
        "limits": {
            "max_images": MAX_IMAGES,
            "max_top_k": MAX_TOP_K,
            "max_volume_slices": MAX_VOLUME_SLICES,
            "max_volume_voxels": MAX_VOLUME_VOXELS,
            "max_forest_agents": MAX_FOREST_AGENTS,
            "max_debate_rounds": MAX_DEBATE_ROUNDS,
            "volume_path_enabled": _Server.volume_dir is not None,
        },
        "models_loaded": bool(_Server.resources is not None and _Server.resources._agents is not None),
        "disclaimer": DISCLAIMER,
    }


@mcp.tool()
def classify_slices(
    images: list[str],
    task: Task,
    pipeline: PipelineName = "standard",
    top_k: int = 3,
    aggregation: Aggregation = "max_abnormal",
    forest_n_agents: int = 3,
    debate_rounds: int = 1,
    include_fhir: bool = True,
    metadata: Optional[dict] = None,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Classify one study from prepared 2D slices.

    images: base64-encoded PNG/JPEG slices (raw base64 or data: URLs), in slice
        order. Slices should already be windowed/normalised like a viewer would
        show them (CT: brain window; MR: intensity-normalised).
    task: which classifier to use.
    pipeline: standard | forest | debate.
    top_k: slices (ranked by CNN abnormality) that get the full agent pipeline.
    aggregation: max_abnormal (any abnormal slice → abnormal study) or majority.
    metadata: optional caller context (e.g. StudyInstanceUID) echoed into FHIR.

    Returns the study-level prediction, per-slice agent outputs and, if
    include_fhir, one FHIR R4 bundle per analysed slice.
    """
    if not images:
        raise ToolError("images is empty")
    if len(images) > MAX_IMAGES:
        raise ToolError(f"at most {MAX_IMAGES} images per request")
    # Validate everything before writing inputs to disk.
    config = _pipeline_config(
        pipeline, volume=False, overrides=_overrides(top_k, aggregation, forest_n_agents, debate_rounds)
    )

    workdir, paths = _save_images(images, "images")
    context = PipelineContext(task=task, metadata={**(metadata or {}), "image_paths": paths})
    return _run(config, context, ctx, include_fhir, cleanup=[str(workdir)])


@mcp.tool()
def classify_volume(
    task: Task,
    modality: Literal["CT", "MR"],
    volume_b64: Optional[str] = None,
    volume_path: Optional[str] = None,
    pipeline: PipelineName = "standard",
    n_slices: int = 32,
    top_k: int = 3,
    aggregation: Aggregation = "max_abnormal",
    forest_n_agents: int = 3,
    debate_rounds: int = 1,
    rescale_slope: Optional[float] = None,
    rescale_intercept: Optional[float] = None,
    include_fhir: bool = True,
    metadata: Optional[dict] = None,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Classify one study from a 3D volume, shape (slices, H, W), axial-first.

    Give exactly one of:
      volume_b64: base64 of a .npy or .npz file (npz: key "volume" or the first array);
      volume_path: a .npy/.npz path on this server, inside its --volume_dir.
    modality: CT (windowed in HU, brain window) or MR (percentile-normalised).
    rescale_slope / rescale_intercept: DICOM rescale tags if the CT volume is
        stored as raw values rather than HU.
    n_slices: slices sampled from the tissue-containing range.
    top_k, pipeline, aggregation: as in classify_slices.
    """
    if (volume_b64 is None) == (volume_path is None):
        raise ToolError("give exactly one of volume_b64 or volume_path")

    if volume_path is not None:
        if _Server.volume_dir is None:
            raise ToolError("volume_path is disabled on this server (start it with --volume_dir)")
        path = Path(volume_path).resolve()
        if not path.is_relative_to(_Server.volume_dir):
            raise ToolError("volume_path must be inside the server's --volume_dir")
        if not path.is_file():
            raise ToolError(f"volume_path not found: {volume_path}")
        raw = path.read_bytes()
    else:
        raw = _decode_b64(volume_b64, "volume_b64")
    volume = _load_volume(raw)

    meta = dict(metadata or {})
    if rescale_slope is not None:
        meta["rescale_slope"] = rescale_slope
    if rescale_intercept is not None:
        meta["rescale_intercept"] = rescale_intercept

    config = _pipeline_config(
        pipeline,
        volume=True,
        overrides=_overrides(top_k, aggregation, forest_n_agents, debate_rounds, n_slices=n_slices),
    )
    context = PipelineContext(task=task, volume=volume, modality=modality, metadata=meta)
    return _run(config, context, ctx, include_fhir, cleanup=[])


# ── HTTP auth ─────────────────────────────────────────────────────────────────


class BearerTokenMiddleware:
    """Reject HTTP requests without `Authorization: Bearer <token>`."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self.expected):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")],
                    }
                )
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MCP endpoint for the neuroimaging classifier")
    p.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--path", default="/mcp", help="HTTP path of the MCP endpoint")
    p.add_argument("--max_body_mb", type=int, default=256, help="Max HTTP request size (slices/volumes are base64)")
    p.add_argument("--volume_dir", help="Enable classify_volume(volume_path=...) for files under this directory")
    p.add_argument("--keep_files", action="store_true", help="Keep decoded inputs and slice PNGs (default: delete)")
    p.add_argument("--lazy_load", action="store_true", help="Load models on first request instead of at startup")
    p.add_argument(
        "--allow_missing_checkpoints", action="store_true",
        help="Start even if a task checkpoint is missing (CNN falls back to ImageNet weights, BiomedCLIP to zero-shot)",
    )
    return p.parse_known_args(argv)


def _check_checkpoints(model_cfg) -> list[str]:
    """Fetch any missing task checkpoint now and return the ones still missing.

    The agents would otherwise fall back silently on the first request (CNN to
    ImageNet weights, BiomedCLIP to zero-shot, SAM3 skipped) and still return a
    confident-looking diagnosis.
    """
    from config import CHECKPOINT_SOURCE, TASKS, download_hf_checkpoint

    wanted = [(t, "cnn", model_cfg.cnn_checkpoints.get(t)) for t in TASKS]
    wanted += [(t, "biomedclip", model_cfg.biomedclip_probe_checkpoints.get(t)) for t in TASKS]
    wanted.append(("tumor_segmentation", "sam3", model_cfg.sam3_linear_probe_checkpoint))

    missing = []
    for task, kind, path in wanted:
        if path is None:
            missing.append(f"{task}/{kind}: no checkpoint configured")
            continue
        if Path(path).exists():
            continue
        if CHECKPOINT_SOURCE == "local":
            missing.append(f"{task}/{kind}: {path} (CHECKPOINT_SOURCE=local)")
            continue
        try:
            download_hf_checkpoint(task, kind, path, caller="mcp_server")
        except Exception as exc:
            missing.append(f"{task}/{kind}: {path} ({type(exc).__name__}: {str(exc)[:200]})")
    return missing


def main(argv=None):
    args, rest = parse_args(argv)
    is_loopback = args.host in ("127.0.0.1", "localhost", "::1")
    token = os.environ.get(TOKEN_ENV, "")
    if args.transport == "streamable-http" and not token and not is_loopback:
        sys.exit(f"Refusing to serve on {args.host} without auth: set {TOKEN_ENV}.")

    # stdout is the protocol channel under stdio; keep startup logs off it.
    log_stream = sys.stderr if args.transport == "stdio" else sys.stdout
    with contextlib.redirect_stdout(log_stream):
        from dotenv import load_dotenv

        load_dotenv()
        import run_pipeline
        from pipeline.resources import Resources

        cfg = run_pipeline.build_config(run_pipeline.parse_args(["--image", "<mcp>", *rest]))
        missing = _check_checkpoints(cfg.model)
        if missing:
            listing = "\n  ".join(missing)
            if not args.allow_missing_checkpoints:
                sys.exit(
                    f"[mcp_server] Missing task checkpoints:\n  {listing}\n"
                    "Put them under checkpoints/ (/models/checkpoints in Docker), allow the HF "
                    "download, or pass --allow_missing_checkpoints to serve degraded results."
                )
            print(f"[mcp_server] WARNING: serving with missing checkpoints:\n  {listing}", flush=True)
        _Server.resources = Resources(cfg)
        _Server.input_root = Path(cfg.output_dir) / "mcp_inputs"
        _Server.volume_dir = Path(args.volume_dir).resolve() if args.volume_dir else None
        _Server.keep_files = args.keep_files
        if not args.lazy_load:
            _Server.resources.agents  # load all models now, not on the first request

    if args.transport == "stdio":
        # While serving, the SDK points fd 1 at stderr, but text still sitting in
        # sys.stdout's buffer (pipeline prints without flush=True) would be flushed
        # after fd 1 is restored, i.e. onto the protocol channel. Flush every line.
        sys.stdout.reconfigure(line_buffering=True)
        mcp.run("stdio")
        return

    import uvicorn

    app = mcp.streamable_http_app(
        streamable_http_path=args.path,
        max_request_body_size=args.max_body_mb * 1024 * 1024,
        host=args.host,
    )
    if token:
        app = BearerTokenMiddleware(app, token)
    else:
        print(f"[mcp_server] WARNING: no {TOKEN_ENV} set — unauthenticated, loopback only.", flush=True)
    print(f"[mcp_server] serving MCP on http://{args.host}:{args.port}{args.path}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
