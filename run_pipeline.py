"""
Entry point for the multi-agent neuroimaging pipeline.

Usage examples:
  # Single image:
  python run_pipeline.py --image data/processed/1/2.jpg --task binary_tumor

  # Full evaluation across all datasets:
  python run_pipeline.py --eval \
    --binary_tumor_dir  data/test/binary_tumor \
    --multiclass_dir    data/test/multiclass_tumor \
    --ms_dir            data/test/ms \
    --stroke_dir        data/test/stroke

  # With custom CNN checkpoints:
  python run_pipeline.py --image image.jpg --task binary_tumor \
    --cnn_binary_tumor checkpoints/densenet169_binary_tumor.pt

  # With BiomedCLIP linear probe (layer-6 or concat-fusion head):
  python run_pipeline.py --image image.jpg --task binary_tumor \
    --clip_binary_tumor checkpoints/biomedclip_probe_binary_tumor.pt

  # With few-shot examples (one image per class prepended to MedGemma triage):
  python run_pipeline.py --image image.jpg --task binary_tumor \
    --few_shot --few_shot_data_dir /path/to/data

  # Force Apple Silicon GPU:
  python run_pipeline.py --image image.jpg --task binary_tumor --device mps

  # Faster Mac evaluation: use MPS and skip final MedGemma report generation.
  python run_pipeline.py --tumor_eval --tumor_eval_dir data/Br35H \
    --task binary_tumor --label_map br35h --device mps --skip_report

Checkpoints:
  CNN checkpoints should be PyTorch state dicts saved as:
      torch.save({"model_state_dict": model.state_dict(), ...}, path)
  or plain state dicts:
      torch.save(model.state_dict(), path)
"""

import argparse
import json
import os
from pathlib import Path

from agents.debate import ADVOCATES, DebateOrchestrator, parse_advocates
from agents.forest import ROLE_NAMES, VOTE_MODES, parse_roles
from config import (
    CHECKPOINT_SOURCE,
    DEFAULT_CONFIG,
    ModelConfig,
    PipelineConfig,
    RoutingConfig,
    download_hf_checkpoint,
    resolve_torch_device,
)
from eval.evaluate import compare_configurations, load_test_split, run_single
from pipeline.graph import build_debate_pipeline, build_forest_pipeline, build_pipeline
from eval.tumor_eval import (
    LABEL_MAPS,
    TASK_DEFAULT_LABEL_MAP,
    ResumeConfigMismatch,
    _samples_from_list,
    build_run_config,
    check_resume_compatible,
    load_dataset,
    load_image_list,
    run_dataset_eval,
)
from dotenv import load_dotenv

load_dotenv()


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Multi-agent neuroimaging pipeline")

    # Mode
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Path to a single input image")
    mode.add_argument("--eval", action="store_true", help="Run full evaluation")
    mode.add_argument(
        "--tumor_eval",
        action="store_true",
        help=(
            "Run full pipeline on a single tumor dataset; writes rich JSONL with "
            "outputs from every model. Resumes from partial runs automatically."
        ),
    )
    mode.add_argument(
        "--dataset_eval",
        action="store_true",
        help=(
            "Run resumable rich JSONL evaluation on any single class-folder "
            "dataset, including MS and stroke."
        ),
    )

    # Single image
    p.add_argument(
        "--task",
        type=str,
        choices=["binary_tumor", "multiclass_tumor", "ms", "stroke"],
        help="Classification task",
    )

    # Evaluation datasets
    p.add_argument("--binary_tumor_dir", type=str, default=None)
    p.add_argument("--multiclass_dir", type=str, default=None)
    p.add_argument("--ms_dir", type=str, default=None)
    p.add_argument("--stroke_dir", type=str, default=None)

    # Resumable single-dataset eval mode
    p.add_argument(
        "--tumor_eval_dir",
        type=str,
        default=None,
        help="Dataset root for --tumor_eval: <dir>/<class>/<image.*> (e.g. data/processed)",
    )
    p.add_argument(
        "--tumor_eval_output",
        type=str,
        default=None,
        help="Output JSONL path for --tumor_eval (default: outputs/eval/<task>_tumor_eval.jsonl)",
    )
    p.add_argument(
        "--dataset_eval_dir",
        type=str,
        default=None,
        help="Dataset root for --dataset_eval: <dir>/<class>/<image.*>",
    )
    p.add_argument(
        "--dataset_eval_output",
        type=str,
        default=None,
        help="Output JSONL path for --dataset_eval",
    )
    p.add_argument(
        "--image_list",
        type=str,
        default=None,
        help=(
            "For resumable eval: run exactly these images, in this order. A text file "
            "(one image_path per line) or a prior run's .jsonl (its image_path values). "
            "Paths as recorded (relative to the repo root). --max_samples applies after."
        ),
    )
    p.add_argument(
        "--allow_missing_checkpoints",
        action="store_true",
        help=(
            "Eval modes refuse to start when a task's CNN / BiomedCLIP probe (or the SAM3 "
            "probe on tumor tasks) checkpoint is missing; this flag runs anyway, with "
            "ImageNet / zero-shot / no-SAM3 fallbacks."
        ),
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="For resumable eval: stop after this many images total across all runs.",
    )
    p.add_argument(
        "--label_map",
        type=str,
        default="auto",
        choices=["auto", *LABEL_MAPS.keys(), "none"],
        help=(
            "For resumable eval: label mapping preset. "
            "'auto' selects a default from --task. "
            "'figshare3' maps 1/2/3 → meningioma/glioma/pituitary. "
            "'br35h' maps yes/no → brain tumor MRI/normal brain MRI. "
            "'ms_binary' and 'stroke_binary' map common binary folders. "
            "'none' uses raw folder names as labels."
        ),
    )

    # CNN checkpoint overrides
    p.add_argument("--cnn_binary_tumor", type=str, default=None)
    p.add_argument("--cnn_multiclass", type=str, default=None)
    p.add_argument("--cnn_ms", type=str, default=None)
    p.add_argument("--cnn_stroke", type=str, default=None)

    # BiomedCLIP linear probe checkpoint overrides (layer-6 or concat-fusion heads)
    p.add_argument("--clip_binary_tumor", type=str, default=None,
                   help="BiomedCLIP probe checkpoint for binary_tumor task")
    p.add_argument("--clip_multiclass", type=str, default=None,
                   help="BiomedCLIP probe checkpoint for multiclass_tumor task")
    p.add_argument("--clip_ms", type=str, default=None,
                   help="BiomedCLIP probe checkpoint for ms task")
    p.add_argument("--clip_stroke", type=str, default=None,
                   help="BiomedCLIP probe checkpoint for stroke task")
    p.add_argument(
        "--sam3_probe",
        type=str,
        default=None,
        help="Path to SAM3 linear probe checkpoint (checkpoints/sam3_probe.pth)",
    )
    p.add_argument(
        "--sam3_bpe_path",
        type=str,
        default=None,
        help="Path to SAM3 BPE vocabulary (sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz)",
    )

    # Routing thresholds
    p.add_argument("--sam3_threshold", type=float, default=0.70)
    p.add_argument("--human_threshold", type=float, default=0.45)
    p.add_argument(
        "--always_run_sam3",
        action="store_true",
        help="Force SAM3 routing on every non-normal case, regardless of confidence",
    )
    p.add_argument(
        "--always_run_biomedclip",
        action="store_true",
        help="Force BiomedCLIP routing on every case, regardless of confidence",
    )

    # Calibration
    p.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help=(
            'JSON file with per-task temperatures, e.g. {"binary_tumor": 1.3, "ms": 0.9}. '
            "Fitted via TemperatureScaler.fit() on a held-out validation set."
        ),
    )

    # Few-shot examples for MedGemma triage
    p.add_argument(
        "--few_shot",
        action="store_true",
        help="Prepend one example image per class to MedGemma's triage prompt",
    )
    p.add_argument(
        "--few_shot_data_dir",
        type=str,
        default=None,
        help="Root directory for resolving few_shot_examples.csv image paths",
    )

    # Eval optimisation
    p.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help=(
            "Override compute device, e.g. --device mps on Apple Silicon. "
            "Defaults to CUDA, then Apple MPS, then CPU."
        ),
    )
    p.add_argument(
        "--skip_report",
        action="store_true",
        help=(
            "Skip MedGemma report generation (saves ~5–9 s/image). Not accuracy-"
            "neutral: the final class then falls back to the CNN prediction."
        ),
    )
    p.add_argument(
        "--medgemma_model",
        type=str,
        default=None,
        help=(
            "Override the MedGemma model ID (other-backbone ablation, e.g. "
            f"google/medgemma-27b-it). Default: {DEFAULT_CONFIG.model.medgemma_model_id}."
        ),
    )
    p.add_argument(
        "--triage_only",
        action="store_true",
        help=(
            "Stop after triage / forest_triage; the triage becomes the final prediction "
            "(no specialists, no report, no FHIR). Standard or forest mode only."
        ),
    )
    p.add_argument(
        "--load_4bit",
        action="store_true",
        help=(
            "Load MedGemma with 4-bit NF4 quantization (CUDA only). Use on GPUs "
            "with <12 GB VRAM. Also enabled by MEDGEMMA_4BIT=1 in the environment."
        ),
    )

    # Explainability
    p.add_argument(
        "--generate_explainability",
        action="store_true",
        help="Run Grad-CAM++ and Integrated Gradients after CNN classification",
    )

    # Pipeline mode (System B / C)
    p.add_argument(
        "--pipeline_mode",
        type=str,
        choices=["standard", "debate", "forest"],
        default="standard",
        help=(
            "Pipeline variant: 'standard' (baseline), "
            "'debate' (System B — multi-agent debate), "
            "'forest' (System C — agent forest with majority vote)."
        ),
    )
    p.add_argument(
        "--debate_rounds",
        type=int,
        default=1,
        help="Number of debate rounds for --pipeline_mode debate (1–3, default 1).",
    )
    p.add_argument(
        "--debate_advocates",
        type=str,
        default=",".join(ADVOCATES),
        help=(
            "Comma list of debate advocates, a subset of cnn,clip,sam "
            "(default: all three). E.g. cnn,clip drops the SAM3 advocate."
        ),
    )
    p.add_argument(
        "--forest_n_agents",
        type=int,
        default=3,
        help="Number of forest agents for --pipeline_mode forest (default 3).",
    )
    p.add_argument(
        "--forest_roles",
        type=str,
        default=None,
        help=(
            f"Comma list of forest roles from {','.join(ROLE_NAMES)}, cycled to "
            "--forest_n_agents (default: all four in that order). Homogeneous control: "
            "--forest_roles radiologist --forest_n_agents 4 --forest_temperature 0.7"
        ),
    )
    p.add_argument(
        "--forest_temperature",
        type=float,
        default=0.0,
        help="Forest sampling temperature; 0 = greedy decoding (default, published runs).",
    )
    p.add_argument(
        "--forest_top_p",
        type=float,
        default=1.0,
        help="Forest nucleus-sampling top_p (only used with --forest_temperature > 0).",
    )
    p.add_argument(
        "--forest_seed",
        type=int,
        default=0,
        help="Forest sampling seed; agent i uses seed + i (default 0).",
    )
    p.add_argument(
        "--forest_vote",
        type=str,
        choices=list(VOTE_MODES),
        default="majority",
        help=(
            "Forest aggregation over canonical labels: 'majority' (ballot count) or "
            "'confidence' (summed diagnosis_confidence). Ties: first label in agent order."
        ),
    )

    # Output
    p.add_argument("--output_dir", type=str, default="outputs")

    args = p.parse_args(argv)
    _validate_args(p, args)
    return args


def _validate_args(p: argparse.ArgumentParser, args) -> None:
    if not 1 <= args.debate_rounds <= DebateOrchestrator.MAX_ROUNDS:
        p.error(f"--debate_rounds must be in 1..{DebateOrchestrator.MAX_ROUNDS}, got {args.debate_rounds}")
    if args.forest_n_agents < 1:
        p.error(f"--forest_n_agents must be >= 1, got {args.forest_n_agents}")
    if args.forest_temperature < 0:
        p.error("--forest_temperature must be >= 0")
    if not 0 < args.forest_top_p <= 1:
        p.error("--forest_top_p must be in (0, 1]")
    try:
        args.forest_roles = parse_roles(args.forest_roles)
        args.debate_advocates = parse_advocates(args.debate_advocates)
    except ValueError as exc:
        p.error(str(exc))
    if args.pipeline_mode == "forest":
        roles = args.forest_roles or ROLE_NAMES
        assigned = [roles[i % len(roles)] for i in range(args.forest_n_agents)]
        if args.forest_temperature == 0 and len(set(assigned)) < len(assigned):
            p.error(
                f"forest roles {assigned} contain duplicates with --forest_temperature 0: "
                "greedy duplicates cast identical votes. Use distinct roles / fewer agents, "
                "or a temperature > 0."
            )
    if args.triage_only and args.pipeline_mode == "debate":
        p.error("--triage_only: debate's triage is the standard one; use --pipeline_mode standard")


def build_config(args) -> PipelineConfig:
    default_model_cfg = DEFAULT_CONFIG.model
    cnn_checkpoints = default_model_cfg.cnn_checkpoints.copy()
    overrides = {
        "binary_tumor": args.cnn_binary_tumor,
        "multiclass_tumor": args.cnn_multiclass,
        "ms": args.cnn_ms,
        "stroke": args.cnn_stroke,
    }
    cnn_checkpoints.update(
        {task: path for task, path in overrides.items() if path is not None}
    )

    temperatures = default_model_cfg.cnn_temperatures.copy()
    if args.calibration_file:
        cal_path = Path(args.calibration_file)
        if cal_path.exists():
            temperatures.update(json.loads(cal_path.read_text()))
            print(f"[calibration] Loaded temperatures from {cal_path}: {temperatures}")
        else:
            print(f"[calibration] File not found: {cal_path} — using T=1.0 defaults")

    device = resolve_torch_device(
        args.device or default_model_cfg.device,
        caller="run_pipeline",
    )

    clip_checkpoints = default_model_cfg.biomedclip_probe_checkpoints.copy()
    clip_overrides = {
        "binary_tumor":     args.clip_binary_tumor,
        "multiclass_tumor": args.clip_multiclass,
        "ms":               args.clip_ms,
        "stroke":           args.clip_stroke,
    }
    clip_checkpoints.update(
        {task: path for task, path in clip_overrides.items() if path is not None}
    )

    # 4-bit NF4 for MedGemma: CLI flag or MEDGEMMA_4BIT=1 (lets a batch/container
    # run flip it without editing config.py).
    use_4bit = args.load_4bit or os.environ.get("MEDGEMMA_4BIT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_4bit:
        print("[run_pipeline] MedGemma 4-bit NF4 quantization enabled.")

    model_cfg = ModelConfig(
        medgemma_model_id=args.medgemma_model or default_model_cfg.medgemma_model_id,
        cnn_checkpoints=cnn_checkpoints,
        use_4bit_quantization=use_4bit,
        biomedclip_probe_checkpoints=clip_checkpoints,
        sam3_linear_probe_checkpoint=(
            args.sam3_probe or default_model_cfg.sam3_linear_probe_checkpoint
        ),
        sam3_bpe_path=args.sam3_bpe_path or default_model_cfg.sam3_bpe_path,
        cnn_temperatures=temperatures,
        use_few_shot=args.few_shot,
        few_shot_data_dir=args.few_shot_data_dir,
        device=device.type,
        prefer_cuda_for_vision=(args.device is None or args.device == "cuda"),
    )

    routing_cfg = RoutingConfig(
        always_run_sam3=args.always_run_sam3,
        always_run_biomedclip=args.always_run_biomedclip,
        sam3_threshold=args.sam3_threshold,
        human_review_threshold=args.human_threshold,
    )
    return PipelineConfig(
        model=model_cfg,
        routing=routing_cfg,
        output_dir=args.output_dir,
        generate_explainability=args.generate_explainability,
        skip_report=args.skip_report,
        pipeline_mode=args.pipeline_mode,
        triage_only=args.triage_only,
        forest_n_agents=args.forest_n_agents,
        forest_roles=args.forest_roles,
        forest_temperature=args.forest_temperature,
        forest_top_p=args.forest_top_p,
        forest_seed=args.forest_seed,
        forest_vote=args.forest_vote,
        debate_rounds=args.debate_rounds,
        debate_advocates=tuple(args.debate_advocates),
    )


def _resolve_label_map(label_map_name: str, task: str) -> dict[str, str]:
    if label_map_name == "none":
        return {}
    if label_map_name == "auto":
        default_name = TASK_DEFAULT_LABEL_MAP.get(task)
        return LABEL_MAPS.get(default_name, {})
    return LABEL_MAPS[label_map_name]


def _mode_suffix(cfg: PipelineConfig) -> str:
    """Output-name suffix for a non-baseline run ("" for the standard pipeline, so
    the paper's baseline file names are unchanged)."""
    parts = []
    if cfg.pipeline_mode == "forest":
        parts.append(f"forest_n{cfg.forest_n_agents}")
        if cfg.forest_roles and tuple(cfg.forest_roles) != ROLE_NAMES:
            parts.append("-".join(cfg.forest_roles))
        if cfg.forest_temperature > 0:
            parts.append(f"t{cfg.forest_temperature:g}")
            if cfg.forest_top_p != 1.0:
                parts.append(f"p{cfg.forest_top_p:g}")
            if cfg.forest_seed != 0:
                parts.append(f"s{cfg.forest_seed}")
        if cfg.forest_vote != "majority":
            parts.append(f"v{cfg.forest_vote}")
    elif cfg.pipeline_mode == "debate":
        parts.append(f"debate_r{cfg.debate_rounds}")
        if tuple(cfg.debate_advocates) != ADVOCATES:
            parts.append("-".join(cfg.debate_advocates))
    if cfg.triage_only:
        parts.append("triage_only")
    if cfg.model.medgemma_model_id != DEFAULT_CONFIG.model.medgemma_model_id:
        parts.append(cfg.model.medgemma_model_id.rsplit("/", 1)[-1])
    return "".join(f"_{p}" for p in parts)


def _default_resumable_eval_output(args, cfg: PipelineConfig, task: str) -> Path:
    suffix = "tumor_eval" if args.tumor_eval else "dataset_eval"
    shot = "_few_shot" if args.few_shot else ""
    return Path(f"{cfg.output_dir}/eval/{task}{shot}_{suffix}{_mode_suffix(cfg)}.jsonl")


def _missing_checkpoints(cfg: PipelineConfig, tasks) -> list[str]:
    """Checkpoints an eval of `tasks` needs that are still absent after a download attempt."""
    model = cfg.model
    needed = []
    for task in tasks:
        needed.append((task, "cnn", model.cnn_checkpoints.get(task)))
        needed.append((task, "biomedclip", model.biomedclip_probe_checkpoints.get(task)))
        if task in cfg.routing.sam3_eligible_tasks:
            needed.append(("tumor_segmentation", "sam3", model.sam3_linear_probe_checkpoint))
    missing = []
    for hf_task, kind, path in dict.fromkeys(needed):
        if path is None:
            missing.append(f"{hf_task}/{kind}: not configured")
            continue
        if not Path(path).exists() and CHECKPOINT_SOURCE != "local":
            try:
                download_hf_checkpoint(hf_task, kind, path, caller="run_pipeline")
            except Exception as exc:
                print(f"[run_pipeline] download of {hf_task}/{kind} failed: {exc}")
        if not Path(path).exists():
            missing.append(f"{hf_task}/{kind}: {path}")
    return missing


def _prepare_resumable_eval(args, cfg: PipelineConfig) -> dict:
    """Resolve and validate a --tumor_eval/--dataset_eval run before any model loads."""
    data_dir = args.dataset_eval_dir or args.tumor_eval_dir
    if not data_dir:
        flag = "--dataset_eval_dir" if args.dataset_eval else "--tumor_eval_dir"
        raise SystemExit(f"Error: {flag} is required")
    task = args.task or ("multiclass_tumor" if args.tumor_eval else None)
    if task is None:
        raise SystemExit("Error: --task is required with --dataset_eval")

    output_file = Path(
        args.dataset_eval_output
        or args.tumor_eval_output
        or _default_resumable_eval_output(args, cfg, task)
    )
    image_list, list_info = None, None
    try:
        if args.image_list:
            image_list, list_info = load_image_list(args.image_list)
            print(f"[run_pipeline] --image_list: {list_info['n']} images from {list_info['path']}")
            _samples_from_list(load_dataset(data_dir, task), image_list)
        run_config = build_run_config(cfg, image_list=list_info)
        check_resume_compatible(output_file, run_config)
    except (ResumeConfigMismatch, ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(f"[run_pipeline] Output: {output_file}")
    return {
        "data_dir": data_dir,
        "task": task,
        "output_file": output_file,
        "label_map": _resolve_label_map(args.label_map, task),
        "max_samples": args.max_samples,
        "run_config": run_config,
        "image_list": image_list,
    }


def main():
    args = parse_args()
    cfg = build_config(args)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    resumable = None
    if args.eval or args.tumor_eval or args.dataset_eval:
        if args.eval:
            eval_tasks = [t for t, d in (
                ("binary_tumor", args.binary_tumor_dir),
                ("multiclass_tumor", args.multiclass_dir),
                ("ms", args.ms_dir),
                ("stroke", args.stroke_dir),
            ) if d]
        else:
            resumable = _prepare_resumable_eval(args, cfg)
            eval_tasks = [resumable["task"]]
        missing = _missing_checkpoints(cfg, eval_tasks)
        if missing:
            msg = "Missing checkpoints for this eval:\n  " + "\n  ".join(missing)
            if not args.allow_missing_checkpoints:
                raise SystemExit(
                    f"Error: {msg}\nRefusing to run an eval on fallback weights; "
                    "pass --allow_missing_checkpoints to run anyway."
                )
            print(f"[run_pipeline] WARNING: {msg}\n  (continuing: --allow_missing_checkpoints)")

    mode = args.pipeline_mode
    if mode == "debate":
        app = build_debate_pipeline(cfg)
    elif mode == "forest":
        app = build_forest_pipeline(cfg)
    else:
        app = build_pipeline(cfg)

    if args.image:
        # ── Single image mode ─────────────────────────────────────────────────
        if not args.task:
            print("Error: --task is required with --image")
            return
        run_single(
            app,
            args.image,
            args.task,
            verbose=True,
            output_dir=cfg.output_dir,
            save_output=True,
        )

    elif resumable is not None:
        # ── Single dataset eval mode (JSONL, resumable, all models) ──────────
        run_dataset_eval(app=app, **resumable)

    elif args.eval:
        # ── Full evaluation mode ──────────────────────────────────────────────
        task_dirs = {
            "binary_tumor": args.binary_tumor_dir,
            "multiclass_tumor": args.multiclass_dir,
            "ms": args.ms_dir,
            "stroke": args.stroke_dir,
        }
        test_datasets = {}
        for task, directory in task_dirs.items():
            if directory and Path(directory).exists():
                samples = load_test_split(directory, task)
                test_datasets[task] = samples
                print(
                    f"Loaded {len(samples)} test samples for '{task}' from {directory}"
                )
            else:
                print(f"Skipping '{task}': no directory provided or not found.")

        if not test_datasets:
            print(
                "No valid test datasets found. Provide at least one --*_dir argument."
            )
            return

        summary = compare_configurations(
            configs={"agent_pipeline": app},
            test_datasets=test_datasets,
            output_dir=f"{cfg.output_dir}/eval",
        )

        print("\n=== Evaluation Summary ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
