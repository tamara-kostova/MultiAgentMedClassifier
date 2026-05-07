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
from pathlib import Path

from config import (
    DEFAULT_CONFIG,
    ModelConfig,
    PipelineConfig,
    RoutingConfig,
    resolve_torch_device,
)
from eval.evaluate import compare_configurations, load_test_split, run_single
from eval.tumor_eval import LABEL_MAPS, run_tumor_eval
from pipeline.graph import build_pipeline
from dotenv import load_dotenv

load_dotenv()


def parse_args():
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

    # Tumor-eval mode
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
        "--max_samples",
        type=int,
        default=None,
        help="For --tumor_eval: stop after this many images total across all runs.",
    )
    p.add_argument(
        "--label_map",
        type=str,
        default="figshare3",
        choices=["figshare3", "br35h", "none"],
        help=(
            "For --tumor_eval: label mapping preset. "
            "'figshare3' maps 1/2/3 → meningioma/glioma/pituitary (default). "
            "'br35h' maps yes/no → brain tumor MRI/normal brain MRI. "
            "'none' uses raw folder names as labels."
        ),
    )

    # CNN checkpoint overrides
    p.add_argument("--cnn_binary_tumor", type=str, default=None)
    p.add_argument("--cnn_multiclass", type=str, default=None)
    p.add_argument("--cnn_ms", type=str, default=None)
    p.add_argument("--cnn_stroke", type=str, default=None)
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
        help="Skip MedGemma report generation (eval mode — saves ~5–9 s/image)",
    )

    # Explainability
    p.add_argument(
        "--generate_explainability",
        action="store_true",
        help="Run Grad-CAM++ and Integrated Gradients after CNN classification",
    )

    # Output
    p.add_argument("--output_dir", type=str, default="outputs")

    return p.parse_args()


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

    model_cfg = ModelConfig(
        cnn_checkpoints=cnn_checkpoints,
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
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

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

    elif args.tumor_eval:
        # ── Tumor dataset eval mode (JSONL, resumable, all models) ───────────
        if not args.tumor_eval_dir:
            print("Error: --tumor_eval requires --tumor_eval_dir")
            return
        output_file = Path(
            args.tumor_eval_output
            or f"{cfg.output_dir}/eval/{args.task or 'multiclass_tumor'}_tumor_eval.jsonl"
        )
        label_map = LABEL_MAPS.get(args.label_map) if args.label_map != "none" else None
        run_tumor_eval(
            app=app,
            data_dir=args.tumor_eval_dir,
            task=args.task or "multiclass_tumor",
            output_file=output_file,
            label_map=label_map,
            max_samples=args.max_samples,
        )

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
