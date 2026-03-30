"""
Entry point for the multi-agent neuroimaging pipeline.

Usage examples:
  # Single image:
  python run_pipeline.py --image image.jpg --task binary_tumor

  # Full evaluation across all datasets:
  python run_pipeline.py --eval \
    --binary_tumor_dir  data/test/binary_tumor \
    --multiclass_dir    data/test/multiclass_tumor \
    --ms_dir            data/test/ms \
    --stroke_dir        data/test/stroke

  # With custom CNN checkpoints:
  python run_pipeline.py --image image.jpg --task binary_tumor \
    --cnn_binary_tumor checkpoints/densenet169_binary_tumor.pt

Checkpoints:
  CNN checkpoints should be PyTorch state dicts saved as:
      torch.save({"model_state_dict": model.state_dict(), ...}, path)
  or plain state dicts:
      torch.save(model.state_dict(), path)
"""

import argparse
from pathlib import Path

from config import DEFAULT_CONFIG, ModelConfig, PipelineConfig, RoutingConfig
from eval.evaluate import compare_configurations, load_test_split, run_single
from pipeline.graph import build_pipeline
from pipeline.state import initial_state
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent neuroimaging pipeline")

    # Mode
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Path to a single input image")
    mode.add_argument("--eval", action="store_true", help="Run full evaluation")

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

    model_cfg = ModelConfig(
        cnn_checkpoints=cnn_checkpoints,
        sam3_linear_probe_checkpoint=(
            args.sam3_probe or default_model_cfg.sam3_linear_probe_checkpoint
        ),
        sam3_bpe_path=args.sam3_bpe_path or default_model_cfg.sam3_bpe_path,
    )
    routing_cfg = RoutingConfig(
        always_run_sam3=args.always_run_sam3,
        sam3_threshold=args.sam3_threshold,
        human_review_threshold=args.human_threshold,
    )
    return PipelineConfig(
        model=model_cfg,
        routing=routing_cfg,
        output_dir=args.output_dir,
        generate_explainability=args.generate_explainability,
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
