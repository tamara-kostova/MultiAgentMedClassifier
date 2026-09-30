"""
Run a stage pipeline (pipeline/runner.py) from a YAML config.

    # one or more prepared slices
    python run_stages.py --config configs/pipelines/standard.yaml \
        --task binary_tumor --image scan.png [more.png ...]

    # a volume (.npy, axial-first); from_files is swapped for neuro_axial
    python run_stages.py --config configs/pipelines/forest.yaml \
        --task stroke --volume head_ct.npy --modality CT

Any other flag is forwarded to run_pipeline's config builder
(--load_4bit, --calibration_file, --generate_explainability, --output_dir, ...).
Prints the study-level result as JSON.
"""

import argparse
import json
import sys

import numpy as np

from pipeline.context import PipelineContext
from pipeline.registry import parse_spec
from pipeline.runner import Pipeline, load_config

TASKS = ["binary_tumor", "multiclass_tumor", "ms", "stroke"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a stage pipeline from a YAML config")
    p.add_argument("--config", required=True, help="Pipeline YAML, e.g. configs/pipelines/standard.yaml")
    p.add_argument("--task", required=True, choices=TASKS)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", nargs="+", help="One or more 2D slice images")
    src.add_argument("--volume", help="Volume as .npy, shape (slices, H, W), axial-first")
    p.add_argument("--modality", choices=["CT", "MR"], help="Needed for --volume windowing")
    p.add_argument("--metadata", help="JSON file merged into context.metadata (e.g. rescale_slope, UIDs)")
    p.add_argument("--list_stages", action="store_true", help="Print registered stage names and exit")
    return p.parse_known_args(argv)


def main(argv=None):
    if "--list_stages" in (argv if argv is not None else sys.argv[1:]):
        from pipeline.registry import available

        print("\n".join(available()))
        return

    args, rest = parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    import run_pipeline
    from pipeline.resources import Resources

    # --image placeholder satisfies run_pipeline's required mode group (same trick as ui/demo.py).
    cfg = run_pipeline.build_config(run_pipeline.parse_args(["--image", "<stages>", *rest]))

    config = load_config(args.config)
    metadata = json.loads(open(args.metadata).read()) if args.metadata else {}
    context = PipelineContext(task=args.task, metadata=metadata)

    if args.volume:
        context.volume = np.load(args.volume)
        context.modality = args.modality
        config["stages"] = [
            "inference.slice_extraction.neuro_axial"
            if parse_spec(spec)[0] == "inference.slice_extraction.from_files"
            else spec
            for spec in config["stages"]
        ]
    else:
        context.metadata["image_paths"] = args.image

    pipe = Pipeline.from_config(config, resources=Resources(cfg))
    print(f"[run_stages] {pipe.name}: {' → '.join(pipe.stage_names())}", flush=True)
    context = pipe.run(context)

    print(json.dumps({"study_result": context.study_result, "timings": context.timings}, indent=2, default=str))


if __name__ == "__main__":
    main()
