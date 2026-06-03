#!/usr/bin/env python3
"""
CLI entry point for the research orchestration layer.

Loads all model agents once, then runs a research experiment family through
the LangGraph research orchestrator (plan → run → analyze → report).

Usage:
    python run_research.py \\
        --family ablation \\
        --binary_tumor_dir data/test/binary_tumor \\
        --multiclass_dir   data/test/multiclass_tumor \\
        --ms_dir           data/test/ms \\
        --stroke_dir       data/test/stroke

    python run_research.py --family threshold_sweep --binary_tumor_dir data/test/binary_tumor

Experiment families:
    threshold_sweep       sam3_threshold ∈ [0.50–0.85] (8 points)
    human_review_sweep    human_review_threshold ∈ [0.30–0.55] (6 points)
    ablation              full / no_sam3 / always_sam3 / no_biomedclip (4 points)
    biomedclip_threshold  biomedclip_rerank_threshold ∈ [0.50–0.80] (7 points)

Outputs land in: outputs/research/{family}_{timestamp}/
  sweep_manifest.json          — metadata for every sweep point
  results/{experiment_id}/     — comparison_summary.csv + all_predictions.csv
  analysis/sweep_summary.csv   — merged across all points
  analysis/report.md           — Markdown report with analysis tables
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from config import DEFAULT_CONFIG
from pipeline.graph import load_agents
from experiments.experiments import EXPERIMENT_FAMILIES
from experiments.graph import ResearchState, build_research_pipeline


def main():
    p = argparse.ArgumentParser(
        description="Research orchestrator for the multi-agent neuroimaging pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--family",
        required=True,
        choices=sorted(EXPERIMENT_FAMILIES.keys()),
        help="Experiment family to run",
    )
    p.add_argument("--binary_tumor_dir", type=str, default=None,
                   help="Test directory for binary_tumor task")
    p.add_argument("--multiclass_dir", type=str, default=None,
                   help="Test directory for multiclass_tumor task")
    p.add_argument("--ms_dir", type=str, default=None,
                   help="Test directory for ms task")
    p.add_argument("--stroke_dir", type=str, default=None,
                   help="Test directory for stroke task")
    p.add_argument("--output_dir", type=str, default="outputs/research",
                   help="Root output directory (default: outputs/research)")
    args = p.parse_args()

    dataset_dirs: dict[str, str] = {}
    if args.binary_tumor_dir:
        dataset_dirs["binary_tumor"] = args.binary_tumor_dir
    if args.multiclass_dir:
        dataset_dirs["multiclass_tumor"] = args.multiclass_dir
    if args.ms_dir:
        dataset_dirs["ms"] = args.ms_dir
    if args.stroke_dir:
        dataset_dirs["stroke"] = args.stroke_dir

    if not dataset_dirs:
        p.error("At least one dataset directory (--binary_tumor_dir / --multiclass_dir / "
                "--ms_dir / --stroke_dir) must be provided.")

    print("Loading pipeline agents (once, shared across all sweep points)...")
    agents = load_agents(DEFAULT_CONFIG)

    app = build_research_pipeline()

    initial_state: ResearchState = {
        "experiment_family": args.family,
        "dataset_dirs": dataset_dirs,
        "output_base": args.output_dir,
        "base_cfg": DEFAULT_CONFIG,
        "preloaded_agents": agents,
        "results_dir": "",
        "sweep_summary": None,
        "analysis": {},
        "report_md": "",
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"Research run complete.")
    print(f"Report : {final_state['results_dir']}/analysis/report.md")
    print(f"Summary: {final_state['results_dir']}/analysis/sweep_summary.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
