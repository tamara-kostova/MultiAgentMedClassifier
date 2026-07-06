#!/usr/bin/env python3
"""
CLI entry point for the research orchestration layer.

Loads all model agents once, then runs a research experiment family through
the LangGraph research orchestrator (plan → run → analyze → report).

Usage:
    # Full family sweep:
    python run_research.py --family ablation --multiclass_dir data/eval_multiclass

    # Single config only + quick, class-balanced subset:
    python run_research.py --family debate_rounds --points debate_r2 \\
        --multiclass_dir data/eval_multiclass --max_samples 30
    python run_research.py --family agent_forest --points forest_n4 \\
        --multiclass_dir data/eval_multiclass --max_samples 30

    # List the experiment_ids available for a family:
    python run_research.py --family debate_rounds --list_points

Experiment families:
    threshold_sweep       sam3_threshold ∈ [0.50–0.85] (8 points)
    human_review_sweep    human_review_threshold ∈ [0.30–0.55] (6 points)
    ablation              full / no_sam3 / always_sam3 / no_biomedclip (4 points)
    biomedclip_threshold  biomedclip_rerank_threshold ∈ [0.50–0.80] (7 points)
    agent_forest          System C — forest size N ∈ {1, 3, 4} (3 points)
    debate_rounds         System B — debate rounds R ∈ {1, 2, 3} (3 points)

Use --points to run a subset of a family, and --max_samples to cap the dataset
for a fast run. Outputs land in: outputs/research/{family}_{timestamp}/
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
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap each task to the first N samples (class-balanced) for a "
                        "quick run. Default: use all samples.")
    p.add_argument("--points", nargs="+", default=None, metavar="EXPERIMENT_ID",
                   help="Run only these sweep points by experiment_id instead of the "
                        "whole family, e.g. --points debate_r2  (single 2-round debate) "
                        "or --points forest_n4  (single 4-agent forest). "
                        "Run with --list_points to see the ids for a family.")
    p.add_argument("--list_points", action="store_true",
                   help="Print the sweep points (experiment_ids) for --family and exit.")
    args = p.parse_args()

    if args.list_points:
        print(f"Sweep points for family '{args.family}':")
        for pt in EXPERIMENT_FAMILIES[args.family]:
            print(f"  {pt.experiment_id:16s} {pt.description}")
        return

    if args.points:
        available = {pt.experiment_id for pt in EXPERIMENT_FAMILIES[args.family]}
        unknown = [pid for pid in args.points if pid not in available]
        if unknown:
            p.error(f"Unknown experiment_id(s) {unknown} for family '{args.family}'. "
                    f"Available: {sorted(available)}")

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
        "max_samples": args.max_samples,
        "point_ids": args.points,
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
