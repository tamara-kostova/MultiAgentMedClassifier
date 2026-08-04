"""
Sweep runner for experiment families.

Iterates over SweepPoints in a family, reassembles the LangGraph pipeline
with each point's RoutingConfig overrides (reusing pre-loaded model agents),
and calls compare_configurations() to collect per-point metrics.

All results land under:
  {output_dir}/
  ├── sweep_manifest.json         — metadata for every sweep point
  ├── results/{experiment_id}/   — comparison_summary.csv + all_predictions.csv
  └── analysis/sweep_summary.csv — merged across all points
"""

import dataclasses
import json
import time
from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG, PipelineConfig
from eval.evaluate import compare_configurations
from pipeline.graph import assemble_debate_pipeline, assemble_forest_pipeline, assemble_pipeline
from experiments.experiments import EXPERIMENT_FAMILIES, SweepPoint


def _apply_overrides(base_cfg: PipelineConfig, overrides: dict) -> PipelineConfig:
    """Return a new PipelineConfig with the given RoutingConfig fields replaced."""
    new_routing = dataclasses.replace(base_cfg.routing, **overrides)
    return dataclasses.replace(base_cfg, routing=new_routing)


def run_experiment_family(
    family_name: str,
    test_datasets: dict[str, list[dict]],
    output_dir: str,
    preloaded_agents: tuple,
    base_cfg: PipelineConfig = None,
    only_points: list[str] = None,
) -> pd.DataFrame:
    """
    Run every SweepPoint in a family and return a merged summary DataFrame.

    Args:
        family_name:       Key into EXPERIMENT_FAMILIES.
        test_datasets:     {task: list of sample dicts} — from load_test_split().
        output_dir:        Root directory for this family's results.
        preloaded_agents:  (medgemma, cnn, sam3, clip) from load_agents().
        base_cfg:          Base PipelineConfig; defaults to DEFAULT_CONFIG.
        only_points:       If given, run only the sweep points whose experiment_id
                           is in this list (e.g. ["debate_r2"] for a single config).

    Returns:
        DataFrame with one row per (experiment_id, task), columns from
        comparison_summary.csv plus experiment_id / description / override_* cols.
    """
    base_cfg = base_cfg or DEFAULT_CONFIG
    sweep_points: list[SweepPoint] = EXPERIMENT_FAMILIES[family_name]

    if only_points:
        wanted = set(only_points)
        sweep_points = [pt for pt in sweep_points if pt.experiment_id in wanted]
        if not sweep_points:
            available = [pt.experiment_id for pt in EXPERIMENT_FAMILIES[family_name]]
            raise ValueError(
                f"No sweep points in family '{family_name}' match {sorted(wanted)}. "
                f"Available: {available}"
            )

    family_dir = Path(output_dir)
    results_dir = family_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    medgemma, cnn, sam3, clip = preloaded_agents

    manifest: dict = {
        "family": family_name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_sweep_points": len(sweep_points),
        "sweep_points": [],
    }

    all_summaries: list[pd.DataFrame] = []

    for point in sweep_points:
        print(f"\n{'='*60}")
        print(f"Sweep point : {point.experiment_id}")
        print(f"Description : {point.description}")
        if point.routing_overrides:
            print(f"Overrides   : {point.routing_overrides}")
        print("=" * 60)

        point_dir = results_dir / point.experiment_id
        point_dir.mkdir(parents=True, exist_ok=True)

        cfg = _apply_overrides(base_cfg, point.routing_overrides)
        mode = point.pipeline_mode
        kwargs = point.pipeline_kwargs or {}
        if mode == "debate":
            app = assemble_debate_pipeline(medgemma, cnn, sam3, clip, cfg, **kwargs)
        elif mode == "forest":
            app = assemble_forest_pipeline(medgemma, cnn, sam3, clip, cfg, **kwargs)
        else:
            app = assemble_pipeline(medgemma, cnn, sam3, clip, cfg)

        t0 = time.perf_counter()
        summary = compare_configurations(
            {point.experiment_id: app},
            test_datasets,
            output_dir=str(point_dir),
        )
        elapsed = time.perf_counter() - t0

        # Tag rows with sweep metadata for later merging
        summary["experiment_id"] = point.experiment_id
        summary["description"] = point.description
        for k, v in point.routing_overrides.items():
            summary[f"override_{k}"] = v

        all_summaries.append(summary)

        manifest["sweep_points"].append({
            "experiment_id": point.experiment_id,
            "description": point.description,
            "routing_overrides": point.routing_overrides,
            "elapsed_s": round(elapsed, 1),
            "result_dir": str(point_dir),
        })
        print(f"  Completed in {elapsed:.1f}s")

    # Merge and persist
    merged = pd.concat(all_summaries, ignore_index=True)
    analysis_dir = family_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(analysis_dir / "sweep_summary.csv", index=False)

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (family_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nSweep complete — {len(sweep_points)} points. Results in {family_dir}")
    return merged
