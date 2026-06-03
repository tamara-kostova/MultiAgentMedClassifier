"""
LangGraph research orchestrator — outer-loop graph that plans, executes,
analyzes, and reports on systematic pipeline experiments.

Graph (linear, 4 nodes):
  plan_experiments → run_experiments → analyze_results → write_report → END

State fields:
  experiment_family   — which family to run (key into EXPERIMENT_FAMILIES)
  dataset_dirs        — {task: directory path}
  output_base         — root dir; family subdir is created inside it
  base_cfg            — PipelineConfig used as the base for all sweep points
  preloaded_agents    — (medgemma, cnn, sam3, clip) from load_agents()
  results_dir         — set by plan_experiments, consumed by subsequent nodes
  sweep_summary       — pd.DataFrame set by run_experiments
  analysis            — dict of DataFrames set by analyze_results
  report_md           — Markdown string set by write_report
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from eval.evaluate import load_test_split
from experiments.analysis import (
    ablation_summary,
    calibration_by_routing_path,
    calibration_per_task,
    load_sweep_predictions,
    medgemma_agreement_analysis,
    per_class_failure_breakdown,
    routing_distribution,
    sensitivity_specificity_table,
)
from experiments.experiments import EXPERIMENT_FAMILIES
from experiments.runner import run_experiment_family


# ── State ─────────────────────────────────────────────────────────────────────


class ResearchState(TypedDict):
    experiment_family: str
    dataset_dirs: dict[str, str]
    output_base: str
    base_cfg: Any                   # PipelineConfig
    preloaded_agents: Any           # tuple from load_agents()
    results_dir: str
    sweep_summary: Optional[Any]    # pd.DataFrame
    analysis: dict                  # {name: pd.DataFrame}
    report_md: str


# ── Node functions ─────────────────────────────────────────────────────────────


def plan_experiments(state: ResearchState) -> dict:
    """Validate the family, create a timestamped output directory, log the plan."""
    family = state["experiment_family"]
    if family not in EXPERIMENT_FAMILIES:
        raise ValueError(
            f"Unknown experiment family '{family}'. "
            f"Available: {sorted(EXPERIMENT_FAMILIES.keys())}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{state['output_base']}/{family}_{timestamp}"

    points = EXPERIMENT_FAMILIES[family]
    lines = [
        f"\n{'#'*60}",
        f"# Research plan: {family}",
        f"#   Sweep points : {len(points)}",
        f"#   Output dir   : {results_dir}",
        f"#   Datasets     : {sorted(state['dataset_dirs'].keys())}",
        f"#   Points:",
    ]
    for p in points:
        lines.append(f"#     [{p.experiment_id}] {p.description}")
    lines.append("#" * 60)
    print("\n".join(lines))

    return {"results_dir": results_dir}


def run_experiments(state: ResearchState) -> dict:
    """Load test samples and run all sweep points via runner.run_experiment_family."""
    test_datasets = {
        task: load_test_split(d, task)
        for task, d in state["dataset_dirs"].items()
    }
    sweep_summary = run_experiment_family(
        family_name=state["experiment_family"],
        test_datasets=test_datasets,
        output_dir=state["results_dir"],
        preloaded_agents=state["preloaded_agents"],
        base_cfg=state["base_cfg"],
    )
    return {"sweep_summary": sweep_summary}


def analyze_results(state: ResearchState) -> dict:
    """Run all four analysis functions on the sweep outputs."""
    summary_df: pd.DataFrame = state["sweep_summary"]
    preds_df = load_sweep_predictions(f"{state['results_dir']}/results")

    analysis: dict[str, pd.DataFrame] = {}

    if summary_df is not None and not summary_df.empty:
        analysis["sensitivity_specificity"] = sensitivity_specificity_table(summary_df)
        analysis["ablation"] = ablation_summary(summary_df)

    if not preds_df.empty:
        analysis["routing_distribution"] = routing_distribution(preds_df)
        analysis["calibration_by_path"] = calibration_by_routing_path(preds_df)
        analysis["medgemma_agreement"] = medgemma_agreement_analysis(preds_df)
        analysis["calibration_per_task"] = calibration_per_task(preds_df)
        analysis["per_class_failures"] = per_class_failure_breakdown(preds_df)

    return {"analysis": analysis}


def write_report(state: ResearchState) -> dict:
    """Render analysis DataFrames to a Markdown report and save it."""
    family = state["experiment_family"]
    analysis = state["analysis"]
    results_dir = state["results_dir"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# Research Report: {family}",
        f"\nGenerated: {ts}  |  Output: `{results_dir}`",
    ]

    try:
        import tabulate as _  # noqa: F401
        _render = lambda df: df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        _render = lambda df: df.to_string(index=False)

    _TABLE_ORDER = [
        ("sensitivity_specificity", "## Sensitivity–Specificity Trade-off"),
        ("ablation",                "## Ablation Study"),
        ("routing_distribution",    "## Routing Distribution"),
        ("calibration_by_path",     "## Calibration by Routing Path"),
        ("medgemma_agreement",      "## MedGemma–CNN Agreement Analysis"),
        ("calibration_per_task",    "## Calibration per Task"),
        ("per_class_failures",      "## Per-Class Failure Breakdown"),
    ]

    for key, heading in _TABLE_ORDER:
        if key in analysis and not analysis[key].empty:
            sections.append(f"\n{heading}\n")
            sections.append(_render(analysis[key]))

    report_md = "\n".join(sections)
    report_path = Path(results_dir) / "analysis" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md)
    print(f"\nReport written to {report_path}")

    return {"report_md": report_md}


# ── Graph assembly ─────────────────────────────────────────────────────────────


def build_research_pipeline():
    """
    Build and compile the 4-node linear research orchestrator graph.

    Returns a compiled LangGraph app that accepts ResearchState and produces
    a Markdown report plus structured analysis DataFrames.
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan_experiments", plan_experiments)
    workflow.add_node("run_experiments", run_experiments)
    workflow.add_node("analyze_results", analyze_results)
    workflow.add_node("write_report", write_report)

    workflow.set_entry_point("plan_experiments")
    workflow.add_edge("plan_experiments", "run_experiments")
    workflow.add_edge("run_experiments", "analyze_results")
    workflow.add_edge("analyze_results", "write_report")
    workflow.add_edge("write_report", END)

    return workflow.compile()
