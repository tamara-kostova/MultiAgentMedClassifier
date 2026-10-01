"""
Evaluation protocol for the multi-agent neuroimaging pipeline.

Compares three configurations on all four datasets:
  1. Baseline:       Best CNN per task from cnns.tex (no agent routing)
  2. Static pipeline: SAM3→CNN always (from sam3_pipeline.tex)
  3. Agent pipeline:  LangGraph MedGemma router → optimal specialist

Metrics:
  Standard:  Accuracy, F1 (macro), ROC-AUC
  Agent-specific:
    - Routing correctness: % cases where MedGemma routing matches oracle-optimal path
    - Specificity on normal scans (key: does routing fix the 67.1%→41.3% collapse?)
    - SAM3 invocation rate: what % of cases get routed to segmentation?
    - Latency per routing path
"""

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from config import DEFAULT_CONFIG, TASKS
from eval.labels import canonical_label
from eval.metrics import compute_ece  # noqa: F401  (re-exported: `from eval.evaluate import compute_ece`)
from eval.tumor_eval import (  # read-only shared constants
    IMAGE_EXTENSIONS,
    LABEL_MAPS,
    TASK_DEFAULT_LABEL_MAP,
    _DEFAULT_EXCLUDE_DIRS,
)
from pipeline.state import NeuroimagingState, initial_state


def _canonical_true_label(raw_label: str, task: str) -> str:
    """
    Map a raw dataset folder name to its canonical evaluation label, mirroring
    the rich-JSONL path in eval/tumor_eval.py: apply the task's default label
    map (e.g. figshare "1"→meningioma, Br35H "yes"→brain tumor) first, then
    canonicalize (collapses wording/case, folds subtypes to "tumor" for the
    binary task, etc.). Without this, metrics compare model outputs like
    "meningioma" against raw folder names like "1" and score near-zero.
    """
    label_map = LABEL_MAPS.get(TASK_DEFAULT_LABEL_MAP.get(task, ""), {})
    mapped = label_map.get(raw_label, raw_label)
    return canonical_label(mapped, task)

ConfigName = Literal["baseline_cnn", "static_sam3_cnn", "agent_pipeline"]


# ── Calibration utilities ──────────────────────────────────────────────────────


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for CNN logits (Guo et al., 2017).

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(logits_val, labels_val)         # calibrate on held-out val set
        calibrated_probs = scaler.calibrate(logits_test)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "TemperatureScaler":
        """Optimise temperature via NLL on the validation set."""
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        self.eval()
        print(
            f"[TemperatureScaler] T = {self.temperature.item():.4f}  "
            f"(T>1 ⟹ was over-confident, T<1 ⟹ was under-confident)"
        )
        return self

    def calibrate(self, logits: torch.Tensor) -> np.ndarray:
        """Return calibrated probabilities as a numpy array."""
        with torch.no_grad():
            return torch.softmax(self.forward(logits), dim=-1).cpu().numpy()


# ── Oracle routing labels for routing-correctness metric ─────────────────────
# Based on CNN confidence on held-out data: if CNN confidence < threshold,
# SAM3-guided path would have been optimal; else direct CNN is optimal.
# These are computed empirically during evaluation.


def compute_oracle_routing(cnn_result: dict, routing_cfg=None) -> str:
    """Assign ground-truth optimal route based on CNN output."""
    cfg = routing_cfg or DEFAULT_CONFIG.routing
    conf = cnn_result["confidence"]
    task_needs_fine_grain = cnn_result.get("task") == "multiclass_tumor"

    if conf < cfg.human_review_threshold:
        return "human_review"
    elif conf < cfg.sam3_threshold and not task_needs_fine_grain:
        return "sam3_then_cnn"
    elif conf < cfg.biomedclip_rerank_threshold and task_needs_fine_grain:
        return "biomedclip"
    else:
        return "cnn_direct"


# ── Dataset loader ────────────────────────────────────────────────────────────


def load_test_split(dataset_dir: str, task: str) -> list[dict]:
    """
    Load test split from a directory structured as:
        <dataset_dir>/<class_name>/<image_file>

    Non-recursive (images must sit directly in the class folder). Skips class
    folders named in eval.tumor_eval._DEFAULT_EXCLUDE_DIRS (e.g. stroke's
    lesion-painted OVERLAY) or ending in "_mask", matches extensions
    case-insensitively (.png/.jpg/.jpeg), and returns a sorted list.

    Returns list of {"image_path": str, "label": str, "task": str}
    """
    dataset_path = Path(dataset_dir)
    samples = []
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.endswith("_mask") or class_dir.name in _DEFAULT_EXCLUDE_DIRS:
            continue
        for img_file in sorted(class_dir.iterdir()):
            if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if img_file.stem.endswith("_mask"):
                continue
            samples.append(
                {
                    "image_path": str(img_file),
                    "label": class_dir.name,
                    "task": task,
                }
            )
    return sorted(samples, key=lambda s: (s["label"], s["image_path"]))


# ── Single-configuration evaluator ───────────────────────────────────────────


class PipelineEvaluator:
    """Runs a compiled LangGraph app over a test set and collects metrics."""

    def __init__(self, app, config_name: ConfigName):
        self.app = app
        self.config_name = config_name

    def run(self, samples: list[dict], output_path: str | Path) -> pd.DataFrame:
        """
        Run the pipeline on all samples and stream per-sample results to JSONL.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                print(
                    f"  [{i+1}/{len(samples)}] {Path(sample['image_path']).name}",
                    end="\r",
                )
                state = initial_state(sample["image_path"], sample["task"])

                t0 = time.perf_counter()
                final_state: NeuroimagingState = self.app.invoke(state)
                latency = time.perf_counter() - t0

                predicted_class = final_state.get("final_predicted_class", "unknown")
                row = {
                    "image_path": sample["image_path"],
                    "true_label": sample["label"],
                    "task": sample["task"],
                    "predicted_class": predicted_class,
                    # Canonical labels so metrics survive folder-name vs model-output
                    # wording differences (see _canonical_true_label). compute_metrics
                    # and the sweep analysis functions prefer these when present.
                    "true_label_canonical": _canonical_true_label(
                        sample["label"], sample["task"]
                    ),
                    "predicted_class_canonical": canonical_label(
                        predicted_class, sample["task"]
                    ),
                    "final_confidence": final_state.get("final_confidence", 0.0),
                    "routing_decision": final_state.get("routing_decision", "unknown"),
                    "routing_confidence": final_state.get("routing_confidence", 0.0),
                    "routing_path": " → ".join(final_state.get("routing_path", [])),
                    "requires_human_review": final_state.get(
                        "requires_human_review", False
                    ),
                    "latency_s": latency,
                    "config": self.config_name,
                    "medgemma_class": (
                        final_state["medgemma_diagnosis"].get("diagnosis_name")
                        if final_state.get("medgemma_diagnosis") else None
                    ),
                    # multiclass_tumor scores the subtype (diagnosis_detailed)
                    "medgemma_class_detailed": (
                        final_state["medgemma_diagnosis"].get("diagnosis_detailed")
                        if final_state.get("medgemma_diagnosis") else None
                    ),
                    # Forest fields (None in standard / debate pipelines)
                    "dissent_rate": (
                        final_state["forest_consensus"].get("dissent_rate")
                        if final_state.get("forest_consensus") else None
                    ),
                    "vote_fraction": (
                        final_state["forest_consensus"].get("vote_fraction")
                        if final_state.get("forest_consensus") else None
                    ),
                    # Debate fields (None in standard / forest pipelines)
                    "debate_rounds_completed": final_state.get("debate_rounds_completed"),
                    "debate_round_changed": (
                        final_state["debate_verdict"].get("round_changed")
                        if final_state.get("debate_verdict") else None
                    ),
                    # Per-round verdicts when the debate agent exposes them (newer
                    # versions); analysis prefers these over the judge's own flag.
                    "debate_round_verdicts": (
                        final_state.get("debate_round_verdicts")
                        or (final_state.get("debate_verdict") or {}).get("round_verdicts")
                    ),
                }
                f.write(json.dumps(row) + "\n")

        print()
        return pd.read_json(output_path, lines=True)


# ── Metric computation ────────────────────────────────────────────────────────


def score_frame(df: pd.DataFrame, task: str | None = None) -> pd.DataFrame:
    """Per-row scoring identical to the paper path (eval_analysis strict scoring).

    Adds columns: _true (prep'd canonical truth), _pred (prep'd canonical
    prediction, "unknown" for an abstention), _correct (abstain = wrong) and
    _conf (final_confidence; only a missing value becomes 0.5).
    """
    from eval.eval_analysis import prep  # lazy: keeps matplotlib out of module import

    out = df.copy()
    tasks = out["task"] if "task" in out.columns else pd.Series([task] * len(out), index=out.index)
    if task is not None:
        tasks = pd.Series([task] * len(out), index=out.index)

    def _true(row, t):
        raw = row.get("true_label_canonical")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
            raw = _canonical_true_label(str(row.get("true_label", "")), t)
        return prep(canonical_label(raw, t), t)

    def _pred(row, t):
        raw = row.get("predicted_class")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raw = row.get("predicted_class_canonical")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raw = ""
        lbl = canonical_label(raw, t)
        return prep(lbl, t) if lbl else "unknown"

    out["_true"] = [_true(r, t) for (_, r), t in zip(out.iterrows(), tasks)]
    out["_pred"] = [_pred(r, t) for (_, r), t in zip(out.iterrows(), tasks)]
    out["_correct"] = ((out["_true"] == out["_pred"]) & (out["_pred"] != "unknown")).astype(float)
    conf = pd.to_numeric(out.get("final_confidence"), errors="coerce") if "final_confidence" in out else None
    out["_conf"] = np.clip(conf.fillna(0.5).values, 0, 1) if conf is not None else 0.5
    return out


def compute_metrics(df: pd.DataFrame, task: str | None = None) -> dict:
    """
    Compute standard and agent-specific metrics from a results DataFrame.

    Scoring is the paper path's: eval.labels.canonical_label + eval_analysis.prep
    strict binarization (binary tasks collapse to pos|normal; naming another
    pathology is a negative; "other abnormalities" -> "abnormal" is NOT "normal").
    Abstentions (empty/unknown prediction) count as wrong. Accuracy, F1, ROC-AUC,
    specificity and ECE all use the same per-row predictions.
    """
    if task is None and "task" in df.columns and len(df):
        task = str(df["task"].iloc[0])
    sc = score_frame(df, task)
    sc = sc[sc["_true"] != "unknown"]
    y_true = sc["_true"].tolist()
    y_pred = sc["_pred"].tolist()
    classes = sorted(set(y_true))
    labels = sorted(set(y_true) | (set(y_pred) - {"unknown"}))

    metrics = {
        "n_scored": len(sc),
        "n_abstained": int((sc["_pred"] == "unknown").sum()),
        "accuracy": float(sc["_correct"].mean()) if len(sc) else float("nan"),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }

    # ROC-AUC (binary tasks): positive-class score from final_confidence of the
    # same canonical prediction; an abstention scores 0.5.
    if len(classes) == 2:
        positive_class = (
            [c for c in classes if c != "normal"][0] if "normal" in classes else classes[1]
        )
        y_scores = [
            c if p == positive_class else (1 - c if p != "unknown" else 0.5)
            for p, c in zip(sc["_pred"], sc["_conf"])
        ]
        y_binary = [1 if t == positive_class else 0 for t in y_true]
        try:
            metrics["roc_auc"] = roc_auc_score(y_binary, y_scores)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    # ── Agent-specific metrics ────────────────────────────────────────────────

    # Specificity on normal scans: canonical "normal" only ("abnormal" is not normal).
    normal_mask = sc["_true"] == "normal"
    if normal_mask.any():
        metrics["normal_specificity"] = float((sc.loc[normal_mask, "_pred"] == "normal").mean())

    # SAM3 invocation rate
    if "routing_path" in df.columns:
        metrics["sam3_invocation_rate"] = (df["routing_path"].fillna("").str.contains("sam3")).mean()

    # Human review rate
    if "requires_human_review" in df.columns:
        from eval.metrics import parse_bool
        metrics["human_review_rate"] = float(np.mean(
            [parse_bool(v) is True for v in df["requires_human_review"]]))

    # Average latency per path
    if "latency_s" in df.columns:
        metrics["mean_latency_s"] = df["latency_s"].mean()
        if "routing_decision" in df.columns:
            metrics["latency_by_path"] = (
                df.groupby("routing_decision")["latency_s"].mean().to_dict()
            )

    # Calibration: ECE (eval.metrics, first bin [0, 0.1]) on the same predictions
    confidences = np.asarray(sc["_conf"], dtype=float)
    correct = sc["_correct"].values.astype(float)
    if len(sc):
        metrics["mean_confidence"] = float(confidences.mean())
        metrics["mean_accuracy"] = float(correct.mean())
        from eval.metrics import check_ece_bound
        metrics["ece"] = check_ece_bound(confidences, correct)
        # Keep the old MAE-based approximation for backwards-compatibility comparison
        metrics["ece_approx"] = float(np.abs(confidences - correct).mean())

    return metrics


# ── Multi-configuration comparison ───────────────────────────────────────────


def compare_configurations(
    configs: dict,  # {"config_name": compiled_app}
    test_datasets: dict,  # {"task": [{"image_path": str, "label": str}, ...]}
    output_dir: str = "outputs/eval",
) -> pd.DataFrame:
    """
    Run all configurations on all datasets and produce a comparison table.

    Args:
        configs: dict of config_name → compiled LangGraph app
        test_datasets: dict of task → list of test samples
        output_dir: where to save results CSV and JSON

    Returns:
        DataFrame with one row per (config, task) combination.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_results_df = []

    for config_name, app in configs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print("=" * 60)
        evaluator = PipelineEvaluator(app, config_name)

        for task, samples in test_datasets.items():
            print(f"\n  Task: {task} ({len(samples)} samples)")
            task_output_path = Path(output_dir) / f"{config_name}_{task}_predictions.jsonl"
            results_df = evaluator.run(samples, task_output_path)
            all_results_df.append(results_df)

            metrics = compute_metrics(results_df, task)

            row = {
                "config": config_name,
                "task": task,
                "n_samples": len(samples),
                "n_scored": metrics.get("n_scored", len(samples)),
                "n_abstained": metrics.get("n_abstained", 0),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics.get("roc_auc", float("nan")),
                "normal_specificity": metrics.get("normal_specificity", float("nan")),
                "sam3_invocation_rate": metrics.get("sam3_invocation_rate", 0.0),
                "human_review_rate": metrics.get("human_review_rate", 0.0),
                "mean_latency_s": metrics.get("mean_latency_s", 0.0),
                "ece": metrics.get("ece", float("nan")),
                "ece_approx": metrics.get("ece_approx", float("nan")),
            }
            all_rows.append(row)
            print(
                f"    Acc={row['accuracy']:.3f}  F1={row['f1_macro']:.3f}  "
                f"Spec={row['normal_specificity']:.3f}  "
                f"SAM3-rate={row['sam3_invocation_rate']:.2f}  "
                f"Latency={row['mean_latency_s']:.2f}s"
            )

            # Save per-config per-task classification report
            report_path = Path(output_dir) / f"{config_name}_{task}_report.txt"
            report_path.write_text(metrics["classification_report"])

    summary = pd.DataFrame(all_rows)
    summary_path = Path(output_dir) / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Save all raw predictions
    all_preds = pd.concat(all_results_df, ignore_index=True)
    all_preds.to_csv(Path(output_dir) / "all_predictions.csv", index=False)

    return summary


# ── Quick single-image demo ───────────────────────────────────────────────────


def run_single(
    app,
    image_path: str,
    task: str,
    verbose: bool = True,
    output_dir: str | None = None,
    save_output: bool = False,
) -> NeuroimagingState:
    state = initial_state(image_path, task)
    result = app.invoke(state)
    route = " → ".join(result.get("routing_path", []))
    # ── Generated files section ───────────────────────────────────────
    files_lines = []

    seg = result.get("segmentation_result") or {}
    if seg.get("mask_path"):
        files_lines.append(f"  segmentation mask:   {seg['mask_path']}")
    if seg.get("guided_image_path"):
        files_lines.append(f"  segmentation guided: {seg['guided_image_path']}")

    expl = result.get("explainability_result") or {}
    if expl.get("gradcam_pp"):
        files_lines.append(f"  gradcam++:           {expl['gradcam_pp']}")
    if expl.get("integrated_gradients"):
        files_lines.append(f"  integrated grads:    {expl['integrated_gradients']}")

    fhir = result.get("fhir_report") or {}

    fhir_bundle_id = fhir.get("id")
    if fhir_bundle_id:
        files_lines.append(f"  FHIR Bundle:           {fhir_bundle_id}")

    fhir_report_id = None
    for entry in fhir.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "DiagnosticReport":
            fhir_report_id = resource.get("id")
            break

    if fhir_report_id:
        files_lines.append(f"  FHIR DiagnosticReport: {fhir_report_id}")

    if output_dir and fhir_report_id:
        fhir_dir = Path(output_dir) / "fhir"

        fhir_path = next(
            (str(p) for p in fhir_dir.glob(f"fhir_report-{fhir_report_id[:5]}*.json")),
            None,
        )
        if fhir_path:
            files_lines.append(f"  FHIR JSON:             {fhir_path}")

    files_section = ("\nGenerated files:\n" + "\n".join(files_lines)) if files_lines else ""

    # ── IoU line ──────────────────────────────────────────────────────
    iou = result.get("saliency_sam3_iou")
    iou_line = f"GradCAM++/SAM3 IoU: {iou:.3f}\n" if iou is not None else ""

    review_flag = '⚠  FLAGGED FOR HUMAN REVIEW\n' if result.get('requires_human_review') else ''

    summary = (
        f"{'='*60}\n"
        f"Image: {image_path}\n"
        f"Task:  {task}\n"
        f"Route: {route}\n"
        f"Prediction: {result.get('final_predicted_class')} "
        f"(conf={result.get('final_confidence', 0):.3f})\n"
        f"{iou_line}"
        f"{review_flag}"
        f"\nReport:\n{result.get('final_report', 'N/A')}\n"
        f"{files_section}\n"
        f"{'='*60}"
    )

    if verbose:
        print(f"\n{summary}")

    if save_output and output_dir:
        report_dir = Path(output_dir) / "single_runs"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{Path(image_path).stem}_{task}_report.txt"
        report_path.write_text(summary + "\n", encoding="utf-8")
        print(f"[run_single] Saved report to {report_path}")

    return result
