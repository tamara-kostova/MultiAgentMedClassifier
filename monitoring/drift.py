"""
Evidently AI drift detection for the neuroimaging pipeline.

Compares a current prediction window against a reference baseline
and generates HTML reports for confidence drift, routing distribution
shift, and class prediction bias.

Usage:
    from monitoring.drift import DriftDetector
    from monitoring.store import PredictionStore

    store = PredictionStore("monitoring/predictions.db")
    ref_df = pd.read_csv("outputs/eval/all_predictions.csv")
    cur_df = store.get_recent_inferences(100)

    detector = DriftDetector(reports_dir="monitoring/reports")
    result = detector.run_full_report(cur_df, ref_df)
    print(result)

CLI:
    python -m monitoring.drift \\
        --db monitoring/predictions.db \\
        --reference outputs/eval/all_predictions.csv \\
        --out monitoring/reports/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DriftResult:
    drift_detected: bool
    drift_score: float          # highest per-column drift score (0.0–1.0)
    drifted_columns: list[str]  # columns where drift was flagged
    report_html_path: str       # absolute path to saved HTML report
    n_reference: int
    n_current: int
    timestamp: str

    def __str__(self) -> str:
        status = "DRIFT DETECTED" if self.drift_detected else "No drift"
        return (
            f"[{status}]  score={self.drift_score:.3f}  "
            f"n_ref={self.n_reference}  n_cur={self.n_current}  "
            f"drifted_cols={self.drifted_columns}\n"
            f"Report: {self.report_html_path}"
        )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _ts_tag() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


# Columns used for drift analysis
_NUMERIC_COLS = [
    "final_confidence",
    "routing_confidence",
    "saliency_sam3_iou",
    "seg_dice_estimate",
    "clip_top_score",
    "total_latency_s",
]
_CATEGORICAL_COLS = [
    "routing_decision",
    "final_predicted_class",
    "task",
]

# Column name mapping from eval CSV → inference_log SQLite
_EVAL_COL_MAP = {
    "predicted_class": "final_predicted_class",
    "latency_s": "total_latency_s",
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename eval CSV columns to match the SQLite schema."""
    return df.rename(columns=_EVAL_COL_MAP)


class DriftDetector:
    """
    Wraps Evidently AI to compare prediction windows.

    Reference baseline = either a CSV file (e.g. outputs/eval/all_predictions.csv)
    or the oldest N rows from the SQLite prediction store.
    """

    def __init__(
        self,
        reference_csv: Optional[str] = None,
        reports_dir: str = "monitoring/reports",
        drift_threshold: float = 0.5,
    ):
        self.reference_csv = reference_csv
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

    def _load_reference(self, store=None) -> pd.DataFrame:
        """Load reference dataframe from CSV or SQLite store."""
        if self.reference_csv and Path(self.reference_csv).exists():
            df = pd.read_csv(self.reference_csv)
            return _normalize_df(df)
        if store is not None:
            return store.get_reference_window(200)
        raise ValueError(
            "No reference data available. Provide reference_csv or a PredictionStore."
        )

    def run_confidence_drift(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> DriftResult:
        """
        Evidently DataDriftPreset on numeric columns.
        Saves HTML report to reports_dir/confidence_drift_{ts}.html.
        """
        return self._run_report(
            current_df,
            reference_df,
            columns=[c for c in _NUMERIC_COLS if c in current_df.columns and c in reference_df.columns],
            report_prefix="confidence_drift",
        )

    def run_routing_drift(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> DriftResult:
        """
        Evidently DataDriftPreset on categorical routing/prediction columns.
        """
        return self._run_report(
            current_df,
            reference_df,
            columns=[c for c in _CATEGORICAL_COLS if c in current_df.columns and c in reference_df.columns],
            report_prefix="routing_drift",
        )

    def run_full_report(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        store=None,
        task: Optional[str] = None,
    ) -> DriftResult:
        """
        Run all drift checks combined into one HTML report.

        Args:
            current_df:   Recent inference DataFrame from PredictionStore.
            reference_df: Baseline DataFrame. If None, loaded from reference_csv or store.
            store:        PredictionStore (used as fallback reference source).
            task:         Optional task filter applied to both DataFrames.
        """
        if reference_df is None:
            reference_df = self._load_reference(store)

        if task:
            current_df = current_df[current_df.get("task", pd.Series()) == task].copy()
            reference_df = reference_df[reference_df.get("task", pd.Series()) == task].copy()

        ref_norm = _normalize_df(reference_df)
        cur_norm = _normalize_df(current_df)

        all_cols = [
            c for c in _NUMERIC_COLS + _CATEGORICAL_COLS
            if c in cur_norm.columns and c in ref_norm.columns
        ]
        return self._run_report(cur_norm, ref_norm, all_cols, "full_drift")

    def _run_report(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        columns: list[str],
        report_prefix: str,
    ) -> DriftResult:
        """Core method: run Evidently report for a set of columns."""
        try:
            from evidently import ColumnMapping  # type: ignore
            from evidently.report import Report  # type: ignore
            from evidently.metric_preset import DataDriftPreset  # type: ignore
            from evidently.metrics import ColumnDriftMetric  # type: ignore
        except ImportError as e:
            raise ImportError(
                "evidently is required for drift detection. "
                "Install it with: pip install evidently"
            ) from e

        if not columns:
            raise ValueError("No common columns found between current and reference DataFrames.")

        # Evidently needs at least a few rows
        if len(current_df) < 5 or len(reference_df) < 5:
            report_path = self.reports_dir / f"{report_prefix}_insufficient_data.html"
            report_path.write_text(
                "<html><body><h2>Insufficient data for drift analysis</h2>"
                f"<p>current={len(current_df)}, reference={len(reference_df)}</p></body></html>"
            )
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drifted_columns=[],
                report_html_path=str(report_path.resolve()),
                n_reference=len(reference_df),
                n_current=len(current_df),
                timestamp=_now_iso(),
            )

        # Build column mapping
        col_map = ColumnMapping()

        # Build report with per-column drift metrics + overall preset
        metrics = [DataDriftPreset(columns=columns)]

        report = Report(metrics=metrics)

        cur_subset = current_df[columns].copy()
        ref_subset = reference_df[columns].copy()

        report.run(reference_data=ref_subset, current_data=cur_subset, column_mapping=col_map)

        # Save HTML
        ts = _ts_tag()
        html_path = self.reports_dir / f"{report_prefix}_{ts}.html"
        report.save_html(str(html_path))

        # Extract drift results
        result_dict = report.as_dict()
        drifted_cols = []
        max_drift_score = 0.0

        try:
            for metric_result in result_dict.get("metrics", []):
                res = metric_result.get("result", {})
                # DataDriftPreset result structure
                drift_by_col = res.get("drift_by_columns", {})
                for col, col_res in drift_by_col.items():
                    score = col_res.get("drift_score", 0.0) or 0.0
                    max_drift_score = max(max_drift_score, score)
                    if col_res.get("drift_detected", False):
                        drifted_cols.append(col)
        except Exception:
            pass  # parsing error — still return the HTML report

        return DriftResult(
            drift_detected=len(drifted_cols) > 0,
            drift_score=round(max_drift_score, 4),
            drifted_columns=drifted_cols,
            report_html_path=str(html_path.resolve()),
            n_reference=len(reference_df),
            n_current=len(current_df),
            timestamp=_now_iso(),
        )

    def detect_class_bias(
        self,
        current_df: pd.DataFrame,
        expected_class_freq: Optional[dict] = None,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Check for class prediction frequency bias.

        Computes the frequency of each predicted class in current_df.
        Compares against expected_class_freq (or reference_df distribution or uniform prior).

        Returns:
            {
                "bias_detected": bool,
                "dominant_class": str,
                "dominant_freq": float,
                "expected_freq": float,
                "class_frequencies": {class: freq, ...},
            }
        """
        col = "final_predicted_class"
        if col not in current_df.columns or current_df.empty:
            return {"bias_detected": False, "dominant_class": None, "dominant_freq": 0.0}

        counts = current_df[col].dropna().value_counts(normalize=True)
        if counts.empty:
            return {"bias_detected": False, "dominant_class": None, "dominant_freq": 0.0}

        dominant_class = counts.index[0]
        dominant_freq = float(counts.iloc[0])

        # Derive expected frequency
        if expected_class_freq:
            n_classes = len(expected_class_freq)
            expected = expected_class_freq.get(dominant_class, 1.0 / max(n_classes, 1))
        elif reference_df is not None and col in reference_df.columns:
            ref_counts = reference_df[col].dropna().value_counts(normalize=True)
            expected = float(ref_counts.get(dominant_class, 1.0 / max(len(ref_counts), 1)))
        else:
            n_classes = len(counts)
            expected = 1.0 / max(n_classes, 1)

        return {
            "bias_detected": dominant_freq > 0.80,
            "dominant_class": dominant_class,
            "dominant_freq": round(dominant_freq, 4),
            "expected_freq": round(float(expected), 4),
            "class_frequencies": {k: round(float(v), 4) for k, v in counts.items()},
        }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Run Evidently drift report")
    parser.add_argument("--db", default="monitoring/predictions.db")
    parser.add_argument("--reference", default=None, help="Path to reference CSV")
    parser.add_argument("--out", default="monitoring/reports")
    parser.add_argument("--task", default=None)
    parser.add_argument("--n_current", type=int, default=200)
    args = parser.parse_args()

    from monitoring.store import PredictionStore

    store = PredictionStore(args.db)
    current_df = store.get_recent_inferences(args.n_current, task=args.task)

    reference_df = None
    if args.reference and Path(args.reference).exists():
        reference_df = _normalize_df(pd.read_csv(args.reference))

    detector = DriftDetector(
        reference_csv=args.reference,
        reports_dir=args.out,
    )
    result = detector.run_full_report(current_df, reference_df=reference_df, store=store)
    print(result)

    # Persist to store
    ref_df = reference_df if reference_df is not None else store.get_reference_window(200)
    store.log_drift_report(
        window_start=current_df["timestamp"].min() if "timestamp" in current_df.columns else "",
        window_end=current_df["timestamp"].max() if "timestamp" in current_df.columns else "",
        n_current=result.n_current,
        n_reference=result.n_reference,
        report_html_path=result.report_html_path,
        drift_detected=result.drift_detected,
        drift_score=result.drift_score,
    )


if __name__ == "__main__":
    _cli()
