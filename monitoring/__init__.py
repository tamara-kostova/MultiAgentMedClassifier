"""
MLOps monitoring layer for the neuroimaging pipeline.

The central integration point is MonitoringContext — a context manager
that wraps a single pipeline invocation with validation, logging,
persistence, and alerting. All monitoring is opt-in via --monitor.

Usage in run_pipeline.py:
    ctx = MonitoringContext(cfg.monitoring)
    with ctx.run(image_path, task):
        result = run_single(app, image_path, task, ...)
    ctx.finalize(state_before, result, latency_s)
"""

import time
import uuid
from contextlib import contextmanager
from typing import Generator, Optional

from monitoring.logger import configure_pipeline_logger, log_inference
from monitoring.store import PredictionStore
from monitoring.validator import MRIImageValidator, ValidationResult
from monitoring.alerts import AlertEngine


class MonitoringContext:
    """
    Wraps a single pipeline invocation with the full monitoring stack.

    Components activated on construction:
        - Structured JSON logger (monitoring/logs/pipeline.log)
        - SQLite prediction store (monitoring/predictions.db)
        - Input image validator (PIL-based, 10 checks)
        - Alert engine (8 default rules, evaluated after each run)

    The drift detector (Evidently) is triggered separately every
    cfg.drift_window_size inferences via trigger_drift_check().
    """

    def __init__(self, monitoring_cfg):
        """
        Args:
            monitoring_cfg: MonitoringConfig instance from config.py.
        """
        self.cfg = monitoring_cfg
        self.logger = configure_pipeline_logger(monitoring_cfg.log_dir)
        self.store = PredictionStore(monitoring_cfg.db_path)
        self.validator = MRIImageValidator() if monitoring_cfg.validate_inputs else None
        self.alert_engine = AlertEngine(store=self.store, logger=self.logger)

        self._run_id: Optional[str] = None
        self._t0: Optional[float] = None
        self._validation_result: Optional[ValidationResult] = None

    @contextmanager
    def run(self, image_path: str, task: str) -> Generator[str, None, None]:
        """
        Context manager for a single inference.

        Performs pre-run input validation and yields the run_id.
        Call finalize() after app.invoke() to persist results.

        Example:
            with ctx.run(image_path, task) as run_id:
                result = run_single(app, image_path, task, ...)
            ctx.finalize(state_before, result)
        """
        self._run_id = uuid.uuid4().hex
        self._t0 = time.perf_counter()

        # Input validation
        if self.validator:
            self._validation_result = self.validator.validate(image_path)
            if not self._validation_result.passed:
                msg = f"Input validation failed for {image_path}: {self._validation_result.failures}"
                self.logger.warning(msg)
                if self.cfg.validation_fail_action == "abort":
                    raise ValueError(msg)

        try:
            yield self._run_id
        except Exception:
            self.logger.exception(
                f"Pipeline error for run_id={self._run_id} image={image_path}"
            )
            raise

    def finalize(
        self,
        state_before: dict,
        state_after: dict,
        true_label: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> None:
        """
        Call after app.invoke() completes.

        Logs the inference record, persists to SQLite, evaluates alert rules,
        and optionally triggers a drift check.

        Args:
            state_before: Initial state dict (image_path, task).
            state_after:  Final NeuroimagingState dict returned by app.invoke().
            true_label:   Ground-truth class label (available in eval mode).
            is_correct:   Whether the prediction was correct (eval mode).
        """
        if self._t0 is None or self._run_id is None:
            return

        latency_s = time.perf_counter() - self._t0

        # Structured log
        log_inference(self.logger, self._run_id, state_before, state_after, latency_s)

        # Persist to SQLite
        self.store.log_inference(
            run_id=self._run_id,
            state_before=state_before,
            state_after=state_after,
            latency_s=latency_s,
            true_label=true_label,
            is_correct=is_correct,
        )

        # Alert evaluation (use last window_size rows)
        if self.cfg.enable_alerts:
            try:
                recent = self.store.get_recent_inferences(self.cfg.alert_window_size)
                self.alert_engine.evaluate(recent)
            except Exception as e:
                self.logger.warning(f"Alert evaluation failed: {e}")

        # Periodic drift check
        try:
            total = self.store.count()
            if total > 0 and total % self.cfg.drift_window_size == 0:
                self.trigger_drift_check()
        except Exception as e:
            self.logger.warning(f"Drift check trigger failed: {e}")

        # Reset run state
        self._run_id = None
        self._t0 = None
        self._validation_result = None

    def trigger_drift_check(self) -> None:
        """
        Run Evidently drift detection and persist the report.
        Called automatically every cfg.drift_window_size inferences,
        or manually by the dashboard "Refresh Drift Report" button.
        """
        try:
            from monitoring.drift import DriftDetector

            current_df = self.store.get_recent_inferences(self.cfg.drift_window_size)
            reference_df = None
            if self.cfg.reference_csv:
                import pandas as pd
                reference_df = pd.read_csv(self.cfg.reference_csv)

            detector = DriftDetector(
                reference_csv=self.cfg.reference_csv,
                reports_dir=self.cfg.reports_dir,
            )
            result = detector.run_full_report(
                current_df, reference_df=reference_df, store=self.store
            )

            self.store.log_drift_report(
                window_start=(
                    current_df["timestamp"].min()
                    if "timestamp" in current_df.columns and not current_df.empty
                    else ""
                ),
                window_end=(
                    current_df["timestamp"].max()
                    if "timestamp" in current_df.columns and not current_df.empty
                    else ""
                ),
                n_current=result.n_current,
                n_reference=result.n_reference,
                report_html_path=result.report_html_path,
                drift_detected=result.drift_detected,
                drift_score=result.drift_score,
            )

            if result.drift_detected:
                self.logger.warning(
                    "drift_detected",
                    extra={"data": {
                        "drifted_columns": result.drifted_columns,
                        "drift_score": result.drift_score,
                        "report": result.report_html_path,
                    }},
                )
        except Exception as e:
            self.logger.warning(f"Drift detection failed: {e}")
