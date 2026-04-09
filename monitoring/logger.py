"""
Structured JSON logger for the neuroimaging pipeline.

Replaces ad-hoc print() calls with rotating JSON log files.
Every inference emits a single structured record with all state fields.
"""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


LOG_VERSION = "1.0"
_LOGGER_NAME = "neuroimaging.pipeline"


class _JSONFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data"):
            payload["data"] = record.data
        return json.dumps(payload, default=str)


def configure_pipeline_logger(
    log_dir: str = "monitoring/logs",
    log_level: int = logging.INFO,
    also_stdout: bool = False,
) -> logging.Logger:
    """
    Configure and return the pipeline logger (idempotent).

    Args:
        log_dir:      Directory where rotating log files are written.
        log_level:    Logging level.
        also_stdout:  If True, also emit to stdout.

    Returns:
        logging.Logger named "neuroimaging.pipeline"
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(log_level)
    fmt = _JSONFormatter()

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "pipeline.log"
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.propagate = False
    return logger


def _extract_state_fields(state: dict) -> dict:
    """Pull the monitoring-relevant fields from a NeuroimagingState dict."""
    seg = state.get("segmentation_result") or {}
    cls = state.get("classification_result") or {}
    clip = state.get("biomedclip_result") or {}
    verif = state.get("verification_result") or {}

    return {
        "image_path": state.get("image_path"),
        "task": state.get("task"),
        "routing_decision": state.get("routing_decision"),
        "routing_confidence": state.get("routing_confidence"),
        "routing_path": " → ".join(state.get("routing_path") or []),
        "suspected_pathology": state.get("suspected_pathology"),
        "final_predicted_class": state.get("final_predicted_class"),
        "final_confidence": state.get("final_confidence"),
        "requires_human_review": state.get("requires_human_review", False),
        # CNN
        "cnn_predicted_class": cls.get("predicted_class"),
        "cnn_confidence": cls.get("confidence"),
        "cnn_temperature": cls.get("temperature"),
        # BiomedCLIP
        "clip_top_label": clip.get("top_label"),
        "clip_top_score": clip.get("top_score"),
        # SAM3
        "seg_dice_estimate": seg.get("dice_estimate"),
        "seg_mask_path": seg.get("mask_path"),
        # Explainability
        "saliency_sam3_iou": state.get("saliency_sam3_iou"),
        # Verification
        "verification_agreement": verif.get("agreement"),
        "verification_saliency_plausible": verif.get("saliency_plausible"),
    }


def log_inference(
    logger: logging.Logger,
    run_id: str,
    state_before: dict,
    state_after: dict,
    latency_s: float,
) -> None:
    """
    Emit one structured inference log record.

    Args:
        logger:       Configured logger instance.
        run_id:       UUID hex string for this run.
        state_before: Initial state dict (image_path, task).
        state_after:  Final NeuroimagingState dict after invoke().
        latency_s:    Wall-clock time for the full invoke() call.
    """
    fields = _extract_state_fields(state_after)
    fields["run_id"] = run_id
    fields["total_latency_s"] = round(latency_s, 4)
    fields["log_version"] = LOG_VERSION

    record = logger.makeRecord(
        _LOGGER_NAME,
        logging.INFO,
        fn="",
        lno=0,
        msg="inference_complete",
        args=(),
        exc_info=None,
    )
    record.data = fields
    logger.handle(record)


def log_node_event(
    logger: logging.Logger,
    run_id: str,
    node_name: str,
    event: str,
    payload: Optional[dict] = None,
) -> None:
    """
    Emit a node-level tracing event (enter / exit / error).

    Args:
        logger:    Configured logger instance.
        run_id:    UUID hex string for the current run.
        node_name: Name of the pipeline node (e.g. "triage").
        event:     One of "enter", "exit", "error".
        payload:   Optional extra data (latency_s, routing_decision, etc.).
    """
    data: dict[str, Any] = {
        "run_id": run_id,
        "node": node_name,
        "event": event,
    }
    if payload:
        data.update(payload)

    level = logging.ERROR if event == "error" else logging.DEBUG
    record = logger.makeRecord(
        _LOGGER_NAME,
        level,
        fn="",
        lno=0,
        msg=f"node_{event}",
        args=(),
        exc_info=None,
    )
    record.data = data
    logger.handle(record)
