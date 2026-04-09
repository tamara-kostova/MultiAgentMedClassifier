"""
SQLite persistence layer for inference results, alerts, and drift reports.

Uses stdlib sqlite3 — no SQLAlchemy dependency required.
Thread safety: create one PredictionStore instance per thread.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_log (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                          TEXT NOT NULL,
    timestamp                       TEXT NOT NULL,
    image_path                      TEXT NOT NULL,
    task                            TEXT NOT NULL,

    routing_decision                TEXT,
    routing_confidence              REAL,
    routing_path                    TEXT,
    suspected_pathology             TEXT,

    final_predicted_class           TEXT,
    final_confidence                REAL,
    requires_human_review           INTEGER,

    cnn_predicted_class             TEXT,
    cnn_confidence                  REAL,
    cnn_all_probs                   TEXT,
    cnn_temperature                 REAL,

    clip_top_label                  TEXT,
    clip_top_score                  REAL,

    seg_dice_estimate               REAL,
    seg_mask_path                   TEXT,

    saliency_sam3_iou               REAL,

    verification_agreement          INTEGER,
    verification_saliency_plausible INTEGER,

    total_latency_s                 REAL,

    ece                             REAL,
    is_correct                      INTEGER,
    true_label                      TEXT
);

CREATE TABLE IF NOT EXISTS alert_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    alert_type  TEXT NOT NULL,
    severity    TEXT NOT NULL,
    message     TEXT NOT NULL,
    value       REAL,
    threshold   REAL
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    window_start     TEXT,
    window_end       TEXT,
    n_current        INTEGER,
    n_reference      INTEGER,
    report_html_path TEXT,
    drift_detected   INTEGER,
    drift_score      REAL
);
"""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class PredictionStore:
    """
    Thin wrapper around a SQLite database for persisting inference results.

    All public methods are synchronous. The database file and parent
    directories are created automatically on first use.
    """

    def __init__(self, db_path: str = "monitoring/predictions.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── Writes ────────────────────────────────────────────────────────────────

    def log_inference(
        self,
        run_id: str,
        state_before: dict,
        state_after: dict,
        latency_s: float,
        true_label: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> int:
        """
        Insert one inference record. Returns the new row id.

        Extracts fields from state_after matching the inference_log schema.
        cnn_all_probs is JSON-serialized. None values are stored as NULL.
        """
        seg = state_after.get("segmentation_result") or {}
        cls = state_after.get("classification_result") or {}
        clip = state_after.get("biomedclip_result") or {}
        verif = state_after.get("verification_result") or {}

        routing_path = state_after.get("routing_path") or []
        if isinstance(routing_path, list):
            routing_path = " → ".join(routing_path)

        all_probs = cls.get("all_probs")
        all_probs_json = json.dumps(all_probs, default=str) if all_probs else None

        def _int_or_none(v):
            if v is None:
                return None
            return int(bool(v))

        row = (
            run_id,
            _now_iso(),
            state_after.get("image_path") or state_before.get("image_path", ""),
            state_after.get("task") or state_before.get("task", ""),
            state_after.get("routing_decision"),
            state_after.get("routing_confidence"),
            routing_path,
            state_after.get("suspected_pathology"),
            state_after.get("final_predicted_class"),
            state_after.get("final_confidence"),
            _int_or_none(state_after.get("requires_human_review")),
            cls.get("predicted_class"),
            cls.get("confidence"),
            all_probs_json,
            cls.get("temperature"),
            clip.get("top_label"),
            clip.get("top_score"),
            seg.get("dice_estimate"),
            seg.get("mask_path"),
            state_after.get("saliency_sam3_iou"),
            _int_or_none(verif.get("agreement")),
            _int_or_none(verif.get("saliency_plausible")),
            round(latency_s, 4),
            None,  # ece — filled later via batch job or eval
            _int_or_none(is_correct),
            true_label,
        )

        sql = """
            INSERT INTO inference_log (
                run_id, timestamp, image_path, task,
                routing_decision, routing_confidence, routing_path, suspected_pathology,
                final_predicted_class, final_confidence, requires_human_review,
                cnn_predicted_class, cnn_confidence, cnn_all_probs, cnn_temperature,
                clip_top_label, clip_top_score,
                seg_dice_estimate, seg_mask_path,
                saliency_sam3_iou,
                verification_agreement, verification_saliency_plausible,
                total_latency_s,
                ece, is_correct, true_label
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._connect() as conn:
            cur = conn.execute(sql, row)
            return cur.lastrowid

    def log_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        value: float,
        threshold: float,
    ) -> None:
        """Insert one alert record."""
        sql = """
            INSERT INTO alert_log (timestamp, alert_type, severity, message, value, threshold)
            VALUES (?,?,?,?,?,?)
        """
        with self._connect() as conn:
            conn.execute(sql, (_now_iso(), alert_type, severity, message, value, threshold))

    def log_drift_report(
        self,
        window_start: str,
        window_end: str,
        n_current: int,
        n_reference: int,
        report_html_path: str,
        drift_detected: bool,
        drift_score: float,
    ) -> None:
        """Insert one drift report record."""
        sql = """
            INSERT INTO drift_reports
                (timestamp, window_start, window_end, n_current, n_reference,
                 report_html_path, drift_detected, drift_score)
            VALUES (?,?,?,?,?,?,?,?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                _now_iso(), window_start, window_end,
                n_current, n_reference, report_html_path,
                int(drift_detected), drift_score,
            ))

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_recent_inferences(
        self,
        n: int = 500,
        task: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the n most recent inference records as a DataFrame."""
        if task:
            sql = """
                SELECT * FROM inference_log
                WHERE task = ?
                ORDER BY id DESC LIMIT ?
            """
            params = (task, n)
        else:
            sql = "SELECT * FROM inference_log ORDER BY id DESC LIMIT ?"
            params = (n,)

        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        return df

    def get_reference_window(self, n: int = 200) -> pd.DataFrame:
        """Return the oldest n records as the reference distribution."""
        sql = "SELECT * FROM inference_log ORDER BY id ASC LIMIT ?"
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=(n,))

    def get_alerts(self, n: int = 50) -> pd.DataFrame:
        """Return the n most recent alerts as a DataFrame."""
        sql = "SELECT * FROM alert_log ORDER BY id DESC LIMIT ?"
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=(n,))

    def get_drift_reports(self, n: int = 20) -> pd.DataFrame:
        """Return the n most recent drift report records."""
        sql = "SELECT * FROM drift_reports ORDER BY id DESC LIMIT ?"
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=(n,))

    def get_summary_stats(self, task: Optional[str] = None) -> dict:
        """
        Aggregate stats for the dashboard header.

        Returns dict with:
          total_inferences, human_review_rate, mean_confidence,
          mean_latency_s, sam3_invocation_rate, last_updated
        """
        where = "WHERE task = ?" if task else ""
        params = (task,) if task else ()
        sql = f"""
            SELECT
                COUNT(*)                                     AS total,
                AVG(requires_human_review)                   AS hr_rate,
                AVG(final_confidence)                        AS mean_conf,
                AVG(total_latency_s)                         AS mean_lat,
                AVG(CASE WHEN routing_path LIKE '%sam3%' THEN 1.0 ELSE 0.0 END) AS sam3_rate,
                MAX(timestamp)                               AS last_ts
            FROM inference_log {where}
        """
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()

        if row is None or row["total"] == 0:
            return {
                "total_inferences": 0,
                "human_review_rate": 0.0,
                "mean_confidence": 0.0,
                "mean_latency_s": 0.0,
                "sam3_invocation_rate": 0.0,
                "last_updated": "—",
            }
        return {
            "total_inferences": row["total"],
            "human_review_rate": round(row["hr_rate"] or 0.0, 4),
            "mean_confidence": round(row["mean_conf"] or 0.0, 4),
            "mean_latency_s": round(row["mean_lat"] or 0.0, 3),
            "sam3_invocation_rate": round(row["sam3_rate"] or 0.0, 4),
            "last_updated": row["last_ts"] or "—",
        }

    def count(self) -> int:
        """Total number of inference records."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM inference_log").fetchone()[0]
