"""
Rule-based alert engine for the neuroimaging pipeline.

Evaluates a set of AlertRules against a recent window of predictions
from the PredictionStore. Fires alerts when thresholds are crossed,
with per-rule cooldowns to avoid alert storms.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class AlertRule:
    name: str               # e.g. "human_review_rate"
    severity: str           # "warning" | "critical"
    threshold: float
    condition: str          # "gt" | "lt"
    window_size: int = 50   # evaluate over last N inferences
    cooldown_minutes: int = 30


@dataclass
class FiredAlert:
    rule_name: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: str


_DEFAULT_RULES: list[AlertRule] = [
    AlertRule("human_review_rate",  "warning",  0.30, "gt", window_size=50,  cooldown_minutes=30),
    AlertRule("human_review_rate",  "critical", 0.50, "gt", window_size=50,  cooldown_minutes=15),
    AlertRule("mean_confidence",    "warning",  0.60, "lt", window_size=50,  cooldown_minutes=30),
    AlertRule("mean_confidence",    "critical", 0.45, "lt", window_size=50,  cooldown_minutes=15),
    AlertRule("ece",                "warning",  0.15, "gt", window_size=100, cooldown_minutes=60),
    AlertRule("ece",                "critical", 0.25, "gt", window_size=100, cooldown_minutes=30),
    AlertRule("class_bias",         "warning",  0.80, "gt", window_size=100, cooldown_minutes=60),
]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins (same implementation as eval/evaluate.py)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece)


class AlertEngine:
    """
    Evaluates AlertRules against recent prediction windows.

    Usage:
        engine = AlertEngine(store=store, logger=logger)
        fired = engine.evaluate(recent_df)
    """

    def __init__(
        self,
        rules: Optional[list[AlertRule]] = None,
        store=None,
        logger: Optional[logging.Logger] = None,
        notify_fn: Optional[Callable[[FiredAlert], None]] = None,
    ):
        self.rules = rules if rules is not None else _DEFAULT_RULES
        self.store = store
        self.logger = logger or logging.getLogger("neuroimaging.pipeline")
        self.notify_fn = notify_fn
        # {(rule_name, severity): last_fired_datetime_utc}
        self._last_fired: dict[tuple, datetime] = {}

    def evaluate(self, recent_df: pd.DataFrame) -> list[FiredAlert]:
        """
        Evaluate all rules against recent_df.
        Returns list of FiredAlert objects that fired this call.
        """
        if recent_df.empty:
            return []

        metrics = self._compute_metrics(recent_df)
        fired: list[FiredAlert] = []

        for rule in self.rules:
            if not self._cooldown_ok(rule):
                continue

            value = metrics.get(rule.name)
            if value is None:
                continue

            triggered = (rule.condition == "gt" and value > rule.threshold) or \
                        (rule.condition == "lt" and value < rule.threshold)

            if not triggered:
                continue

            direction = "above" if rule.condition == "gt" else "below"
            alert = FiredAlert(
                rule_name=rule.name,
                severity=rule.severity,
                message=(
                    f"[{rule.severity.upper()}] {rule.name} = {value:.4f} "
                    f"({direction} threshold {rule.threshold}) "
                    f"over last {rule.window_size} inferences."
                ),
                value=value,
                threshold=rule.threshold,
                timestamp=_now_iso(),
            )
            fired.append(alert)
            self._last_fired[(rule.name, rule.severity)] = datetime.now(tz=timezone.utc)

            if self.store:
                self.store.log_alert(
                    alert_type=rule.name,
                    severity=rule.severity,
                    message=alert.message,
                    value=value,
                    threshold=rule.threshold,
                )
            self.logger.warning(
                "alert_fired",
                extra={"data": {
                    "rule": rule.name, "severity": rule.severity,
                    "value": value, "threshold": rule.threshold,
                }},
            )
            if self.notify_fn:
                try:
                    self.notify_fn(alert)
                except Exception as e:
                    self.logger.error(f"notify_fn failed: {e}")

        return fired

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        """Compute aggregate metrics for alert evaluation."""
        metrics: dict[str, float] = {}

        # human review rate
        if "requires_human_review" in df.columns:
            metrics["human_review_rate"] = df["requires_human_review"].fillna(0).astype(float).mean()

        # mean confidence
        if "final_confidence" in df.columns:
            metrics["mean_confidence"] = df["final_confidence"].dropna().mean()

        # ECE (only when ground truth labels are available)
        if "final_confidence" in df.columns and "is_correct" in df.columns:
            labeled = df.dropna(subset=["final_confidence", "is_correct"])
            if len(labeled) >= 10:
                confs = labeled["final_confidence"].values.astype(float)
                correct = labeled["is_correct"].values.astype(float)
                metrics["ece"] = _compute_ece(confs, correct)

        # class bias: dominant predicted class frequency
        if "final_predicted_class" in df.columns:
            counts = df["final_predicted_class"].dropna().value_counts(normalize=True)
            if not counts.empty:
                metrics["class_bias"] = float(counts.iloc[0])

        return metrics

    def _cooldown_ok(self, rule: AlertRule) -> bool:
        """Returns True if the rule is not in cooldown."""
        last = self._last_fired.get((rule.name, rule.severity))
        if last is None:
            return True
        elapsed = datetime.now(tz=timezone.utc) - last
        return elapsed > timedelta(minutes=rule.cooldown_minutes)

    def check_routing_shift(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> Optional[FiredAlert]:
        """
        Chi-squared test on routing_decision distribution vs. reference.
        Returns a FiredAlert if the distribution has shifted significantly (p < 0.05),
        or None otherwise.
        """
        try:
            from scipy.stats import chisquare  # type: ignore
        except ImportError:
            return None

        if "routing_decision" not in current_df.columns or current_df.empty or reference_df.empty:
            return None

        categories = sorted(
            set(current_df["routing_decision"].dropna()) |
            set(reference_df["routing_decision"].dropna())
        )
        if not categories:
            return None

        cur_counts = current_df["routing_decision"].value_counts()
        ref_counts = reference_df["routing_decision"].value_counts()

        # Align to same categories
        cur_freq = np.array([cur_counts.get(c, 0) for c in categories], dtype=float)
        ref_freq = np.array([ref_counts.get(c, 0) for c in categories], dtype=float)

        if cur_freq.sum() == 0 or ref_freq.sum() == 0:
            return None

        # Scale reference to match current total
        expected = ref_freq / ref_freq.sum() * cur_freq.sum()
        expected = np.maximum(expected, 1e-6)  # avoid division by zero

        _, p_value = chisquare(cur_freq, f_exp=expected)
        if p_value < 0.05:
            return FiredAlert(
                rule_name="routing_shift",
                severity="warning",
                message=(
                    f"[WARNING] routing_decision distribution has shifted significantly "
                    f"from reference (chi-squared p={p_value:.4f} < 0.05). "
                    f"Current distribution: {dict(zip(categories, cur_freq.astype(int)))}."
                ),
                value=p_value,
                threshold=0.05,
                timestamp=_now_iso(),
            )
        return None


# ── Optional notifier factories ───────────────────────────────────────────────

def make_webhook_notifier(url: str) -> Callable[[FiredAlert], None]:
    """Returns a notify_fn that POSTs alert JSON to a webhook URL."""
    import urllib.request

    def notify(alert: FiredAlert) -> None:
        payload = json.dumps({
            "rule": alert.rule_name,
            "severity": alert.severity,
            "message": alert.message,
            "value": alert.value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp,
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)

    return notify


import json  # noqa: E402 (needed by make_webhook_notifier)
