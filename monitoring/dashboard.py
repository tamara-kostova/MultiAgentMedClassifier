"""
Streamlit monitoring dashboard for the neuroimaging pipeline.

Reads from the SQLite prediction store — no model imports, no GPU.

Launch:
    streamlit run monitoring/dashboard.py

Or with a custom database path:
    streamlit run monitoring/dashboard.py -- --db monitoring/predictions.db
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MultiAgentMedClassifier — MLOps",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CLI argument for db path (passed after `--`) ──────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default="monitoring/predictions.db")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

_cli_args = _parse_args()
DB_DEFAULT = _cli_args.db

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_inferences(db_path: str, n: int = 2000, task: str = "all") -> pd.DataFrame:
    """Load recent inferences from SQLite."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        if task != "all":
            df = pd.read_sql_query(
                "SELECT * FROM inference_log WHERE task=? ORDER BY id DESC LIMIT ?",
                conn, params=(task, n),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM inference_log ORDER BY id DESC LIMIT ?",
                conn, params=(n,),
            )
        conn.close()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df
    except Exception as e:
        st.error(f"Could not load database: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_alerts(db_path: str, n: int = 100) -> pd.DataFrame:
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT * FROM alert_log ORDER BY id DESC LIMIT ?", conn, params=(n,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_drift_reports(db_path: str) -> pd.DataFrame:
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT * FROM drift_reports ORDER BY id DESC LIMIT 20", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_summary_stats(db_path: str, task: str = "all") -> dict:
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        where = "WHERE task=?" if task != "all" else ""
        params = (task,) if task != "all" else ()
        row = conn.execute(f"""
            SELECT
                COUNT(*) AS total,
                AVG(requires_human_review) AS hr_rate,
                AVG(final_confidence) AS mean_conf,
                AVG(total_latency_s) AS mean_lat,
                AVG(CASE WHEN routing_path LIKE '%sam3%' THEN 1.0 ELSE 0.0 END) AS sam3_rate,
                MAX(timestamp) AS last_ts
            FROM inference_log {where}
        """, params).fetchone()
        conn.close()
        if row is None or row[0] == 0:
            return {"total_inferences": 0, "human_review_rate": 0.0,
                    "mean_confidence": 0.0, "mean_latency_s": 0.0,
                    "sam3_invocation_rate": 0.0, "last_updated": "—"}
        return {
            "total_inferences": row[0],
            "human_review_rate": round(row[1] or 0.0, 3),
            "mean_confidence": round(row[2] or 0.0, 3),
            "mean_latency_s": round(row[3] or 0.0, 2),
            "sam3_invocation_rate": round(row[4] or 0.0, 3),
            "last_updated": row[5] or "—",
        }
    except Exception:
        return {"total_inferences": 0, "human_review_rate": 0.0,
                "mean_confidence": 0.0, "mean_latency_s": 0.0,
                "sam3_invocation_rate": 0.0, "last_updated": "—"}


# ── KPI row ───────────────────────────────────────────────────────────────────

def render_kpi_row(stats: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("Total Inferences", stats["total_inferences"])
    cols[1].metric("Human Review Rate", f"{stats['human_review_rate']:.1%}")
    cols[2].metric("Mean Confidence", f"{stats['mean_confidence']:.3f}")
    cols[3].metric("Mean Latency", f"{stats['mean_latency_s']:.2f}s")
    cols[4].metric("SAM3 Rate", f"{stats['sam3_invocation_rate']:.1%}")
    cols[5].metric("Last Updated", str(stats["last_updated"])[:19] if stats["last_updated"] != "—" else "—")


# ── Tab 1: Overview ───────────────────────────────────────────────────────────

def render_overview_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No inference data yet. Run the pipeline with `--monitor` to populate.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Volume Over Time")
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            ts_df = df.set_index("timestamp").resample("1h").size().rename("count").reset_index()
            ts_df.columns = ["timestamp", "count"]
            st.line_chart(ts_df.set_index("timestamp")["count"])
        else:
            st.bar_chart(pd.Series([len(df)], index=["total"], name="Inferences"))

    with col2:
        st.subheader("Routing Decision Breakdown")
        if "routing_decision" in df.columns:
            counts = df["routing_decision"].dropna().value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Routing Distribution")
            st.pyplot(fig)
            plt.close(fig)


# ── Tab 2: Confidence & Calibration ──────────────────────────────────────────

def render_confidence_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confidence Distribution")
        if "final_confidence" in df.columns:
            task_options = ["all"] + sorted(df["task"].dropna().unique().tolist())
            selected_task = st.selectbox("Filter by task", task_options, key="conf_task")
            plot_df = df if selected_task == "all" else df[df["task"] == selected_task]
            confs = plot_df["final_confidence"].dropna()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(confs, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.axvline(confs.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={confs.mean():.3f}")
            ax.set_xlabel("Final Confidence")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Rolling Mean Confidence")
        if "final_confidence" in df.columns and len(df) >= 5:
            roll = df.sort_values("id")["final_confidence"].rolling(window=20, min_periods=1).mean()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(roll.values, color="#4C72B0")
            ax.axhline(0.60, color="orange", linestyle="--", linewidth=1, label="warn threshold 0.60")
            ax.axhline(0.45, color="red", linestyle="--", linewidth=1, label="critical threshold 0.45")
            ax.set_xlabel("Inference #")
            ax.set_ylabel("Confidence (rolling 20)")
            ax.legend(fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

    # Reliability diagram (requires ground truth)
    if "is_correct" in df.columns and "final_confidence" in df.columns:
        labeled = df.dropna(subset=["is_correct", "final_confidence"])
        if len(labeled) >= 20:
            st.subheader("Reliability Diagram (Calibration)")
            confs = labeled["final_confidence"].values
            correct = labeled["is_correct"].values.astype(float)
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            bin_accs, bin_confs, bin_counts = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (confs > lo) & (confs <= hi)
                if mask.sum() > 0:
                    bin_accs.append(correct[mask].mean())
                    bin_confs.append(confs[mask].mean())
                    bin_counts.append(mask.sum())

            if bin_accs:
                ece = sum(
                    (bin_counts[i] / len(confs)) * abs(bin_accs[i] - bin_confs[i])
                    for i in range(len(bin_accs))
                )
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#4C72B0", label="Model")
                ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Reliability Diagram  (ECE={ece:.4f})")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)


# ── Tab 3: Routing & Human Review ─────────────────────────────────────────────

def render_routing_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Routing Path Frequency")
        if "routing_decision" in df.columns:
            counts = df["routing_decision"].dropna().value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            counts.plot.bar(ax=ax, color="#4C72B0", edgecolor="white")
            ax.set_ylabel("Count")
            ax.set_xlabel("")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Human Review Rate (Rolling 20)")
        if "requires_human_review" in df.columns and len(df) >= 5:
            roll = df.sort_values("id")["requires_human_review"].astype(float).rolling(
                window=20, min_periods=1
            ).mean()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(roll.values, color="#DD8452")
            ax.axhline(0.30, color="orange", linestyle="--", linewidth=1, label="warn 30%")
            ax.axhline(0.50, color="red", linestyle="--", linewidth=1, label="critical 50%")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Inference #")
            ax.set_ylabel("Human Review Rate")
            ax.legend(fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Recent Cases Flagged for Human Review")
    if "requires_human_review" in df.columns:
        flagged = df[df["requires_human_review"].astype(bool)].sort_values("id", ascending=False)
        if flagged.empty:
            st.success("No cases currently flagged.")
        else:
            display_cols = [c for c in ["timestamp", "image_path", "task",
                                         "final_predicted_class", "final_confidence",
                                         "routing_decision"] if c in flagged.columns]
            st.dataframe(flagged[display_cols].head(20), use_container_width=True)


# ── Tab 4: Drift Analysis ─────────────────────────────────────────────────────

def render_drift_tab(df: pd.DataFrame, db_path: str) -> None:
    st.subheader("Drift Detection")

    drift_reports_df = load_drift_reports(db_path)

    col1, col2 = st.columns([2, 1])
    with col1:
        reference_csv = st.text_input(
            "Reference CSV path (outputs/eval/all_predictions.csv)",
            value="outputs/eval/all_predictions.csv",
        )
        if st.button("Run Drift Report"):
            if df.empty:
                st.warning("No current data in store.")
            else:
                with st.spinner("Running Evidently drift analysis..."):
                    try:
                        from monitoring.drift import DriftDetector
                        from monitoring.store import PredictionStore

                        ref_df = None
                        if Path(reference_csv).exists():
                            ref_df = pd.read_csv(reference_csv)

                        store = PredictionStore(db_path)
                        detector = DriftDetector(
                            reference_csv=reference_csv if Path(reference_csv).exists() else None,
                            reports_dir="monitoring/reports",
                        )
                        result = detector.run_full_report(df, reference_df=ref_df, store=store)
                        store.log_drift_report(
                            window_start=str(df["timestamp"].min()) if "timestamp" in df.columns else "",
                            window_end=str(df["timestamp"].max()) if "timestamp" in df.columns else "",
                            n_current=result.n_current,
                            n_reference=result.n_reference,
                            report_html_path=result.report_html_path,
                            drift_detected=result.drift_detected,
                            drift_score=result.drift_score,
                        )
                        st.cache_data.clear()
                        if result.drift_detected:
                            st.error(
                                f"Drift detected! Score={result.drift_score:.3f}  "
                                f"Columns: {result.drifted_columns}"
                            )
                        else:
                            st.success(f"No drift detected. Score={result.drift_score:.3f}")
                    except ImportError:
                        st.error("evidently is not installed. Run: pip install evidently")
                    except Exception as e:
                        st.error(f"Drift detection failed: {e}")

    with col2:
        if not drift_reports_df.empty:
            st.caption("Recent drift reports")
            st.dataframe(
                drift_reports_df[["timestamp", "drift_detected", "drift_score", "n_current"]].head(10),
                use_container_width=True,
            )

    # Class bias chart
    st.subheader("Predicted Class Frequencies")
    if "final_predicted_class" in df.columns and not df.empty:
        counts = df["final_predicted_class"].dropna().value_counts(normalize=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["#e74c3c" if v > 0.8 else "#4C72B0" for v in counts.values]
        counts.plot.barh(ax=ax, color=colors)
        ax.axvline(0.8, color="red", linestyle="--", linewidth=1, label="bias threshold 80%")
        ax.set_xlabel("Frequency")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        if counts.iloc[0] > 0.8:
            st.warning(
                f"Class bias detected: '{counts.index[0]}' accounts for "
                f"{counts.iloc[0]:.1%} of predictions."
            )

    # Embed latest Evidently HTML report
    if not drift_reports_df.empty:
        latest_path = drift_reports_df.iloc[0]["report_html_path"]
        if latest_path and Path(str(latest_path)).exists():
            st.subheader("Latest Evidently Report")
            html_content = Path(str(latest_path)).read_text(encoding="utf-8")
            components.html(html_content, height=600, scrolling=True)


# ── Tab 5: Alerts ─────────────────────────────────────────────────────────────

def render_alerts_tab(db_path: str) -> None:
    alerts_df = load_alerts(db_path)
    st.subheader("Recent Alerts")

    if alerts_df.empty:
        st.success("No alerts fired.")
        return

    # Colour code by severity
    def _highlight(row):
        color = "#ffeeba" if row.get("severity") == "warning" else "#f8d7da"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        alerts_df[["timestamp", "alert_type", "severity", "message", "value", "threshold"]],
        use_container_width=True,
    )

    # Summary counts
    col1, col2 = st.columns(2)
    warn_count = (alerts_df["severity"] == "warning").sum()
    crit_count = (alerts_df["severity"] == "critical").sum()
    col1.metric("Warning Alerts", int(warn_count))
    col2.metric("Critical Alerts", int(crit_count), delta_color="inverse")


# ── Tab 6: Per-Task Accuracy ──────────────────────────────────────────────────

def render_accuracy_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available.")
        return

    if "is_correct" not in df.columns or df["is_correct"].isna().all():
        st.info(
            "Ground-truth labels not available. "
            "Run `--eval` with `--monitor` to populate accuracy metrics."
        )
        return

    labeled = df.dropna(subset=["is_correct", "task"])
    if labeled.empty:
        st.info("No labeled data yet.")
        return

    st.subheader("Per-Task Accuracy")
    task_acc = labeled.groupby("task")["is_correct"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    task_acc.plot.bar(ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="green", linestyle="--", linewidth=1, label="90% target")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    plt.close(fig)

    # Confusion matrix per task
    if "final_predicted_class" in df.columns and "true_label" in df.columns:
        task_options = sorted(labeled["task"].dropna().unique().tolist())
        selected = st.selectbox("Confusion matrix for task", task_options, key="cm_task")
        sub = labeled[labeled["task"] == selected].dropna(subset=["true_label", "final_predicted_class"])
        if len(sub) >= 5:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # type: ignore
            classes = sorted(sub["true_label"].unique())
            cm = confusion_matrix(sub["true_label"], sub["final_predicted_class"], labels=classes)
            fig, ax = plt.subplots(figsize=(max(4, len(classes)), max(3, len(classes))))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot(ax=ax, colorbar=False)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🧠 MultiAgentMedClassifier — MLOps Dashboard")

    # Sidebar
    st.sidebar.header("Settings")
    db_path = st.sidebar.text_input("Database path", DB_DEFAULT)
    task_filter = st.sidebar.selectbox(
        "Task filter", ["all", "binary_tumor", "multiclass_tumor", "ms", "stroke"]
    )
    n_rows = st.sidebar.slider("Max rows to load", 100, 5000, 1000, step=100)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")

    # KPI
    stats = get_summary_stats(db_path, task=task_filter)
    render_kpi_row(stats)
    st.divider()

    # Load data
    df = load_inferences(db_path, n=n_rows, task=task_filter)

    # Tabs
    tabs = st.tabs([
        "Overview",
        "Confidence & Calibration",
        "Routing & Human Review",
        "Drift Analysis",
        "Alerts",
        "Per-Task Accuracy",
    ])
    with tabs[0]:
        render_overview_tab(df)
    with tabs[1]:
        render_confidence_tab(df)
    with tabs[2]:
        render_routing_tab(df)
    with tabs[3]:
        render_drift_tab(df, db_path)
    with tabs[4]:
        render_alerts_tab(db_path)
    with tabs[5]:
        render_accuracy_tab(df)

    if auto_refresh:
        import time
        time.sleep(30)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
