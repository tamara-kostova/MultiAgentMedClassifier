"""
Analysis utilities for sweep results.

Each function takes DataFrames produced by runner.py and returns a
thesis-ready DataFrame (suitable for to_markdown() or to_latex()).

Functions:
  routing_distribution         — routing decision counts/pct per task & experiment
  sensitivity_specificity_table — threshold → specificity + sam3_rate + accuracy
  calibration_by_routing_path  — ECE per routing path per task & experiment
  ablation_summary             — component contribution pivot table
  per_class_failure_breakdown  — per-class precision/recall/F1 and error rates
  load_sweep_predictions       — merge all_predictions.csv from all sweep subdirs
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from eval.evaluate import compute_ece


# ── Loader ────────────────────────────────────────────────────────────────────


def load_sweep_predictions(results_dir: str) -> pd.DataFrame:
    """
    Load and merge all_predictions.csv from every sweep-point subdirectory.
    Adds 'experiment_id' column inferred from subdirectory name.
    """
    dfs = []
    for subdir in sorted(Path(results_dir).iterdir()):
        pred_file = subdir / "all_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            df["experiment_id"] = subdir.name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Analysis functions ────────────────────────────────────────────────────────


def routing_distribution(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Routing decision counts and share (%) per experiment_id, task, and routing_decision.

    Use this to show that the routing logic is active and task-dependent.
    """
    counts = (
        preds_df.groupby(["experiment_id", "task", "routing_decision"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(["experiment_id", "task"])["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)
    return counts.sort_values(["experiment_id", "task", "routing_decision"])


def sensitivity_specificity_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics across tasks per experiment point.

    Returns one row per experiment_id with columns:
      accuracy, f1_macro, normal_specificity, sam3_invocation_rate,
      human_review_rate, ece, and any override_* columns.

    Use this to show the sensitivity–specificity trade-off when sweeping thresholds.
    """
    metric_cols = [
        "accuracy", "f1_macro", "normal_specificity",
        "sam3_invocation_rate", "human_review_rate", "ece",
    ]
    available = ["experiment_id"] + [c for c in metric_cols if c in summary_df.columns]
    grouped = (
        summary_df[available]
        .groupby("experiment_id")
        .mean(numeric_only=True)
        .reset_index()
    )

    # Attach override columns (one value per experiment_id — same across tasks)
    override_cols = [c for c in summary_df.columns if c.startswith("override_")]
    if override_cols:
        overrides = (
            summary_df[["experiment_id", "description"] + override_cols]
            .drop_duplicates("experiment_id")
        )
        grouped = grouped.merge(overrides, on="experiment_id", how="left")

    return grouped.sort_values("experiment_id").reset_index(drop=True)


def calibration_by_routing_path(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    ECE per routing path for each (experiment_id, task) combination.

    Requires at least 5 samples per path cell; cells with fewer are skipped.

    Use this to show whether SAM3-routed cases are better or worse calibrated
    than directly-classified cases.
    """
    rows = []
    for (exp_id, task, path), grp in preds_df.groupby(
        ["experiment_id", "task", "routing_decision"]
    ):
        if len(grp) < 5:
            continue
        confs = grp["final_confidence"].values
        correct = (grp["true_label"] == grp["predicted_class"]).values.astype(float)
        rows.append({
            "experiment_id": exp_id,
            "task": task,
            "routing_path": path,
            "n": len(grp),
            "mean_confidence": round(float(confs.mean()), 4),
            "accuracy": round(float(correct.mean()), 4),
            "ece": round(compute_ece(confs, correct), 4),
        })
    return pd.DataFrame(rows).sort_values(
        ["experiment_id", "task", "routing_path"]
    ).reset_index(drop=True)


def per_class_failure_breakdown(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-class precision, recall, F1, and confusion details per (experiment_id, task).

    For each true class, shows:
      - n            : total samples of that class
      - correct      : correctly predicted
      - precision    : TP / (TP + FP)
      - recall       : TP / (TP + FN)
      - f1           : harmonic mean
      - top_confusion: the most common wrong prediction (and its count)

    Useful for answering: which tumor subtypes benefit most from SAM3/BiomedCLIP,
    and where does the pipeline still systematically fail?
    """
    rows = []
    for (exp_id, task), grp in preds_df.groupby(["experiment_id", "task"]):
        y_true = grp["true_label"].tolist()
        y_pred = grp["predicted_class"].tolist()
        classes = sorted(set(y_true))

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )

        for i, cls in enumerate(classes):
            cls_mask = grp["true_label"] == cls
            cls_grp = grp[cls_mask]
            n = len(cls_grp)
            correct = (cls_grp["predicted_class"] == cls).sum()

            # Most common misclassification
            wrong = cls_grp.loc[cls_grp["predicted_class"] != cls, "predicted_class"]
            if len(wrong) > 0:
                top_conf_cls = wrong.value_counts().index[0]
                top_conf_n = wrong.value_counts().iloc[0]
                top_confusion = f"{top_conf_cls} ({top_conf_n})"
            else:
                top_confusion = "—"

            rows.append({
                "experiment_id": exp_id,
                "task": task,
                "true_class": cls,
                "n": n,
                "correct": int(correct),
                "precision": round(float(prec[i]), 3),
                "recall": round(float(rec[i]), 3),
                "f1": round(float(f1[i]), 3),
                "top_confusion": top_confusion,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_id", "task", "true_class"])
        .reset_index(drop=True)
    )


def medgemma_agreement_analysis(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agreement between MedGemma's triage diagnosis and the CNN/final prediction.

    Requires 'medgemma_class' column (added to all_predictions.csv after eval/evaluate.py
    was updated). Returns an empty DataFrame gracefully if the column is absent.

    For each (experiment_id, task):
      - n_agree / n_disagree : sample counts
      - agree_pct            : % where MedGemma class == predicted_class
      - accuracy_when_agree  : fraction correct when they agree
      - accuracy_when_disagree: fraction correct when they disagree (min 5 samples)
      - disagree_delta       : accuracy_when_disagree - accuracy_when_agree
                               (negative = disagreement predicts errors → supports human review)
    """
    if "medgemma_class" not in preds_df.columns:
        return pd.DataFrame()

    df = preds_df.dropna(subset=["medgemma_class"]).copy()
    df["_mg"] = df["medgemma_class"].str.strip().str.lower()
    df["_pred"] = df["predicted_class"].str.strip().str.lower()
    df["_agree"] = df["_mg"] == df["_pred"]
    df["_correct"] = df["true_label"].str.strip().str.lower() == df["_pred"]

    rows = []
    for (exp_id, task), grp in df.groupby(["experiment_id", "task"]):
        agree = grp[grp["_agree"]]
        disagree = grp[~grp["_agree"]]

        n_agree = len(agree)
        n_disagree = len(disagree)
        total = len(grp)

        acc_agree = float(agree["_correct"].mean()) if n_agree > 0 else float("nan")
        acc_disagree = (
            float(disagree["_correct"].mean()) if n_disagree >= 5 else float("nan")
        )
        delta = (
            round(acc_disagree - acc_agree, 4)
            if not (acc_agree != acc_agree or acc_disagree != acc_disagree)
            else float("nan")
        )

        rows.append({
            "experiment_id": exp_id,
            "task": task,
            "n_agree": n_agree,
            "n_disagree": n_disagree,
            "agree_pct": round(n_agree / total * 100, 1) if total > 0 else float("nan"),
            "accuracy_when_agree": round(acc_agree, 4),
            "accuracy_when_disagree": round(acc_disagree, 4),
            "disagree_delta": round(delta, 4) if delta == delta else float("nan"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_id", "task"])
        .reset_index(drop=True)
    )


def calibration_per_task(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    ECE and overconfidence per (experiment_id, task), collapsed across routing paths.

    Requires at least 10 samples per cell.

    Columns: n, mean_confidence, accuracy, ece, overconfidence
    overconfidence = mean_confidence - accuracy (positive = model too confident)

    Use this to show the MS calibration problem: high confidence despite low accuracy.
    """
    rows = []
    for (exp_id, task), grp in preds_df.groupby(["experiment_id", "task"]):
        if len(grp) < 10:
            continue
        confs = grp["final_confidence"].values
        correct = (grp["true_label"] == grp["predicted_class"]).values.astype(float)
        mean_conf = float(confs.mean())
        acc = float(correct.mean())
        rows.append({
            "experiment_id": exp_id,
            "task": task,
            "n": len(grp),
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(acc, 4),
            "ece": round(compute_ece(confs, correct), 4),
            "overconfidence": round(mean_conf - acc, 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_id", "task"])
        .reset_index(drop=True)
    )


def forest_voting_analysis(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agent Forest voting quality analysis.

    Requires 'dissent_rate' and 'vote_fraction' columns in preds_df
    (written by evaluate.py when forest_consensus is present in state).

    Returns per-(experiment_id, task) breakdown of:
      - mean_dissent_rate : average fraction of agents that disagreed
      - unanimous_pct     : % of cases where all agents agreed
      - accuracy_unanimous: accuracy on unanimous cases
      - accuracy_split    : accuracy on split-vote cases (≥ 5 samples)

    Use this to show whether agent agreement correlates with correctness
    (connects to Li et al.: larger forests reduce uncertainty on hard cases).
    """
    if "dissent_rate" not in preds_df.columns:
        return pd.DataFrame()

    rows = []
    for (exp_id, task), grp in preds_df.groupby(["experiment_id", "task"]):
        unanimous = grp[grp["dissent_rate"] == 0.0]
        split = grp[grp["dissent_rate"] > 0.0]
        correct = grp["true_label"] == grp["predicted_class"]

        rows.append({
            "experiment_id": exp_id,
            "task": task,
            "n": len(grp),
            "mean_dissent_rate": round(float(grp["dissent_rate"].mean()), 4),
            "unanimous_pct": round(len(unanimous) / len(grp) * 100, 1) if len(grp) > 0 else float("nan"),
            "accuracy_unanimous": round(float((grp.loc[unanimous.index, "true_label"] == grp.loc[unanimous.index, "predicted_class"]).mean()), 4) if len(unanimous) > 0 else float("nan"),
            "accuracy_split": round(float((grp.loc[split.index, "true_label"] == grp.loc[split.index, "predicted_class"]).mean()), 4) if len(split) >= 5 else float("nan"),
            "accuracy_overall": round(float(correct.mean()), 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_id", "task"])
        .reset_index(drop=True)
    )


def debate_round_analysis(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-Agent Debate round analysis.

    Requires 'debate_rounds_completed' and 'debate_round_changed' columns
    (written by evaluate.py when debate_verdict is present in state).

    Returns per-(experiment_id, task) breakdown of:
      - pct_verdict_changed : % of cases where round 2+ changed the verdict
      - ece_changed         : ECE on cases where verdict changed
      - ece_unchanged       : ECE on cases where verdict held (≥ 5 samples each)
      - accuracy_changed    : accuracy on verdict-changed cases
      - accuracy_unchanged  : accuracy on verdict-stable cases

    Use this to test Du et al.'s claim that additional rounds improve reasoning
    on hard cases and whether verdict instability correlates with miscalibration.
    """
    if "debate_rounds_completed" not in preds_df.columns:
        return pd.DataFrame()

    rows = []
    for (exp_id, task), grp in preds_df.groupby(["experiment_id", "task"]):
        multi_round = grp[grp["debate_rounds_completed"] > 1]
        if len(multi_round) == 0:
            changed = pd.DataFrame()
            unchanged = grp
        else:
            changed = grp[grp.get("debate_round_changed", pd.Series(False, index=grp.index))]
            unchanged = grp[~grp.index.isin(changed.index)]

        confs_all = grp["final_confidence"].values
        correct_all = (grp["true_label"] == grp["predicted_class"]).values.astype(float)

        def _ece_safe(subset):
            if len(subset) < 5:
                return float("nan")
            c = subset["final_confidence"].values
            ok = (subset["true_label"] == subset["predicted_class"]).values.astype(float)
            return round(compute_ece(c, ok), 4)

        rows.append({
            "experiment_id": exp_id,
            "task": task,
            "n": len(grp),
            "pct_verdict_changed": round(len(changed) / len(grp) * 100, 1) if len(grp) > 0 else float("nan"),
            "accuracy_changed": round(float((changed["true_label"] == changed["predicted_class"]).mean()), 4) if len(changed) >= 5 else float("nan"),
            "accuracy_unchanged": round(float((unchanged["true_label"] == unchanged["predicted_class"]).mean()), 4) if len(unchanged) >= 5 else float("nan"),
            "ece_changed": _ece_safe(changed),
            "ece_unchanged": _ece_safe(unchanged),
            "ece_overall": round(compute_ece(confs_all, correct_all), 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_id", "task"])
        .reset_index(drop=True)
    )


def ablation_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Component contribution table: all metrics per (experiment_id, task).

    Use this to compare full_pipeline vs no_sam3 vs always_sam3 vs no_biomedclip.
    """
    cols = [
        "experiment_id", "task", "accuracy", "f1_macro",
        "normal_specificity", "sam3_invocation_rate",
        "human_review_rate", "ece",
    ]
    available = [c for c in cols if c in summary_df.columns]
    return (
        summary_df[available]
        .sort_values(["task", "experiment_id"])
        .reset_index(drop=True)
    )
