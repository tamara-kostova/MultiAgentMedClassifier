"""
Gradio UI for the Multi-Agent Neuroimaging Classifier.

Usage:
    python app.py
    python app.py --load_4bit --calibration_file temps.json
    # then open http://localhost:7860 in a browser

Any `run_pipeline.py` config flag (checkpoints, --device, --load_4bit,
--calibration_file, --few_shot, thresholds, --pipeline_mode / --debate_rounds /
--forest_n_agents as UI defaults, ...) is accepted, so the UI runs with exactly
the configuration the CLI would. MEDGEMMA_4BIT=1 is honoured as well.

The models (MedGemma + CNN + SAM3 + BiomedCLIP) are loaded once in a background
thread so the UI appears immediately. All three pipeline variants — Standard,
Debate (System B) and Forest (System C) — are assembled from those same agents,
so switching mode never reloads weights.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from html import escape
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Pipeline registry (agents loaded once in background) ──────────────────────

_cfg = None
_agents: tuple | None = None
_pipeline_error: str | None = None
_pipeline_ready = threading.Event()
_pipelines: dict[tuple, object] = {}
_pipelines_lock = threading.Lock()


def _load_agents_in_background(cfg) -> None:
    global _agents, _pipeline_error
    try:
        from pipeline.graph import load_agents

        _agents = load_agents(cfg)
    except Exception as exc:
        _pipeline_error = f"{type(exc).__name__}: {exc}"
    finally:
        _pipeline_ready.set()


def _get_pipeline(mode: str, debate_rounds: int, forest_n_agents: int):
    """Assemble (and cache) the compiled graph for a mode from the shared agents."""
    from pipeline.graph import (
        assemble_debate_pipeline,
        assemble_forest_pipeline,
        assemble_pipeline,
    )

    if mode == "debate":
        key = ("debate", int(debate_rounds))
    elif mode == "forest":
        key = ("forest", int(forest_n_agents))
    else:
        key = ("standard",)

    with _pipelines_lock:
        if key not in _pipelines:
            if mode == "debate":
                app = assemble_debate_pipeline(*_agents, _cfg, rounds=key[1])
            elif mode == "forest":
                app = assemble_forest_pipeline(*_agents, _cfg, n_agents=key[1])
            else:
                app = assemble_pipeline(*_agents, _cfg)
            _pipelines[key] = app
        return _pipelines[key]


def _linear_node_order(app) -> list[str]:
    """Walk the compiled graph from START to END (all variants are linear)."""
    graph = app.get_graph()
    nexts = {edge.source: edge.target for edge in graph.edges}
    order, node = [], nexts.get("__start__")
    while node and node != "__end__" and node not in order:
        order.append(node)
        node = nexts.get(node)
    return order


# ── Choices ───────────────────────────────────────────────────────────────────

_TASK_CHOICES = [
    ("Binary Tumor  (tumor / normal)", "binary_tumor"),
    ("Multiclass Tumor  (meningioma / glioma / pituitary / ...)", "multiclass_tumor"),
    ("Multiple Sclerosis  (MS / normal FLAIR)", "ms"),
    ("Stroke  (ischemic / normal CT)", "stroke"),
]

_MODE_CHOICES = [
    ("Standard", "standard"),
    ("Debate (System B)", "debate"),
    ("Forest (System C)", "forest"),
]

_MODE_NOTES = {
    "standard": "Single MedGemma triage, specialist tools, verification and a fused report.",
    "debate": "Three MedGemma advocates argue for CNN, BiomedCLIP and SAM3; a MedGemma judge decides.",
    "forest": "N role-specialized MedGemma agents vote at triage; the rest of the pipeline is unchanged.",
}


# ── Pipeline stage metadata ───────────────────────────────────────────────────

_NODE_LABELS = {
    "triage": "MedGemma Triage",
    "forest_triage": "Forest Triage",
    "cnn_classify": "CNN Classify",
    "sam3_segment": "SAM3 Segment",
    "biomedclip": "BiomedCLIP",
    "explainability": "Explainability",
    "verification": "Verification",
    "debate": "Debate",
    "report": "Report",
    "fhir_output": "FHIR Export",
}


def _node_detail(node_name: str, update: dict) -> str:
    """Return a one-line human-readable summary of a node's state update."""
    if node_name == "triage":
        pathology = update.get("suspected_pathology") or ""
        conf = update.get("routing_confidence", 0.0)
        return f"Suspected: {pathology or '—'}  ·  Confidence {conf:.0%}"
    if node_name == "forest_triage":
        cons = update.get("forest_consensus") or {}
        winner = cons.get("winner_detailed") or cons.get("winner") or "—"
        return (
            f"Consensus: {winner}  ·  Agreement {cons.get('vote_fraction', 0.0):.0%}"
            f"  ·  Dissent {cons.get('dissent_rate', 0.0):.0%}"
        )
    if node_name == "cnn_classify":
        cr = update.get("classification_result") or {}
        return f"Predicted: {cr.get('predicted_class', '—')}  ·  Confidence {cr.get('confidence', 0.0):.0%}"
    if node_name == "sam3_segment":
        seg = update.get("segmentation_result") or {}
        if seg.get("skipped"):
            return "Skipped (task not eligible for SAM3)"
        return "Mask and bounding-box overlay generated"
    if node_name == "biomedclip":
        br = update.get("biomedclip_result") or {}
        return f"Top label: {br.get('top_label', '—')}  ·  Score {br.get('top_score', 0.0):.2f}"
    if node_name == "explainability":
        expl = update.get("explainability_result")
        if not expl:
            return "No saliency maps generated"
        iou = update.get("saliency_sam3_iou")
        if iou is not None:
            return f"Grad-CAM++ + IG generated  ·  GradCAM/SAM3 IoU {iou:.3f}"
        if update.get("sam3_mask_empty"):
            return "Grad-CAM++ + IG generated  ·  SAM3 mask empty (IoU skipped)"
        return "Grad-CAM++ and Integrated Gradients generated"
    if node_name == "verification":
        vr = update.get("verification_result") or {}
        agreement = vr.get("agreement")
        if agreement is None:
            return "No saliency map — skipped"
        return "MedGemma agrees with CNN" if agreement else "MedGemma disagrees — confidence capped"
    if node_name in ("report", "debate"):
        cls = update.get("final_predicted_class") or "—"
        conf = update.get("final_confidence", 0.0)
        tag = "  ·  ⚠ Human review required" if update.get("requires_human_review") else ""
        return f"Final: {cls}  ·  Confidence {conf:.0%}{tag}"
    if node_name == "fhir_output":
        return "FHIR R4 bundle written"
    return ""


def _build_progress_html(
    order: list[str],
    completed: list[str],
    timings: dict[str, float],
    running: bool,
    last_node: str | None,
    last_update: dict | None,
) -> str:
    """Render the stepper for the selected mode plus the latest node's detail."""
    done = set(completed)
    active = next((n for n in order if n not in done), None) if running else None

    steps = []
    for key in order:
        label = escape(_NODE_LABELS.get(key, key))
        if key in done:
            secs = timings.get(key)
            time_html = f'<span class="step-time">{secs:.1f}s</span>' if secs is not None else ""
            steps.append(
                f'<div class="pipeline-step step-done"><span class="step-dot">✓</span>'
                f'<span class="step-label">{label}</span>{time_html}</div>'
            )
        elif key == active:
            steps.append(
                '<div class="pipeline-step step-active"><span class="step-dot">'
                f'<span class="step-pulse"></span></span><span class="step-label">{label}</span></div>'
            )
        else:
            steps.append(
                f'<div class="pipeline-step step-pending"><span class="step-dot"></span>'
                f'<span class="step-label">{label}</span></div>'
            )

    detail_html = ""
    if last_node and last_update is not None:
        detail = _node_detail(last_node, last_update)
        if detail:
            detail_html = (
                '<div class="node-detail">'
                f'<span class="node-detail-label">{escape(_NODE_LABELS.get(last_node, last_node))}</span>'
                f'<span class="node-detail-text">{escape(detail)}</span>'
                "</div>"
            )
    total = sum(timings.values())
    total_html = (
        f'<span class="pipeline-total">{"Elapsed" if running else "Total"} {total:.1f}s</span>'
        if timings
        else ""
    )
    return f'<div class="pipeline-track">{"".join(steps)}{total_html}{detail_html}</div>'


_PROGRESS_PLACEHOLDER = (
    '<div class="pipeline-track pipeline-track-idle">'
    '<span class="step-label">Run the pipeline to see live stage progress.</span></div>'
)


# ── UI copy helpers ───────────────────────────────────────────────────────────

_HERO_HTML = """
<div class="hero-card">
  <div class="hero-main">
    <div class="hero-kicker">Neuroimaging review workspace</div>
    <h1>Multi-Agent Neuroimaging Classifier</h1>
    <p>
      Upload a scan, select the task and pipeline mode, and review the prediction,
      each agent's evidence, and the report in one focused workspace.
    </p>
  </div>
  <div class="hero-meta">
    <div class="hero-meta-title">Agents &amp; tools</div>
    <div class="hero-chips">
      <span>MedGemma</span>
      <span>CNN</span>
      <span>SAM3</span>
      <span>BiomedCLIP</span>
      <span>Grad-CAM++</span>
      <span>Debate</span>
      <span>Forest</span>
      <span>FHIR</span>
    </div>
  </div>
</div>
"""

_SUMMARY_PLACEHOLDER = """
<div class="notice-card notice-neutral empty-card">
  <div class="notice-title">Awaiting analysis</div>
  <p>The prediction summary, review status, and route details will appear here after the pipeline runs.</p>
</div>
"""

_REPORT_PLACEHOLDER_HTML = """
<div class="report-card report-empty">
  <p>Run the pipeline to generate MedGemma's narrative diagnostic summary. The completed note will appear here in full.</p>
</div>
"""

_AGENTS_PLACEHOLDER = """
<div class="notice-card notice-neutral">
  <p>Each agent's output (MedGemma triage, CNN probabilities, BiomedCLIP ranking, SAM3, verification) appears here as soon as that stage finishes.</p>
</div>
"""

_ENSEMBLE_PLACEHOLDER = {
    "forest": '<div class="notice-card notice-neutral"><p>The forest ballot and consensus appear here once forest triage finishes.</p></div>',
    "debate": '<div class="notice-card notice-neutral"><p>The advocate arguments and the judge\'s verdict appear here once the debate finishes.</p></div>',
}

_FHIR_PLACEHOLDER = """
<div class="fhir-card fhir-neutral">
  <span class="fhir-label">FHIR status</span>
  <strong>—</strong>
  <p>The FHIR R4 bundle (Patient, ImagingStudy, Observation, DiagnosticReport) appears here after export.</p>
</div>
"""


def _load_css() -> str:
    css_path = Path(__file__).resolve().parent / "styles" / "app.css"
    return css_path.read_text(encoding="utf-8")


def _build_notice_html(message: str, tone: str = "neutral", detail: str | None = None) -> str:
    detail_html = f"<pre>{escape(detail)}</pre>" if detail else ""
    return f'<div class="notice-card notice-{tone}"><p>{escape(message)}</p>{detail_html}</div>'


def _build_status_html(message: str, tone: str, detail: str | None = None) -> str:
    detail_html = f"<span>{escape(detail)}</span>" if detail else ""
    return f'<div class="status-card status-{tone}"><strong>{escape(message)}</strong>{detail_html}</div>'


def _fmt_label(value) -> str:
    text = str(value or "").replace("_", " ").strip()
    return text.title() if text else "—"


def _is_null(value) -> bool:
    return str(value or "").strip().lower() in ("", "none", "null", "nan")


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


# ── Summary card ──────────────────────────────────────────────────────────────

def _build_summary_html(state: dict, mode: str) -> str:
    prediction = state.get("final_predicted_class")
    provisional = False
    if not prediction:
        cnn = state.get("classification_result") or {}
        if cnn.get("predicted_class"):
            prediction, confidence, provisional = cnn["predicted_class"], cnn.get("confidence", 0.0), True
        elif state.get("suspected_pathology"):
            prediction, confidence, provisional = state["suspected_pathology"], state.get("routing_confidence", 0.0), True
        else:
            return _SUMMARY_PLACEHOLDER
    else:
        confidence = state.get("final_confidence", 0.0)

    review = state.get("requires_human_review", False)
    iou = state.get("saliency_sam3_iou")
    route = " → ".join(state.get("routing_path", []))

    if provisional:
        tone, badge = "summary-provisional", "Provisional — pipeline still running"
    elif review:
        tone, badge = "summary-alert", "Flagged for human review"
    else:
        tone, badge = "summary-clear", "Within review threshold"

    if mode == "debate":
        source = "Debate judge"
    elif state.get("final_medgemma_diagnosis"):
        source = "Fused MedGemma report"
    else:
        # report_node fell back down its priority chain (CNN → BiomedCLIP → triage)
        source = "Fallback (no fused diagnosis)"
    if provisional:
        source = "CNN (interim)" if state.get("classification_result") else "Triage (interim)"

    metrics = [
        ("Confidence", _fmt_pct(confidence)),
        ("Review", "Pending" if provisional else ("Required" if review else "Not required")),
        ("Decided by", source),
    ]
    if iou is not None:
        metrics.append(("Saliency/SAM3 IoU", f"{iou:.3f}"))
    icd = (state.get("final_medgemma_diagnosis") or {}).get("icd10_code")
    if not _is_null(icd):
        metrics.append(("ICD-10", str(icd)))

    metrics_html = "".join(
        f'<div class="summary-metric"><span class="metric-label">{escape(k)}</span><strong>{escape(v)}</strong></div>'
        for k, v in metrics
    )
    return f"""
    <div class="summary-card {tone}">
      <div class="summary-top">
        <span class="summary-eyebrow">Pipeline result · {escape(dict((v, k) for k, v in _MODE_CHOICES).get(mode, mode))}</span>
        <span class="summary-badge">{escape(badge)}</span>
      </div>
      <h2>{escape(_fmt_label(prediction))}</h2>
      <div class="summary-metrics">{metrics_html}</div>
      <div class="summary-route">
        <span class="metric-label">Route</span>
        <strong>{escape(route or "—")}</strong>
      </div>
    </div>
    """


# ── Agents tab ────────────────────────────────────────────────────────────────

def _bar_rows(pairs: list[tuple[str, float]], highlight, fmt: str = "pct") -> str:
    highlight = {highlight} if isinstance(highlight, str) or highlight is None else set(highlight)
    if not pairs:
        return ""
    top = max((max(v, 0.0) for _, v in pairs), default=0.0) or 1.0
    rows = []
    for label, value in pairs:
        width = max(value, 0.0) / top * 100 if fmt == "score" else max(0.0, min(value, 1.0)) * 100
        shown = f"{value:.1%}" if fmt == "pct" else f"{value:.3f}"
        cls = " bar-top" if label in highlight else ""
        rows.append(
            f'<div class="bar-row{cls}"><span class="bar-label">{escape(_fmt_label(label))}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.1f}%"></span></span>'
            f'<span class="bar-value">{shown}</span></div>'
        )
    return f'<div class="bar-list">{"".join(rows)}</div>'


def _kv_rows(pairs: list[tuple[str, object]]) -> str:
    items = [
        f"<div><dt>{escape(k)}</dt><dd>{escape(str(v))}</dd></div>"
        for k, v in pairs
        if not _is_null(v)
    ]
    return f'<dl class="kv-grid">{"".join(items)}</dl>' if items else ""


def _agent_card(title: str, subtitle: str, body: str, tone: str = "") -> str:
    return (
        f'<div class="agent-card {tone}"><div class="agent-head">'
        f'<span class="agent-title">{escape(title)}</span>'
        f'<span class="agent-sub">{escape(subtitle)}</span></div>{body}</div>'
    )


def _diagnosis_kv(dx: dict) -> str:
    seq = " · ".join(
        str(dx.get(k)) for k in ("modality", "specialized_sequence", "plane") if not _is_null(dx.get(k))
    )
    return _kv_rows([
        ("Diagnosis", _fmt_label(dx.get("diagnosis_name"))),
        ("Detail", dx.get("diagnosis_detailed")),
        ("Confidence", _fmt_pct(dx.get("diagnosis_confidence"))),
        ("ICD-10", dx.get("icd10_code")),
        ("Severity", dx.get("severity_score")),
        ("Acquisition", seq),
    ])


def _build_agents_html(state: dict, mode: str) -> str:
    cards = []

    dx = state.get("medgemma_diagnosis")
    if dx:
        if mode == "forest":
            # medgemma_diagnosis is the first winning agent's read, not the ballot —
            # the Forest tab shows the actual vote.
            title, sub = "Forest triage", "Read of the first agent voting for the consensus"
            note = state.get("routing_reasoning") or ""
        else:
            title, sub, note = "MedGemma triage", "First-pass read of the raw scan", ""
        body = _diagnosis_kv(dx) + (f'<p class="agent-note">{escape(note)}</p>' if note else "")
        cards.append(_agent_card(title, sub, body))

    cnn = state.get("classification_result")
    if cnn:
        probs = sorted((cnn.get("all_probs") or {}).items(), key=lambda kv: kv[1], reverse=True)
        body = _bar_rows(probs, cnn.get("predicted_class")) or _kv_rows(
            [("Predicted", cnn.get("predicted_class")), ("Confidence", _fmt_pct(cnn.get("confidence")))]
        )
        cards.append(_agent_card("CNN classifier", f"Predicted {_fmt_label(cnn.get('predicted_class'))}", body))

    clip = state.get("biomedclip_result")
    if clip:
        pairs = list(zip(clip.get("ranked_labels") or [], clip.get("scores") or []))
        body = _bar_rows([(l, float(s)) for l, s in pairs], clip.get("top_label"), fmt="score")
        cards.append(_agent_card("BiomedCLIP", "Layer-6 ranking (ViT-B/16)", body or _kv_rows([("Top", clip.get("top_label"))])))

    seg = state.get("segmentation_result")
    if seg:
        if seg.get("skipped"):
            body = '<p class="agent-note">SAM3 is only eligible for tumor tasks; skipped for this task.</p>'
        else:
            bbox = seg.get("bbox")
            body = _kv_rows([
                ("Bounding box", ", ".join(str(int(v)) for v in bbox) if bbox else "No lesion found"),
                ("Mask empty", "Yes" if state.get("sam3_mask_empty") else ("No" if state.get("explainability_result") else None)),
                ("Grad-CAM++ IoU", f"{state['saliency_sam3_iou']:.3f}" if state.get("saliency_sam3_iou") is not None else None),
            ])
            bbox_dx = state.get("medgemma_bbox_diagnosis")
            if bbox_dx:
                body += '<p class="agent-note">MedGemma re-read on the bbox overlay:</p>' + _diagnosis_kv(bbox_dx)
        cards.append(_agent_card("SAM3 segmentation", "Linear-probe lesion mask", body))

    vr = state.get("verification_result")
    if vr:
        agree = vr.get("agreement")
        body = _kv_rows([
            ("Verdict", "Agrees with CNN" if agree else "Disagrees with CNN"),
            ("Saliency plausible", "Yes" if vr.get("saliency_plausible") else "No"),
            ("Alternative", vr.get("alternative_diagnosis")),
            ("Confidence", _fmt_pct(vr.get("verification_confidence"))),
        ])
        if vr.get("reasoning"):
            body += f'<p class="agent-note">{escape(vr["reasoning"])}</p>'
        cards.append(
            _agent_card("MedGemma verification", "CNN prediction vs Grad-CAM++ map", body, "" if agree else "agent-warn")
        )

    if not cards:
        return _AGENTS_PLACEHOLDER
    return f'<div class="agent-grid">{"".join(cards)}</div>'


# ── Forest / Debate tab ───────────────────────────────────────────────────────

_ADVOCATE_LABELS = {"cnn": "CNN advocate", "clip": "BiomedCLIP advocate", "sam": "SAM3 advocate"}


def _build_forest_html(state: dict) -> str:
    cons = state.get("forest_consensus")
    votes = state.get("forest_votes") or []
    if not cons:
        return _ENSEMBLE_PLACEHOLDER["forest"]
    winner = cons.get("winner_detailed") or cons.get("winner")
    field = cons.get("vote_field") or "diagnosis_name"
    counts = sorted((cons.get("vote_counts") or {}).items(), key=lambda kv: kv[1], reverse=True)
    n = cons.get("n_agents") or len(votes) or 1
    head = _kv_rows([
        ("Consensus", _fmt_label(winner)),
        ("Agreement", _fmt_pct(cons.get("vote_fraction"))),
        ("Dissent rate", _fmt_pct(cons.get("dissent_rate"))),
        ("Confidence-weighted", _fmt_pct(cons.get("confidence_weighted_confidence"))),
        ("Agents", n),
    ])
    tally = _bar_rows([(k, v / n) for k, v in counts], {cons.get("winner"), cons.get("winner_detailed")})
    rows = "".join(
        "<tr>"
        f"<td>{escape(_fmt_label(v.get('role')))}</td>"
        f"<td>{escape(_fmt_label(v.get('diagnosis_name')))}</td>"
        f"<td>{escape('' if _is_null(v.get('diagnosis_detailed')) else str(v.get('diagnosis_detailed')))}</td>"
        f"<td>{_fmt_pct(v.get('diagnosis_confidence'))}</td>"
        "</tr>"
        for v in votes
    )
    table = (
        '<table class="vote-table"><thead><tr><th>Role</th><th>Diagnosis</th><th>Detail</th>'
        f"<th>Confidence</th></tr></thead><tbody>{rows}</tbody></table>"
        if rows
        else ""
    )
    return (
        _agent_card("Forest consensus", f"Majority vote on {field}", head + tally)
        + _agent_card("Ballot", "One row per role-specialized MedGemma agent", table)
    )


def _build_debate_html(state: dict) -> str:
    verdict = state.get("debate_verdict")
    args = state.get("debate_arguments") or []
    if not verdict:
        return _ENSEMBLE_PLACEHOLDER["debate"]
    warn = ""
    if verdict.get("judge_parse_failed"):
        warn = (
            '<p class="agent-note agent-note-warn">The judge output could not be parsed in '
            f"{verdict.get('judge_parse_failed_rounds', 1)} round(s); the verdict fell back to the triage guess.</p>"
        )
    head = _kv_rows([
        ("Winner", _fmt_label(verdict.get("winner"))),
        ("Detail", verdict.get("winner_detailed")),
        ("Confidence", _fmt_pct(verdict.get("confidence"))),
        ("Rounds", verdict.get("rounds_completed")),
        ("Changed in last round", "Yes" if verdict.get("round_changed") else "No"),
    ])
    reason = f'<p class="agent-note">{escape(verdict["reason"])}</p>' if verdict.get("reason") else ""
    cards = [_agent_card("Judge verdict", "MedGemma judge", head + reason + warn,
                         "agent-warn" if verdict.get("judge_parse_failed") else "")]

    rounds: dict[int, list[dict]] = {}
    for arg in args:
        rounds.setdefault(arg.get("round", 1), []).append(arg)
    for rnd in sorted(rounds):
        items = "".join(
            f'<div class="debate-arg"><span class="debate-role">{escape(_ADVOCATE_LABELS.get(a.get("role"), a.get("role", "")))}</span>'
            f'<p>{escape(str(a.get("argument", ""))).replace(chr(10), "<br>")}</p></div>'
            for a in rounds[rnd]
        )
        cards.append(_agent_card(f"Round {rnd}", "Advocate arguments", items))
    return "".join(cards)


def _build_ensemble_html(state: dict, mode: str) -> str:
    if mode == "forest":
        return _build_forest_html(state)
    if mode == "debate":
        return _build_debate_html(state)
    return ""


# ── Report + FHIR ─────────────────────────────────────────────────────────────

def _build_report_html(state: dict) -> str:
    report = state.get("final_report")
    if not report or report.strip() == "No report generated.":
        return _REPORT_PLACEHOLDER_HTML

    header = ""
    dx = state.get("final_medgemma_diagnosis")
    if dx:
        header = f'<div class="report-meta">{_diagnosis_kv(dx)}</div>'
    paragraphs = "".join(
        f"<p>{escape(chunk).replace(chr(10), '<br>')}</p>"
        for chunk in report.strip().split("\n\n")
        if chunk.strip()
    )
    return f'<div class="report-card report-ready">{header}{paragraphs or f"<p>{escape(report)}</p>"}</div>'


def _fhir_status(bundle: dict) -> str:
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "DiagnosticReport":
            return resource.get("status", "unknown").upper()
    return "N/A"


def _build_fhir_html(bundle: dict | None) -> str:
    if not bundle:
        return _FHIR_PLACEHOLDER
    status = _fhir_status(bundle)
    if status == "FINAL":
        tone, copy = "fhir-final", "Diagnostic report is marked final and ready for release."
    elif status == "PRELIMINARY":
        tone, copy = "fhir-pending", "Diagnostic report is preliminary and still needs clinician review."
    else:
        tone, copy = "fhir-neutral", "FHIR export completed without a standard release state."
    return (
        f'<div class="fhir-card {tone}"><span class="fhir-label">FHIR status</span>'
        f"<strong>{escape(status)}</strong><p>{escape(copy)}</p></div>"
    )


def _write_fhir_download(bundle: dict) -> str:
    report_id = next(
        (
            e["resource"].get("id")
            for e in bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == "DiagnosticReport"
        ),
        None,
    ) or bundle.get("id", "bundle")
    out = Path(tempfile.gettempdir()) / "neuro_ui_fhir" / f"fhir_{report_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return str(out)


def _collect_images(state: dict) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    seg = state.get("segmentation_result") or {}
    expl = state.get("explainability_result") or {}
    for path, caption in (
        (seg.get("guided_image_path"), "SAM3 bounding-box overlay"),
        (seg.get("mask_path"), "SAM3 segmentation mask"),
        (expl.get("gradcam_pp"), "Grad-CAM++"),
        (expl.get("integrated_gradients"), "Integrated Gradients"),
    ):
        if path and Path(path).exists():
            images.append((path, caption))
    return images


# ── Core inference function ───────────────────────────────────────────────────

def _render(state: dict, mode: str, progress_html: str, final: bool, summary_override: str | None = None) -> tuple:
    """All output panels for the current state; `final` gates the FHIR download."""
    bundle = state.get("fhir_report")
    return (
        progress_html,
        summary_override or _build_summary_html(state, mode),
        _build_agents_html(state, mode),
        _build_ensemble_html(state, mode) or "",
        _collect_images(state),
        _build_report_html(state),
        _build_fhir_html(bundle),
        bundle,
        _write_fhir_download(bundle) if (final and bundle) else None,
    )


def _render_notice(mode: str, message: str, tone: str = "neutral", detail: str | None = None) -> tuple:
    return _render({}, mode, _PROGRESS_PLACEHOLDER, False, _build_notice_html(message, tone, detail))


def run_pipeline(
    image_path: str | None,
    task: str,
    mode: str,
    debate_rounds: int,
    forest_n_agents: int,
) -> Generator[tuple, None, None]:
    """Streaming generator called by Gradio on button click.

    Re-renders every output panel after each pipeline node, so the stepper, the
    provisional summary, agent cards and saliency images fill in as they arrive.
    """
    if image_path is None:
        yield _render_notice(mode, "Please upload a brain scan image first.")
        return
    if not _pipeline_ready.is_set():
        yield _render_notice(mode, "Models are still loading. The Run button unlocks once they are ready.")
        return
    if _pipeline_error or _agents is None:
        yield _render_notice(mode, "Pipeline failed to load.", "error", _pipeline_error)
        return

    try:
        app = _get_pipeline(mode, debate_rounds, forest_n_agents)
        order = _linear_node_order(app)
    except Exception as exc:
        yield _render_notice(mode, "Could not assemble the pipeline.", "error", f"{type(exc).__name__}: {exc}")
        return

    from pipeline.state import initial_state

    suffix = Path(image_path).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copy(image_path, tmp.name)
        tmp_path = tmp.name

    completed: list[str] = []
    timings: dict[str, float] = {}
    last_node, last_update = None, None
    state = dict(initial_state(tmp_path, task))

    yield _render(state, mode, _build_progress_html(order, completed, timings, True, None, None), False)

    try:
        t_prev = time.perf_counter()
        for chunk in app.stream(state):
            for node_name, update in chunk.items():
                now = time.perf_counter()
                timings[node_name] = now - t_prev
                t_prev = now
                state.update(update or {})
                completed.append(node_name)
                last_node, last_update = node_name, update or {}
            progress = _build_progress_html(order, completed, timings, True, last_node, last_update)
            yield _render(state, mode, progress, False)
    except Exception as exc:
        progress = _build_progress_html(order, completed, timings, False, last_node, last_update)
        yield _render(
            state, mode, progress, False,
            _build_notice_html("Inference error — partial results are shown below.", "error", f"{type(exc).__name__}: {exc}"),
        )
        return
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    progress = _build_progress_html(order, completed, timings, False, last_node, last_update)
    yield _render(state, mode, progress, True)


# ── Model status polling ──────────────────────────────────────────────────────

def poll_model_status():
    """Update the status pill and unlock Run once models are ready; stops the timer."""
    if not _pipeline_ready.is_set():
        return (
            _build_status_html("Loading models in background.", "loading", "This may take 1-2 minutes on first run."),
            gr.Button(interactive=False, value="Loading models…"),
            gr.Timer(active=True),
        )
    if _pipeline_error or _agents is None:
        return (
            _build_status_html("Models failed to load.", "error", _pipeline_error),
            gr.Button(interactive=False, value="Pipeline unavailable"),
            gr.Timer(active=False),
        )
    device = getattr(getattr(_cfg, "model", None), "device", "")
    quant = "4-bit NF4" if getattr(getattr(_cfg, "model", None), "use_4bit_quantization", False) else "bf16"
    return (
        _build_status_html("Models loaded and ready.", "ready", f"MedGemma {quant} · {device}"),
        gr.Button(interactive=True, value="Run Pipeline"),
        gr.Timer(active=False),
    )


def _on_mode_change(mode: str):
    return (
        gr.Slider(visible=mode == "debate"),
        gr.Slider(visible=mode == "forest"),
        f'<p class="mode-note">{escape(_MODE_NOTES.get(mode, ""))}</p>',
        gr.Tab(visible=mode in ("debate", "forest"), label="Debate" if mode == "debate" else "Forest"),
        _ENSEMBLE_PLACEHOLDER.get(mode, ""),
    )


def _example_images() -> list[str]:
    exts = {".png", ".jpg", ".jpeg"}
    found = [p for p in sorted((Path(__file__).resolve().parent / "examples").glob("*")) if p.suffix.lower() in exts]
    scan = _REPO_ROOT / "scan.jpg"
    if scan.exists():
        found.insert(0, scan)
    return [str(p) for p in found]


# ── Gradio theme + layout ─────────────────────────────────────────────────────

_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("IBM Plex Sans"),
).set(
    body_background_fill="#f4f6f8",
    body_background_fill_dark="#f4f6f8",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#dde3e9",
    block_border_color_dark="#dde3e9",
    block_border_width="1px",
    block_radius="18px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="white",
    button_primary_border_color="#2563eb",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#f8fafc",
    input_border_color="#c8d4de",
    input_border_color_dark="#c8d4de",
    input_radius="14px",
)


def _section_head(title: str, note: str) -> gr.HTML:
    return gr.HTML(
        f'<div class="section-head"><div class="section-title">{escape(title)}</div>'
        f'<div class="section-note">{escape(note)}</div></div>'
    )


def create_demo(default_mode: str = "standard", default_rounds: int = 1, default_agents: int = 3) -> gr.Blocks:
    with gr.Blocks(title="Multi-Agent Neuroimaging Classifier") as app:
        gr.HTML(_HERO_HTML)

        model_status = gr.HTML(elem_id="status-box")
        status_timer = gr.Timer(2.0)

        with gr.Row(equal_height=False, elem_classes=["panel-row"]):
            with gr.Column(scale=1, min_width=320, elem_classes=["panel-card"]):
                gr.HTML(
                    """
                    <div class="panel-kicker">Input</div>
                    <h2 class="panel-title">Prepare the scan</h2>
                    <p class="panel-copy">
                      Upload the study image, confirm the task and pipeline mode, and start the staged review.
                    </p>
                    """
                )
                _section_head("Study image", "Drag in an MRI or CT scan, or click the frame to browse")
                image_input = gr.Image(type="filepath", show_label=False, height=260, elem_id="scan-input")

                _section_head("Classification task", "Choose the relevant diagnostic pathway")
                task_input = gr.Dropdown(
                    choices=_TASK_CHOICES, value="binary_tumor", show_label=False, elem_id="task-select"
                )

                _section_head("Pipeline mode", "Standard, advocate debate, or agent forest")
                mode_input = gr.Radio(
                    choices=_MODE_CHOICES, value=default_mode, show_label=False, elem_id="mode-select"
                )
                mode_note = gr.HTML(f'<p class="mode-note">{escape(_MODE_NOTES[default_mode])}</p>')
                rounds_input = gr.Slider(
                    1, 3, value=default_rounds, step=1, label="Debate rounds",
                    visible=default_mode == "debate", elem_classes=["mode-slider"],
                )
                agents_input = gr.Slider(
                    1, 8, value=default_agents, step=1, label="Forest agents",
                    info="Roles cycle radiologist → conservative → emergency → differential.",
                    visible=default_mode == "forest", elem_classes=["mode-slider"],
                )

                run_btn = gr.Button(
                    "Loading models…", variant="primary", size="lg", elem_id="run-btn", interactive=False
                )

                examples = _example_images()
                if examples:
                    gr.Examples(
                        examples=[[p, "binary_tumor"] for p in examples],
                        inputs=[image_input, task_input],
                        label="Example scans",
                    )

            with gr.Column(scale=2, elem_classes=["panel-card"]):
                gr.HTML(
                    """
                    <div class="panel-kicker">Output</div>
                    <h2 class="panel-title">Review the analysis</h2>
                    <p class="panel-copy">
                      Results fill in stage by stage: the summary is provisional until the report (or judge) finishes.
                    </p>
                    """
                )
                progress_out = gr.HTML(_PROGRESS_PLACEHOLDER, elem_id="progress-out")
                summary_out = gr.HTML(_SUMMARY_PLACEHOLDER, elem_id="summary-out")

                with gr.Tabs(elem_id="result-tabs"):
                    with gr.Tab("Visual evidence"):
                        _section_head("Visual evidence", "SAM3 mask, Grad-CAM++, and Integrated Gradients")
                        gallery_out = gr.Gallery(
                            show_label=False, columns=2, height=520, object_fit="contain", elem_id="gallery-out"
                        )
                    with gr.Tab("Agents"):
                        _section_head("Agent outputs", "What each model contributed to the decision")
                        agents_out = gr.HTML(_AGENTS_PLACEHOLDER, elem_id="agents-out")
                    with gr.Tab(
                        "Debate" if default_mode == "debate" else "Forest",
                        visible=default_mode in ("debate", "forest"),
                    ) as ensemble_tab:
                        ensemble_out = gr.HTML(_ENSEMBLE_PLACEHOLDER.get(default_mode, ""), elem_id="ensemble-out")
                    with gr.Tab("Clinical report"):
                        _section_head("Clinical report", "Readable narrative output from MedGemma")
                        report_out = gr.HTML(_REPORT_PLACEHOLDER_HTML, elem_id="report-out")
                    with gr.Tab("FHIR"):
                        fhir_out = gr.HTML(_FHIR_PLACEHOLDER, elem_id="fhir-out")
                        fhir_file = gr.File(label="Download FHIR R4 bundle", interactive=False, elem_id="fhir-file")
                        fhir_json = gr.JSON(label="Bundle", elem_id="fhir-json")

        mode_input.change(
            fn=_on_mode_change,
            inputs=mode_input,
            outputs=[rounds_input, agents_input, mode_note, ensemble_tab, ensemble_out],
        )
        run_btn.click(
            fn=run_pipeline,
            inputs=[image_input, task_input, mode_input, rounds_input, agents_input],
            outputs=[
                progress_out, summary_out, agents_out, ensemble_out, gallery_out,
                report_out, fhir_out, fhir_json, fhir_file,
            ],
            concurrency_limit=1,  # one GPU, one pipeline run at a time
        )
        status_timer.tick(fn=poll_model_status, outputs=[model_status, run_btn, status_timer])
        app.load(fn=poll_model_status, outputs=[model_status, run_btn, status_timer])

    return app


def launch(argv: list[str] | None = None) -> None:
    """Parse run_pipeline config flags, start model loading, and serve the UI."""
    global _cfg
    from run_pipeline import build_config, parse_args

    # --image is a placeholder that satisfies the CLI's required mode group;
    # explainability is always on in the UI so the evidence tab is populated.
    args = parse_args(["--image", "<ui>", "--generate_explainability", *(argv or [])])
    _cfg = build_config(args)
    Path(_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    threading.Thread(target=_load_agents_in_background, args=(_cfg,), daemon=True).start()

    demo = create_demo(
        default_mode=args.pipeline_mode,
        default_rounds=max(1, min(args.debate_rounds, 3)),
        default_agents=max(1, min(args.forest_n_agents, 8)),
    )
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        theme=_theme,
        css=_load_css(),
        allowed_paths=[str(Path(_cfg.output_dir).resolve())],
    )
