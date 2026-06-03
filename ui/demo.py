"""
Gradio UI for the Multi-Agent Neuroimaging Classifier.

Usage:
    python app.py
    # then open http://localhost:7860 in a browser

The pipeline (MedGemma + CNN + SAM3 + BiomedCLIP) is loaded once in a
background thread so the UI appears immediately; a status line tells you
when the models are ready.
"""

import os
import shutil
import tempfile
import threading
from html import escape
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ── Pipeline singleton (loaded once in background) ────────────────────────────

_pipeline = None
_pipeline_error: str | None = None
_pipeline_ready = threading.Event()


def _init_pipeline() -> None:
    global _pipeline, _pipeline_error
    try:
        from config import DEFAULT_CONFIG, PipelineConfig
        from pipeline.graph import build_pipeline

        cfg = PipelineConfig(
            model=DEFAULT_CONFIG.model,
            routing=DEFAULT_CONFIG.routing,
            output_dir="outputs",
            generate_explainability=True,
        )
        _pipeline = build_pipeline(cfg)
    except Exception as exc:
        _pipeline_error = str(exc)
    finally:
        _pipeline_ready.set()


threading.Thread(target=_init_pipeline, daemon=True).start()


# ── Task display helpers ──────────────────────────────────────────────────────

_TASK_LABELS = {
    "binary_tumor": "Binary Tumor  (tumor / normal)",
    "multiclass_tumor": "Multiclass Tumor  (meningioma / glioma / pituitary / ...)",
    "ms": "Multiple Sclerosis  (MS / normal FLAIR)",
    "stroke": "Stroke  (ischemic / normal CT)",
}
_TASK_KEYS = list(_TASK_LABELS.keys())
_TASK_DISPLAY = [_TASK_LABELS[k] for k in _TASK_KEYS]


# ── Pipeline stage metadata ───────────────────────────────────────────────────

_NODE_ORDER = [
    "triage",
    "cnn_classify",
    "sam3_segment",
    "biomedclip",
    "explainability",
    "verification",
    "report",
    "fhir_output",
]

_NODE_LABELS = {
    "triage": "MedGemma Triage",
    "cnn_classify": "CNN Classify",
    "sam3_segment": "SAM3 Segment",
    "biomedclip": "BiomedCLIP",
    "explainability": "Explainability",
    "explainability_skipped": "Explainability",
    "verification": "Verification",
    "report": "Report",
    "fhir_output": "FHIR Export",
}


def _node_detail(node_name: str, update: dict) -> str:
    """Return a one-line human-readable summary of a node's state update."""
    if node_name == "triage":
        pathology = update.get("suspected_pathology") or ""
        conf = update.get("routing_confidence", 0.0)
        return f"Suspected: {pathology or '—'}  ·  Routing confidence {conf:.0%}"
    if node_name == "cnn_classify":
        cr = update.get("classification_result") or {}
        cls = cr.get("predicted_class", "—")
        conf = cr.get("confidence", 0.0)
        return f"Predicted: {cls}  ·  Confidence {conf:.0%}"
    if node_name == "sam3_segment":
        seg = update.get("segmentation_result") or {}
        if seg.get("skipped"):
            return "Skipped (task not eligible for SAM3)"
        return "Mask and bounding-box overlay generated"
    if node_name == "biomedclip":
        br = update.get("biomedclip_result") or {}
        label = br.get("top_label", "—")
        score = br.get("top_score", 0.0)
        return f"Top label: {label}  ·  Score {score:.2f}"
    if node_name in ("explainability", "explainability_skipped"):
        iou = update.get("saliency_sam3_iou")
        empty = update.get("sam3_mask_empty", False)
        if node_name == "explainability_skipped":
            return "Skipped (eval mode)"
        if iou is not None:
            return f"Grad-CAM++ + IG generated  ·  GradCAM/SAM3 IoU {iou:.3f}"
        if empty:
            return "Grad-CAM++ + IG generated  ·  SAM3 mask empty (IoU skipped)"
        return "Grad-CAM++ and Integrated Gradients generated"
    if node_name == "verification":
        vr = update.get("verification_result") or {}
        agreement = vr.get("agreement")
        if agreement is None:
            return "No saliency map — skipped"
        return "MedGemma agrees with CNN" if agreement else "MedGemma disagrees — confidence capped"
    if node_name == "report":
        cls = update.get("final_predicted_class") or "—"
        conf = update.get("final_confidence", 0.0)
        review = update.get("requires_human_review", False)
        tag = "  ·  ⚠ Human review required" if review else ""
        return f"Final: {cls}  ·  Confidence {conf:.0%}{tag}"
    if node_name == "fhir_output":
        return "Diagnostic Report written"
    return ""


def _build_progress_html(completed: list[str], current: str | None) -> str:
    """Render the pipeline stepper and last-node detail card."""
    completed_set = set(completed)

    steps_html = []
    for key in _NODE_ORDER:
        label = _NODE_LABELS.get(key, key)
        norm_current = (
            "explainability" if current in ("explainability", "explainability_skipped") else current
        )
        norm_key = "explainability" if key == "explainability_skipped" else key

        if norm_key in {("explainability" if c in ("explainability", "explainability_skipped") else c) for c in completed_set}:
            state_cls = "step-done"
            dot = "✓"
        elif norm_key == norm_current:
            state_cls = "step-active"
            dot = '<span class="step-pulse"></span>'
        else:
            state_cls = "step-pending"
            dot = ""

        steps_html.append(
            f'<div class="pipeline-step {state_cls}">'
            f'<span class="step-dot">{dot}</span>'
            f'<span class="step-label">{escape(label)}</span>'
            f"</div>"
        )

    steps = "\n".join(steps_html)
    return f'<div class="pipeline-track">{steps}</div>'


_PROGRESS_PLACEHOLDER = '<div class="pipeline-track pipeline-track-idle"><span class="step-label">Run the pipeline to see live stage progress.</span></div>'


# ── UI copy helpers ──────────────────────────────────────────────────────────

_HERO_HTML = """
<div class="hero-card">
  <div class="hero-main">
    <div class="hero-kicker">Neuroimaging review workspace</div>
    <h1>Multi-Agent Neuroimaging Classifier</h1>
    <p>
      Upload a scan, select the task, and review the prediction, visual evidence,
      and report in one focused workspace.
    </p>
  </div>
  <div class="hero-meta">
    <div class="hero-meta-title">Pipeline stages</div>
    <div class="hero-chips">
      <span>MedGemma</span>
      <span>CNN</span>
      <span>SAM3</span>
      <span>BiomedCLIP</span>
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


def _load_css() -> str:
    css_path = Path(__file__).resolve().parent / "styles" / "app.css"
    return css_path.read_text(encoding="utf-8")


def _build_notice_html(message: str, tone: str = "neutral", detail: str | None = None) -> str:
    detail_html = f"<pre>{escape(detail)}</pre>" if detail else ""
    return (
        f'<div class="notice-card notice-{tone}">'
        f"<p>{escape(message)}</p>"
        f"{detail_html}"
        "</div>"
    )


def _build_status_html(message: str, tone: str, detail: str | None = None) -> str:
    detail_html = f"<span>{escape(detail)}</span>" if detail else ""
    return (
        f'<div class="status-card status-{tone}">'
        f"<strong>{escape(message)}</strong>"
        f"{detail_html}"
        "</div>"
    )


def _format_prediction_label(prediction: str) -> str:
    return prediction.replace("_", " ").strip().title()


def _build_summary_html(
    prediction: str,
    confidence: float,
    review: bool,
    route: str,
    iou: float | None,
) -> str:
    badge_text = "Flagged for human review" if review else "Within review threshold"
    review_text = "Required" if review else "Not required"
    iou_metric = (
        ""
        if iou is None
        else (
            '<div class="summary-metric">'
            '<span class="metric-label">IoU</span>'
            f"<strong>{iou:.3f}</strong>"
            "</div>"
        )
    )

    return f"""
    <div class="summary-card {'summary-alert' if review else 'summary-clear'}">
      <div class="summary-top">
        <span class="summary-eyebrow">Pipeline result</span>
        <span class="summary-badge">{escape(badge_text)}</span>
      </div>
      <h2>{escape(_format_prediction_label(prediction))}</h2>
      <div class="summary-metrics">
        <div class="summary-metric">
          <span class="metric-label">Confidence</span>
          <strong>{confidence:.1%}</strong>
        </div>
        <div class="summary-metric">
          <span class="metric-label">Review</span>
          <strong>{escape(review_text)}</strong>
        </div>
        {iou_metric}
      </div>
      <div class="summary-route">
        <span class="metric-label">Route</span>
        <strong>{escape(route or "Unavailable")}</strong>
      </div>
    </div>
    """


def _build_fhir_html(fhir_status: str) -> str:
    if fhir_status == "FINAL":
        tone = "fhir-final"
        copy = "Diagnostic report is marked final and ready for release."
    elif fhir_status == "PRELIMINARY":
        tone = "fhir-pending"
        copy = "Diagnostic report is preliminary and still needs clinician review."
    else:
        tone = "fhir-neutral"
        copy = "FHIR export completed without a standard release state."

    return f"""
    <div class="fhir-card {tone}">
      <span class="fhir-label">FHIR status</span>
      <strong>{escape(fhir_status)}</strong>
      <p>{escape(copy)}</p>
    </div>
    """


def _build_report_html(report: str | None) -> str:
    if not report or report.strip() == "No report generated.":
        return _REPORT_PLACEHOLDER_HTML

    paragraphs = [
        f"<p>{escape(chunk).replace(chr(10), '<br>')}</p>"
        for chunk in report.strip().split("\n\n")
        if chunk.strip()
    ]
    body = "".join(paragraphs) or f"<p>{escape(report)}</p>"
    return f'<div class="report-card report-ready">{body}</div>'


# ── Core inference function ───────────────────────────────────────────────────

def run_pipeline(
    image_path: str | None, task_display: str
) -> Generator[tuple, None, None]:
    """Streaming generator called by Gradio on button click.

    Yields (progress_html, summary_html, report_html, images, fhir_html) after
    each pipeline node so the UI updates in real time.
    """
    if image_path is None:
        yield (
            _PROGRESS_PLACEHOLDER,
            _build_notice_html("Please upload a brain scan image first."),
            _REPORT_PLACEHOLDER_HTML,
            [],
            "",
        )
        return

    if not _pipeline_ready.wait(timeout=600):
        yield (
            _PROGRESS_PLACEHOLDER,
            _build_notice_html(
                "Models are still loading. Please wait a moment and try again.",
                tone="neutral",
            ),
            _REPORT_PLACEHOLDER_HTML,
            [],
            "",
        )
        return

    if _pipeline_error:
        yield (
            _PROGRESS_PLACEHOLDER,
            _build_notice_html("Pipeline failed to load.", tone="error", detail=_pipeline_error),
            _REPORT_PLACEHOLDER_HTML,
            [],
            "",
        )
        return

    if not _pipeline:
        yield (
            _PROGRESS_PLACEHOLDER,
            _build_notice_html("Pipeline not available.", tone="error"),
            _REPORT_PLACEHOLDER_HTML,
            [],
            "",
        )
        return

    task = _TASK_KEYS[_TASK_DISPLAY.index(task_display)]

    from pipeline.state import initial_state

    suffix = Path(image_path).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copy(image_path, tmp.name)
        tmp_path = tmp.name

    completed_nodes: list[str] = []
    accumulated: dict = {}

    try:
        state = initial_state(tmp_path, task)
        accumulated = dict(state)

        for chunk in _pipeline.stream(state):
            for node_name, update in chunk.items():
                accumulated.update(update)
                completed_nodes.append(node_name)

                detail = _node_detail(node_name, update)
                label = _NODE_LABELS.get(node_name, node_name)
                detail_html = (
                    f'<div class="node-detail">'
                    f'<span class="node-detail-label">{escape(label)}</span>'
                    f'<span class="node-detail-text">{escape(detail)}</span>'
                    f"</div>"
                    if detail
                    else ""
                )
                progress_html = (
                    _build_progress_html(completed_nodes[:-1], node_name) + detail_html
                )

                yield (
                    progress_html,
                    _SUMMARY_PLACEHOLDER,
                    _REPORT_PLACEHOLDER_HTML,
                    [],
                    "",
                )

    except Exception as exc:
        yield (
            _PROGRESS_PLACEHOLDER,
            _build_notice_html("Inference error.", tone="error", detail=str(exc)),
            _REPORT_PLACEHOLDER_HTML,
            [],
            "",
        )
        return
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result = accumulated
    prediction = result.get("final_predicted_class") or "unknown"
    confidence = result.get("final_confidence", 0.0)
    review = result.get("requires_human_review", False)
    route = " → ".join(result.get("routing_path", []))
    report = result.get("final_report") or "No report generated."
    report_html = _build_report_html(report)
    iou = result.get("saliency_sam3_iou")
    summary_html = _build_summary_html(prediction, confidence, review, route, iou)

    images: list[tuple[str, str]] = []
    seg = result.get("segmentation_result") or {}
    if seg.get("guided_image_path") and Path(seg["guided_image_path"]).exists():
        images.append((seg["guided_image_path"], "SAM3 bounding-box overlay"))
    if seg.get("mask_path") and Path(seg["mask_path"]).exists():
        images.append((seg["mask_path"], "SAM3 segmentation mask"))

    expl = result.get("explainability_result") or {}
    if expl.get("gradcam_pp") and Path(expl["gradcam_pp"]).exists():
        images.append((expl["gradcam_pp"], "Grad-CAM++"))
    if expl.get("integrated_gradients") and Path(expl["integrated_gradients"]).exists():
        images.append((expl["integrated_gradients"], "Integrated Gradients"))

    fhir = result.get("fhir_report") or {}
    fhir_status = "N/A"
    for entry in fhir.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "DiagnosticReport":
            fhir_status = resource.get("status", "unknown").upper()
            break

    fhir_html = _build_fhir_html(fhir_status)
    final_progress = _build_progress_html(completed_nodes, None)
    yield final_progress, summary_html, report_html, images, fhir_html


def get_model_status() -> str:
    if _pipeline_ready.is_set():
        if _pipeline_error:
            return _build_status_html("Models failed to load.", "error", _pipeline_error)
        return _build_status_html("Models loaded and ready.", "ready")
    return _build_status_html(
        "Loading models in background.",
        "loading",
        "This may take 1-2 minutes on first run.",
    )


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


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Multi-Agent Neuroimaging Classifier") as app:
        gr.HTML(_HERO_HTML)

        model_status = gr.HTML(get_model_status(), elem_id="status-box")

        with gr.Row(equal_height=False, elem_classes=["panel-row"]):
            with gr.Column(scale=1, min_width=320, elem_classes=["panel-card"]):
                gr.HTML(
                    """
                    <div class="panel-kicker">Input</div>
                    <h2 class="panel-title">Prepare the scan</h2>
                    <p class="panel-copy">
                      Upload the study image, confirm the task, and start the staged review pipeline.
                    </p>
                    """
                )
                gr.HTML(
                    """
                    <div class="section-head">
                      <div class="section-title">Study image</div>
                      <div class="section-note">Drag in an MRI or CT scan, or click the frame to browse</div>
                    </div>
                    """
                )
                image_input = gr.Image(
                    type="filepath",
                    show_label=False,
                    height=260,
                    elem_id="scan-input",
                )
                gr.HTML(
                    """
                    <div class="section-head">
                      <div class="section-title">Classification task</div>
                      <div class="section-note">Choose the relevant diagnostic pathway</div>
                    </div>
                    """
                )
                task_input = gr.Dropdown(
                    choices=_TASK_DISPLAY,
                    value=_TASK_DISPLAY[0],
                    show_label=False,
                    elem_id="task-select",
                )
                run_btn = gr.Button(
                    "Run Pipeline",
                    variant="primary",
                    size="lg",
                    elem_id="run-btn",
                )

            with gr.Column(scale=2, elem_classes=["panel-card"]):
                gr.HTML(
                    """
                    <div class="panel-kicker">Output</div>
                    <h2 class="panel-title">Review the analysis</h2>
                    <p class="panel-copy">
                      The predicted class, report text, and visual evidence are grouped for faster review.
                    </p>
                    """
                )
                progress_out = gr.HTML(_PROGRESS_PLACEHOLDER, elem_id="progress-out")
                summary_out = gr.HTML(_SUMMARY_PLACEHOLDER, elem_id="summary-out")
                fhir_out = gr.HTML("", elem_id="fhir-out")
                with gr.Row(equal_height=False, elem_classes=["output-grid"]):
                    with gr.Column(scale=5, min_width=260):
                        gr.HTML(
                            """
                            <div class="section-head">
                              <div class="section-title">Clinical report</div>
                              <div class="section-note">Readable narrative output from MedGemma</div>
                            </div>
                            """
                        )
                        report_out = gr.HTML(_REPORT_PLACEHOLDER_HTML, elem_id="report-out")
                    with gr.Column(scale=10, min_width=280):
                        gr.HTML(
                            """
                            <div class="section-head">
                              <div class="section-title">Visual evidence</div>
                              <div class="section-note">SAM3 mask, Grad-CAM++, and Integrated Gradients</div>
                            </div>
                            """
                        )
                        gallery_out = gr.Gallery(
                            show_label=False,
                            columns=2,
                            height=520,
                            object_fit="contain",
                            elem_id="gallery-out",
                        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[image_input, task_input],
            outputs=[progress_out, summary_out, report_out, gallery_out, fhir_out],
        )

        app.load(fn=get_model_status, outputs=model_status)

    return app


demo = create_demo()


def launch() -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=_theme,
        css=_load_css(),
    )
