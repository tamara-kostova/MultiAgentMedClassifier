"""
Minimal Gradio GUI for the Multi-Agent Neuroimaging Classifier.

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
from pathlib import Path

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
    "multiclass_tumor": "Multiclass Tumor  (meningioma / glioma / pituitary / …)",
    "ms": "Multiple Sclerosis  (MS / normal FLAIR)",
    "stroke": "Stroke  (ischemic / normal CT)",
}
_TASK_KEYS = list(_TASK_LABELS.keys())
_TASK_DISPLAY = [_TASK_LABELS[k] for k in _TASK_KEYS]


# ── Core inference function ───────────────────────────────────────────────────

def run_pipeline(image_path: str | None, task_display: str) -> tuple:
    """Called by Gradio on button click. Returns (summary_md, report, images, fhir_md)."""

    if image_path is None:
        return "Please upload a brain scan image first.", "", [], ""

    if not _pipeline_ready.wait(timeout=600):
        return "Models are still loading — please wait and try again.", "", [], ""

    if _pipeline_error:
        return f"**Pipeline failed to load:**\n```\n{_pipeline_error}\n```", "", [], ""

    if not _pipeline:
        return "Pipeline not available.", "", [], ""

    task = _TASK_KEYS[_TASK_DISPLAY.index(task_display)]

    from pipeline.state import initial_state

    suffix = Path(image_path).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copy(image_path, tmp.name)
        tmp_path = tmp.name

    try:
        state = initial_state(tmp_path, task)
        result = _pipeline.invoke(state)
    except Exception as exc:
        return f"**Inference error:**\n```\n{exc}\n```", "", [], ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Build outputs ─────────────────────────────────────────────────────────
    prediction = result.get("final_predicted_class") or "unknown"
    confidence = result.get("final_confidence", 0.0)
    review = result.get("requires_human_review", False)
    route = " → ".join(result.get("routing_path", []))
    report = result.get("final_report") or "No report generated."
    iou = result.get("saliency_sam3_iou")

    flag_md = (
        "\n\n> ⚠️ **FLAGGED FOR HUMAN REVIEW** — confidence below threshold or model disagreement"
        if review
        else ""
    )
    iou_md = f"\n- **GradCAM++ / SAM3 IoU:** `{iou:.3f}`" if iou is not None else ""

    summary_md = (
        f"## {prediction}\n"
        f"**Confidence:** {confidence:.1%}{flag_md}\n\n"
        f"---\n"
        f"- **Route:** `{route}`"
        f"{iou_md}"
    )

    # Generated images (SAM3 + explainability)
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

    # FHIR status line
    fhir = result.get("fhir_report") or {}
    fhir_status = "N/A"
    for entry in fhir.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "DiagnosticReport":
            fhir_status = resource.get("status", "unknown").upper()
            break

    if fhir_status == "FINAL":
        fhir_md = f"**FHIR status:** `{fhir_status}` — report cleared for release"
    elif fhir_status == "PRELIMINARY":
        fhir_md = f"**FHIR status:** `{fhir_status}` — pending human review"
    else:
        fhir_md = f"**FHIR status:** `{fhir_status}`"

    return summary_md, report, images, fhir_md


def get_model_status() -> str:
    if _pipeline_ready.is_set():
        if _pipeline_error:
            return f"Models failed to load: {_pipeline_error}"
        return "Models loaded and ready."
    return "Loading models in background — this may take 1–2 minutes on first run..."


# ── Gradio layout ─────────────────────────────────────────────────────────────

_CSS = """
#status-box { font-size: 0.85rem; color: #666; }
#result-heading { margin-top: 0; }
"""

with gr.Blocks(title="Multi-Agent Neuroimaging Classifier", css=_CSS) as demo:
    gr.Markdown(
        "# 🧠 Multi-Agent Neuroimaging Classifier\n"
        "Upload a brain scan, choose the classification task, then click **Run Pipeline**. "
        "The pipeline runs: MedGemma triage → CNN → SAM3 segmentation → BiomedCLIP → "
        "Grad-CAM++ explainability → MedGemma report."
    )

    model_status = gr.Markdown(get_model_status(), elem_id="status-box")

    gr.Markdown("---")

    with gr.Row():
        # ── Left panel: inputs ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            image_input = gr.Image(
                type="filepath",
                label="Brain scan (MRI / CT)",
                height=280,
            )
            task_input = gr.Dropdown(
                choices=_TASK_DISPLAY,
                value=_TASK_DISPLAY[0],
                label="Classification task",
            )
            run_btn = gr.Button("▶  Run Pipeline", variant="primary", size="lg")

        # ── Right panel: outputs ──────────────────────────────────────────────
        with gr.Column(scale=2):
            summary_out = gr.Markdown(
                "Results will appear here after running the pipeline.",
                elem_id="result-heading",
            )
            report_out = gr.Textbox(
                label="Clinical report (MedGemma)",
                lines=8,
                interactive=False,
                placeholder="MedGemma's free-text diagnostic summary appears here.",
            )
            gallery_out = gr.Gallery(
                label="Generated images  (SAM3 mask · Grad-CAM++ · Integrated Gradients)",
                columns=2,
                height=280,
                object_fit="contain",
                show_label=True,
            )
            fhir_out = gr.Markdown("")

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, task_input],
        outputs=[summary_out, report_out, gallery_out, fhir_out],
    )

    # Refresh status when the user returns focus to the tab
    demo.load(fn=get_model_status, outputs=model_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
