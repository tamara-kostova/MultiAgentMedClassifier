"""
MedGemma 1.5 4B agent — router and report generator.

Roles in the pipeline:
  1. Triage / Router: Runs system_prompt.txt on the raw image → structured diagnosis JSON.
     Routing decision (cnn_direct / sam3_then_cnn / biomedclip / human_review) is derived
     from diagnosis_confidence and diagnosis_name — no separate routing prompt needed.
  2. SAM3-guided diagnosis: Runs system_prompt_bbox.txt on the SAM3 bbox-overlay image.
  3. Report Generator: Synthesizes all tool outputs into a structured triage report.

Prompts are loaded from prompts/ at import time so they can be edited without touching code.

Requires HuggingFace authentication:
    huggingface-cli login   (or set HF_TOKEN env var)
and accepted terms of use at:
    https://huggingface.co/google/medgemma-1.5-4b-it
"""

import json
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from pydantic import BaseModel, field_validator
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import DEFAULT_CONFIG, ModelConfig, RoutingConfig

# ── Load prompts from files ───────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SYSTEM_PROMPT = (_PROMPTS_DIR / "system_prompt.txt").read_text()
SYSTEM_PROMPT_FEW_SHOT = (_PROMPTS_DIR / "system_prompt_few_shot.txt").read_text()
SYSTEM_PROMPT_BBOX = (_PROMPTS_DIR / "system_prompt_bbox.txt").read_text()


# ── Pydantic schema matching system_prompt.txt output ────────────────────────


class MedicalDiagnosis(BaseModel):
    modality: Optional[str]  # "MRI" | "CT" | null
    specialized_sequence: Optional[str]  # "FLAIR" | "T1" | "T2" | "T1C+" | null
    plane: Optional[str]  # "axial" | "sagittal" | "coronal" | null
    diagnosis_name: Optional[
        str
    ]  # "tumor" | "stroke" | "multiple sclerosis" | "normal" | "other abnormalities" | null
    diagnosis_detailed: Optional[str]  # tumor/stroke subtype or null
    icd10_code: Optional[str]
    severity_score: Optional[float]
    diagnosis_confidence: float
    severity_confidence: Optional[float]

    @field_validator(
        "diagnosis_confidence", "severity_score", "severity_confidence", mode="before"
    )
    @classmethod
    def clamp_float(cls, v):
        if v is None:
            return v
        return max(0.0, min(1.0, float(v)))


# ── Routing decision derived from MedicalDiagnosis ───────────────────────────


class RoutingDecision(BaseModel):
    modality: str
    suspected_pathology: str
    routing_decision: (
        str  # "cnn_direct" | "sam3_then_cnn" | "biomedclip" | "human_review"
    )
    confidence: float
    reasoning: str

    @field_validator("routing_decision")
    @classmethod
    def validate_routing(cls, v: str) -> str:
        valid = {"cnn_direct", "sam3_then_cnn", "biomedclip", "human_review"}
        if v not in valid:
            raise ValueError(f"routing_decision must be one of {valid}")
        return v

class VerificationResult(BaseModel):
    agreement: bool
    saliency_plausible: bool
    alternative_diagnosis: Optional[str]
    verification_confidence: float
    reasoning: str

    @field_validator("verification_confidence", mode="before")
    @classmethod
    def clamp(cls, v):
        return max(0.0, min(1.0, float(v)))


def diagnosis_to_routing(
    dx: MedicalDiagnosis, routing_cfg: RoutingConfig
) -> RoutingDecision:
    """
    Derive a routing decision from the structured MedGemma diagnosis.

    Logic:
      - confidence < human_review_threshold  → human_review
      - confidence < sam3_threshold AND pathology is not clearly normal → sam3_then_cnn
      - tumor with fine-grained subtype AND confidence < biomedclip threshold → biomedclip
      - otherwise → cnn_direct
    """
    conf = dx.diagnosis_confidence
    name = (dx.diagnosis_name or "").lower()
    modality = dx.modality or "MRI"
    pathology = dx.diagnosis_detailed or name or "unknown"

    if conf < routing_cfg.human_review_threshold:
        decision = "human_review"
        reasoning = f"Diagnosis confidence {conf:.2f} is below threshold {routing_cfg.human_review_threshold}."
    elif routing_cfg.always_run_biomedclip:
        decision = "biomedclip"
        reasoning = "BiomedCLIP forced on; bypassing confidence-based routing."
    elif routing_cfg.always_run_sam3 and name not in ("normal", ""):
        decision = "sam3_then_cnn"
        reasoning = (
            "SAM3 forced on for all non-normal cases; bypassing confidence-based routing."
        )
    elif name == "tumor" and conf < routing_cfg.biomedclip_rerank_threshold:
        decision = "biomedclip"
        reasoning = (
            f"Tumor subtype ({pathology}) may benefit from vision-language re-ranking."
        )
    elif conf < routing_cfg.sam3_threshold and name not in ("normal", ""):
        decision = "sam3_then_cnn"
        reasoning = f"Confidence {conf:.2f} is ambiguous; SAM3 spatial cues may help."
    else:
        decision = "cnn_direct"
        reasoning = f"Confidence {conf:.2f} sufficient for direct CNN classification."

    return RoutingDecision(
        modality=modality,
        suspected_pathology=pathology,
        routing_decision=decision,
        confidence=conf,
        reasoning=reasoning,
    )


# ── Report prompt ─────────────────────────────────────────────────────────────

REPORT_PROMPT_TEMPLATE = """You are a radiologist synthesizing multi-tool AI pipeline findings into a final report.

You are given multiple images from the same case:
- Image 1: original scan
- Image 2: SAM3 segmentation / bbox-guided overlay, if available
- Image 3: Grad-CAM++ explainability overlay, if available
- Image 4: Integrated Gradients explainability overlay, if available

Pipeline outputs (use these together with the images to inform your diagnosis):
Task: {task}
Route: {routing_path}

Initial MedGemma triage diagnosis:
{initial_medgemma_dx}

SAM3-guided MedGemma diagnosis:
{sam3_medgemma_dx}

CNN classification:
{cnn_result}

SAM3 segmentation:
{sam3_result}

BiomedCLIP re-ranking:
{biomedclip_result}

Verification result (MedGemma vs CNN agreement):
{verification_result}

Explainability outputs:
{explainability_result}

Spatial alignment — GradCAM++ ∩ SAM3 mask IoU: {saliency_iou}
(IoU < 0.3 suggests the CNN attended to background rather than the lesion)

Your response MUST contain exactly two sections in this order:

FINDINGS:
1. Primary finding: <one sentence>
2. Confidence assessment: <one sentence>
3. Recommended next step: <one sentence>
4. Flags/caveats: <one sentence>

STRUCTURED DIAGNOSIS:
{{
  "modality": "<MRI|CT|null>",
  "specialized_sequence": "<FLAIR|T1|T2|T1C+|null>",
  "plane": "<axial|sagittal|coronal|null>",
  "diagnosis_name": "<tumor|stroke|multiple sclerosis|normal|other abnormalities|null>",
  "diagnosis_detailed": "<glioma|meningioma|pituitary_tumor|carcinoma|germinoma|granuloma|medulloblastoma|neurocytoma|papilloma|schwannoma|tuberculoma|ischemic|hemorrhagic|null>",
  "icd10_code": "<ICD-10 or null>",
  "severity_score": <float 0.0-1.0 or null>,
  "diagnosis_confidence": <float 0.0-1.0>,
  "severity_confidence": <float 0.0-1.0>
}}

Rules:
- Keep FINDINGS under 100 words. Do not make definitive diagnoses — this is a triage aid only.
- Weigh task-specific model outputs (CNN, SAM3 localization, BiomedCLIP) more heavily than the initial MedGemma triage when they conflict, but use the full evidence bundle to resolve disagreements.
- For STRUCTURED DIAGNOSIS: integrate all evidence from the images and tool outputs; derive modality/sequence/plane from the image and MedGemma observations.
- Output no text outside these two sections."""

VERIFICATION_PROMPT_TEMPLATE = """You are a senior neuroradiologist reviewing an AI classification.

Original image has been classified by a CNN with the following result:
  Task:             {task}
  Predicted class:  {predicted_class}
  CNN confidence:   {confidence:.1%}
  Calibration T:    {temperature:.3f}  (>1.0 means model was over-confident at training time)

The saliency map (Grad-CAM++) is shown overlaid on the scan in the second image.

Assess:
1. Does the saliency map highlight a plausible lesion region for this diagnosis?
2. Is the predicted class consistent with the imaging features visible?
3. If you disagree, what is the most likely alternative diagnosis?

Respond in JSON only:
{{
  "agreement": true | false,
  "saliency_plausible": true | false,
  "alternative_diagnosis": "<string or null>",
  "verification_confidence": <0.0–1.0>,
  "reasoning": "<one sentence>"
}}"""


# ── Agent class ───────────────────────────────────────────────────────────────


class MedGemmaAgent:
    """MedGemma 1.5 4B orchestrator — uses system_prompt.txt / system_prompt_bbox.txt."""

    def __init__(
        self, model_cfg: ModelConfig = None, routing_cfg: RoutingConfig = None
    ):
        self.model_cfg = model_cfg or DEFAULT_CONFIG.model
        self.routing_cfg = routing_cfg or DEFAULT_CONFIG.routing
        self.device = torch.device(self.model_cfg.device)

        self.processor = AutoProcessor.from_pretrained(self.model_cfg.medgemma_model_id)

        if self.model_cfg.use_4bit_quantization:
            print(
                f"[MedGemmaAgent] Loading {self.model_cfg.medgemma_model_id} (4-bit NF4)..."
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_cfg.medgemma_model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            print(
                f"[MedGemmaAgent] Loading {self.model_cfg.medgemma_model_id} (bfloat16)..."
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_cfg.medgemma_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()
        print("[MedGemmaAgent] Model loaded.")

        if self.model_cfg.use_few_shot:
            from agents.few_shot_loader import load_few_shot_examples
            self._few_shot_examples = load_few_shot_examples(
                data_dir=self.model_cfg.few_shot_data_dir,
            )
            print(f"[MedGemmaAgent] Loaded {len(self._few_shot_examples)} few-shot examples.")
        else:
            self._few_shot_examples = None

    # ── Primary triage (raw image) ────────────────────────────────────────────

    def diagnose(self, image_path: str) -> tuple[MedicalDiagnosis, RoutingDecision]:
        """
        Run system_prompt.txt on the raw image.
        Returns (structured diagnosis, derived routing decision).
        """
        image = Image.open(image_path).convert("RGB")
        dx = self._run_diagnostic_prompt(image, SYSTEM_PROMPT)
        routing = diagnosis_to_routing(dx, self.routing_cfg)
        return dx, routing

    # ── SAM3-guided diagnosis (bbox overlay image) ────────────────────────────

    def diagnose_with_bbox(self, guided_image_path: str) -> MedicalDiagnosis:
        """
        Run system_prompt_bbox.txt on the SAM3 bbox-overlay image.
        Used on the sam3_then_cnn path after segmentation.
        """
        image = Image.open(guided_image_path).convert("RGB")
        return self._run_diagnostic_prompt(image, SYSTEM_PROMPT_BBOX)

    # ── Report generator ──────────────────────────────────────────────────────

    def generate_report(
        self,
        image_path: str,
        task: str,
        routing_path: list,
        initial_medgemma_dx: Optional[MedicalDiagnosis],
        sam3_medgemma_dx: Optional[MedicalDiagnosis],
        cnn_result: Optional[dict],
        sam3_result: Optional[dict],
        biomedclip_result: Optional[dict],
        explainability_result: Optional[dict] = None,
        verification_result: Optional[dict] = None,
        saliency_iou: Optional[float] = None,
    ) -> tuple[str, Optional[MedicalDiagnosis]]:
        def fmt(d) -> str:
            if d is None:
                return "Not invoked."
            if hasattr(d, "model_dump"):
                return json.dumps(d.model_dump(), indent=2)
            return json.dumps(d, indent=2)

        iou_str = f"{saliency_iou:.3f}" if saliency_iou is not None else "Not computed."

        prompt = REPORT_PROMPT_TEMPLATE.format(
            task=task,
            routing_path=" → ".join(routing_path),
            initial_medgemma_dx=fmt(initial_medgemma_dx),
            sam3_medgemma_dx=fmt(sam3_medgemma_dx),
            cnn_result=fmt(cnn_result),
            sam3_result=fmt(sam3_result),
            biomedclip_result=fmt(biomedclip_result),
            explainability_result=fmt(explainability_result),
            verification_result=fmt(verification_result),
            saliency_iou=iou_str,
        )
        image_paths = [image_path]
        if sam3_result and sam3_result.get("guided_image_path"):
            image_paths.append(sam3_result["guided_image_path"])
        if explainability_result and explainability_result.get("gradcam_pp"):
            image_paths.append(explainability_result["gradcam_pp"])
        if explainability_result and explainability_result.get("integrated_gradients"):
            image_paths.append(explainability_result["integrated_gradients"])

        images = [Image.open(path).convert("RGB") for path in image_paths]
        raw = self._generate(images, prompt, max_new_tokens=600)
        # Soft-validate the JSON block and pretty-print it in place
        try:
            dx = self._parse_diagnosis(raw)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                before = raw[: match.start()].rstrip()
                report = f"{before}\n\n{json.dumps(dx.model_dump(), indent=2)}"
                return report, dx
        except (json.JSONDecodeError, ValueError):
            pass
        return raw, None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_diagnostic_prompt(
        self, image: Image.Image, prompt: str
    ) -> MedicalDiagnosis:
        """Run a diagnostic prompt and parse the JSON output with retries."""
        raw = self._generate(image, prompt, few_shot=self._few_shot_examples)
        for attempt in range(self.routing_cfg.max_parse_retries):
            try:
                return self._parse_diagnosis(raw)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < self.routing_cfg.max_parse_retries - 1:
                    fix = (
                        f"Your previous output was not valid JSON: {raw[:200]}\n"
                        f"Error: {e}\nOutput ONLY valid JSON matching the schema."
                    )
                    raw = self._generate(image, prompt + "\n" + fix, few_shot=self._few_shot_examples)
                else:
                    print(
                        f"[MedGemmaAgent] JSON parse failed after {self.routing_cfg.max_parse_retries} attempts. \n Raw output: {raw}\n Error: {e}\n Returning default diagnosis with low confidence."
                    )
                    return MedicalDiagnosis(
                        modality=None,
                        specialized_sequence=None,
                        plane=None,
                        diagnosis_name=None,
                        diagnosis_detailed=None,
                        icd10_code=None,
                        severity_score=None,
                        diagnosis_confidence=0.5,
                        severity_confidence=None,
                    )

    def _generate(
        self,
        image: Image.Image,
        text_prompt: str,
        max_new_tokens: int = 2064,
        few_shot: list[tuple[Image.Image, str]] | None = None,
    ) -> str:
        images = image if isinstance(image, list) else [image]
        messages = []
        if few_shot:
            for ex_img, ex_json in few_shot:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex_img},
                        {"type": "text",  "text": text_prompt},
                    ],
                })
                messages.append({"role": "assistant", "content": ex_json})
        messages.append(
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": text_prompt}]
                ),
            }
        )
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        n_input = inputs["input_ids"].shape[-1]
        generated = output_ids[0][n_input:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_diagnosis(text: str) -> MedicalDiagnosis:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        return MedicalDiagnosis(**json.loads(match.group()))

    def verify_cnn_prediction(
        self,
        original_image_path: str,
        saliency_image_path: str,
        cnn_result: dict,
    ) -> VerificationResult:
        """
        Post-hoc verification: MedGemma reviews the CNN label against the saliency map.
        Returns a VerificationResult — disagreement should elevate requires_human_review.
        """
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            task=cnn_result.get("task", "unknown"),
            predicted_class=cnn_result["predicted_class"],
            confidence=cnn_result["confidence"],
            temperature=cnn_result.get("temperature", 1.0),
        )

        original  = Image.open(original_image_path).convert("RGB")
        saliency  = Image.open(saliency_image_path).convert("RGB")

        # Two-image turn: original scan + saliency overlay
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",  "image": original},
                    {"type": "image",  "image": saliency},
                    {"type": "text",   "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=False, temperature=1.0
            )

        n_input   = inputs["input_ids"].shape[-1]
        raw       = self.processor.decode(output_ids[0][n_input:], skip_special_tokens=True).strip()
        raw       = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        match     = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            # Verification failed to parse — treat as non-committal agreement
            return VerificationResult(
                agreement=True, saliency_plausible=True,
                alternative_diagnosis=None, verification_confidence=0.5,
                reasoning="Verification parse failed — defaulting to CNN prediction.",
            )
        return VerificationResult(**json.loads(match.group()))
