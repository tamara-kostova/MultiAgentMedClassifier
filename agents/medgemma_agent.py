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
    elif conf < routing_cfg.sam3_threshold and name not in ("normal", ""):
        decision = "sam3_then_cnn"
        reasoning = f"Confidence {conf:.2f} is ambiguous; SAM3 spatial cues may help."
    elif name == "tumor" and conf < routing_cfg.biomedclip_rerank_threshold:
        decision = "biomedclip"
        reasoning = (
            f"Tumor subtype ({pathology}) may benefit from vision-language re-ranking."
        )
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

REPORT_PROMPT_TEMPLATE = """You are a neuroimaging AI assistant generating a structured triage report. Summarize the following findings concisely for clinical review.

Task: {task}
Routing path used: {routing_path}
MedGemma initial diagnosis:
{medgemma_dx}

CNN classification result:
{cnn_result}

SAM3 segmentation result:
{sam3_result}

BiomedCLIP similarity result:
{biomedclip_result}

Generate a brief structured report with:
1. Primary finding
2. Confidence assessment
3. Recommended next step
4. Any flags or caveats

Keep it under 150 words. Do not make definitive diagnoses — this is a triage aid only."""


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
        medgemma_dx: Optional[MedicalDiagnosis],
        cnn_result: Optional[dict],
        sam3_result: Optional[dict],
        biomedclip_result: Optional[dict],
    ) -> str:
        def fmt(d) -> str:
            if d is None:
                return "Not invoked."
            if hasattr(d, "model_dump"):
                return json.dumps(d.model_dump(), indent=2)
            return json.dumps(d, indent=2)

        prompt = REPORT_PROMPT_TEMPLATE.format(
            task=task,
            routing_path=" → ".join(routing_path),
            medgemma_dx=fmt(medgemma_dx),
            cnn_result=fmt(cnn_result),
            sam3_result=fmt(sam3_result),
            biomedclip_result=fmt(biomedclip_result),
        )
        image = Image.open(image_path).convert("RGB")
        return self._generate(image, prompt, max_new_tokens=300)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_diagnostic_prompt(
        self, image: Image.Image, prompt: str
    ) -> MedicalDiagnosis:
        """Run a diagnostic prompt and parse the JSON output with retries."""
        raw = self._generate(image, prompt)
        for attempt in range(self.routing_cfg.max_parse_retries):
            try:
                return self._parse_diagnosis(raw)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < self.routing_cfg.max_parse_retries - 1:
                    fix = (
                        f"Your previous output was not valid JSON: {raw[:200]}\n"
                        f"Error: {e}\nOutput ONLY valid JSON matching the schema."
                    )
                    raw = self._generate(image, prompt + "\n" + fix)
                else:
                    print(
                        f"[MedGemmaAgent] JSON parse failed after {self.routing_cfg.max_parse_retries} attempts. Using fallback."
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
        self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
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
