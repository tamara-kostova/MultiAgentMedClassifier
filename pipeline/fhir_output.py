"""
Serialise NeuroimagingState into a FHIR R4 Bundle containing:
  1. Patient          — anonymous subject
  2. ImagingStudy     — the MRI/CT scan
  3. Observation      — AI prediction (CNN / BiomedCLIP output)
  4. DiagnosticReport — MedGemma narrative + references Observation

Spec refs:
  https://hl7.org/fhir/R4/bundle.html
  https://hl7.org/fhir/R4/diagnosticreport.html
  https://hl7.org/fhir/R4/observation.html
  https://hl7.org/fhir/R4/imagingstudy.html

SNOMED CT codes:
  393563007 — glioma          189372002 — meningioma
  254940001 — glioblastoma    127024001 — pituitary adenoma
  126952004 — brain tumor     17621005  — normal
  230690007 — stroke          24700007  — multiple sclerosis
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# SNOMED CT lookup — extend as needed
SNOMED = {
    "glioma":             "393563007",
    "glioblastoma":       "254940001",
    "meningioma":         "189372002",
    "pituitary":          "127024001",
    "tumor":              "126952004",
    "normal":             "17621005",
    "stroke":             "230690007",
    "multiple sclerosis": "24700007",
    "ms":                 "24700007",
}

# LOINC
LOINC_BRAIN_MRI = "24590-2"   # MRI Brain
LOINC_AI_RESULT = "84892-9"   # AI interpretation

# Modality → ImagingStudy.modality DICOM code
DICOM_MODALITY = {"MRI": "MR", "CT": "CT"}


# ── Individual resource builders ──────────────────────────────────────────────

def _build_patient(patient_id: str) -> dict:
    """Minimal anonymous Patient resource."""
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]
        },
    }


def _build_imaging_study(study_id: str, patient_id: str, state: dict) -> dict:
    """ImagingStudy representing the MRI/CT scan."""
    metadata = state.get("metadata") or {}
    modality_value = metadata.get("modality") or (state.get("medgemma_diagnosis") or {}).get(
        "modality", "MRI"
    )
    modality_code = DICOM_MODALITY.get(modality_value, modality_value if modality_value in {"MR", "CT"} else "MR")
    description = (
        metadata.get("study_description")
        or metadata.get("series_description")
        or metadata.get("source_image_path")
        or state.get("image_path", "unknown")
    )
    return {
        "resourceType": "ImagingStudy",
        "id": study_id,
        "status": "available",
        "subject": {"reference": f"Patient/{patient_id}"},
        "modality": [{
            "system": "http://dicom.nema.org/resources/ontology/DCM",
            "code": modality_code,
        }],
        "description": description,
    }


def _build_observation(obs_id: str, patient_id: str, study_id: str, state: dict) -> dict:
    """
    Observation capturing the AI prediction (CNN or BiomedCLIP).
    valueCodeableConcept holds the SNOMED-coded predicted class.
    component holds confidence, temperature (calibration), and IoU.
    """
    predicted = (state.get("final_predicted_class") or "unknown").lower()
    confidence = state.get("final_confidence", 0.0)
    requires_review = state.get("requires_human_review", False)
    snomed_code = SNOMED.get(predicted, "404684003")  # 404684003 = clinical finding

    cnn = state.get("classification_result") or {}
    temperature = cnn.get("temperature", 1.0)
    iou = state.get("saliency_sam3_iou")

    obs = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "preliminary" if requires_review else "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": LOINC_AI_RESULT,
                "display": "AI Interpretation",
            }],
            "text": "AI Neuroimaging Classification",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "derivedFrom": [{"reference": f"ImagingStudy/{study_id}"}],
        "valueCodeableConcept": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "display": predicted,
            }]
        },
        "component": [
            {
                "code": {"text": "AI confidence"},
                "valueDecimal": round(confidence, 4),
            },
            {
                "code": {"text": "Calibration temperature (T)"},
                "valueDecimal": round(temperature, 4),
            },
        ],
    }

    if iou is not None:
        obs["component"].append({
            "code": {"text": "GradCAM++ / SAM3 mask IoU"},
            "valueDecimal": round(iou, 4),
        })

    return obs


def _build_diagnostic_report(
    report_id: str,
    patient_id: str,
    study_id: str,
    obs_id: str,
    state: dict,
) -> dict:
    """
    DiagnosticReport summarising all pipeline outputs.
    References the Observation for the AI prediction.
    """
    predicted = (state.get("final_predicted_class") or "unknown").lower()
    requires_review = state.get("requires_human_review", False)
    routing_path = state.get("routing_path", [])
    verification = state.get("verification_result")
    snomed_code = SNOMED.get(predicted, "404684003")

    report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"]
        },
        "status": "preliminary" if requires_review else "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "RAD",
                "display": "Radiology",
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": LOINC_BRAIN_MRI,
                "display": "MRI Brain",
            }]
        },
        "subject":       {"reference": f"Patient/{patient_id}"},
        "imagingStudy":  [{"reference": f"ImagingStudy/{study_id}"}],
        "result":        [{"reference": f"Observation/{obs_id}"}],
        "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
        "conclusion": state.get("final_report", ""),
        "conclusionCode": [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "display": predicted,
            }]
        }],
        "extension": [
            {
                "url": "https://example.org/fhir/StructureDefinition/ai-routing-path",
                "valueString": " → ".join(routing_path),
            },
            {
                "url": "https://example.org/fhir/StructureDefinition/ai-requires-human-review",
                "valueBoolean": requires_review,
            },
        ],
    }

    if verification:
        report["extension"].append({
            "url": "https://example.org/fhir/StructureDefinition/ai-verification",
            "extension": [
                {"url": "agreement",   "valueBoolean": verification.get("agreement")},
                {"url": "reasoning",   "valueString":  verification.get("reasoning", "")},
                {"url": "alternative", "valueString":  verification.get("alternative_diagnosis") or ""},
            ],
        })

    atlas = state.get("atlas_enrichment")
    if atlas:
        report["extension"].append({
            "url": "https://example.org/fhir/StructureDefinition/ebrains-atlas-assignment",
            "extension": [
                {"url": "parcellation",     "valueString":  "Julich-Brain 3.0"},
                {"url": "assigned-region",  "valueString":  atlas.get("assigned_region", "")},
                {"url": "hemisphere",       "valueString":  atlas.get("hemisphere", "")},
                {"url": "mni-coordinates",  "valueString":  str(atlas.get("mni_coords", []))},
                {"url": "top-candidates",   "valueString":  str(atlas.get("assignment_scores", [])[:3])},
            ],
        })

    saliency = state.get("explainability_result") or {}
    if saliency:
        report["presentedForm"] = [
            {"url": f"file://{path}", "title": name}
            for name, path in saliency.items() if path
        ]

    return report


# ── Public entry point ────────────────────────────────────────────────────────

def build_diagnostic_report(state: dict, output_dir: Optional[str] = None) -> dict:
    """
    Build a FHIR R4 Bundle (Patient + ImagingStudy + Observation + DiagnosticReport)
    from the final pipeline state.

    Returns the Bundle dict and optionally saves it as JSON to output_dir.
    """
    # Shared IDs for cross-resource references
    patient_id = f"patient-{uuid.uuid4().hex[:8]}"
    study_id   = f"study-{uuid.uuid4().hex[:8]}"
    obs_id     = f"obs-{uuid.uuid4().hex[:8]}"
    report_id  = f"report-{uuid.uuid4().hex[:8]}"

    patient = _build_patient(patient_id)
    study   = _build_imaging_study(study_id, patient_id, state)
    obs     = _build_observation(obs_id, patient_id, study_id, state)
    report  = _build_diagnostic_report(report_id, patient_id, study_id, obs_id, state)

    bundle = {
        "resourceType": "Bundle",
        "id": f"bundle-{uuid.uuid4().hex[:8]}",
        "type": "collection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": [
            {"resource": patient},
            {"resource": study},
            {"resource": obs},
            {"resource": report},
        ],
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / f"fhir_{report_id[:12]}.json"
        out.write_text(json.dumps(bundle, indent=2))
        print(f"[FHIR] Bundle written → {out}")

    return bundle
