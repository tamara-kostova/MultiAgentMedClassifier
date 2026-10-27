"""
Load task-aware few-shot examples from few_shot_examples.csv.

Returns a list of (PIL.Image, json_str) pairs ready to be injected as
prior conversation turns in MedGemma's _generate call. Missing image files
are silently skipped so the pipeline degrades gracefully.
"""

import csv
import json
from pathlib import Path

from PIL import Image

_CSV_PATH = Path(__file__).parent.parent / "few_shot_examples.csv"

# Fixed ICD-10 codes derivable from class + subclass
_ICD10 = {
    ("multiple sclerosis", None): "G35",
    ("stroke", "ischemic"): "I63",
    ("stroke", "hemorrhagic"): "I61",
    ("tumor", "glioma"): "C71",
    ("tumor", "meningioma"): "D32",
    ("tumor", "neurocytoma"): "D43.2",
    ("tumor", "schwannoma"): "D33.3",
    ("tumor", "pituitary_tumor"): "D35.2",
    ("tumor", "carcinoma"): "C71.9",
    ("tumor", "medulloblastoma"): "C71.6",
    ("tumor", "papilloma"): "D43.0",
    ("tumor", "granuloma"): "G06.0",
    ("tumor", "tuberculoma"): "G06.0",
    ("tumor", "germinoma"): "C71.9",
}

# Per-class severity hints for the few-shot ground-truth JSON
_SEVERITY = {
    "normal":              (None,  None),
    "multiple sclerosis":  (0.55,  0.70),
    "other abnormalities": (0.45,  0.60),
    "stroke":              (0.65,  0.75),
    "tumor":               (0.70,  0.72),
}

# Normalise subclass spellings that differ between the CSV and the schema
_SUBCLASS_MAP = {
    "neurocitoma": "neurocytoma",
    "pituitary":   "pituitary_tumor",
}

_TASK_FILTERS = {
    "binary_tumor": {
        "classes": {"normal", "tumor"},
        "datasets": {"figshare", "images-17", "images-44c"},
    },
    "multiclass_tumor": {
        "classes": {"normal", "tumor"},
        "datasets": {"figshare", "images-17", "images-44c"},
    },
    "ms": {
        "classes": {"normal", "multiple sclerosis"},
        "datasets": {"sclerosis"},
    },
    "stroke": {
        "classes": {"normal", "stroke"},
        "datasets": {"aisd", "stroke"},
    },
}


def _parse_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def _row_matches_task(row: dict, task: str | None) -> bool:
    if not task or task not in _TASK_FILTERS:
        return True
    filters = _TASK_FILTERS[task]
    cls = row.get("class", "").lower()
    dataset = row.get("dataset", "").lower()
    return cls in filters["classes"] and dataset in filters["datasets"]


def _selection_key(row: dict, task: str | None) -> str:
    cls = row.get("class", "")
    subclass = row.get("subclass", "").lower()
    if task in {"multiclass_tumor", "stroke"} and subclass:
        return f"{cls}:{_SUBCLASS_MAP.get(subclass, subclass)}"
    return cls


def load_few_shot_examples(
    data_dir: str | None,
    csv_path: Path = _CSV_PATH,
    task: str | None = None,
) -> list[tuple[Image.Image, str]]:
    """
    Returns task-relevant (image, json_str) pairs.

    For binary tasks this selects one example per class. For multiclass tumor
    and stroke it keeps one example per available subtype as well, because
    MedGemma must emit diagnosis_detailed for those tasks.
    """
    rows = [row for row in _parse_csv(csv_path) if _row_matches_task(row, task)]
    if not rows:
        rows = _parse_csv(csv_path)

    seen: set[str] = set()
    selected: list[dict] = []
    for row in rows:
        key = _selection_key(row, task)
        if key and key not in seen:
            seen.add(key)
            selected.append(row)

    examples: list[tuple[Image.Image, str]] = []
    base = Path(data_dir) if data_dir else None

    for row in selected:
        cls        = row["class"]
        subclass   = row.get("subclass") or None
        modality   = row.get("modality") or None
        seq        = row.get("modality_subtype") or None
        plane      = row.get("plane") or None
        rel_path   = row.get("file_path", "")

        if subclass:
            subclass = _SUBCLASS_MAP.get(subclass.lower(), subclass.lower()) or None

        # Sequence is only valid for MRI
        if modality and modality.upper() == "CT":
            seq = None

        # Resolve image path
        img_path = None
        if base and rel_path:
            candidate = base / rel_path
            if candidate.exists():
                img_path = candidate
        if img_path is None:
            if rel_path:
                print(
                    f"[few_shot_loader] Skipping '{cls}' example — "
                    f"image not found: {rel_path} (data_dir={data_dir})"
                )
            continue

        # Build ground-truth JSON
        sev_score, sev_conf = _SEVERITY.get(cls, (None, None))
        icd10 = _ICD10.get((cls, subclass)) or _ICD10.get((cls, None))

        gt = {
            "modality":            modality,
            "specialized_sequence": seq or None,
            "plane":               plane or None,
            "diagnosis_name":      cls,
            "diagnosis_detailed":  subclass,
            "icd10_code":          icd10,
            "severity_score":      sev_score,
            "diagnosis_confidence": 0.95,
            "severity_confidence": sev_conf,
        }

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"[few_shot_loader] Could not open {img_path}: {exc}")
            continue

        examples.append((image, json.dumps(gt, indent=2)))

    return examples
