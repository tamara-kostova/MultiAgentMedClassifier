"""Single source of truth for label canonicalization.

Every scorer (eval_analysis, tumor_eval, evaluate, paired_system_comparison, the
stage pipeline's aggregation) imports canonical_label from here, so a stored
canonical label and a re-computed one can never disagree. This is the version
eval_analysis.py used for the paper tables. Keep this module free of heavy imports:
pipeline stages import it at request time.
"""

TUMOR_SUBTYPES = (
    "carcinoma", "germinoma", "glioma", "granuloma", "medulloblastoma",
    "meningioma", "neurocytoma", "papilloma", "schwannoma", "tuberculoma",
)
STROKE_TOKENS = ("stroke", "ischemic", "ischemia", "hemorrhag", "bleeding", "infarct")
# MedGemma names MS lesions descriptively rather than using the disease name, e.g.
# "demyelinating plaque", "periventricular white matter lesion". On a FLAIR MS
# benchmark these are MS assertions, so they must canonicalize to "ms" or strict
# scoring counts them as negatives.
MS_TOKENS = ("multiple sclerosis", "demyelinat", "white matter lesion", "white matter plaque")


def canonical_label(value: object, task: str | None = None) -> str:
    """Normalize labels/predictions so metrics survive wording differences.

    The prompt schema uses the literal string "null" as the sentinel for an
    indeterminable field, so it normalizes to "" (absent) rather than being
    treated as a class name.
    """
    text = str(value or "").strip().lower()
    if not text or text in ("none", "null", "nan"):
        return ""
    n = (text.replace("_", " ").replace("-", " ")
              .replace("/", " ").replace("tumour", "tumor"))
    # "abnormal" / "other abnormalities" contains the substring "normal" — catch it
    # first, or an abnormality assertion is scored as a normal read. The debate
    # judge's schema offers "other abnormalities" as a verdict.
    if "abnormal" in n:
        return "abnormal"
    if "normal" in n or "control" in n:
        return "normal"
    if n == "ms" or n.startswith("ms ") or any(tok in n for tok in MS_TOKENS):
        return "ms"
    if any(tok in n for tok in STROKE_TOKENS):
        return "stroke"
    subtype = next((s for s in TUMOR_SUBTYPES if s in n), None)
    tumorish = (subtype is not None or "pituitary" in n or "brain tumor" in n
                or n == "tumor" or " tumor" in n)
    if task == "binary_tumor" and tumorish:
        return "tumor"
    if "pituitary" in n:
        return "pituitary_tumor"
    if subtype:
        return subtype
    if tumorish:
        return "tumor"
    return n
