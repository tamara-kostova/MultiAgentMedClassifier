"""
Analysis of MedGemma free-text final_report fields from tumor_eval JSONL.

Usage:
    python eval/report_analysis.py \
        --jsonl outputs/eval/binary_tumor_tumor_eval.jsonl \
        [--output_dir outputs/analysis/reports]

Sections
────────
1.  Report structure inventory
        (section presence, length distribution)
2.  Embedded structured-diagnosis extraction + consistency check
        (JSON block inside report text vs final_medgemma_diagnosis state field)
3.  Hallucination analysis — tumor findings in normal scans
        (false-positive clinical language in FINDINGS section)
4.  Missed-finding analysis — absent findings in tumor reports
5.  Key clinical phrase frequency by true class + correctness
6.  Recommendation patterns ("contrast-enhanced MRI", etc.)
7.  Internal text↔diagnosis consistency
        (does FINDINGS text agree with the report's own structured diagnosis?)
8.  Report length analysis (by class, correctness, confidence)
9.  Report quality scoring (composite)
Outputs: CSVs + analysis_report_text.md
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── label helpers ─────────────────────────────────────────────────────────────

GT_MAP = {"yes": "tumor", "no": "normal"}

TUMOR_FINDING_WORDS = [
    "lesion", "mass", "tumor", "tumour", "glioma", "meningioma", "pituitary",
    "schwannoma", "granuloma", "cyst", "mass effect", "midline shift",
    "edema", "oedema", "necrosis", "enhancement", "ring-enhancing",
    "heterogeneous", "hypointense lesion", "hyperintense lesion",
    "abnormal signal", "space-occupying",
]

NORMAL_FINDING_WORDS = [
    "normal", "no abnormality", "no lesion", "no mass", "unremarkable",
    "within normal limits", "no evidence of tumor", "no tumor",
]

RECOMMENDATION_PHRASES = [
    "contrast-enhanced mri", "further evaluation", "biopsy",
    "follow-up", "neurosurgical", "oncology", "radiation",
    "additional imaging", "clinical correlation",
]

SECTIONS = ["FINDINGS", "STRUCTURED DIAGNOSIS", "Confidence assessment",
            "Recommended next step", "Flags/caveats"]


# ── parsing helpers ───────────────────────────────────────────────────────────

def split_findings(report: str) -> str:
    """Extract text between FINDINGS: and the next all-caps section header."""
    if not report:
        return ""
    # Match FINDINGS section up to STRUCTURED DIAGNOSIS or another ALL-CAPS header
    m = re.search(
        r'FINDINGS[:\s]*(.*?)(?=\n[A-Z][A-Z\s/]+:|STRUCTURED DIAGNOSIS|\Z)',
        report, re.DOTALL | re.IGNORECASE
    )
    return m.group(1).strip() if m else ""


def extract_embedded_json(report: str) -> dict | None:
    """Parse the JSON block embedded after STRUCTURED DIAGNOSIS in the report text."""
    m = re.search(r'STRUCTURED DIAGNOSIS[^\{]*(\{[^}]+\})', report, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def has_section(report: str, section: str) -> bool:
    return bool(re.search(re.escape(section), report, re.IGNORECASE))


def count_phrases(text: str, phrases: list[str]) -> dict[str, bool]:
    tl = text.lower()
    return {p: (p in tl) for p in phrases}


def to_binary(label) -> str:
    if label is None:
        return "unknown"
    s = str(label).lower().strip()
    if s in ("normal", "normal brain mri"):
        return "normal"
    if s in ("nan", "none", "null", ""):
        return "unknown"
    return "tumor"


# ── section printer ───────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print("═" * 72)


# ── loader ────────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["gt"] = df["true_label"].map(GT_MAP)
    df["cnn_binary"] = df["cnn_predicted_class"].apply(to_binary)
    df["final_report"] = df["final_report"].fillna("")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# §1  Report structure inventory
# ══════════════════════════════════════════════════════════════════════════════

def section_structure(df: pd.DataFrame) -> pd.DataFrame:
    print_section("1. Report structure inventory")

    has_report = (df["final_report"].str.len() > 10).sum()
    print(f"  Reports present : {has_report} / {len(df)}")

    rows = []
    for sec in SECTIONS:
        n = df["final_report"].apply(lambda r: has_section(r, sec)).sum()
        rows.append({"section": sec, "n_present": n,
                     "pct": round(n / len(df) * 100, 1)})
    struct_df = pd.DataFrame(rows)
    print(struct_df.to_string(index=False))

    # Length distribution
    df["report_len"] = df["final_report"].str.len()
    lengths = df["report_len"]
    print(f"\n  Length (chars): mean={lengths.mean():.0f}  median={lengths.median():.0f}  "
          f"min={lengths.min()}  max={lengths.max()}  std={lengths.std():.0f}")
    for cls in ["tumor", "normal"]:
        l = lengths[df["gt"] == cls]
        print(f"    {cls:6s}: mean={l.mean():.0f}  median={l.median():.0f}")

    return struct_df


# ══════════════════════════════════════════════════════════════════════════════
# §2  Embedded JSON extraction + consistency with state field
# ══════════════════════════════════════════════════════════════════════════════

def section_embedded_json(df: pd.DataFrame) -> pd.DataFrame:
    print_section("2. Embedded structured-diagnosis JSON — extraction + consistency")

    embedded = df["final_report"].apply(extract_embedded_json)
    n_extracted = embedded.notna().sum()
    print(f"  Embedded JSON parsed: {n_extracted} / {len(df)}")

    # Compare diagnosis_name in embedded JSON vs final_medgemma_diagnosis state field
    state_diag = df["final_medgemma_diagnosis"].apply(
        lambda d: d.get("diagnosis_name") if isinstance(d, dict) else None
    )
    embedded_diag = embedded.apply(
        lambda d: d.get("diagnosis_name") if isinstance(d, dict) else None
    )

    both_present = embedded.notna() & state_diag.notna()
    match = (embedded_diag[both_present].astype(str) ==
             state_diag[both_present].astype(str))
    print(f"  diagnosis_name match (embedded vs state): "
          f"{match.sum()} / {both_present.sum()}  ({match.mean()*100:.1f}%)")

    # Mismatches
    mismatches = df[both_present & ~match][["true_label", "gt"]].copy()
    mismatches["embedded_diag"] = embedded_diag[both_present & ~match].values
    mismatches["state_diag"]    = state_diag[both_present & ~match].values
    if len(mismatches):
        print(f"\n  Mismatch examples (first 10):")
        print(mismatches.head(10).to_string(index=False))

    # Field-by-field comparison
    compare_fields = ["modality", "specialized_sequence", "plane",
                      "diagnosis_name", "diagnosis_detailed",
                      "diagnosis_confidence", "severity_score"]
    field_rows = []
    for field in compare_fields:
        emb_vals = embedded.apply(
            lambda d: str(d.get(field)) if isinstance(d, dict) else "missing")
        st_vals = df["final_medgemma_diagnosis"].apply(
            lambda d: str(d.get(field)) if isinstance(d, dict) else "missing")
        both = (emb_vals != "missing") & (st_vals != "missing")
        agree = (emb_vals[both] == st_vals[both]).mean() if both.any() else float("nan")
        field_rows.append({"field": field,
                           "n_both_present": int(both.sum()),
                           "agreement_rate": round(float(agree), 4)})
    field_df = pd.DataFrame(field_rows)
    print("\n  Field-by-field agreement (embedded report JSON vs state field):")
    print(field_df.to_string(index=False))

    return field_df, embedded


# ══════════════════════════════════════════════════════════════════════════════
# §3  Hallucination analysis — tumor language in normal reports
# ══════════════════════════════════════════════════════════════════════════════

def section_hallucination(df: pd.DataFrame) -> pd.DataFrame:
    print_section("3. Hallucination analysis — tumor findings in normal scan reports")

    findings_text = df["final_report"].apply(split_findings)
    df = df.copy()
    df["findings_text"] = findings_text

    normal_mask = df["gt"] == "normal"
    tumor_mask  = df["gt"] == "tumor"

    rows = []
    for phrase in TUMOR_FINDING_WORDS:
        in_normal = findings_text[normal_mask].str.lower().str.contains(
            re.escape(phrase), regex=True).sum()
        in_tumor  = findings_text[tumor_mask].str.lower().str.contains(
            re.escape(phrase), regex=True).sum()
        rows.append({
            "phrase": phrase,
            "in_normal_reports": in_normal,
            "pct_normal": round(in_normal / normal_mask.sum() * 100, 1),
            "in_tumor_reports": in_tumor,
            "pct_tumor": round(in_tumor / tumor_mask.sum() * 100, 1),
        })

    halluc_df = pd.DataFrame(rows).sort_values("in_normal_reports", ascending=False)
    print(halluc_df.to_string(index=False))

    # Per-sample hallucination score: # tumor words in FINDINGS of a normal scan
    def halluc_score(text: str) -> int:
        tl = text.lower()
        return sum(1 for w in TUMOR_FINDING_WORDS if w in tl)

    df["halluc_score"] = findings_text.apply(halluc_score)
    normal_scores = df.loc[normal_mask, "halluc_score"]
    tumor_scores  = df.loc[tumor_mask,  "halluc_score"]

    print(f"\n  Hallucination score (# tumor phrases in FINDINGS):")
    print(f"    Normal scans: mean={normal_scores.mean():.2f}  "
          f"median={normal_scores.median():.1f}  max={normal_scores.max()}")
    print(f"    Tumor  scans: mean={tumor_scores.mean():.2f}  "
          f"median={tumor_scores.median():.1f}  max={tumor_scores.max()}")

    # Hallucination rate by CNN correctness
    cnn_correct = (df["cnn_binary"] == df["gt"])
    for label, mask in [("CNN correct", cnn_correct), ("CNN wrong", ~cnn_correct)]:
        sub = df[mask & normal_mask]["halluc_score"]
        if len(sub):
            print(f"    Normal + {label}: n={len(sub)}  mean_halluc={sub.mean():.2f}")

    return halluc_df, df


# ══════════════════════════════════════════════════════════════════════════════
# §4  Missed findings — absent tumor language in tumor reports
# ══════════════════════════════════════════════════════════════════════════════

def section_missed_findings(df: pd.DataFrame) -> pd.DataFrame:
    print_section("4. Missed-finding analysis — absent tumor language in tumor reports")

    findings_text = df["final_report"].apply(split_findings)
    tumor_mask = df["gt"] == "tumor"

    # A tumor report "misses" if the FINDINGS section has no tumor-related words
    def finding_score(text: str) -> int:
        tl = text.lower()
        return sum(1 for w in TUMOR_FINDING_WORDS if w in tl)

    tumor_findings_scores = findings_text[tumor_mask].apply(finding_score)
    missed = (tumor_findings_scores == 0).sum()
    print(f"  Tumor scans where FINDINGS has ZERO tumor-related phrases: "
          f"{missed} / {tumor_mask.sum()}  ({missed/tumor_mask.sum()*100:.1f}%)")

    # FINDINGS phrase presence for tumor scans
    rows = []
    for phrase in TUMOR_FINDING_WORDS:
        n = findings_text[tumor_mask].str.lower().str.contains(
            re.escape(phrase), regex=True).sum()
        rows.append({
            "phrase": phrase,
            "n_in_tumor_findings": n,
            "pct_tumor": round(n / tumor_mask.sum() * 100, 1),
        })
    missed_df = pd.DataFrame(rows).sort_values("n_in_tumor_findings", ascending=False)
    print(missed_df.to_string(index=False))
    return missed_df


# ══════════════════════════════════════════════════════════════════════════════
# §5  Key phrase frequency by true class + correctness
# ══════════════════════════════════════════════════════════════════════════════

def section_phrase_frequency(df: pd.DataFrame) -> pd.DataFrame:
    print_section("5. Key clinical phrase frequency by true class + prediction correctness")

    cnn_correct = (df["cnn_binary"] == df["gt"])
    all_phrases = TUMOR_FINDING_WORDS + NORMAL_FINDING_WORDS + RECOMMENDATION_PHRASES

    rows = []
    for phrase in all_phrases:
        present = df["final_report"].str.lower().str.contains(
            re.escape(phrase), regex=True)
        row = {"phrase": phrase, "total": int(present.sum())}
        for cls in ["tumor", "normal"]:
            m = df["gt"] == cls
            row[f"in_{cls}"] = int(present[m].sum())
            row[f"pct_{cls}"] = round(present[m].mean() * 100, 1)
        for label, mask in [("correct", cnn_correct), ("wrong", ~cnn_correct)]:
            row[f"in_cnn_{label}"] = int(present[mask].sum())
        rows.append(row)

    freq_df = pd.DataFrame(rows).sort_values("total", ascending=False)
    print(freq_df[["phrase", "total", "in_tumor", "pct_tumor",
                   "in_normal", "pct_normal"]].to_string(index=False))
    return freq_df


# ══════════════════════════════════════════════════════════════════════════════
# §6  Recommendation patterns
# ══════════════════════════════════════════════════════════════════════════════

def section_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    print_section("6. Recommendation patterns")

    rows = []
    for phrase in RECOMMENDATION_PHRASES:
        present = df["final_report"].str.lower().str.contains(
            re.escape(phrase), regex=True)
        tumor_rate  = present[df["gt"] == "tumor"].mean()
        normal_rate = present[df["gt"] == "normal"].mean()
        rows.append({
            "phrase": phrase,
            "total": int(present.sum()),
            "pct_all": round(present.mean() * 100, 1),
            "pct_tumor": round(tumor_rate * 100, 1),
            "pct_normal": round(normal_rate * 100, 1),
            "tumor_vs_normal_diff": round((tumor_rate - normal_rate) * 100, 1),
        })
    rec_df = pd.DataFrame(rows).sort_values("total", ascending=False)
    print(rec_df.to_string(index=False))
    return rec_df


# ══════════════════════════════════════════════════════════════════════════════
# §7  Internal text↔diagnosis consistency
# ══════════════════════════════════════════════════════════════════════════════

def section_internal_consistency(df: pd.DataFrame, embedded: pd.Series) -> pd.DataFrame:
    print_section("7. Internal text↔diagnosis consistency")

    findings_text = df["final_report"].apply(split_findings)

    # Embedded structured diagnosis binary prediction
    emb_binary = embedded.apply(
        lambda d: to_binary(d.get("diagnosis_name")) if isinstance(d, dict) else "unknown"
    )

    # Does the FINDINGS text "sound like" the embedded diagnosis?
    def text_implies_tumor(text: str) -> bool:
        tl = text.lower()
        return any(w in tl for w in TUMOR_FINDING_WORDS)

    def text_implies_normal(text: str) -> bool:
        tl = text.lower()
        return any(w in tl for w in NORMAL_FINDING_WORDS)

    text_says_tumor  = findings_text.apply(text_implies_tumor)
    text_says_normal = findings_text.apply(text_implies_normal)

    rows = []
    for emb_cls in ["tumor", "normal"]:
        emb_mask = emb_binary == emb_cls
        n = emb_mask.sum()
        if n == 0:
            continue
        text_agree_tumor  = text_says_tumor[emb_mask].sum()
        text_agree_normal = text_says_normal[emb_mask].sum()
        contradicted = (
            (text_says_normal[emb_mask] & (emb_cls == "tumor")) |
            (text_says_tumor[emb_mask]  & (emb_cls == "normal"))
        ).sum()
        rows.append({
            "embedded_diagnosis": emb_cls,
            "n": int(n),
            "findings_mentions_tumor": int(text_agree_tumor),
            "findings_mentions_normal": int(text_agree_normal),
            "contradicted": int(contradicted),
            "contradiction_rate": round(contradicted / n, 4),
        })
    consist_df = pd.DataFrame(rows)
    print(consist_df.to_string(index=False))

    # Cases where FINDINGS text and embedded JSON disagree with each other
    contradiction = (
        (text_says_tumor  & (emb_binary == "normal")) |
        (text_says_normal & (emb_binary == "tumor"))
    )
    print(f"\n  Total FINDINGS↔embedded-JSON contradictions: {contradiction.sum()}")
    for cls in ["tumor", "normal"]:
        m = (df["gt"] == cls) & contradiction
        print(f"    true={cls}: {m.sum()}")

    return consist_df


# ══════════════════════════════════════════════════════════════════════════════
# §8  Report length analysis
# ══════════════════════════════════════════════════════════════════════════════

def section_length_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print_section("8. Report length analysis")

    df = df.copy()
    df["report_len"] = df["final_report"].str.len()
    cnn_correct = (df["cnn_binary"] == df["gt"])

    rows = []
    for label, mask in [
        ("all",            pd.Series([True] * len(df), index=df.index)),
        ("true_tumor",     df["gt"] == "tumor"),
        ("true_normal",    df["gt"] == "normal"),
        ("cnn_correct",    cnn_correct),
        ("cnn_wrong",      ~cnn_correct),
        ("tp",             (df["gt"] == "tumor")  & (df["cnn_binary"] == "tumor")),
        ("fp",             (df["gt"] == "normal") & (df["cnn_binary"] == "tumor")),
        ("tn",             (df["gt"] == "normal") & (df["cnn_binary"] == "normal")),
        ("fn",             (df["gt"] == "tumor")  & (df["cnn_binary"] == "normal")),
    ]:
        sub = df.loc[mask, "report_len"]
        if len(sub) == 0:
            continue
        rows.append({
            "subset": label,
            "n": len(sub),
            "mean": round(sub.mean(), 0),
            "median": round(sub.median(), 0),
            "std": round(sub.std(), 0),
            "min": sub.min(),
            "max": sub.max(),
        })
    len_df = pd.DataFrame(rows)
    print(len_df.to_string(index=False))

    # Correlation between report length and CNN confidence
    conf = df["cnn_confidence"].values
    length = df["report_len"].values
    corr = float(np.corrcoef(conf, length)[0, 1])
    print(f"\n  Pearson r(length, cnn_confidence): {corr:.4f}")

    return len_df


# ══════════════════════════════════════════════════════════════════════════════
# §9  Report quality scoring
# ══════════════════════════════════════════════════════════════════════════════

def section_quality_score(df: pd.DataFrame, embedded: pd.Series) -> pd.DataFrame:
    print_section("9. Report quality scoring (composite, 0–5)")

    """
    Quality criteria (1 point each):
      1. Has FINDINGS section
      2. Has parseable embedded STRUCTURED DIAGNOSIS JSON
      3. Embedded JSON diagnosis_name is not null
      4. No hallucination: normal scan → FINDINGS has no tumor words
         OR tumor scan → FINDINGS has at least one tumor word
      5. Has a recommendation (contrast-enhanced MRI or further evaluation etc)
    """
    findings_text = df["final_report"].apply(split_findings)

    def tumor_in_findings(text): return any(w in text.lower() for w in TUMOR_FINDING_WORDS)
    def has_recommendation(report): return any(p in report.lower() for p in RECOMMENDATION_PHRASES)

    scores = []
    detail_rows = []
    for i, row in df.iterrows():
        report = row["final_report"]
        gt     = row["gt"]
        emb    = embedded.iloc[i] if i < len(embedded) else None

        c1 = int(has_section(report, "FINDINGS"))
        c2 = int(isinstance(emb, dict))
        c3 = int(isinstance(emb, dict) and emb.get("diagnosis_name") not in (None, "null", "None"))
        ft = tumor_in_findings(findings_text.iloc[i])
        if gt == "normal":
            c4 = int(not ft)   # no hallucination
        else:
            c4 = int(ft)        # finding corroborated
        c5 = int(has_recommendation(report))

        total = c1 + c2 + c3 + c4 + c5
        scores.append(total)
        detail_rows.append({
            "has_findings": c1, "has_embedded_json": c2,
            "json_diagnosis_not_null": c3, "finding_correct": c4,
            "has_recommendation": c5, "quality_score": total,
        })

    df_q = pd.DataFrame(detail_rows, index=df.index)
    df_q["gt"] = df["gt"].values
    df_q["cnn_correct"] = (df["cnn_binary"] == df["gt"]).values

    print(f"  Mean quality score: {df_q['quality_score'].mean():.3f} / 5")
    for cls in ["tumor", "normal"]:
        sub = df_q[df_q["gt"] == cls]["quality_score"]
        print(f"    {cls:6s}: mean={sub.mean():.3f}  min={sub.min()}  max={sub.max()}")

    # Score distribution
    dist = df_q["quality_score"].value_counts().sort_index()
    print("\n  Score distribution:")
    for score, n in dist.items():
        print(f"    {score}/5 : {n:4d}  ({n/len(df)*100:.1f}%)")

    # Criterion breakdown
    print("\n  Criterion pass rates:")
    for col in ["has_findings", "has_embedded_json", "json_diagnosis_not_null",
                "finding_correct", "has_recommendation"]:
        rate = df_q[col].mean()
        print(f"    {col:30s}: {rate*100:.1f}%")

    return df_q[["quality_score", "has_findings", "has_embedded_json",
                 "json_diagnosis_not_null", "finding_correct", "has_recommendation", "gt"]]


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    jsonl_path: str, df: pd.DataFrame,
    struct_df, field_df, halluc_df, missed_df,
    freq_df, rec_df, consist_df, len_df, quality_df,
    out_dir: Path,
):
    n = len(df)
    n_tumor  = int((df["gt"] == "tumor").sum())
    n_normal = int((df["gt"] == "normal").sum())

    mean_q_tumor  = quality_df[quality_df["gt"] == "tumor"]["quality_score"].mean()
    mean_q_normal = quality_df[quality_df["gt"] == "normal"]["quality_score"].mean()

    top_halluc = halluc_df.head(5)
    top_rec    = rec_df.head(5)

    lines = [
        "# MedGemma Report Analysis",
        "",
        f"**Source:** `{jsonl_path}`  ",
        f"**N:** {n} ({n_tumor} tumor / {n_normal} normal)  ",
        "",
        "---",
        "",
        "## 1. Report structure",
        "",
        struct_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 2. Embedded JSON consistency (report text vs state field)",
        "",
        field_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 3. Hallucination — tumor language in NORMAL scan reports",
        "",
        halluc_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 4. Missed findings — tumor language in TUMOR scan reports",
        "",
        missed_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 5. Key phrase frequency by true class",
        "",
        freq_df[["phrase","total","in_tumor","pct_tumor","in_normal","pct_normal"]].to_markdown(index=False),
        "",
        "---",
        "",
        "## 6. Recommendation patterns",
        "",
        top_rec.to_markdown(index=False),
        "",
        "---",
        "",
        "## 7. Internal text↔diagnosis consistency",
        "",
        consist_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 8. Report length by subset",
        "",
        len_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 9. Report quality scores",
        "",
        f"- Mean quality score (tumor scans): **{mean_q_tumor:.3f} / 5**",
        f"- Mean quality score (normal scans): **{mean_q_normal:.3f} / 5**",
        "",
        quality_df["quality_score"].value_counts().sort_index().reset_index(
            ).rename(columns={"index":"score","quality_score":"n"}
            ).to_markdown(index=False),
        "",
        "---",
        "",
        "## Summary of report quality issues",
        "",
        "| Issue | Finding |",
        "|-------|---------|",
        f"| Hallucination rate (normal scans) | Top phrase: '{top_halluc.iloc[0]['phrase']}' appears in {top_halluc.iloc[0]['pct_normal']:.1f}% of normal reports |",
        f"| contrast-enhanced MRI recommendation | {rec_df[rec_df['phrase']=='contrast-enhanced mri'].iloc[0]['pct_normal']:.1f}% of normal scans get this recommendation |"
        if 'contrast-enhanced mri' in rec_df['phrase'].values else "",
        f"| Internal JSON agreement | diagnosis_name agrees in {field_df[field_df['field']=='diagnosis_name'].iloc[0]['agreement_rate']*100:.1f}% of cases |"
        if 'diagnosis_name' in field_df['field'].values else "",
        "",
    ]

    report = "\n".join(lines)
    out_path = out_dir / "analysis_report_text.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Report written to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyse MedGemma final_report text fields")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", default="outputs/analysis/reports")
    args = parser.parse_args()

    df  = load(args.jsonl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    struct_df                   = section_structure(df)
    field_df, embedded          = section_embedded_json(df)
    halluc_df, df               = section_hallucination(df)
    missed_df                   = section_missed_findings(df)
    freq_df                     = section_phrase_frequency(df)
    rec_df                      = section_recommendations(df)
    consist_df                  = section_internal_consistency(df, embedded)
    len_df                      = section_length_analysis(df)
    quality_df                  = section_quality_score(df, embedded)

    # ── save CSVs ──────────────────────────────────────────────────────────────
    saves = {
        "report_structure.csv":         struct_df,
        "embedded_json_consistency.csv": field_df,
        "hallucination_phrases.csv":    halluc_df,
        "missed_findings.csv":          missed_df,
        "phrase_frequency.csv":         freq_df,
        "recommendations.csv":          rec_df,
        "internal_consistency.csv":     consist_df,
        "report_length.csv":            len_df,
        "report_quality.csv":           quality_df,
    }
    for fname, data in saves.items():
        if data is not None and not data.empty:
            data.to_csv(out / fname, index=False)

    generate_report(
        jsonl_path=args.jsonl, df=df,
        struct_df=struct_df, field_df=field_df,
        halluc_df=halluc_df, missed_df=missed_df,
        freq_df=freq_df, rec_df=rec_df,
        consist_df=consist_df, len_df=len_df,
        quality_df=quality_df, out_dir=out,
    )

    print(f"\n{'─'*72}")
    print(f"All outputs in {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
