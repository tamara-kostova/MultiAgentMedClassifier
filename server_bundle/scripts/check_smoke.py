"""
Check the rows written by server_bundle/10_smoke.sh before committing GPU-days.

Every v2 mode has run on a few images; this verifies that each output row carries
what the analysis and the paper need (provenance, explainability, per-round debate
verdicts, sampled forest votes, ...), and projects the wall clock of the full runs
from the measured latency. Exit code 0 and "SMOKE OK" mean the long runs can start.

    python server_bundle/scripts/check_smoke.py --dir outputs/eval/v2/smoke --n 6
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

FAILURES: list[str] = []
WARNINGS: list[str] = []

# Full-run steps per system and the number of tasks each runs on.
FULL_RUN_STEPS = {"base": 4, "forest": 4, "debate": 4, "homog": 4}
# A parse-failure rate above this on a handful of images means the prompts/token
# budgets are broken on this setup, not that MedGemma is occasionally verbose.
MAX_PARSE_FAIL_RATE = 1 / 3


def fail(name: str, msg: str) -> None:
    print(f"  [FAIL] {name}: {msg}")
    FAILURES.append(f"{name}: {msg}")


def warn(name: str, msg: str) -> None:
    print(f"  [warn] {name}: {msg}")
    WARNINGS.append(f"{name}: {msg}")


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rate(rows: list[dict], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / max(1, len(rows))


def check_common(name: str, rows: list[dict], n: int) -> None:
    if len(rows) != n:
        fail(name, f"{len(rows)} rows, expected {n}")
    errors = [r.get("error") for r in rows if r.get("error")]
    if errors:
        fail(name, f"{len(errors)} rows with error, e.g. {errors[0][:200]}")
    if any(not r.get("predicted_class") for r in rows if not r.get("error")):
        fail(name, "rows without a prediction")
    rc = [r.get("run_config") or {} for r in rows]
    if not all(rc):
        fail(name, "rows without run_config")
        return
    if any(c.get("git_commit") is None for c in rc):
        fail(name, "run_config.git_commit is None — CODE_VERSION missing from the bundle")
    if any(c.get("git_dirty") for c in rc):
        fail(name, "run_config.git_dirty is true — bundle was packed from uncommitted code")
    if any((c.get("image_list") or {}).get("sha256") is None for c in rc):
        fail(name, "run_config.image_list missing — run was not restricted to the published images")
    for key in ("triage_parse_failed", "verification_parse_failed", "report_parse_failed",
                "judge_parse_failed"):
        if rate(rows, key) > MAX_PARSE_FAIL_RATE:
            fail(name, f"{key} on {rate(rows, key):.0%} of rows")
        elif rate(rows, key) > 0:
            warn(name, f"{key} on {rate(rows, key):.0%} of rows")


def check_explainability(name: str, rows: list[dict]) -> None:
    rc = rows[0].get("run_config") or {}
    if not rc.get("generate_explainability"):
        fail(name, "generate_explainability is off — systems would not be matched")
    if any(not r.get("gradcam_pp_path") for r in rows):
        fail(name, "rows without a Grad-CAM++ map")


def check_verification(name: str, rows: list[dict]) -> None:
    missing = [r for r in rows if r.get("verification_agreement") is None
               and not r.get("verification_parse_failed")]
    if missing:
        fail(name, f"{len(missing)} rows without a verification result")


def check_sam3_empty_mask(name: str, rows: list[dict]) -> None:
    bad = [r for r in rows if r.get("sam3_mask_empty") and r.get("sam3_bbox")]
    if bad:
        fail(name, "empty SAM3 mask still has a bbox (full-frame box bug)")


def check_debate(name: str, rows: list[dict], task: str) -> None:
    for r in rows:
        verdicts = r.get("debate_round_verdicts") or []
        if len(verdicts) != 2:
            fail(name, f"debate_round_verdicts has {len(verdicts)} rounds, expected 2")
            break
    advocates = [r.get("debate_advocates") or [] for r in rows]
    if task == "stroke" and any("sam" in a for a in advocates):
        fail(name, "SAM3 advocate took part on stroke (SAM3 is ineligible there)")
    if task == "binary_tumor" and not all("sam" in a for a in advocates):
        fail(name, "SAM3 advocate missing on binary_tumor")
    if any(r.get("debate_confidence") is not None and not 0 <= r["debate_confidence"] <= 1
           for r in rows):
        fail(name, "debate confidence outside [0, 1]")


def check_forest(name: str, rows: list[dict], sampled: bool, temperature: float) -> None:
    for r in rows:
        votes = r.get("forest_votes") or []
        if len(votes) != 4:
            fail(name, f"{len(votes)} forest votes, expected 4")
            return
        if sampled:
            if any(v.get("temperature") != temperature for v in votes):
                fail(name, f"vote temperature != {temperature}")
                return
            if len({v.get("seed") for v in votes}) != 4:
                fail(name, "the 4 sampled agents do not have distinct seeds")
                return
    if sampled:
        roles = {v.get("role") for r in rows for v in r["forest_votes"]}
        if roles != {"radiologist"}:
            fail(name, f"homogeneous forest has roles {sorted(roles)}")
        if not (rows[0].get("run_config") or {}).get("triage_only"):
            fail(name, "homogeneous forest is not triage_only")
        varied = any(
            len({(v.get("diagnosis_name"), v.get("diagnosis_detailed"), v.get("diagnosis_confidence"))
                 for v in r["forest_votes"]}) > 1
            for r in rows
        )
        if not varied:
            warn(name, "all 4 sampled votes are identical on every image — is sampling active?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.jsonl"))
    if not files:
        print(f"no smoke outputs in {args.dir}")
        return 1

    latency: dict[str, list[float]] = {}
    for path in files:
        stem = path.stem                      # e.g. binary_tumor_debate
        task, system = stem.rsplit("_", 1)
        name = f"{system}/{task}"
        print(f"\n── {name}  ({path})")
        rows = load(path)
        check_common(name, rows, args.n)
        if not rows:
            continue
        if system in ("base", "forest", "debate"):
            check_explainability(name, rows)
        if system in ("base", "forest"):
            check_verification(name, rows)
        if task.endswith("tumor"):
            check_sam3_empty_mask(name, rows)
        if system == "debate":
            check_debate(name, rows, task)
        if system == "forest":
            check_forest(name, rows, sampled=False, temperature=args.temperature)
        if system == "homog":
            check_forest(name, rows, sampled=True, temperature=args.temperature)
        lat = [r["latency_s"] for r in rows if r.get("latency_s")]
        if lat:
            med = statistics.median(lat)
            latency.setdefault(system, []).append(med)
            print(f"  median latency {med:.1f} s/image → {med * 500 / 3600:.1f} h per 500 images")

    print("\n── Projected wall clock for the full runs (500 images each, one GPU)")
    total = 0.0
    for system, n_tasks in FULL_RUN_STEPS.items():
        if system in latency:
            hours = statistics.mean(latency[system]) * 500 / 3600 * n_tasks
            total += hours
            print(f"  {system:8s} x{n_tasks}: ~{hours:.0f} h")
    print(f"  total     : ~{total:.0f} GPU-hours")

    print()
    for w in WARNINGS:
        print(f"  warning: {w}")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAILED: {f}")
        print("\n  SMOKE FAILED — do not start the long runs. Send logs/ back.")
        return 1
    print("  SMOKE OK — the long runs can be started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
