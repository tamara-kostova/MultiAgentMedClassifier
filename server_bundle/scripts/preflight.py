"""
Preflight check for the faculty GPU server runs.

Verifies — in this order, cheapest first — that everything the six evaluation runs
need is present and working, then actually pushes one image through the Forest
pipeline and one through the Debate pipeline. Prints a measured seconds/image
figure so the wall clock of the full runs can be projected before committing a
night of GPU time.

Run it through the container from the project root:

    bash server_bundle/00_preflight.sh

Exit code 0 and a final "PREFLIGHT OK" line mean the long runs can be started.
Any FAIL line means they should not be, and the log should go back to Tamara.

Model weights are loaded exactly once and shared by both smoke tests, so this uses
the same ~14 GB of VRAM as a real run (a real run is one process, one MedGemma).
"""

import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def section(title: str) -> None:
    print()
    print("─" * 78)
    print(f"  {title}")
    print("─" * 78)


def ok(msg: str) -> None:
    print(f"  [ OK ]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL]   {msg}")
    FAILURES.append(msg)


def warn(msg: str) -> None:
    print(f"  [warn]   {msg}")
    WARNINGS.append(msg)


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


# ── 1. Software environment ───────────────────────────────────────────────────
def check_environment() -> None:
    section("1. Software environment")
    print(f"  python           {sys.version.split()[0]}  ({sys.executable})")

    import torch

    print(f"  torch            {torch.__version__}  (CUDA build {torch.version.cuda})")
    try:
        import transformers

        print(f"  transformers     {transformers.__version__}")
        import numpy

        print(f"  numpy            {numpy.__version__}")
        if numpy.__version__.startswith("2."):
            fail(
                "numpy 2.x detected — SAM3 requires numpy<2. Host site-packages are "
                "probably leaking in; PYTHONNOUSERSITE=1 must be set."
            )
        import open_clip

        print(f"  open_clip        {open_clip.__version__}")
        import pandas

        print(f"  pandas           {pandas.__version__}")
    except Exception as exc:  # pragma: no cover - environment probe
        fail(f"library import failed: {exc}")

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False — was the container run with --nv?")
        return

    n_gpu = torch.cuda.device_count()
    print(f"  visible GPUs     {n_gpu}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"    GPU {i}: {props.name}  {total_gb:.1f} GB  sm_{props.major}{props.minor}")
        if i == 0:
            if total_gb < 12:
                warn(
                    f"GPU 0 has only {total_gb:.1f} GB — set LOAD_4BIT=1 in "
                    "server_bundle/config.env before starting the runs."
                )
            elif total_gb < 15:
                warn(
                    f"GPU 0 has {total_gb:.1f} GB — enough for MedGemma in bfloat16, but "
                    "tight together with SAM3. Set LOAD_4BIT=1 if a run dies with OOM."
                )
            else:
                ok(f"GPU 0 memory sufficient ({total_gb:.1f} GB)")

    try:
        x = torch.empty(1024, 1024, device="cuda")
        del x
        torch.cuda.empty_cache()
        ok("CUDA allocation works")
    except Exception as exc:
        fail(f"CUDA allocation failed: {exc}")

    print(f"  HF_HOME          {os.environ.get('HF_HOME', 'unset')}")
    print(f"  HF_HUB_OFFLINE   {os.environ.get('HF_HUB_OFFLINE', 'unset')}")
    print(f"  CHECKPOINT_SOURCE{os.environ.get('CHECKPOINT_SOURCE', 'unset'):>9}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        warn("HF_HUB_OFFLINE is not 1 — the run may try to reach the internet.")


# ── 2. Datasets ───────────────────────────────────────────────────────────────
DATASETS = [
    ("binary_tumor", "DATA_BR35H", "data/Br35H", "br35h"),
    ("multiclass_tumor", "DATA_FIGSHARE", "data/figshare", "figshare3"),
    ("ms", "DATA_MS", "data/sclerosis/MS", "ms_binary"),
    ("stroke", "DATA_STROKE", "data/stroke/Brain_Stroke_CT_Dataset", "stroke_binary"),
]


def check_datasets() -> dict[str, str]:
    section("2. Datasets")
    from eval.tumor_eval import (
        LABEL_MAPS,
        _class_counts,
        _interleave_by_label,
        load_dataset,
    )

    max_samples = int(os.environ.get("PREFLIGHT_MAX_SAMPLES", "500"))
    usable: dict[str, str] = {}

    for task, env_name, default_dir, label_map_name in DATASETS:
        data_dir = os.environ.get(env_name, default_dir)
        if not Path(data_dir).is_dir():
            fail(f"{task}: directory not found: {data_dir}")
            continue

        samples = load_dataset(data_dir, task)
        if not samples:
            fail(f"{task}: no images found under {data_dir}/<class>/")
            continue

        counts = _class_counts(samples)
        selected = _interleave_by_label(samples)[:max_samples]
        sel_counts = _class_counts(selected)
        label_map = LABEL_MAPS.get(label_map_name, {})
        unmapped = [c for c in counts if c not in label_map]

        print(f"  {task:<18} {data_dir}")
        print(f"    {len(samples)} images   classes={counts}")
        print(f"    first {max_samples} selected: {sel_counts}")
        print(f"    example: {selected[0]['image_path']}")
        if unmapped:
            fail(
                f"{task}: class folder(s) {unmapped} are not in label map "
                f"'{label_map_name}' — labels would be wrong. Check the folder names."
            )
        else:
            ok(f"{task}: {len(selected)} images selected, all class folders recognised")
            usable[task] = data_dir

    return usable


# ── 3. Checkpoints ────────────────────────────────────────────────────────────
def check_checkpoints() -> None:
    section("3. Checkpoints")
    from config import DEFAULT_CONFIG

    model_cfg = DEFAULT_CONFIG.model
    required = {
        f"CNN {task}": path for task, path in model_cfg.cnn_checkpoints.items()
    }
    required.update(
        {
            f"BiomedCLIP probe {task}": path
            for task, path in model_cfg.biomedclip_probe_checkpoints.items()
        }
    )

    for name, path in sorted(required.items()):
        p = Path(path)
        if p.is_file():
            ok(f"{name:<32} {path}  ({human_size(p.stat().st_size)})")
        else:
            fail(f"{name:<32} MISSING: {path}")

    sam3_probe = Path(model_cfg.sam3_linear_probe_checkpoint or "")
    if sam3_probe.is_file():
        ok(f"{'SAM3 probe':<32} {sam3_probe}  ({human_size(sam3_probe.stat().st_size)})")
    else:
        warn(
            f"SAM3 probe missing: {sam3_probe} — segmentation will be skipped, which "
            "breaks comparability with the finished Forest tumour runs."
        )


# ── 4. SAM3 availability ──────────────────────────────────────────────────────
def check_sam3() -> None:
    section("4. SAM3 (segmentation — needed for the two tumour debate runs)")
    from config import DEFAULT_CONFIG

    bpe = Path(DEFAULT_CONFIG.model.sam3_bpe_path or "")
    if bpe.is_file():
        ok(f"BPE vocabulary present: {bpe}")
    else:
        warn(f"BPE vocabulary missing: {bpe} — SAM3 will be skipped.")

    import agents.sam3_tool as sam3_tool

    if getattr(sam3_tool, "_SAM_AVAILABLE", False):
        ok("`import sam3.model_builder` works (vendored ./sam3 found)")
    else:
        warn(
            "SAM3 source not importable: "
            f"{getattr(sam3_tool, '_SAM_IMPORT_ERROR', 'unknown error')} — "
            "segmentation will be skipped for the tumour runs."
        )

    weights = Path(os.environ.get("HF_HOME", "hf_cache")) / "hub" / "models--facebook--sam3"
    if weights.is_dir():
        ok(f"SAM3 backbone weights cached: {weights}")
    else:
        warn(f"SAM3 backbone weights not in the offline cache: {weights}")


# ── 5+6. End-to-end smoke tests ───────────────────────────────────────────────
def smoke_tests(usable: dict[str, str]) -> None:
    section("5. Loading models (this is the real test of the offline cache)")

    from config import DEFAULT_CONFIG, ModelConfig, PipelineConfig, RoutingConfig
    from eval.tumor_eval import (
        LABEL_MAPS,
        _interleave_by_label,
        extract_row,
        load_dataset,
    )
    from pipeline.graph import (
        assemble_debate_pipeline,
        assemble_forest_pipeline,
        load_agents,
    )
    from pipeline.state import initial_state

    use_4bit = os.environ.get("MEDGEMMA_4BIT", "").lower() in ("1", "true", "yes")
    model_cfg = ModelConfig(use_4bit_quantization=use_4bit)
    cfg = PipelineConfig(model=model_cfg, routing=RoutingConfig(), output_dir="outputs")

    t0 = time.perf_counter()
    try:
        medgemma, cnn, sam3, clip = load_agents(cfg)
    except Exception as exc:
        fail(f"model loading failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    ok(f"all agents loaded in {time.perf_counter() - t0:.0f} s")

    import torch

    if torch.cuda.is_available():
        print(
            f"  VRAM in use: {torch.cuda.memory_allocated()/1024**3:.1f} GB allocated, "
            f"{torch.cuda.memory_reserved()/1024**3:.1f} GB reserved"
        )

    # Pick a task that actually has data; prefer a tumour task so SAM3 is exercised.
    for task in ("multiclass_tumor", "binary_tumor", "stroke", "ms"):
        if task in usable:
            smoke_task, smoke_dir = task, usable[task]
            break
    else:
        fail("no usable dataset — cannot run the end-to-end smoke tests")
        return

    label_map = LABEL_MAPS.get(
        {"binary_tumor": "br35h", "multiclass_tumor": "figshare3",
         "ms": "ms_binary", "stroke": "stroke_binary"}[smoke_task],
        {},
    )
    samples = _interleave_by_label(load_dataset(smoke_dir, smoke_task))[:2]

    for label, build, out_name in (
        ("Forest (N=4)", lambda: assemble_forest_pipeline(medgemma, cnn, sam3, clip, cfg, n_agents=4),
         "outputs/eval/_preflight_forest.jsonl"),
        ("Debate (R=2)", lambda: assemble_debate_pipeline(medgemma, cnn, sam3, clip, cfg, rounds=2),
         "outputs/eval/_preflight_debate.jsonl"),
    ):
        section(f"6. End-to-end smoke test — {label} on 1 {smoke_task} image")
        try:
            app = build()
            sample = samples[0]
            t0 = time.perf_counter()
            final_state = app.invoke(initial_state(sample["image_path"], smoke_task))
            latency = time.perf_counter() - t0
            row = extract_row(sample, final_state, latency, label_map)

            Path(out_name).parent.mkdir(parents=True, exist_ok=True)
            import json

            with open(out_name, "w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            print(f"    image              {sample['image_path']}")
            print(f"    true label         {row['true_label_canonical']}")
            print(f"    prediction         {row['predicted_class']!r} -> {row['predicted_class_canonical']!r}")
            print(f"    confidence         {row['final_confidence']}")
            print(f"    routing path       {row['routing_path']}")
            print(f"    SAM3 mask          {row['sam3_mask_path']}  (skipped={row['sam3_skipped']})")
            print(f"    BiomedCLIP         {row['biomedclip_top_label']!r} ({row['biomedclip_mode']})")
            if "Forest" in label:
                print(f"    dissent_rate       {row['dissent_rate']}")
                print(f"    vote_fraction      {row['vote_fraction']}")
            else:
                print(f"    debate rounds      {row['debate_rounds_completed']}")
                print(f"    verdict changed    {row['debate_round_changed']}")
                print(f"    debate winner      {row['debate_winner']}")
            print(f"    latency            {latency:.1f} s/image")

            if row["error"] is not None:
                fail(f"{label}: pipeline returned an error: {row['error']}")
            elif not row["predicted_class"]:
                fail(f"{label}: no prediction produced")
            else:
                ok(f"{label} ran end to end in {latency:.1f} s")
                per_500 = latency * 500 / 3600
                print(f"    → projected {per_500:.1f} h for 500 images on this GPU")
                if "Forest" in label:
                    print(f"    → the two Forest runs would take about {2 * per_500:.0f} h in total")
                else:
                    print(f"    → the four Debate runs would take about {4 * per_500:.0f} h in total")
                if per_500 > 20:
                    warn(
                        f"{label} projects {per_500:.0f} h per 500 images. Consider lowering "
                        "MAX_SAMPLES in server_bundle/config.env (e.g. to 300)."
                    )
            print(f"    record written to  {out_name}")
        except Exception as exc:
            fail(f"{label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()


def main() -> int:
    print("=" * 78)
    print("  MultiAgentMedClassifier — PREFLIGHT")
    print(f"  project root: {PROJECT_ROOT}")
    print("=" * 78)

    check_environment()
    usable = check_datasets()
    check_checkpoints()
    check_sam3()
    if not FAILURES:
        smoke_tests(usable)
    else:
        section("5+6. Skipping model loading and smoke tests")
        print("  Earlier checks failed — fix those first.")

    section("Summary")
    for w in WARNINGS:
        print(f"  warning: {w}")
    if FAILURES:
        print()
        for f_msg in FAILURES:
            print(f"  FAILED: {f_msg}")
        print()
        print("  PREFLIGHT FAILED — do not start the long runs.")
        print("  Please send this log (logs/00_preflight.log) back to Tamara.")
        return 1

    print()
    print("  PREFLIGHT OK — the evaluation runs can be started.")
    if WARNINGS:
        print("  (Warnings above are not blocking, but worth sending back.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
