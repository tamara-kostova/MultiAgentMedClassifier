# PLAN — Faculty GPU server bundle (Singularity + .py scripts + data)

**Status:** in progress. Created 2026-07-29. Pick this file up to resume cold.

## Why this exists

To run the remaining Forest / Debate experiments on
GPU servers **from an account**. Requirements, from email:

1. Send the code as **`.py` scripts** plus the **data**; she launches them.
2. If execution is multi-step, send **exactly what to run and in what order**.
3. Results must be **saved to a file — she prefers CSV or TSV** — she sends them back when done.
4. A **Singularity container** must be provided (she builds via sylabs.io).
   No library/version restrictions; the only hard constraint is **CUDA 12.6 compatibility**.
5. No interactive debugging: whatever we send must run unattended.

Deliverable: a self-contained tarball + a `container.def` + numbered step scripts + a
Macedonian/English runbook, all under `server_bundle/`.

## Decisions already made (confirmed with Tamara)

| Question | Decision |
|---|---|
| Which runs | **Only the 6 remaining runs** from README "Guide — reproducing paper-comparable Forest / Debate results" (Forest N=4 on stroke + ms; Debate R=2 on all four tasks). No baseline re-runs. |
| Model weights | **Ship a prepacked offline HF cache** (`hf_cache/`). No HF token, no gated-model access and no internet needed on the compute node. |
| SAM3 | **Include it.** The two tumour debate runs have a SAM3 advocate, and the finished Forest tumour runs had SAM3 active (mask written for 500/500 and 465/465), so dropping it would break that comparison. |

## Facts established by inspection (do not re-derive)

**Known-good library versions** — from `.venv` on this machine, i.e. the environment that
produced the existing JSONLs. The container pins these, swapping only the CUDA build:

- Python **3.12.3**
- `torch==2.10.0`, `torchvision==0.25.0` → container uses **`+cu126` wheels**
  (verified present for cp312 at `https://download.pytorch.org/whl/cu126`)
- `transformers==5.3.0`, `accelerate==1.13.0`, `huggingface_hub==1.7.2`,
  `tokenizers==0.22.2`, `safetensors==0.7.0`, `bitsandbytes==0.49.2`
- `open_clip_torch==3.3.0`, `timm==1.0.25`, `numpy==1.26.4` (SAM3 requires numpy<2),
  `scipy==1.15.3`, `scikit-learn==1.8.0`, `pandas==3.0.1`, `matplotlib==3.10.8`,
  `seaborn==0.13.2`, `tabulate==0.10.0`
- `langgraph==1.1.3`, `langchain-core==1.2.20`, `fhir.resources==8.2.0`,
  `python-dotenv==1.2.2`, `opencv-python(-headless)==4.11.0.86`, `pillow==12.1.1`,
  `einops==0.8.2`, `ftfy==6.1.1`, `regex==2026.2.28`, `iopath==0.1.10`, `tqdm==4.67.3`
- NOT needed on the server: `gradio`, `albumentations`, `outlines`, `decord` — only
  `ui/demo.py` imports those, and it is not part of the eval path.

**Base image** — `docker://nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (tag verified to
exist; `-devel-` also exists if nvcc is ever needed). Ubuntu 24.04 ships Python 3.12.
Install into `/opt/venv` because 24.04's pip is PEP-668 managed. Set
**`PYTHONNOUSERSITE=1`** in `%environment` — Singularity bind-mounts `$HOME`, and this host
has a broken `~/.local` numpy 2.x that would otherwise shadow the container's numpy 1.26.

**Models to prepack into `hf_cache/`** (on-disk sizes, symlinks not double-counted):
- `google/medgemma-1.5-4b-it` — 8.1 GB (gated), 2 safetensors shards
- `facebook/sam3` — 3.3 GB (gated). **Restricted to `["sam3.pt", "config.json"]`**: the repo
  also ships `model.safetensors`, the same 3.3 GB of weights for the transformers-native
  `Sam3Model` that this pipeline never loads. `sam3.pt` + `config.json` is exactly what the
  cache held when the finished Forest tumour runs wrote masks for 500/500 images.
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` — 753 MB
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` — 260 KB, **restricted to tokenizer
  patterns**. Whole-repo would be 838 MB of PyTorch + Flax BERT weights that open_clip never
  loads (the text tower weights come from the BiomedCLIP checkpoint). Verified: BiomedCLIP
  creates offline with only `config.json` + `tokenizer_config.json` + `vocab.txt` present,
  full 109.7 M-param text tower included. Easy to miss entirely — open_clip needs it for the
  text tokenizer.

Both restrictions were added *after* a first whole-repo prepack run, which is how the two
duplicates were spotted: `hf_cache` came out at 17 GB instead of ~13 GB. Re-seeding after
deleting a repo dir is free — `hardlink_tree` shares inodes with `~/.cache`, verified.

`dir_size()` skips symlinks: the HF cache stores each file once in `blobs/` and links it from
`snapshots/`, so counting both inflates every number (a first run reported MedGemma as
24.1 GB and the whole cache as 40.1 GB).

**Checkpoints to ship** (~800 MB; prune the rest of `checkpoints/`, it holds many unused
architectures):
- `vgg16_MRI_tumor_binary_norm_final.pt` (513 MB), `densenet169_MRI_tumor_multiclass_norm_final.pt`,
  `resnet101_MRI_ms_norm_final.pt`, `densenet169_CT_stroke_binary_norm_final.pt`
- the four `linear_probe_BiomedCLIP_*_best.pt` probes
- `checkpoints/sam3_probe.pth` — **absent on this machine**; prepack must fetch it from
  `tamara-kostova/multiagentmed-tumor-segmentation`
  (`tumor_segmentation/sam3/sam3_linear_probe_tumor_segmentation_best.pt`).

**⚠ `pycocotools` is mandatory too** — `sam3/sam3/visualization_utils.py` and
`sam3/sam3/train/*` import `pycocotools.mask`, and `sam3.model_builder` pulls them in, but
upstream SAM3's `pyproject.toml` does **not** declare it. Same silent-failure mode as the
setuptools issue. Pinned `pycocotools==2.0.11`. Also absent from this machine's `.venv`.
Verified in a clean container build: with `setuptools<81` + `pycocotools`,
`agents.sam3_tool._SAM_AVAILABLE == True`.

**⚠ `setuptools<81` is mandatory** — upstream `sam3/sam3/model_builder.py:8` does
`import pkg_resources`, which setuptools **removed in 81**. `agents/sam3_tool.py` catches the
`ModuleNotFoundError` silently and just reports `_SAM_AVAILABLE=False`, so a newer setuptools
means segmentation is skipped with no error anywhere except preflight's SAM3 line. Pinned in
both `container.def` (the pip bootstrap) and `requirements-server.txt`.
**This also affects this machine**: `.venv` has setuptools 82.0.1, so SAM3 cannot currently
load locally either — run `pip install "setuptools<81"` in `.venv` to restore it. (The
finished Forest tumour runs predate that upgrade, which is why they have masks.)

**SAM3 source** — `agents/sam3_tool.py` puts `<repo>/sam3` on `sys.path` when it exists, so
vendoring a clone of `https://github.com/facebookresearch/sam3` at `./sam3` is enough; no
pip install, no network at build time. Asset `sam3/assets/bpe_simple_vocab_16e6.txt.gz`
confirmed in that repo, matching the config default
`sam3_bpe_path="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"` (relative to repo root, so
runs must use the repo root as cwd). Its deps (timm, ftfy, regex, iopath, tqdm,
typing_extensions, numpy<2) are already in the container.

**VRAM per process** ≈ 14 GB: `build_forest_pipeline`/`build_debate_pipeline` construct
**one shared `MedGemmaAgent`** (~9 GB bf16) — agents are role prompts over the same model,
not N model copies — plus SAM3 (~3.5 GB) and CNN/BiomedCLIP (~1 GB). So: one run per
≥16 GB GPU; two per 40/80 GB GPU. If the server GPU is smaller, `--load_4bit`.

**Data layout on the server** — Ship data in the layout recorded in the existing
JSONL `image_path` fields (confirmed):

| Task | dir | class folders |
|---|---|---|
| `binary_tumor` | `data/Br35H` | `yes`, `no` |
| `multiclass_tumor` | `data/figshare` | `1`, `2`, `3` |
| `ms` | `data/sclerosis/MS` | `MS Axial_crop`, `MS Saggital_crop`, `Control Axial_crop`, `Control Saggital_crop` |
| `stroke` | `data/stroke/Brain_Stroke_CT_Dataset` | `Bleeding`, `Ischemia`, `Normal` (images under `<class>/PNG/`) |

Locally these same datasets currently live under different names — `data/Br35H`,
`data/processed`, `data/MS`, `data/Brain_Stroke_CT_Dataset` — so the packing script must
**rename into the table above** while copying. Local class counts:
Br35H 1500/1500; figshare 708/1426/930; MS 1002/1014/650/761;
stroke Bleeding 1093 / Ischemia 1130 / Normal 4427 (PNG folders).
Pruned data payload ≈ **1.2 GB** (Br35H 72 MB + figshare 150 MB + MS 432 MB + stroke PNG 539 MB).

**Do not ship** `*_mask`, `OVERLAY`, `DICOM`, `MASKS`, `External_Test`, `pred`,
`Br35H-Mask-RCNN` — `eval/tumor_eval.py:_DEFAULT_EXCLUDE_DIRS` skips them anyway, and they
are most of the raw dataset size (stroke drops 1.5 GB → 539 MB).

**⚠ Finding worth carrying into the paper:** the existing stroke baseline
`outputs/eval/stroke_dataset_eval.jsonl` (n=1000) ran on `Bleeding/OVERLAY/*.png`, and
those images have **the lesion painted on in red/green** — verified by opening
`data/Brain_Stroke_CT_Dataset/Bleeding/OVERLAY/10002.png`. That is label leakage, and its
numbers are not comparable to any new stroke run: `OVERLAY` was later added to
`_DEFAULT_EXCLUDE_DIRS`, so new runs read the clean `PNG/` folder. Tamara chose not to
re-run baselines in this campaign — so **the stroke baseline must be re-run (or its numbers
flagged/dropped) before the stroke forest/debate comparison goes in the paper.**

**Total transfer** ≈ 14.5 GB (hf_cache 12.5 + data 1.2 + checkpoints 0.8 + code). Pack as
two tarballs so the container can be built while the big one copies.

## Code changes applied

1. **`run_pipeline.py`** — added `--load_4bit`, also honouring `MEDGEMMA_4BIT=1`, wired into
   `ModelConfig(use_4bit_quantization=...)` in `build_config()`. Needed because the server
   GPU is unknown and `config.py` cannot be edited mid-run.

2. **`pipeline/nodes.py:debate_node`** — on `multiclass_tumor`, `final_predicted_class` now
   prefers `verdict["winner_detailed"]` over `verdict["winner"]`. The judge schema
   (`agents/debate.py:_JUDGE_TEMPLATE`) makes `winner` coarse
   (`tumor|stroke|multiple sclerosis|normal|other abnormalities`) and puts the subtype in
   `winner_detailed`, but the multiclass label space *is* the subtype — so before this fix
   step 6 would have recorded `"tumor"` against `glioma`/`meningioma`/`pituitary_tumor`
   labels and scored ≈0 by construction, wasting ~15 GPU-hours.

3. **`canonical_label` in both `eval/tumor_eval.py` and `eval/eval_analysis.py`** — `"other
   abnormalities"` (a value the judge's schema explicitly offers) used to canonicalize to
   `"normal"`, because *ab-**normal**-ities* contains the substring `normal`. It now maps to
   `"abnormal"`, which under `strict` scoring counts as "target pathology asserted absent"
   (same as naming another pathology) and under `--scoring abnormal` counts as positive —
   both verified. This also touches the existing baselines: 14 records across
   `binary_tumor_tumor_eval.jsonl` (6), `stroke_dataset_eval.jsonl` (6) and
   `ms_dataset_eval.jsonl` (2) hold `"other abnormalities"` and were previously scored as
   normal reads, so any table regenerated from those JSONLs will shift very slightly.

## Files to create under `server_bundle/`

All written. Verification status noted per file.

- [x] `PLAN.md` — this file
- [x] `container.def` — Singularity definition (base image above, `/opt/venv`, pinned wheels,
      import smoke test in `%post`, `PYTHONNOUSERSITE=1`). **Not build-tested** — needs
      apptainer/docker build with network; the base-image tag and every wheel version were
      verified to exist.
- [x] `requirements-server.txt` — the pins above, minus torch/torchvision (installed from the
      cu126 index first)
- [x] `config.env` — the single file to edit on the server: dataset dirs, `MAX_SAMPLES=500`,
      `SIF` path, `LOAD_4BIT=0`
- [x] `_lib.sh` — repo root, `config.env`, `pyrun()` wrapping `singularity exec --nv` with
      `HF_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CHECKPOINT_SOURCE=local`,
      `--pwd $ROOT`, the `DATA_*`/`MEDGEMMA_4BIT` passthrough, plus `log_start`/`log_done`/
      `export_one`/`run_step`. `set -o pipefail` so a crashed run is not masked by `tee`.
- [x] `00_preflight.sh` + `scripts/preflight.py` — versions/GPU/VRAM, dataset dirs + class
      counts + label-map coverage, checkpoints, SAM3 status, then 1 image through Forest and
      1 through Debate (agents loaded **once** via `load_agents` + `assemble_*`, so one
      MedGemma, same VRAM as a real run), printing measured s/image and the projected
      wall clock. **Run locally end-to-end** (see Verification below).
- [x] `01_forest_stroke.sh` … `06_debate_multiclass_tumor.sh` — one command each, the README
      commands with the distinct `--*_output` names (never the defaults — resume keys on the
      output file and the defaults hold the paper's baselines)
- [x] `run_all.sh` — steps 00→06 in order, continue-on-failure (but hard-stops if preflight
      fails), per-step logs under `logs/`
- [x] `run_parallel.sh` — assigns steps to GPUs round-robin via `CUDA_VISIBLE_DEVICES`, one
      queue per GPU, preflight once first
- [x] `90_export_results.sh` + `scripts/export_results.py` — JSONL → per-image TSV + summary
      TSV + combined `all_runs_summary.tsv`, and `eval/eval_analysis.py` per JSONL for the CSV
      metric tables. Runs after **every** step, so partial results are always in the
      preferred format. Ends by tarring `outputs/` + `logs/` into `results_<host>_<date>.tar.gz`.
      **Tested on the two finished Forest JSONLs.**
- [x] `scripts/prepack_models.py` — hardlink-seed `hf_cache/hub` from `~/.cache/huggingface/hub`,
      `snapshot_download` to fill gaps, fetch `sam3_probe.pth`, then verify in a child process
      with `HF_HUB_OFFLINE=1`. **Not yet run.**
- [x] `scripts/pack_bundle.sh` — vendor the `sam3/` clone, prune checkpoints to the 9 used
      files, stage + rename data into the server layout while excluding
      `OVERLAY`/`DICOM`/`MASKS`/`*_mask`/`External_Test`, emit `maclf-code-data.tar.gz` +
      `maclf-models.tar` + `SHA256SUMS.txt`. **Not yet run.**
- [x] `README_SERVER.md` — Macedonian + English
- [x] `SEND_CHECKLIST.md` — prepack → pack → transfer → what comes back

## Verification done on this machine

- `bash -n` on all 12 shell scripts, `py_compile` on all 3 python scripts — clean.
- `export_results.py` on `binary_forest_n4.jsonl` + `multiclass_forest_n4.jsonl`: 500/465
  rows, rectangular TSV (51 columns everywhere), and the summary reproduces the known
  numbers — 8.9 and 9.62 GPU-hours, median latency 62.6 / 71.7 s, forest dissent 0.095 /
  0.312. Combined summary written.
- `preflight.py` run locally with `DATA_*` pointed at the local dataset names and
  `MEDGEMMA_4BIT=1` (this machine is a 6 GB GTX 1660 Ti): sections 1–4 all correct, and it
  correctly FAILED with exit 1 at model loading because MedGemma does not fit that card.
- Step scripts dry-run with `SINGULARITY_BIN=echo`: the assembled command line, env vars,
  `--load_4bit` appending, output filenames and env overrides are all correct.
- **`container.def` build-tested via docker** (same base image + identical `%post` steps,
  since apptainer is not installed here): build succeeded, every pin resolved together with
  no conflicts, and the image reports `torch 2.10.0+cu126 / CUDA 12.6`, `transformers 5.3.0`,
  `numpy 1.26.4`, `pandas 3.0.1`, `open_clip 3.3.0`. Note pip pre-installs numpy 2.4.4 and
  pillow 12.2.0 as torch deps and then cleanly downgrades both to the pins.
- **Repo imports inside that container**: `config`, all of `pipeline.*`, all of `agents.*`,
  `explainability.saliency`, `eval.tumor_eval`, `eval.evaluate`, `eval.eval_analysis` and the
  graph builders all import with `PYTHONNOUSERSITE=1`, confirming the pinned list covers
  every real import (dropping gradio / albumentations / outlines / decord is safe).

Still untested: the actual `singularity build` (only the equivalent docker build was run) and
the offline `hf_cache` (needs `prepack_models.py` to run first).

## Fix — SAM3 broken on the server, cause found and fixed (2026-08-05)

`00_preflight.sh` run on 2026-08-04 (`gpgpu01.hpc.local`, A100-80GB) came back
**`PREFLIGHT OK`** but with `SAM3 source not importable: ModuleNotFoundError: No module
named 'sam3.train.data'`. Everything else — environment, all 4 datasets (500/500 selected,
label maps recognised), all 9 checkpoints, the offline `hf_cache`, both smoke tests (Forest
130.9 s/img, Debate 91.5 s/img on multiclass_tumor) — passed. Only SAM3 was broken, and it's
one of the two tumour tasks (`binary_tumor`, `multiclass_tumor`) that needs it.

**Root cause: an rsync anchoring bug in `pack_bundle.sh`'s code copy, not a container/setuptools
issue.** The generic code rsync (`pack_bundle.sh` step 2) used unanchored excludes —
`--exclude='data'`, `--exclude='outputs'`, `--exclude='checkpoints'`, `--exclude='logs'`,
etc. — meant to drop the top-level `data/`, `outputs/`, `checkpoints/`, `logs/` (populated or
created separately later in the script). In rsync, a single-component exclude pattern with no
leading `/` matches that name **at any depth**, not just the transfer root. So
`--exclude='data'` silently dropped `sam3/sam3/train/data/` too — and
`sam3/sam3/model/sam3_image.py` does `from sam3.train.data.collator import BatchedDatapoint`,
so `sam3.model_builder` (and therefore all of SAM3) failed to import on the server the moment
that subpackage was missing. Verified empirically with `rsync --dry-run`: unanchored `data`
drops `sam3/train/data/`; anchored `/data` does not. (`sam3/.git` was already fine — a
pattern with an embedded, non-leading slash is anchored to the transfer root, so that one
never had the bug.)

**Fix applied**: anchored the directory excludes in `pack_bundle.sh` to `/data`, `/outputs`,
`/checkpoints`, `/hf_cache`, `/.git`, `/.venv`, `/.env`, `/.codex`, `/.claude`, `/paper`,
`/papers`, `/presentation`, `/logs` (leading `/` = transfer-root only). Left the genuinely
recursive ones alone (`__pycache__`, `*.pyc`, `*.pdf`, `*.tar(.gz)`, `*.zip`/`*.7z`/`*.rar`,
`*.mat`, `*.jsonl`, `*.sif`) — those are correctly meant to match anywhere.

**Re-verified**: `SKIP_MODELS=1 bash server_bundle/scripts/pack_bundle.sh` rebuilt cleanly —
staged tree still 2.0 GB, image counts unchanged (3000/3064/3427/6650), `sam3/sam3/train/data/`
now present in `maclf-code-data.tar.gz` (checked via `tar tzf`), no stray-file trip. Only
`maclf-code-data.tar.gz` changed (new sha256 `d11ebc26...`); `maclf-models.tar` is untouched
(reused via `SKIP_MODELS=1`, same sha256 as before) — **no need to re-transfer the 12 GB model
tar or the data**, since datasets are staged by a separate rsync call with its own excludes
(`DATA_EXCLUDES`) that was never affected by this bug. Confirmed no `binary_debate_r2.jsonl` /
`multiclass_debate_r2.jsonl` exist yet, so this is a pre-run fix, not a redo — no GPU-hours
were wasted on a broken-SAM3 tumour run. Stroke/MS steps (01, 02, 04, 05) never touch SAM3
(ineligible task, per [[project_sam3_scope]]) and are unaffected either way.

**What to send**: just the new `maclf-code-data.tar.gz` (~1.9 GB) + updated
`SHA256SUMS.txt`. She extracts it over the existing directory (overwrites code + data +
checkpoints, leaves `hf_cache/` alone), re-runs `00_preflight.sh` to confirm section 4 now
says SAM3 imports OK with no warning, then proceeds — no step needs to be redone.


## Ship status (2026-07-29)

- [x] `prepack_models.py` → `PREPACK OK`. `hf_cache/` = **12 GB** after both trims; all five
      offline checks pass; `checkpoints/sam3_probe.pth` fetched (3.9 KB).
- [x] `pack_bundle.sh` → `~/Documents/bundle_out/`:
      `maclf-code-data.tar.gz` **1.9 GB** (gzip barely helps — the payload is JPEG/PNG),
      `maclf-models.tar` **12 GB**, `SHA256SUMS.txt`. Staged tree 2.0 GB.
      Audited: no `.env`, all 9 checkpoints, image counts 1500/1500 · 708/1426/930 ·
      1002/1014/650/761 · 1093/1130/4427 (`PNG/` only), SAM3 BPE asset present and no
      `sam3/.git`, and **52 symlink entries preserved** in the models tar with big blobs as
      real files (MedGemma 4.6 + 3.4 GB, SAM3 3.2 GB, BiomedCLIP 0.7 GB) — the property a
      zip would have destroyed.
      Caught during packing: a 6.8 GB `archive(1).zip` in the repo root inflated the first
      code archive to 8.7 GB; `*.zip`/`papers`/`*.jsonl`/`*.sif` excludes plus a >50 MB
      stray-file guard now prevent it. `SKIP_MODELS=1` reuses the model tar.
- [ ] Transfer ~14 GB + `SHA256SUMS.txt`; reply using the draft in `SEND_CHECKLIST.md`.
- [ ] Delete `~/Documents/bundle_stage/` (2 GB) once the transfer is confirmed.

## Remaining steps to ship
3. Transfer ~14.5 GB + `SHA256SUMS.txt`; 
4. Optional but valuable: ask for the preflight output — it reveals the
   GPU model and count, which decides `LOAD_4BIT` and whether `run_parallel.sh` is worth it.

## Run set (the 6 commands, in execution order)

All resumable and crash-safe (`eval/tumor_eval.py` appends one JSONL line per image and
skips `image_path`s already present with `error: null`). `--max_samples 500` samples
class-balanced by round-robin over class folders.

| # | Run | Output JSONL | Est. wall clock, 500 imgs |
|---|---|---|---|
| 1 | Forest N=4, `stroke` | `outputs/eval/stroke_forest_n4.jsonl` | ~8 h |
| 2 | Forest N=4, `ms` | `outputs/eval/ms_forest_n4.jsonl` | ~8 h |
| 3 | Debate R=2, `binary_tumor` | `outputs/eval/binary_debate_r2.jsonl` | ~13–16 h |
| 4 | Debate R=2, `stroke` | `outputs/eval/stroke_debate_r2.jsonl` | ~12–16 h |
| 5 | Debate R=2, `ms` | `outputs/eval/ms_debate_r2.jsonl` | ~12–16 h |
| 6 | Debate R=2, `multiclass_tumor` | `outputs/eval/multiclass_debate_r2.jsonl` | ~14–16 h |

Estimates are from local medians (forest 62.6 s/img binary, 71.7 s/img multiclass on the
RTX-class GPU that produced the finished runs); a datacentre GPU should be faster, and
preflight's measured latency replaces these guesses. Step 6 is last on purpose: the
multiclass CNN head is 12-class evaluated on 3-class figshare (acc 0.140) and BiomedCLIP
scores 0.163, so both tool advocates are at/below chance there — drop step 6 first if time
runs short.

Already done, not re-run: Forest N=4 `binary_tumor` (n=500) and `multiclass_tumor` (n=465).

## Fix — Debate judge silently never parsed, all 4 debate runs invalid (2026-08-17)

(`results_gpgpu01_20260817_1027/`, all 6 remaining runs finished) came
back with `debate_verdict_change_rate = 0.0` and `accuracy_final` far *below*
`accuracy_cnn_only` on every one of the 4 debate tasks (e.g. stroke: 0.286 final vs. 0.99
CNN-only; multiclass_tumor: 0.368 vs. 0.142). Root cause: `agents/debate.py`'s judge call
(`_JUDGE_TEMPLATE`) never once produced parseable JSON across all 2000 debate images —
confirmed via `grep -c "Judge parse failed" outputs/eval/*_debate_r2.jsonl` → 500/500 on
every file — so every verdict silently fell back to
`state["suspected_pathology"]` (the coarse triage guess) instead of an actual debated
verdict. The `except (json.JSONDecodeError, ValueError)` fallback swallowed this with no
error surfaced anywhere, so `n_errors: 0` in every summary and the bug was invisible short
of reading `outputs/eval/*.jsonl`'s `final_report` text. It was already present in
`00_preflight.log` (`conf=0.50` on both smoke-test rounds) but the preflight check only
verified the pipeline "ran end to end," not that the verdict parsed — so it printed
`PREFLIGHT OK` and the full campaign went ahead on a broken judge.

Likely cause: the judge call was capped at `max_new_tokens=200` — the smallest budget
anywhere in the codebase (advocates get 300, the main triage/report call gets 600) — for
the most complex ask (weigh 3 arguments, reason, emit structured JSON). MedGemma is known
to emit hidden `<unused95>` reasoning chatter before its answer elsewhere in this codebase;
200 tokens plausibly gets consumed before the JSON object closes.

Fix applied (not yet re-verified on a GPU — no local hardware big enough for MedGemma):
- `agents/debate.py`: judge `max_new_tokens` 200 → 500; `_JUDGE_TEMPLATE` now explicitly
  forbids preamble text; parse failures now print the raw truncated output and tag the
  verdict `judge_parse_failed: True` instead of failing silently.
- `eval/tumor_eval.py`: `extract_row` now writes `debate_judge_parse_failed` into every
  JSONL record.
- `server_bundle/scripts/export_results.py`: `all_runs_summary.tsv` now has a
  `judge_parse_failure_rate` column — would have caught this immediately.
- `server_bundle/scripts/preflight.py`: the Debate smoke test now hard-`fail()`s (blocking
  `PREFLIGHT OK`) if `debate_judge_parse_failed` is true on the 1-image smoke test.

**All 4 debate JSONLs in `results_gpgpu01_20260817_1027/` (`binary_debate_r2`,
`multiclass_debate_r2`, `ms_debate_r2`, `stroke_debate_r2`) are invalid and must be
re-run** — they measure "default to triage guess," not agent debate. The CNN-only /
BiomedCLIP / SAM3 / routing / latency columns in the same files are unaffected and remain
usable. The 2 Forest runs (stroke, ms) are a separate code path (`agent_forest.py`, not
`debate.py`) and are not implicated by this bug — no evidence yet either way, should still
re-run preflight once before trusting them, since preflight itself missed this class of bug
for months.

Next step: re-run `00_preflight.sh` on the server with the fix (must show
`judge parse failed False` and a non-trivial, non-0.50 confidence on both smoke rounds)
before re-running the 4 debate scripts (`03`, `04`, `05`, `06`) for real. Send an
updated `maclf-code-data.tar.gz` (`SKIP_MODELS=1 bash server_bundle/scripts/pack_bundle.sh`
— only code changed) the same way as the SAM3 fix above.

## Fix round 2 — the judge thinks past its token budget (2026-08-20)

The 2026-08-19 preflight (`00_preflight.log`, gpgpu01, A100 80 GB) ran with the 500-token
fix above and **still failed to parse the judge on both rounds**:

```
[debate] WARNING: judge parse failed on round 1, raw output:
  '<unused94>thought\nThe user wants me to act as a senior neuroradiologist and evaluate ...'
```

So the 2026-08-17 diagnosis was right about the mechanism but not about the size of it.
MedGemma 1.5 opens a hidden reasoning channel — `<unused94>thought … <unused95>` — before
its visible answer, and on the judge prompt it *always* does. The whole 500-token budget is
spent inside that thought; the JSON is never reached. The "Output ONLY the JSON object"
wording cannot suppress it — the thought channel is opened before the model is "answering"
at all.

The preflight *still* printed `PREFLIGHT OK`, for a second, independent reason:
`DebateOrchestrator.run()` builds a fresh return dict and never copied
`judge_parse_failed` out of the per-round verdict, so `debate_judge_parse_failed` was
always `False` in the JSONL and the guard added on 2026-08-17 could never fire.

Fix applied (again not GPU-verified — no local hardware; tokenizer plumbing was verified,
model behaviour was not):
- `agents/medgemma_agent.py`: new `NO_THINK_PREFILL = "<unused94>thought\n<unused95>"` and a
  `prefill=` argument on `_generate()` / `generate_for_prompt()`. The prefill seeds the
  model's own turn with an already-closed, empty thought block, so generation starts in the
  answer channel. Verified locally that `<unused95>` is a single token (id 101) and that the
  `apply_chat_template(tokenize=False) + prefill` + `add_special_tokens=False` path
  reproduces the template's tokenization exactly.
- `agents/debate.py`: judge and advocates use the prefill. If a judge call still fails to
  parse, it is retried **once** with no prefill and `max_new_tokens=1500` (room for a full
  thought *plus* the JSON), so neither path depends on the prefill working. A parsed object
  without a `winner` field counts as a failure. After `PREFILL_DISABLE_AFTER = 3` failures
  the prefill is dropped for the rest of the process so a doomed first attempt is not paid
  for on every image. Confidence is parsed defensively (`_as_float`).
- `agents/debate.py`: `run()` now returns `judge_parse_failed` / `judge_parse_failed_rounds`,
  which `debate_node` copies into `state["debate_verdict"]` — this is what makes the
  preflight guard and the `judge_parse_failure_rate` TSV column actually work.

What the next preflight must show before the 4 debate runs are restarted:
- no `judge parse failed on round` warnings at all (a warning followed by a successful
  `retry-long` is tolerable but means every image pays double for the judge — report it);
- `judge parse failed False` **and** a confidence that is not exactly `0.50`;
- ideally a `winner_detailed` on the multiclass smoke image.

Note the two `[debate] Round N verdict: tumor conf=0.50 changed=False` lines in the old log
are the signature of this bug: `0.50` is the hard-coded fallback confidence.

Not changed: triage/report/forest calls also run without the prefill (600–2048 tokens) and
have so far produced parseable JSON, so they are left alone — but the same failure mode is
available to them, and `report_node` degrades quietly to raw text when it hits it.

## Open items / risks

- **Server GPU model and count are unknown.** Affects bf16 vs `--load_4bit` and whether
  steps can run in parallel. Preflight print it and have the outputs pasted back. `run_parallel.sh` covers the multi-GPU case without another round trip.
- **Gated weights leave Tamara's control** once `hf_cache/` is shipped (MedGemma + SAM3
  under their respective terms).
- **`sam3/` clone is not on this machine yet** — `pack_bundle.sh` clones it (needs network at
  packing time, not at build/run time).
- **Container build needs internet** (pip + base image) — fine on the sylabs remote builder.
- Local `.venv` is Python 3.12.3 and healthy; the *system* python has a broken numpy/scipy
  pair. Always `source .venv/bin/activate` before running anything locally.
