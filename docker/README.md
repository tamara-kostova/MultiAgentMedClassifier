# Neuroimaging classifier — MCP server image

One container, one MCP tool for the host system: `classify`. Research prototype, not a
medical device.

## Get the image

Public on Docker Hub: [`tamarakostova/neuro-mcp`](https://hub.docker.com/r/tamarakostova/neuro-mcp)
(tags `0.2.0`, `0.1.0`, `latest` = `0.2.0`). No login needed:

```bash
docker pull tamarakostova/neuro-mcp:0.2.0
```

The image holds only code and prompts: no model weights and no tokens. Pin a version rather
than `latest` so an update never changes a running deployment unexpectedly. Coming from
`0.1.0`? Read [Changes in 0.2.0](#changes-in-020) first: one step is needed before the
new image starts.

## Changes in 0.2.0

The `classify` input and its 9-field output are unchanged.

- **One-off step when upgrading:** the image now runs as the unprivileged user `app`
  (uid 1000), and an existing `neuro-models` volume created by 0.1.0 is root-owned. Change its
  owner once before starting 0.2.0, or the server cannot write to `/models`:
  ```bash
  docker run --rm --user root -v neuro-models:/models --entrypoint chown tamarakostova/neuro-mcp:0.2.0 -R 1000:1000 /models
  ```
- **Results can differ for the same scan** because of pipeline bug fixes.
- **Stricter input limits**, checked before anything is decoded and listed by
  `list_capabilities`: `forest_n_agents` at most 4 (was 8); images at most 4096×4096 px and
  32 MB each; volumes at most 1 GB; `metadata` at most 32 scalar keys. Out-of-range requests
  get a readable tool error.
- **Container:** a second volume `/app/outputs` keeps the generated files (SAM3 overlays,
  saliency maps, FHIR JSON) out of the container layer; a healthcheck makes `docker ps` show
  `(healthy)` once the server accepts connections; the build fails if SAM3 is missing, so the
  image can no longer serve `tumor` with segmentation silently skipped.

## Requirements

| | Minimum (tested) | Recommended |
|---|---|---|
| GPU | 8 GB NVIDIA, with `MEDGEMMA_4BIT=1` (peak 7.1 GB during a request) | 16 GB+, bf16 (leave `MEDGEMMA_4BIT` out) |
| RAM | 10 GB free for the container (peak at model load; ~4 GB of process memory once serving) | 16 GB+ |
| Disk | ~12 GB image + ~14 GB `/models` volume | |
| Host | NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (`--gpus all`) | |

Measured on an RTX 4060 Laptop (8 GB): startup ~25 s once weights are cached, ~45 s per image.

## Run

```bash
docker run -d --name neuro-mcp --gpus all --restart unless-stopped \
  --memory 12g \
  -p 8765:8765 \
  -v neuro-models:/models \
  -v neuro-outputs:/app/outputs \
  -e NEURO_MCP_TOKEN=<long random secret> \
  -e HF_TOKEN=<your hugging face token> \
  -e MEDGEMMA_4BIT=1 \
  tamarakostova/neuro-mcp:0.2.0

docker logs -f neuro-mcp      # ready when it prints "Uvicorn running on http://0.0.0.0:8765"
docker ps                     # STATUS shows (healthy) once the port is open
```

The server runs as the unprivileged user `app` (uid 1000). New named volumes inherit the
right ownership automatically. A host bind mount (`-v /srv/models:/models`) must be writable
by uid 1000: `sudo chown -R 1000:1000 /srv/models`.

Upgrading from 0.1.0 needs a one-off `chown` of the existing volume; see
[Changes in 0.2.0](#changes-in-020).

Endpoint: `http://<host>:8765/mcp` (MCP Streamable HTTP). Every request needs the header
`Authorization: Bearer <NEURO_MCP_TOKEN>`. Generate the secret with `openssl rand -hex 32`.

### Environment variables

| Variable | Needed | What it does |
|---|---|---|
| `NEURO_MCP_TOKEN` | **Yes** | Bearer token clients must send. The server refuses to listen on a non-localhost address without it. |
| `HF_TOKEN` | First start only | Hugging Face token for downloading the gated models (below). |
| `MEDGEMMA_4BIT=1` | GPUs < ~12 GB | Loads MedGemma in 4-bit NF4 instead of bf16. |
| `HF_HUB_OFFLINE=1` | Optional, after first start | No network access at all; weights are read from `/models`. Then `HF_TOKEN` can be dropped. |

### Hugging Face token (first start)

No weights are in the image. The first start downloads ~14 GB into the `/models` volume.
Two of the models are **gated**, so `HF_TOKEN` must be **your own** token, from an account
that has accepted both licenses on huggingface.co:

- [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) (Health AI Developer Foundations terms)
- [`facebook/sam3`](https://huggingface.co/facebook/sam3)

A read-only token is enough (huggingface.co → Settings → Access Tokens). Everything else is public:

| Weights | Source | Stored in |
|---|---|---|
| MedGemma 1.5 4B, SAM3 backbone | gated, need `HF_TOKEN` | `/models/hf` |
| BiomedCLIP base | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (public) | `/models/hf` |
| Task CNNs, BiomedCLIP probes, SAM3 probe | `tamara-kostova/multiagentmed-*` (public) | `/models/checkpoints` |

Keep the `neuro-models` volume: later starts reuse it and need no token or network.

### Startup checkpoint check

At startup the server checks all 9 task checkpoints (4 CNNs, 4 BiomedCLIP probes, the SAM3
probe), downloads missing ones, and **exits with a list of what is missing** if any cannot be
fetched. It never serves a task with a missing model. `--allow_missing_checkpoints` overrides
this (the CNN then falls back to ImageNet weights and results are meaningless).

### Operating notes

- One container per GPU. Requests are processed one at a time behind a GPU lock and queue
  otherwise. Set client timeouts generously (30 min), since a volume takes a few minutes.
- Run one worker only: each extra process would load MedGemma again.
- Inputs are deleted after each request. SAM3 overlays, saliency maps and FHIR JSON stay in
  `/app/outputs`. These are derived from patient images. Keep them in the `neuro-outputs`
  volume and prune it to your retention policy, or mount `--tmpfs /app/outputs` to drop them
  when the container stops.
- Extra `mcp_server.py` flags go after the image name, e.g. `tamarakostova/neuro-mcp:0.2.0 --lazy_load`.
- Research prototype, not a medical device. Outputs are not for clinical use.

## Input — `classify` arguments

MCP tool name: **`classify`**. Arguments:

| Field | Type | Required | Values |
|---|---|---|---|
| `task` | string | yes | `tumor` \| `ms` \| `stroke`. `tumor` and `ms` expect **MRI**, `stroke` expects **CT**. `tumor` returns the subtype (`glioma` / `meningioma` / `pituitary_tumor`). |
| `image` | string | one of `image` / `volume` | One 2D slice as base64 PNG or JPEG, either raw base64 or a `data:image/png;base64,...` URL. Windowed as a viewer shows it (not raw DICOM values). |
| `volume` | string | one of `image` / `volume` | A 3D array `(slices, H, W)`, **axial-first**, as base64 of a `.npy` or `.npz` file (key `volume`, or its first array). CT in **HU**. MR can be raw intensities. |

Give exactly one of `image` or `volume`.

Example, one MRI slice:

```json
{
  "task": "tumor",
  "image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAAAAAB..."
}
```

Example, a CT volume:

```json
{
  "task": "stroke",
  "volume": "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0Jywg..."
}
```

For a volume, 32 axial slices are sampled, the CNN screens them, the 3 most abnormal go
through the full pipeline, and the most abnormal diagnosis is returned.

Producing the base64 in Python:

```python
import base64, io, numpy as np
image_b64 = base64.b64encode(open("scan.png", "rb").read()).decode()
buf = io.BytesIO(); np.save(buf, volume.astype(np.float32))       # volume: (slices, H, W)
volume_b64 = base64.b64encode(buf.getvalue()).decode()
```

## Output

The result is one JSON object, as MCP `structuredContent` and as the same JSON in the text content.
Example (a real response for an MRI tumor slice):

```json
{
  "modality": "MRI",
  "specialized_sequence": "T2",
  "plane": "axial",
  "diagnosis_name": "tumor",
  "diagnosis_detailed": "glioma",
  "icd10_code": "C71",
  "severity_score": 0.7,
  "diagnosis_confidence": 0.9,
  "severity_confidence": 0.8
}
```

| Field | Type | Meaning | Values |
|---|---|---|---|
| `modality` | string \| null | Imaging modality | `MRI` \| `CT` |
| `specialized_sequence` | string \| null | MRI sequence (null for CT) | `FLAIR` \| `T1` \| `T2` \| `T1C+` |
| `plane` | string \| null | Slice orientation | `axial` \| `sagittal` \| `coronal` |
| `diagnosis_name` | string \| null | Main diagnosis | `tumor` \| `stroke` \| `multiple sclerosis` \| `normal` \| `other abnormalities` |
| `diagnosis_detailed` | string \| null | Subtype (null when there is none, e.g. normal or MS) | `glioma` \| `meningioma` \| `pituitary_tumor` \| `carcinoma` \| `germinoma` \| `granuloma` \| `medulloblastoma` \| `neurocytoma` \| `papilloma` \| `schwannoma` \| `tuberculoma` \| `ischemic` \| `hemorrhagic` |
| `icd10_code` | string \| null | ICD-10 code of the diagnosis | e.g. `C71`, `I63.9`, `G35` |
| `severity_score` | number \| null | Estimated severity | 0.0 (none) – 1.0 (severe) |
| `diagnosis_confidence` | number | Confidence in the diagnosis | 0.0 – 1.0 |
| `severity_confidence` | number | Confidence in the severity score | 0.0 – 1.0 |

The object follows MedGemma's output schema (`prompts/system_prompt.txt`) and comes from its
final report, which fuses every tool's output (CNN, SAM3, BiomedCLIP, Grad-CAM++). Values are
normalised; anything outside the lists above becomes null, so **check for null**.
`diagnosis_confidence` includes the pipeline's penalty when the Grad-CAM++ map and the SAM3
mask disagree. Treat low confidence (< 0.45) as "needs human review".

Errors (bad base64, unreadable image, a volume that is not 3D, both or neither of
`image`/`volume`, an unknown task, an input over a limit) come back as an MCP tool error
(`isError: true`) with a readable message. A missing or wrong bearer token gets HTTP 401.
Inputs are checked against these limits before any pixel or voxel data is decoded
(`list_capabilities` reports the live values under `limits`):

| Input | Limit |
|---|---|
| Image | PNG or JPEG; ≤ 32 MiB encoded; ≤ 4096 × 4096 = 16,777,216 pixels |
| Images per request (`classify_slices`) | ≤ 64, ≤ 256 MiB encoded in total |
| Volume file (`.npy`/`.npz`, base64 or `volume_path`) | ≤ 1 GiB |
| Volume array | 3D integer or float (no object, structured, complex or pickled arrays), ≤ 268,435,456 voxels and ≤ 1 GiB decoded; float volumes must be finite; `.npz` uses key `volume` or the first array, and an empty archive is an error |
| `volume_path` | only with `--volume_dir`; resolved path inside it, `.npy`/`.npz` suffix |
| `top_k` / `n_slices` / `forest_n_agents` / `debate_rounds` | 1–32 / 1–128 / 1–4 / 1–3 |
| `metadata` | ≤ 32 keys of ≤ 64 characters; values string (≤ 256 characters), number, boolean or null; `image_paths`, `source_paths`, `slice_index` are reserved; `rescale_slope`/`rescale_intercept` must be finite numbers |

The HTTP request body itself is capped by `--max_body_mb` (default 256).

## Calling it

With the reference client (outside the container, or `docker exec` into it):

```bash
NEURO_MCP_TOKEN=<secret> python mcp_client.py --url http://<host>:8765/mcp --task tumor --image scan.png
NEURO_MCP_TOKEN=<secret> python mcp_client.py --url http://<host>:8765/mcp --task stroke --volume ct.npy
```

From Python with the MCP SDK, it is `await client.call_tool("classify", {"task": ..., "image": ...})`.
See `mcp_client.py` for the bearer-token transport. The detailed tools `classify_slices`,
`classify_volume` and `list_capabilities` are still served, for per-slice agent outputs and
FHIR bundles. `classify_slices` also returns MedGemma's free-text report (`final_report`:
numbered findings followed by the same JSON); `classify` returns only the JSON above.
