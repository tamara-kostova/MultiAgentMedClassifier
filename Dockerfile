# MCP server for the neuroimaging classifier (mcp_server.py). See docker/README.md.
#
#   docker build -t neuro-mcp .
#   docker run --gpus all -p 8765:8765 -v neuro-models:/models -v neuro-outputs:/app/outputs \
#       -e NEURO_MCP_TOKEN=... -e HF_TOKEN=... neuro-mcp
#
# Code only. Model weights live in the /models volume: MedGemma and SAM3 are gated
# (HF_TOKEN needed on the first start), the task CNN/BiomedCLIP/SAM3-probe checkpoints
# are public and are fetched into /models/checkpoints on the first start.
#
# python:3.12-slim + the cu126 torch wheels, which bundle their own CUDA runtime, so
# no nvidia/cuda base image is needed. The host needs an NVIDIA driver and the
# NVIDIA Container Toolkit (--gpus all).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/hf \
    CHECKPOINT_SOURCE=hf

# libgomp1: OpenMP for torch/opencv on CPU paths; libglib2.0-0: opencv
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.10.0+cu126 torchvision==0.25.0+cu126
COPY docker/requirements-mcp.txt /tmp/requirements-mcp.txt
RUN pip install -r /tmp/requirements-mcp.txt && rm /tmp/requirements-mcp.txt

# Unprivileged runtime user. /models and /app/outputs are created owned by it, so a new
# named volume mounted there inherits that ownership.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
 && mkdir -p /models/checkpoints /models/hf /app/outputs \
 && chown -R app:app /models /app

WORKDIR /app
# .dockerignore whitelists only the code and prompts the server needs.
COPY --chown=app:app . .
# config.py uses relative checkpoints/... paths; keep them in the volume.
# The SAM3 check fails the build when the vendored sam3/ checkout is missing or broken;
# otherwise the server would start and serve `tumor` with segmentation silently skipped.
RUN ln -s /models/checkpoints /app/checkpoints \
 && python -c "import mcp_server, pipeline.registry as r; r.discover()" \
 && python -c "from agents.sam3_tool import _SAM_AVAILABLE, _SAM_IMPORT_ERROR; assert _SAM_AVAILABLE, _SAM_IMPORT_ERROR"

# transformers 5.3 materialises bf16 tensors on the GPU in a thread pool ahead of
# 4-bit quantisation: MedGemma NF4 peaks >7.5 GB and OOMs on 8 GB GPUs. Sequential
# loading peaks at ~3.2 GB. Load order only; weights and outputs are unchanged.
ENV HF_DEACTIVATE_ASYNC_LOAD=1

USER app

# /app/outputs holds SAM3 overlays, saliency maps and FHIR JSON, i.e. patient-derived data.
# As a volume it stays out of the container's writable layer and can be pruned separately.
VOLUME ["/models", "/app/outputs"]
EXPOSE 8765

# TCP check only (/mcp needs the bearer token). The long start period covers the first
# start, which downloads ~14 GB of weights before the port opens.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8765), 3).close()"
# mkdir at start: with an empty bind mount the checkpoints symlink would dangle and
# every checkpoint download would fail (CNN silently falls back to ImageNet weights).
ENTRYPOINT ["sh", "-c", "mkdir -p /models/checkpoints && exec python mcp_server.py --host 0.0.0.0 --port 8765 \"$@\"", "--"]
CMD []
