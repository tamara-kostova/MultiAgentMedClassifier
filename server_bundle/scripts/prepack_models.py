"""
Prepack every model weight the server needs into a self-contained ./hf_cache.

RUN THIS LOCALLY (on Tamara's machine, with network + HF login), not on the server:

    source .venv/bin/activate
    python server_bundle/scripts/prepack_models.py

It builds ./hf_cache/hub with all four HuggingFace repos the pipeline loads, plus
downloads the SAM3 linear probe into checkpoints/. The server then runs with
HF_HOME=./hf_cache and HF_HUB_OFFLINE=1 — no token, no gated-model access and no
internet needed on the compute node.

Files already present in the local ~/.cache/huggingface/hub are HARD-LINKED into
hf_cache (same filesystem → no extra disk, instant), then snapshot_download fills
any gaps. Total ≈ 12.5 GB.

Verify without touching the network:

    python server_bundle/scripts/prepack_models.py --verify_only
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Repos the pipeline loads at run time: (repo_id, description, allow_patterns).
# allow_patterns=None fetches the whole repo.
REQUIRED_REPOS = [
    ("google/medgemma-1.5-4b-it", "MedGemma 1.5 4B — triage/report/debate/forest (gated)", None),
    (
        "facebook/sam3",
        "SAM3 backbone — segmentation (gated)",
        # The repo carries the same weights twice: sam3.pt (what the `sam3` package
        # loads) and model.safetensors (for the transformers-native Sam3Model, which
        # this pipeline does not use) — 3.3 GB each. sam3.pt + config.json is exactly
        # what the cache held when the finished Forest tumour runs wrote masks for
        # 500/500 images, so that pair is the proven-sufficient set.
        ["sam3.pt", "config.json"],
    ),
    (
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "BiomedCLIP ViT-B/16 — zero-shot / probe features",
        None,
    ),
    (
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "BiomedCLIP text tokenizer — pulled by open_clip, easy to forget",
        # Tokenizer files only. This repo also ships PyTorch/TF/Flax BERT weights
        # (~878 MB) that open_clip never loads: the text tower weights come from
        # the BiomedCLIP checkpoint itself. Verified — BiomedCLIP creates offline
        # with only config.json + tokenizer_config.json + vocab.txt present.
        ["*.json", "*.txt", "*.model", "*tokenizer*"],
    ),
]


def repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def dir_size(path: Path) -> int:
    """On-disk size. The HF cache stores each file once under blobs/ and links it
    from snapshots/, so symlinks must be skipped or every weight is counted twice."""
    if not path.exists():
        return 0
    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file() and not f.is_symlink()
    )


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def hardlink_tree(src: Path, dst: Path) -> bool:
    """Copy src → dst using hard links for file contents (same filesystem only)."""
    try:
        shutil.copytree(src, dst, copy_function=os.link, symlinks=True, dirs_exist_ok=True)
        return True
    except OSError as exc:
        print(f"    hard-link copy not possible ({exc}); falling back to a real copy")
        try:
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
            return True
        except OSError as exc2:
            print(f"    copy failed: {exc2}")
            return False


def seed_from_local_cache(target_hub: Path) -> None:
    """Populate hf_cache/hub from the user's existing HF cache, without re-downloading."""
    default_hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    print(f"\n── Seeding from local cache: {default_hub}")
    if not default_hub.is_dir():
        print("   local cache not found — everything will be downloaded")
        return

    for repo_id, _desc, _patterns in REQUIRED_REPOS:
        name = repo_dir_name(repo_id)
        src, dst = default_hub / name, target_hub / name
        if dst.exists():
            print(f"   {repo_id}: already in hf_cache ({human(dir_size(dst))})")
            continue
        if not src.is_dir():
            print(f"   {repo_id}: not in local cache — will download")
            continue
        print(f"   {repo_id}: linking {human(dir_size(src))} → hf_cache")
        hardlink_tree(src, dst)


def download_repos(target_hub: Path, allow_network: bool) -> list[str]:
    """snapshot_download each repo into target_hub; returns list of failures."""
    from huggingface_hub import snapshot_download

    failures = []
    for repo_id, desc, patterns in REQUIRED_REPOS:
        print(f"\n── {repo_id}")
        print(f"   {desc}")
        if patterns:
            print(f"   (restricted to {patterns})")
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(target_hub),
                local_files_only=not allow_network,
                allow_patterns=patterns,
            )
            print(f"   OK  {path}")
            # Size of the repo dir, not the snapshot: snapshots/ is only symlinks
            # into blobs/, which dir_size deliberately skips.
            print(f"       {human(dir_size(target_hub / repo_dir_name(repo_id)))}")
        except Exception as exc:
            print(f"   FAILED: {type(exc).__name__}: {exc}")
            failures.append(repo_id)
    return failures


def fetch_sam3_probe() -> bool:
    """Download the SAM3 linear probe checkpoint into checkpoints/ (not an HF-cache repo)."""
    from config import DEFAULT_CONFIG, download_hf_checkpoint

    target = Path(DEFAULT_CONFIG.model.sam3_linear_probe_checkpoint or "checkpoints/sam3_probe.pth")
    print(f"\n── SAM3 linear probe → {target}")
    if target.is_file():
        print(f"   already present ({human(target.stat().st_size)})")
        return True
    try:
        download_hf_checkpoint("tumor_segmentation", "sam3", target, caller="prepack")
        print(f"   OK ({human(target.stat().st_size)})")
        return True
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        print("   Segmentation would be skipped on the server, which breaks")
        print("   comparability with the finished Forest tumour runs.")
        return False


VERIFY_SNIPPET = """
import os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
failures = []

def check(label, fn):
    try:
        fn()
        print(f"   OK    {label}")
    except Exception as exc:
        print(f"   FAIL  {label}: {type(exc).__name__}: {exc}")
        failures.append(label)

def medgemma_processor():
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")

def medgemma_config():
    from transformers import AutoConfig
    AutoConfig.from_pretrained("google/medgemma-1.5-4b-it")

def medgemma_weights():
    # Weight shards must all be present; check the index without loading them.
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download("google/medgemma-1.5-4b-it", local_files_only=True))
    shards = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    assert shards, "no weight shards in the snapshot"
    print(f"         {len(shards)} weight shard(s)")

def biomedclip():
    import open_clip
    open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

def sam3_weights():
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download("facebook/sam3", local_files_only=True))
    assert (p / "sam3.pt").is_file(), "sam3.pt missing"

check("MedGemma processor (offline)", medgemma_processor)
check("MedGemma config (offline)", medgemma_config)
check("MedGemma weight shards", medgemma_weights)
check("BiomedCLIP + tokenizer (offline)", biomedclip)
check("SAM3 backbone weights", sam3_weights)
sys.exit(1 if failures else 0)
"""


def verify_offline(cache_dir: Path) -> bool:
    """Re-exec a child process with the offline env vars the server will use."""
    print("\n── Verifying the cache with HF_HUB_OFFLINE=1 (as the server will see it)")
    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(cache_dir),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CHECKPOINT_SOURCE": "local",
            "PYTHONNOUSERSITE": "1",
        }
    )
    env.pop("HF_HUB_CACHE", None)
    proc = subprocess.run(
        [sys.executable, "-c", VERIFY_SNIPPET],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepack HF models for offline server use")
    ap.add_argument("--cache_dir", default="hf_cache", help="Output cache dir (default: hf_cache)")
    ap.add_argument("--verify_only", action="store_true", help="Only verify an existing cache")
    ap.add_argument(
        "--no_network",
        action="store_true",
        help="Do not download; only seed from the local cache and verify",
    )
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    cache_dir = Path(args.cache_dir).resolve()
    target_hub = cache_dir / "hub"
    target_hub.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  Prepacking model weights for the faculty GPU server")
    print(f"  project:    {PROJECT_ROOT}")
    print(f"  cache dir:  {cache_dir}")
    print("=" * 78)

    probe_ok = True
    if not args.verify_only:
        # .env holds HF_TOKEN; gated repos need it for any actual download.
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
        if not args.no_network and not (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ):
            print(
                "\n  note: no HF_TOKEN in the environment. Anything missing from the local\n"
                "        cache cannot be downloaded (MedGemma and SAM3 are gated)."
            )

        seed_from_local_cache(target_hub)
        failures = download_repos(target_hub, allow_network=not args.no_network)
        probe_ok = fetch_sam3_probe()
        if failures:
            print(f"\n  Repos that could not be completed: {failures}")

    ok = verify_offline(cache_dir)

    print("\n" + "=" * 78)
    print(f"  hf_cache size: {human(dir_size(cache_dir))}")
    if ok and probe_ok:
        print("  PREPACK OK — hf_cache/ is complete and loads offline.")
        print("  Next: bash server_bundle/scripts/pack_bundle.sh")
        return 0
    print("  PREPACK INCOMPLETE — see the FAIL lines above.")
    print("  Fix before packing: the server has no internet fallback.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
