#!/usr/bin/env bash
# Build the tarballs to send to the faculty server. RUN LOCALLY, from the project root:
#
#     bash server_bundle/scripts/pack_bundle.sh
#
# Produces, in ../bundle_out/ by default:
#
#   maclf-code-data.tar.gz   code + the 4 datasets + the 9 needed checkpoints  (~2 GB)
#   maclf-models.tar.gz      hf_cache/ — MedGemma, SAM3, BiomedCLIP weights    (~12.5 GB)
#   SHA256SUMS.txt
#
# Two archives on purpose: Martina can extract the small one and start building the
# container while the big one is still copying.
#
# Prerequisite: python server_bundle/scripts/prepack_models.py has been run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/../bundle_out}"
STAGE="${STAGE:-$PROJECT_ROOT/../bundle_stage}"
BUNDLE_NAME="MultiAgentMedClassifier"

# Local source directories for the four datasets. Override via the environment if
# they live elsewhere. They are copied under the names the run scripts expect.
SRC_BR35H="${SRC_BR35H:-data/Br35H}"
SRC_FIGSHARE="${SRC_FIGSHARE:-data/processed}"
SRC_MS="${SRC_MS:-data/MS}"
SRC_STROKE="${SRC_STROKE:-data/Brain_Stroke_CT_Dataset}"

SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"

# Checkpoints actually loaded by the four tasks (the rest of checkpoints/ is unused
# architectures from the earlier benchmarking work).
CHECKPOINTS=(
    vgg16_MRI_tumor_binary_norm_final.pt
    densenet169_MRI_tumor_multiclass_norm_final.pt
    resnet101_MRI_ms_norm_final.pt
    densenet169_CT_stroke_binary_norm_final.pt
    linear_probe_BiomedCLIP_MRI_tumor_binary_norm_best.pt
    linear_probe_BiomedCLIP_MRI_tumor_multiclass_norm_best.pt
    linear_probe_BiomedCLIP_MRI_ms_norm_best.pt
    linear_probe_BiomedCLIP_CT_stroke_binary_norm_best.pt
    sam3_probe.pth
)

# Never ship these: annotation/mask/DICOM folders. eval/tumor_eval.py excludes them
# anyway, and they are most of the raw dataset size. OVERLAY in particular has the
# lesion painted onto the scan — shipping it risks a leaked run.
DATA_EXCLUDES=(
    --exclude='*_mask'
    --exclude='OVERLAY'
    --exclude='DICOM'
    --exclude='MASKS'
    --exclude='External_Test'
    --exclude='pred'
    --exclude='Br35H-Mask-RCNN'
    --exclude='*.mat'
    --exclude='.DS_Store'
)

say() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

# ── 0. sanity ────────────────────────────────────────────────────────────────
say "Checking prerequisites"
missing=0
for d in "$SRC_BR35H" "$SRC_FIGSHARE" "$SRC_MS" "$SRC_STROKE"; do
    if [ -d "$d" ]; then
        echo "   data OK: $d"
    else
        echo "   MISSING data dir: $d"
        missing=1
    fi
done
if [ ! -d hf_cache ]; then
    echo "   MISSING hf_cache/ — run: python server_bundle/scripts/prepack_models.py"
    missing=1
else
    echo "   hf_cache OK: $(du -sh hf_cache | cut -f1)"
fi
for c in "${CHECKPOINTS[@]}"; do
    [ -f "checkpoints/$c" ] || { echo "   MISSING checkpoint: checkpoints/$c"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo; echo "Fix the items above, then re-run."; exit 1; }

# ── 1. vendor SAM3 ───────────────────────────────────────────────────────────
say "Vendoring SAM3 source"
if [ -d sam3/sam3 ]; then
    echo "   ./sam3 already present"
else
    echo "   cloning $SAM3_REPO_URL → ./sam3"
    git clone --depth 1 "$SAM3_REPO_URL" sam3
fi
if [ -f sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz ]; then
    echo "   BPE asset OK"
else
    echo "   WARNING: sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz not found —"
    echo "            SAM3 will be skipped on the server."
fi

# ── 2. stage the tree ────────────────────────────────────────────────────────
say "Staging $STAGE/$BUNDLE_NAME"
rm -rf "$STAGE"
DEST="$STAGE/$BUNDLE_NAME"
mkdir -p "$DEST"

echo "   code"
rsync -a \
    --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.env' --exclude='data' --exclude='outputs' --exclude='checkpoints' \
    --exclude='hf_cache' --exclude='.codex' --exclude='.claude' \
    --exclude='sam3/.git' \
    --exclude='paper' --exclude='papers' --exclude='presentation' --exclude='*.pdf' \
    --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.tgz' --exclude='logs' \
    --exclude='*.zip' --exclude='*.7z' --exclude='*.rar' --exclude='*.mat' \
    --exclude='*.jsonl' --exclude='*.sif' \
    ./ "$DEST/"

echo "   checkpoints (9 files)"
mkdir -p "$DEST/checkpoints"
for c in "${CHECKPOINTS[@]}"; do
    cp -l "checkpoints/$c" "$DEST/checkpoints/$c" 2>/dev/null \
        || cp "checkpoints/$c" "$DEST/checkpoints/$c"
done
cp checkpoints/download_checkpoints.py "$DEST/checkpoints/" 2>/dev/null || true

echo "   data (renamed into the layout the run scripts expect)"
mkdir -p "$DEST/data/sclerosis" "$DEST/data/stroke"
rsync -a "${DATA_EXCLUDES[@]}" "$SRC_BR35H/"    "$DEST/data/Br35H/"
rsync -a "${DATA_EXCLUDES[@]}" "$SRC_FIGSHARE/" "$DEST/data/figshare/"
rsync -a "${DATA_EXCLUDES[@]}" "$SRC_MS/"       "$DEST/data/sclerosis/MS/"
rsync -a "${DATA_EXCLUDES[@]}" "$SRC_STROKE/"   "$DEST/data/stroke/Brain_Stroke_CT_Dataset/"

echo "   empty output dirs"
mkdir -p "$DEST/outputs/eval" "$DEST/outputs/results_tsv" "$DEST/logs"

echo "   container definition at the top level (convenient for the build command)"
cp server_bundle/container.def "$DEST/container.def"
cp server_bundle/requirements-server.txt "$DEST/requirements-server.txt"
chmod +x "$DEST"/server_bundle/*.sh "$DEST"/server_bundle/scripts/*.sh 2>/dev/null || true

echo
echo "   staged image counts:"
for t in Br35H figshare sclerosis/MS stroke/Brain_Stroke_CT_Dataset; do
    n=$(find "$DEST/data/$t" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
    printf '     %-36s %6d images  %s\n' "data/$t" "$n" "$(du -sh "$DEST/data/$t" | cut -f1)"
done

# ── 2b. guard against stray bulk ─────────────────────────────────────────────
# A 6.8 GB archive(1).zip in the repo root once sailed through the exclude list and
# tripled the bundle. Anything large outside data/ and checkpoints/ is suspect.
say "Checking for unexpected large files"
STRAY=$(find "$DEST" -type f -size +50M \
    -not -path "$DEST/data/*" -not -path "$DEST/checkpoints/*" 2>/dev/null || true)
if [ -n "$STRAY" ]; then
    echo "$STRAY" | while read -r f; do
        printf '   %8s  %s\n' "$(du -h "$f" | cut -f1)" "${f#$DEST/}"
    done
    echo
    echo "   These are >50 MB and are neither data nor checkpoints. If any of them"
    echo "   does not belong in the bundle, add an --exclude to this script and re-run."
    echo "   Set ALLOW_STRAY=1 to pack anyway."
    [ "${ALLOW_STRAY:-0}" = "1" ] || { echo; echo "Aborting."; exit 1; }
else
    echo "   none — nothing large outside data/ and checkpoints/"
fi

echo
echo "   staged tree total: $(du -sh "$DEST" | cut -f1)"

# ── 3. archives ──────────────────────────────────────────────────────────────
say "Creating archives in $OUT_DIR"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

tar -C "$STAGE" -czf "$OUT_DIR/maclf-code-data.tar.gz" "$BUNDLE_NAME"
echo "   maclf-code-data.tar.gz  $(du -h "$OUT_DIR/maclf-code-data.tar.gz" | cut -f1)"

# Model cache: already-compressed weights, so store without gzip for speed.
# SKIP_MODELS=1 reuses an existing maclf-models.tar (hf_cache rarely changes, and
# re-writing 12 GB to reshape the code archive is wasted I/O).
if [ "${SKIP_MODELS:-0}" = "1" ] && [ -f "$OUT_DIR/maclf-models.tar" ]; then
    echo "   maclf-models.tar        reused (SKIP_MODELS=1)"
else
    tar -C "$PROJECT_ROOT" -cf "$OUT_DIR/maclf-models.tar" hf_cache
fi
echo "   maclf-models.tar        $(du -h "$OUT_DIR/maclf-models.tar" | cut -f1)"

say "Checksums"
( cd "$OUT_DIR" && sha256sum maclf-code-data.tar.gz maclf-models.tar > SHA256SUMS.txt && cat SHA256SUMS.txt )

cat <<EOF

──────────────────────────────────────────────────────────────────────────────
 Ready to send: $OUT_DIR

 On the server, both archives extract into the SAME directory:

     tar xzf maclf-code-data.tar.gz          # creates $BUNDLE_NAME/
     tar xf  maclf-models.tar -C $BUNDLE_NAME/   # adds $BUNDLE_NAME/hf_cache/

 Then follow $BUNDLE_NAME/server_bundle/README_SERVER.md.

 Staging tree left at $STAGE (delete it when the transfer is confirmed).
──────────────────────────────────────────────────────────────────────────────
EOF
