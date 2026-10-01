"""
Diagnose the 12-class multiclass tumor CNN (DenseNet169) on one or more datasets.

The CNN reported ~99% on its own test split but ~0.14 on 3-class figshare
(data/figshare/{1,2,3}), predicting e.g. pituitary -> granuloma at ~0.99.
This script separates the two usual causes:

  * class-index order bug   -> the best label permutation (Hungarian assignment
                               over the confusion matrix) recovers high accuracy;
  * preprocessing mismatch  -> intensity statistics differ between datasets, and
                               one of the preprocessing variants recovers accuracy.

The model, checkpoint and transform are exactly those of agents/cnn_tool.py
(CNNClassifier._load_model and its transform); the "as_is" variant is
CNNClassifier.classify's input path (PIL .convert("RGB") -> transform).

Usage (run where the data lives):

    python scripts/diagnose_multiclass_cnn.py \\
        --dataset train_test=data/tumor_multiclass/test \\
        --dataset figshare=data/figshare:figshare3 \\
        --max_per_class 100 --out_dir outputs/diagnose_multiclass_cnn

--dataset NAME=DIR[:LABEL_MAP]
    DIR holds one sub-folder per class (images found recursively inside it).
    LABEL_MAP maps folder names to CNN class names:
      auto (default)  folder names normalised, e.g. "Glioma T1" -> glioma,
                      "pituitary" -> pituitary_tumor, "no_tumor" -> normal,
                      Spanish spellings of the 44-class Kaggle set handled;
      figshare3       1=meningioma, 2=glioma, 3=pituitary_tumor;
      inline          "1=meningioma,2=glioma,3=pituitary_tumor";
      a .json file    {"folder": "class", ...}.
    Folders that map to no CNN class are skipped (and listed in the summary).
"""

import argparse
import csv
import dataclasses
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TASK = "multiclass_tumor"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXCLUDE_DIRS = {"overlay", "mask", "masks", "seg", "segmentation"}
VARIANTS = ["as_is", "minmax", "pct1_99", "inverted"]
VARIANT_DOC = {
    "as_is": "exactly CNNClassifier.classify: PIL .convert('RGB') then the CNN transform",
    "minmax": "raw pixel array (16-bit kept) min-max scaled per image to 0-255",
    "pct1_99": "raw pixel array clipped to its 1st-99th percentile, scaled to 0-255",
    "inverted": "255 minus the minmax variant",
}

FIGSHARE3 = {"1": "meningioma", "2": "glioma", "3": "pituitary_tumor"}
# Folder-name aliases -> CNN class (substring match on the normalised name, first hit wins).
ALIASES = [
    ("pituitar", "pituitary_tumor"), ("hipofis", "pituitary_tumor"),
    ("no_tumor", "normal"), ("notumor", "normal"), ("normal", "normal"), ("healthy", "normal"),
    ("meduloblastoma", "medulloblastoma"), ("medulloblastoma", "medulloblastoma"),
    ("neurocitoma", "neurocytoma"), ("neurocytoma", "neurocytoma"),
    ("papiloma", "papilloma"), ("papilloma", "papilloma"),
    ("astrocitoma", "glioma"), ("astrocytoma", "glioma"), ("ependimoma", "glioma"),
    ("ependymoma", "glioma"), ("ganglioglioma", "glioma"), ("glioblastoma", "glioma"),
    ("oligodendroglioma", "glioma"), ("glioma", "glioma"),
    ("meningioma", "meningioma"), ("schwannoma", "schwannoma"), ("tuberculoma", "tuberculoma"),
    ("granuloma", "granuloma"), ("germinoma", "germinoma"), ("carcinoma", "carcinoma"),
]


# ── Arguments and label maps ─────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Diagnose the 12-class multiclass tumor CNN: index-order bug vs preprocessing mismatch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--dataset NAME" + __doc__.split("--dataset NAME", 1)[1],
    )
    p.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR[:LABEL_MAP]",
                   help="Dataset to evaluate (repeatable). LABEL_MAP: auto | figshare3 | 'a=x,b=y' | map.json")
    p.add_argument("--max_per_class", type=int, default=200, help="Images sampled per class (default 200; 0 = all)")
    p.add_argument("--out_dir", default="outputs/diagnose_multiclass_cnn")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: config.py's ModelConfig.device)")
    p.add_argument("--checkpoint", default=None, help="Override the multiclass CNN checkpoint path")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _norm(name: str) -> str:
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    n = re.sub(r"_(t1c\+?|t1|t2|flair)$", "", n)  # "Glioma T1C+" -> "glioma"
    return n.strip("_")


def auto_label(folder: str, classes: list[str]):
    n = _norm(folder)
    if n in classes:
        return n
    return next((cls for alias, cls in ALIASES if alias in n), None)


def parse_dataset(spec: str):
    if "=" not in spec:
        raise SystemExit(f"--dataset {spec!r}: expected NAME=DIR[:LABEL_MAP]")
    name, rest = spec.split("=", 1)
    directory, label_map = rest, "auto"
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if tail and not tail.startswith(("/", "\\")):
            directory, label_map = head, tail
    return name, Path(directory), label_map


def resolve_label_map(label_map: str, folders: list[str], classes: list[str]) -> dict:
    if label_map == "auto":
        return {f: auto_label(f, classes) for f in folders}
    if label_map == "figshare3":
        mapping = FIGSHARE3
    elif label_map.endswith(".json"):
        mapping = json.loads(Path(label_map).read_text())
    elif "=" in label_map:
        mapping = dict(item.split("=", 1) for item in label_map.split(",") if item)
    else:
        raise SystemExit(f"unknown label map {label_map!r} (auto | figshare3 | a=x,b=y | file.json)")
    resolved = {}
    for folder in folders:
        target = mapping.get(folder)
        if target is not None and target not in classes:
            target = auto_label(target, classes)
        resolved[folder] = target
    return resolved


def collect(directory: Path, label_map: str, classes: list[str], max_per_class: int, rng: random.Random):
    if not directory.is_dir():
        raise SystemExit(f"dataset dir not found: {directory}")
    folders = sorted(d.name for d in directory.iterdir() if d.is_dir())
    mapping = resolve_label_map(label_map, folders, classes)
    samples, skipped = [], []
    for folder in folders:
        label = mapping.get(folder)
        if label is None:
            skipped.append(folder)
            continue
        files = sorted(
            f for f in (directory / folder).rglob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
            and "_mask" not in f.stem.lower()
            and not EXCLUDE_DIRS & {part.lower() for part in f.relative_to(directory).parts[:-1]}
        )
        if max_per_class and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))
        samples += [(f, label, folder) for f in files]
    return samples, mapping, skipped


# ── Images and preprocessing variants ────────────────────────────────────────


def raw_array(image):
    """Pixel values without PIL's 8-bit clipping (16-bit PNGs kept), single channel."""
    import numpy as np

    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        return np.asarray(image, dtype=np.float64)
    if image.mode not in ("L", "1"):
        image = image.convert("L")
    return np.asarray(image, dtype=np.float64)


def bit_depth(image, arr) -> str:
    declared = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "LA": 8, "I;16": 16, "I;16B": 16, "I;16L": 16, "I": 32, "F": 32}
    used = int(arr.max()).bit_length() if arr.size and arr.max() >= 0 else 0
    return f"{declared.get(image.mode, '?')}-bit ({used} used)"


def to_uint8(arr):
    import numpy as np

    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return np.round(255.0 * (arr - lo) / (hi - lo)).astype(np.uint8)


def variant_image(variant: str, path: Path):
    """PIL image for one variant; the CNN transform (from cnn_tool) is applied afterwards."""
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    if variant == "as_is":
        return image.convert("RGB")  # == CNNClassifier.classify
    arr = raw_array(image)
    if variant == "minmax":
        out = to_uint8(arr)
    elif variant == "pct1_99":
        lo, hi = np.percentile(arr, [1, 99])
        out = to_uint8(np.clip(arr, lo, hi)) if hi > lo else to_uint8(arr)
    elif variant == "inverted":
        out = 255 - to_uint8(arr)
    else:
        raise ValueError(variant)
    return Image.fromarray(out, mode="L").convert("RGB")


def image_stats(path: Path) -> dict:
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    arr = raw_array(image)
    p1, p50, p99 = np.percentile(arr, [1, 50, 99])
    return {
        "mode": image.mode, "format": image.format, "width": image.width, "height": image.height,
        "bit_depth": bit_depth(image, arr), "min": arr.min(), "max": arr.max(),
        "mean": arr.mean(), "std": arr.std(), "p1": p1, "p50": p50, "p99": p99,
        "frac_zero": float((arr == 0).mean()),
        "frac_ge_255": float((arr >= 255).mean()),  # what PIL's 8-bit conversion would saturate
    }


# ── Metrics ──────────────────────────────────────────────────────────────────


def metrics(probs, logits, true_idx, classes, present):
    """Accuracy, accuracy with softmax restricted to the dataset's classes, best permutation."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred = probs.argmax(1)
    acc = float((pred == true_idx).mean())

    masked = np.full_like(logits, -np.inf)
    masked[:, present] = logits[:, present]
    masked_pred = masked.argmax(1)
    masked_acc = float((masked_pred == true_idx).mean())

    # Confusion over true classes present × all 12 predicted classes.
    confusion = np.zeros((len(present), len(classes)), dtype=int)
    for t, p in zip(true_idx, pred):
        confusion[present.index(t), p] += 1
    rows, cols = linear_sum_assignment(-confusion)
    perm = {classes[present[r]]: classes[c] for r, c in zip(rows, cols)}
    perm_acc = float(confusion[rows, cols].sum() / max(1, len(true_idx)))
    return {
        "accuracy": acc, "masked_accuracy": masked_acc, "best_permutation_accuracy": perm_acc,
        "best_permutation": perm, "confusion": confusion,
        "mean_confidence": float(probs.max(1).mean()),
        "mean_confidence_when_wrong": float(probs.max(1)[pred != true_idx].mean()) if (pred != true_idx).any() else float("nan"),
    }, pred


# ── Main ─────────────────────────────────────────────────────────────────────


def load_cnn(args):
    import torch  # noqa: F401  (imported here so --help works without torch)
    from agents.cnn_tool import CLASS_NAMES, CNNClassifier
    from config import DEFAULT_CONFIG

    model_cfg = DEFAULT_CONFIG.model
    checkpoints = dict(model_cfg.cnn_checkpoints)
    if args.checkpoint:
        checkpoints[TASK] = args.checkpoint
    model_cfg = dataclasses.replace(
        model_cfg, cnn_checkpoints=checkpoints, **({"device": args.device} if args.device else {})
    )
    classifier = CNNClassifier(model_cfg=model_cfg)
    model = classifier._load_model(TASK)
    ckpt = checkpoints.get(TASK)
    if ckpt is None or not Path(ckpt).exists():
        raise SystemExit(f"multiclass checkpoint not found ({ckpt}); refusing to diagnose ImageNet fallback weights")
    temperature = float(model_cfg.cnn_temperatures.get(TASK, 1.0))
    return classifier, model, list(CLASS_NAMES[TASK]), temperature, ckpt


def infer(classifier, model, paths, variant, batch_size, temperature):
    import numpy as np
    import torch

    logits_all = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = [classifier._transform(variant_image(variant, p)) for p in paths[start:start + batch_size]]
            logits = model(torch.stack(batch).to(classifier.device)).float().cpu()
            logits_all.append(logits / temperature)
    logits = torch.cat(logits_all).numpy()
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    return logits.astype(np.float64), probs


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    classifier, model, classes, temperature, ckpt = load_cnn(args)
    print(f"[diagnose] checkpoint {ckpt} on {classifier.device}; classes (index order): {classes}")

    metric_rows, image_rows, intensity_rows = [], [], []
    md = [
        "# Multiclass tumor CNN diagnosis", "",
        f"- checkpoint: `{ckpt}`, temperature {temperature}",
        f"- class index order (agents/cnn_tool.py): {', '.join(f'{i}={c}' for i, c in enumerate(classes))}",
        f"- max_per_class {args.max_per_class}, seed {args.seed}", "",
        "Variants: " + "; ".join(f"**{v}** = {VARIANT_DOC[v]}" for v in args.variants), "",
    ]

    for spec in args.dataset:
        name, directory, label_map = parse_dataset(spec)
        samples, mapping, skipped = collect(directory, label_map, classes, args.max_per_class, rng)
        if not samples:
            print(f"[diagnose] {name}: no images found under {directory}, skipping")
            md += [f"## {name}", "", f"No images found under `{directory}`.", ""]
            continue
        paths = [s[0] for s in samples]
        true_idx = np.array([classes.index(s[1]) for s in samples])
        present = sorted(set(true_idx.tolist()))
        print(f"[diagnose] {name}: {len(samples)} images, classes {[classes[i] for i in present]}")

        md += [f"## {name} (`{directory}`, label map `{label_map}`)", "",
               "Folder → class: " + ", ".join(f"`{f}`→{c}" for f, c in mapping.items() if c),
               *( [f"Skipped folders (no class): {', '.join(f'`{s}`' for s in skipped)}"] if skipped else [] ),
               f"Images per class: " + ", ".join(f"{c}={n}" for c, n in sorted(Counter(s[1] for s in samples).items())),
               ""]

        # Intensity statistics per class.
        per_image_stats = [image_stats(p) for p in paths]
        by_class = defaultdict(list)
        for (path, label, folder), st in zip(samples, per_image_stats):
            by_class[label].append(st)
        md += ["| class | n | mode | bit depth | size (most common) | mean | std | p1 | p50 | p99 | max | frac 0 | frac ≥255 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, stats in sorted(by_class.items()):
            row = {
                "dataset": name, "class": label, "n": len(stats),
                "modes": dict(Counter(s["mode"] for s in stats)),
                "formats": dict(Counter(s["format"] for s in stats)),
                "bit_depths": dict(Counter(s["bit_depth"].split(" ")[0] for s in stats)),
                "sizes": dict(Counter(f"{s['width']}x{s['height']}" for s in stats).most_common(3)),
            }
            for key in ("mean", "std", "p1", "p50", "p99", "min", "max", "frac_zero", "frac_ge_255"):
                row[key] = float(np.mean([s[key] for s in stats]))
            intensity_rows.append(row)
            md.append(
                f"| {label} | {len(stats)} | {row['modes']} | {row['bit_depths']} | {next(iter(row['sizes']))} | "
                f"{row['mean']:.1f} | {row['std']:.1f} | {row['p1']:.1f} | {row['p50']:.1f} | {row['p99']:.1f} | "
                f"{row['max']:.0f} | {row['frac_zero']:.2f} | {row['frac_ge_255']:.2f} |"
            )
        md += ["", "| variant | accuracy | masked to dataset classes | best permutation | mean conf | conf when wrong |",
               "|---|---|---|---|---|---|"]

        per_image = [{"dataset": name, "path": str(p), "folder": f, "true": l, **{k: v for k, v in st.items()}}
                     for (p, l, f), st in zip(samples, per_image_stats)]
        perm_notes = []
        for variant in args.variants:
            logits, probs = infer(classifier, model, paths, variant, args.batch_size, temperature)
            m, pred = metrics(probs, logits, true_idx, classes, present)
            for row, pi, pr in zip(per_image, pred, probs):
                row[f"pred_{variant}"] = classes[pi]
                row[f"conf_{variant}"] = round(float(pr.max()), 4)
            metric_rows.append({
                "dataset": name, "variant": variant, "n": len(paths),
                **{k: round(v, 4) for k, v in m.items() if isinstance(v, float)},
                "best_permutation": json.dumps(m["best_permutation"]),
            })
            with (out / f"confusion_{name}_{variant}.csv").open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["true \\ predicted", *classes])
                for i, t in enumerate(present):
                    writer.writerow([classes[t], *m["confusion"][i].tolist()])
            md.append(
                f"| {variant} | {m['accuracy']:.3f} | {m['masked_accuracy']:.3f} | {m['best_permutation_accuracy']:.3f} | "
                f"{m['mean_confidence']:.3f} | {fmt(m['mean_confidence_when_wrong'])} |"
            )
            moved = {t: p for t, p in m["best_permutation"].items() if t != p}
            if moved:
                perm_notes.append(f"- {variant}: best permutation maps true→predicted " +
                                  ", ".join(f"{t}→{p}" for t, p in m["best_permutation"].items()))
            if variant == args.variants[0]:
                top = []
                for i, t in enumerate(present):
                    row = m["confusion"][i]
                    j = int(row.argmax())
                    top.append(f"{classes[t]}→{classes[j]} ({row[j]}/{row.sum()})")
                md_top = f"Most frequent prediction per true class ({variant}): " + "; ".join(top)
        md += ["", md_top, *perm_notes, ""]
        image_rows += per_image

    write_csv(out / "metrics.csv", metric_rows)
    write_csv(out / "intensity_by_class.csv", [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in intensity_rows])
    write_csv(out / "per_image.csv", image_rows)

    md += [
        "## How to read this", "",
        "- **best permutation ≫ accuracy** (e.g. 0.14 → 0.9) on a dataset the CNN was trained on: the class-index",
        "  order in `agents/cnn_tool.py:_CNN_MULTICLASS_CLASSES` does not match the training label order",
        "  (torchvision `ImageFolder` sorts folder names case-sensitively, so `Glioma`/`_NORMAL` can sort",
        "  before lowercase names). The permutation on the CNN's own 12-class test split gives the fix.",
        "- **masked ≫ accuracy**: the CNN ranks the right class highly but out-of-dataset classes win;",
        "  partial domain shift rather than a pure index bug.",
        "- **a variant ≫ as_is**, or intensity stats (bit depth, max, frac ≥255, p99, image size) clearly",
        "  differ from the training split: preprocessing mismatch (e.g. 16-bit PNGs saturated by PIL's",
        "  `.convert('RGB')`, or the .mat → PNG conversion scaling differently from the Kaggle JPEGs).",
        "- Everything stays low on every variant and permutation, but the training split is fine: the",
        "  images are out of distribution in some other way (orientation, cropping, different slices).",
        "",
        "Files: `metrics.csv`, `confusion_<dataset>_<variant>.csv`, `intensity_by_class.csv`, `per_image.csv`.",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[diagnose] wrote {out}/summary.md and CSVs")


if __name__ == "__main__":
    main()
