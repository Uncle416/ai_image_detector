from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image


FIELDS = [
    "path",
    "label",
    "id",
    "generator",
    "source",
    "split",
    "holdout",
    "sha256",
    "phash",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(f"Saved {len(rows)} rows: {path}")
    print("Labels:", Counter(str(row["label"]) for row in rows))
    print("Splits:", Counter(str(row.get("split", "train")) for row in rows))
    print("Generators:", Counter(str(row.get("generator", "unknown")) for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    try:
        import imagehash
    except ImportError as exc:
        raise ImportError("Install ImageHash to enable perceptual deduplication") from exc
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def image_suffix(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        image_format = (image.format or "PNG").lower()
    return {"jpeg": ".jpg", "tiff": ".tif"}.get(image_format, f".{image_format}")


def safe_id(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return normalized or "sample"


def export_sid(args: argparse.Namespace) -> None:
    try:
        from datasets import Image as DatasetImage
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` before exporting SID_Set") from exc

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    stream = stream.cast_column("image", DatasetImage(decode=False))
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    target = {0: args.per_class, 1: args.per_class}
    counts = Counter()
    rows: list[dict[str, Any]] = []
    scanned = 0

    for row in stream:
        scanned += 1
        if scanned > args.max_scanned:
            break
        label = int(row["label"])
        if label not in target or counts[label] >= target[label]:
            continue

        payload = row["image"]
        if isinstance(payload, dict) and payload.get("bytes") is not None:
            data = payload["bytes"]
        elif isinstance(payload, dict) and payload.get("path"):
            data = Path(payload["path"]).read_bytes()
        else:
            raise RuntimeError("SID_Set image payload did not expose original bytes")

        digest = hashlib.sha256(data).hexdigest()
        suffix = image_suffix(data)
        sample_id = f"sid-{label}-{safe_id(row.get('img_id', scanned))}-{digest[:12]}"
        destination = output_root / str(label) / digest[:2] / f"{sample_id}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)

        rows.append(
            {
                "path": str(destination),
                "label": label,
                "id": sample_id,
                "generator": "sid-real" if label == 0 else "sid-full-synthetic",
                "source": "SID_Set",
                "split": "train",
                "holdout": "0",
                "sha256": digest,
                "phash": "",
            }
        )
        counts[label] += 1
        if counts[0] >= target[0] and counts[1] >= target[1]:
            break
        if len(rows) % 500 == 0:
            print(f"SID_Set exported: real={counts[0]} fake={counts[1]} scanned={scanned}")

    if counts[0] != target[0] or counts[1] != target[1]:
        raise RuntimeError(
            f"SID_Set stream ended before targets were met: {dict(counts)}, scanned={scanned}"
        )
    write_manifest(args.manifest, rows)


def export_imagefolder(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    mapping = {"real": 0, "fake": 1}

    for class_name, label in mapping.items():
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.parent.name.lower() == class_name
        ]
        if len(candidates) < args.per_class:
            raise RuntimeError(
                f"{root} contains only {len(candidates)} {class_name} images; "
                f"requested {args.per_class}"
            )
        for path in rng.sample(sorted(candidates), args.per_class):
            digest = sha256_file(path)
            rows.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "id": f"{args.source}-{label}-{digest[:16]}",
                    "generator": f"{args.source}-{'real' if label == 0 else 'fake'}",
                    "source": args.source,
                    "split": "train",
                    "holdout": "0",
                    "sha256": digest,
                    "phash": "",
                }
            )
    write_manifest(args.manifest, rows)


def normalize_csv_path(value: str) -> tuple[str, ...]:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return PurePosixPath(normalized).parts


def resolve_extracted_path(root: Path, csv_path: str) -> Path | None:
    parts = normalize_csv_path(csv_path)
    matches: list[Path] = []
    for drop in range(min(6, len(parts))):
        candidate = root.joinpath(*parts[drop:])
        if candidate.is_file():
            matches.append(candidate.resolve())
    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous extracted path for {csv_path}: {matches}")
    return matches[0] if matches else None


def wildfake_generator(row: dict[str, str]) -> str:
    return (
        row.get("Weight")
        or row.get("Architecture")
        or row.get("Category")
        or "unknown"
    ).strip()


def is_forbidden_benchmark_row(row: dict[str, str]) -> bool:
    path = row.get("Image_path", "").replace("\\", "/").lower()
    is_dalle_advanced = (
        "/advanced/dalle3/" in path
        or (
            row.get("IsAdvanced") == "1"
            and "dalle" in " ".join(
                [row.get("Architecture", ""), row.get("Weight", ""), row.get("Category", "")]
            ).lower()
        )
    )
    is_coco_val = "/coco2017/val2017/" in path or "/val2017/" in path
    return is_dalle_advanced or is_coco_val


def export_wildfake(args: argparse.Namespace) -> None:
    root = Path(args.image_root).expanduser().resolve()
    train_generators = set(args.train_generators)
    val_generators = set(args.val_generators)
    overlap = train_generators.intersection(val_generators)
    if overlap:
        raise ValueError(f"Generators cannot be in both train and val: {sorted(overlap)}")

    grouped: defaultdict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    missing: list[str] = []
    for csv_path in args.csv:
        for row in read_manifest(csv_path):
            if is_forbidden_benchmark_row(row):
                continue
            generator = wildfake_generator(row)
            if generator in train_generators:
                split = "train"
            elif generator in val_generators:
                split = "val"
            else:
                continue
            label = 1 if row.get("IsFake") == "1" else 0
            path = resolve_extracted_path(root, row["Image_path"])
            if path is None:
                missing.append(row["Image_path"])
                continue
            normalized = dict(row)
            normalized["_resolved_path"] = str(path)
            grouped[(split, label, generator)].append(normalized)

    if missing:
        print(f"Warning: {len(missing)} selected WildFake rows were not extracted")
        for path in missing[:10]:
            print("missing:", path)

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for (split, label, generator), candidates in sorted(grouped.items()):
        limit = min(len(candidates), args.per_generator)
        for row in rng.sample(candidates, limit):
            path = Path(row["_resolved_path"])
            digest = sha256_file(path)
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "id": f"wildfake-{safe_id(generator)}-{row.get('Num', digest[:12])}",
                    "generator": f"wildfake-{generator}",
                    "source": "WildFake",
                    "split": split,
                    "holdout": "0",
                    "sha256": digest,
                    "phash": "",
                }
            )
    if not rows:
        raise RuntimeError("No usable WildFake rows were selected")
    write_manifest(args.manifest, rows)


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def has_within(self, value: int, maximum_distance: int) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= maximum_distance:
                return True
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            stack.extend(
                child for edge, child in node[1].items() if lower <= edge <= upper
            )
        return False


def ensure_fingerprints(row: dict[str, str], use_phash: bool) -> dict[str, str]:
    row = dict(row)
    path = Path(row["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    row["path"] = str(path)
    row["sha256"] = row.get("sha256") or sha256_file(path)
    if use_phash:
        row["phash"] = row.get("phash") or perceptual_hash(path)
    else:
        row["phash"] = row.get("phash", "")
    return row


def balanced_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    output: list[dict[str, str]] = []
    for split in sorted({row.get("split", "train") for row in rows}):
        by_label = {
            label: [
                row
                for row in rows
                if row.get("split", "train") == split and int(row["label"]) == label
            ]
            for label in (0, 1)
        }
        if not by_label[0] or not by_label[1]:
            raise RuntimeError(f"Split {split!r} does not contain both labels")
        limit = min(len(by_label[0]), len(by_label[1]))
        output.extend(rng.sample(by_label[0], limit))
        output.extend(rng.sample(by_label[1], limit))
    rng.shuffle(output)
    return output


def merge_manifests(args: argparse.Namespace) -> None:
    use_phash = args.phash_distance >= 0
    benchmark_rows = [
        ensure_fingerprints(row, use_phash) for row in read_manifest(args.benchmark_manifest)
    ]
    benchmark_sha = {row["sha256"] for row in benchmark_rows}
    benchmark_tree = BKTree()
    if use_phash:
        for row in benchmark_rows:
            benchmark_tree.add(int(row["phash"], 16))

    candidates: list[dict[str, str]] = []
    for manifest in args.inputs:
        candidates.extend(read_manifest(manifest))

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    kept: list[dict[str, str]] = []
    seen_sha: set[str] = set()
    removed = Counter()
    group_counts = Counter()

    for raw_row in candidates:
        row = ensure_fingerprints(raw_row, use_phash)
        if row.get("holdout") == "1":
            removed["holdout"] += 1
            continue
        if row["sha256"] in benchmark_sha:
            removed["benchmark_exact"] += 1
            continue
        if use_phash and benchmark_tree.has_within(int(row["phash"], 16), args.phash_distance):
            removed["benchmark_near"] += 1
            continue
        if row["sha256"] in seen_sha:
            removed["duplicate"] += 1
            continue
        group = (
            row.get("split", "train"),
            int(row["label"]),
            row.get("generator", "unknown"),
        )
        if args.max_per_generator > 0 and group_counts[group] >= args.max_per_generator:
            removed["generator_cap"] += 1
            continue
        seen_sha.add(row["sha256"])
        group_counts[group] += 1
        kept.append(row)

    if args.balance_classes:
        kept = balanced_rows(kept, args.seed)
    print("Removed:", removed)
    write_manifest(args.output, kept)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe V2.1 training manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sid = subparsers.add_parser("sid", help="Stream and materialize a balanced SID_Set subset")
    sid.add_argument("--dataset", default="saberzl/SID_Set")
    sid.add_argument("--dataset-split", default="train")
    sid.add_argument("--output-root", required=True)
    sid.add_argument("--manifest", required=True)
    sid.add_argument("--per-class", type=int, default=10_000)
    sid.add_argument("--shuffle-buffer", type=int, default=10_000)
    sid.add_argument("--max-scanned", type=int, default=210_000)
    sid.add_argument("--seed", type=int, default=42)
    sid.set_defaults(function=export_sid)

    imagefolder = subparsers.add_parser(
        "imagefolder", help="Export a balanced REAL/FAKE ImageFolder subset"
    )
    imagefolder.add_argument("--root", required=True)
    imagefolder.add_argument("--manifest", required=True)
    imagefolder.add_argument("--source", default="cifake")
    imagefolder.add_argument("--per-class", type=int, default=2_500)
    imagefolder.add_argument("--seed", type=int, default=42)
    imagefolder.set_defaults(function=export_imagefolder)

    wildfake = subparsers.add_parser(
        "wildfake", help="Convert selected extracted WildFake generators to a manifest"
    )
    wildfake.add_argument("--csv", nargs="+", required=True)
    wildfake.add_argument("--image-root", required=True)
    wildfake.add_argument("--manifest", required=True)
    wildfake.add_argument("--train-generators", nargs="+", required=True)
    wildfake.add_argument("--val-generators", nargs="*", default=[])
    wildfake.add_argument("--per-generator", type=int, default=2_500)
    wildfake.add_argument("--seed", type=int, default=42)
    wildfake.set_defaults(function=export_wildfake)

    merge = subparsers.add_parser(
        "merge", help="Merge, cap, balance, and deduplicate manifests against the benchmark"
    )
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--benchmark-manifest", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--max-per-generator", type=int, default=10_000)
    merge.add_argument("--phash-distance", type=int, default=4)
    merge.add_argument("--balance-classes", action="store_true")
    merge.add_argument("--seed", type=int, default=42)
    merge.set_defaults(function=merge_manifests)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
