from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
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


def sample_balanced_by_generator(
    rows: list[dict[str, str]],
    total: int,
    seed: int,
    context: str,
    minimum_fraction: float = 1.0,
) -> list[dict[str, str]]:
    """Select a class-balanced target while preventing one generator from dominating."""
    if total <= 0 or total % 2:
        raise ValueError(f"{context} total must be a positive even number, got {total}")
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be in (0, 1]")

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    target_per_label = total // 2
    grouped_by_label: dict[int, defaultdict[str, list[dict[str, str]]]] = {}
    available_by_label: dict[int, int] = {}
    for label in (0, 1):
        groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if int(row["label"]) == label:
                groups[row.get("generator", "unknown")].append(row)
        if not groups:
            raise RuntimeError(f"{context} label={label} has no usable generators")
        grouped_by_label[label] = groups
        available_by_label[label] = sum(len(group) for group in groups.values())

    actual_per_label = min(target_per_label, available_by_label[0], available_by_label[1])
    minimum_per_label = math.ceil(target_per_label * minimum_fraction)
    if actual_per_label < minimum_per_label:
        raise RuntimeError(
            f"{context} cannot meet the allowed quota range: target={total}, "
            f"minimum={minimum_per_label * 2}, available={available_by_label}"
        )
    if actual_per_label < target_per_label:
        print(
            f"Warning: {context} selected {actual_per_label * 2}/{total} rows after "
            f"filtering while preserving class balance"
        )

    for label in (0, 1):
        groups = grouped_by_label[label]
        generator_names = sorted(groups)
        rng.shuffle(generator_names)
        for group in groups.values():
            rng.shuffle(group)

        label_selection: list[dict[str, str]] = []
        positions = Counter()
        while len(label_selection) < actual_per_label:
            made_progress = False
            for generator in generator_names:
                position = positions[generator]
                group = groups[generator]
                if position >= len(group):
                    continue
                label_selection.append(group[position])
                positions[generator] += 1
                made_progress = True
                if len(label_selection) >= actual_per_label:
                    break
            if not made_progress:
                raise RuntimeError(
                    f"{context} label={label} could not meet quota {actual_per_label}"
                )
        selected.extend(label_selection)

    rng.shuffle(selected)
    return selected


def _counter_for_json(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[str, int]:
    counts = Counter("/".join(str(row.get(field, "")) for field in fields) for row in rows)
    return dict(sorted(counts.items()))


def resolve_manifest_paths(
    rows: list[dict[str, str]], root: str | None
) -> list[dict[str, str]]:
    if root is None:
        return rows
    resolved_root = Path(root).expanduser().resolve()
    resolved = []
    for raw_row in rows:
        row = dict(raw_row)
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            row["path"] = str((resolved_root / path).resolve())
        resolved.append(row)
    return resolved


def compose_v3_dataset(args: argparse.Namespace) -> None:
    """Build a source-balanced V3 manifest around per-source target sizes."""
    use_phash = args.phash_distance >= 0
    benchmark_rows = (
        resolve_manifest_paths(
            read_manifest(args.benchmark_manifest), args.benchmark_root
        )
        if args.benchmark_manifest
        else []
    )
    benchmark_rows = [ensure_fingerprints(row, use_phash) for row in benchmark_rows]
    benchmark_sha = {row["sha256"] for row in benchmark_rows}
    benchmark_tree = BKTree()
    if use_phash:
        for row in benchmark_rows:
            benchmark_tree.add(int(row["phash"], 16))

    inputs = [
        ("SID_Set", args.sid_manifest),
        ("CIFAKE", args.cifake_manifest),
        ("WildFake", args.wildfake_manifest),
    ]
    candidates: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    seen_sha: set[str] = set()
    removed = Counter()

    for source, manifest in inputs:
        source_rows = read_manifest(manifest)
        random.Random(args.seed).shuffle(source_rows)
        for raw_row in source_rows:
            if raw_row.get("holdout") == "1":
                removed["holdout"] += 1
                continue
            row = ensure_fingerprints(raw_row, use_phash)
            row["source"] = source
            if row["sha256"] in benchmark_sha:
                removed["benchmark_exact"] += 1
                continue
            if use_phash and benchmark_tree.has_within(
                int(row["phash"], 16), args.phash_distance
            ):
                removed["benchmark_near"] += 1
                continue
            if row["sha256"] in seen_sha:
                removed["duplicate"] += 1
                continue
            seen_sha.add(row["sha256"])
            candidates[source].append(row)

    requested_train = {
        "SID_Set": args.sid_train_total,
        "CIFAKE": args.cifake_train_total,
        "WildFake": args.wildfake_train_total,
    }
    selected: list[dict[str, str]] = []
    for offset, (source, total) in enumerate(requested_train.items()):
        train_rows = [row for row in candidates[source] if row.get("split", "train") == "train"]
        selected.extend(
            sample_balanced_by_generator(
                train_rows,
                total,
                args.seed + offset,
                f"{source} train",
                args.minimum_quota_fraction,
            )
        )

    wildfake_val_candidates = [
        row for row in candidates["WildFake"] if row.get("split", "train") == "val"
    ]
    selected_val = sample_balanced_by_generator(
        wildfake_val_candidates,
        args.wildfake_val_total,
        args.seed + 100,
        "WildFake val",
        args.minimum_quota_fraction,
    )
    selected.extend(selected_val)

    train_generators = {
        row.get("generator", "unknown")
        for row in selected
        if row["source"] == "WildFake" and row.get("split", "train") == "train"
    }
    val_generators = {
        row.get("generator", "unknown")
        for row in selected
        if row["source"] == "WildFake" and row.get("split", "train") == "val"
    }
    overlap = train_generators.intersection(val_generators)
    if overlap:
        raise RuntimeError(f"WildFake train/val generator leakage: {sorted(overlap)}")

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    write_manifest(args.output, selected)

    selected_paths = {Path(row["path"]).resolve() for row in selected}
    selected_bytes = sum(path.stat().st_size for path in selected_paths)
    stats = {
        "schema_version": 1,
        "seed": args.seed,
        "phash_distance": args.phash_distance,
        "minimum_quota_fraction": args.minimum_quota_fraction,
        "requested_train_totals": requested_train,
        "requested_wildfake_val_total": args.wildfake_val_total,
        "selected_rows": len(selected),
        "selected_unique_bytes": selected_bytes,
        "selected_unique_gib": round(selected_bytes / (1024**3), 3),
        "source_split_label": _counter_for_json(selected, ("source", "split", "label")),
        "source_split_generator": _counter_for_json(
            selected, ("source", "split", "generator")
        ),
        "removed": dict(sorted(removed.items())),
        "wildfake_train_generators": sorted(train_generators),
        "wildfake_val_generators": sorted(val_generators),
    }
    stats_path = Path(args.stats_output or f"{args.output}.stats.json").expanduser().resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f"Saved audit statistics: {stats_path}")
    print(f"Selected image bytes (files are referenced, not copied): {stats['selected_unique_gib']} GiB")


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
    parser = argparse.ArgumentParser(description="Prepare leakage-safe V2.1/V3 training manifests")
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

    compose = subparsers.add_parser(
        "compose-v3",
        help="Sample roughly balanced SID/CIFAKE/WildFake targets for V3 training",
    )
    compose.add_argument("--sid-manifest", required=True)
    compose.add_argument("--cifake-manifest", required=True)
    compose.add_argument("--wildfake-manifest", required=True)
    compose.add_argument(
        "--benchmark-manifest",
        help="Optional external benchmark manifest for image-level leakage checks",
    )
    compose.add_argument(
        "--benchmark-root",
        help="Root directory for relative image paths in the benchmark manifest",
    )
    compose.add_argument("--output", required=True)
    compose.add_argument("--stats-output")
    compose.add_argument("--sid-train-total", type=int, default=40_000)
    compose.add_argument("--cifake-train-total", type=int, default=40_000)
    compose.add_argument("--wildfake-train-total", type=int, default=60_000)
    compose.add_argument("--wildfake-val-total", type=int, default=10_000)
    compose.add_argument(
        "--minimum-quota-fraction",
        type=float,
        default=0.90,
        help="Allow balanced post-dedup totals down to this fraction of each target",
    )
    compose.add_argument(
        "--phash-distance",
        type=int,
        default=4,
        help="Benchmark perceptual-hash exclusion distance; use -1 for exact SHA-256 only",
    )
    compose.add_argument("--seed", type=int, default=42)
    compose.set_defaults(function=compose_v3_dataset)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
