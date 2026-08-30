from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable


FIELDS = ["path", "label", "id", "generator", "source", "split"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalized_parts(value: str) -> tuple[str, ...]:
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return PurePosixPath(value).parts


def canonical_key(value: str, label: int) -> tuple[str, ...]:
    parts = normalized_parts(value)
    lowered = [part.lower() for part in parts]
    # COCO archives and prepared directories do not always preserve the
    # coco2017 parent directory, so val2017 is the stable cross-layout anchor.
    anchors = ("dalle",) if label == 1 else ("val2017",)
    for anchor in anchors:
        if anchor in lowered:
            index = lowered.index(anchor)
            return tuple(part.lower() for part in parts[index:])
    raise ValueError(f"Cannot identify a canonical WildFake path: {value}")


def selected_keys(path: Path) -> dict[int, set[tuple[str, ...]]]:
    selected: dict[int, set[tuple[str, ...]]] = {0: set(), 1: set()}
    for row in read_csv(path):
        label = int(row["label"])
        if label in selected:
            selected[label].add(canonical_key(row["path"], label))
    return selected


def archive_index(archive: zipfile.ZipFile, label: int) -> dict[tuple[str, ...], str]:
    index: dict[tuple[str, ...], str] = {}
    duplicates: set[tuple[str, ...]] = set()
    for member in archive.infolist():
        if member.is_dir():
            continue
        try:
            key = canonical_key(member.filename, label)
        except ValueError:
            continue
        if key in index:
            duplicates.add(key)
        else:
            index[key] = member.filename
    if duplicates:
        preview = ["/".join(key) for key in sorted(duplicates)[:5]]
        raise RuntimeError(f"Duplicate archive paths after normalization: {preview}")
    return index


def filter_rows(
    rows: Iterable[dict[str, str]],
    label: int,
    selection: set[tuple[str, ...]] | None,
    coco_count: int,
) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        if int(row.get("IsFake", label)) != label:
            continue
        key = canonical_key(row["Image_path"], label)
        if label == 0 and "val2017" not in key:
            continue
        if selection is not None and key not in selection:
            continue
        normalized = dict(row)
        normalized["_key"] = key
        candidates.append(normalized)

    candidates.sort(key=lambda row: row["_key"])
    if label == 0 and selection is None:
        if len(candidates) < coco_count:
            raise RuntimeError(
                f"COCO val2017 contains only {len(candidates)} rows; requested {coco_count}"
            )
        candidates = candidates[:coco_count]
    return candidates


def extract_group(
    archive_path: Path,
    rows: list[dict[str, str]],
    label: int,
    output_root: Path,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    missing: list[str] = []
    class_dir = "fake" if label == 1 else "real"
    with zipfile.ZipFile(archive_path) as archive:
        index = archive_index(archive, label)
        for position, row in enumerate(rows, start=1):
            key = row["_key"]
            member = index.get(key)
            if member is None:
                missing.append(row["Image_path"])
                continue
            relative = Path(class_dir).joinpath(*key)
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size == 0:
                with archive.open(member) as source, open(destination, "wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            output_rows.append(
                {
                    "path": relative.as_posix(),
                    "label": str(label),
                    "id": f"{'dalle3' if label else 'coco-val2017'}-{row.get('Num', position)}",
                    "generator": "DALL-E Advanced" if label else "COCO val2017",
                    "source": "WildFake demonstration benchmark",
                    "split": "test",
                }
            )
            if position % 1000 == 0:
                print(f"Extracted label={label}: {position}/{len(rows)}")
    if missing:
        raise RuntimeError(
            f"{len(missing)} CSV rows were missing from {archive_path}. First: {missing[:5]}"
        )
    return output_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved manifest: {path}")
    print("Labels:", Counter(row["label"] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the WildFake COCO-val2017 + DALL-E Advanced demo benchmark"
    )
    parser.add_argument("--raw-root", default="data/wildfake_raw")
    parser.add_argument("--output-root", default="data/wildfake_eval")
    parser.add_argument("--manifest", default="data/wildfake_demo.csv")
    parser.add_argument(
        "--selection-manifest",
        help=(
            "Existing benchmark manifest used to reproduce exactly the same image subset. "
            "Strongly recommended when the official 4,998-image COCO selection is available."
        ),
    )
    parser.add_argument(
        "--coco-count",
        type=int,
        default=4998,
        help="Deterministic fallback count when no selection manifest is supplied",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    dalle_zip = raw_root / "Images" / "Diffusion_based" / "DALLE.zip"
    dalle_csv = raw_root / "label_csv_files" / "dalle3.csv"
    coco_zip = raw_root / "Images" / "Real" / "coco.zip"
    coco_csv = raw_root / "label_csv_files" / "real_coco.csv"
    required = [dalle_zip, dalle_csv, coco_zip, coco_csv]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WildFake source files: {missing}")

    selection = (
        selected_keys(Path(args.selection_manifest).expanduser().resolve())
        if args.selection_manifest
        else None
    )
    dalle_rows = filter_rows(
        read_csv(dalle_csv), 1, selection[1] if selection else None, args.coco_count
    )
    coco_rows = filter_rows(
        read_csv(coco_csv), 0, selection[0] if selection else None, args.coco_count
    )
    if len(dalle_rows) != 8843:
        raise RuntimeError(f"Expected 8,843 DALL-E rows, found {len(dalle_rows)}")
    if selection and len(coco_rows) != len(selection[0]):
        raise RuntimeError(
            f"Selection manifest requested {len(selection[0])} COCO rows, found {len(coco_rows)}"
        )
    print(f"Selected: COCO={len(coco_rows)}, DALL-E={len(dalle_rows)}")

    rows = []
    rows.extend(extract_group(coco_zip, coco_rows, 0, output_root))
    rows.extend(extract_group(dalle_zip, dalle_rows, 1, output_root))
    write_manifest(manifest, rows)
    print(f"Images: {output_root}")
    if selection is None:
        print(
            "WARNING: no selection manifest was supplied. The COCO subset is a deterministic "
            "4,998-image fallback, not proof of the organizers' exact two-image exclusion."
        )


if __name__ == "__main__":
    main()
