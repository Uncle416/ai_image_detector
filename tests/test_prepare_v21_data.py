import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from prepare_v21_data import (
    BKTree,
    build_parser,
    compose_v3_dataset,
    is_forbidden_benchmark_row,
    read_manifest,
    resolve_manifest_paths,
    resolve_extracted_path,
    sample_balanced_by_generator,
    write_manifest,
)


def test_benchmark_rows_are_always_forbidden() -> None:
    dalle = {
        "Image_path": "./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/x.jpg",
        "IsAdvanced": "1",
        "Architecture": "DALLE",
    }
    coco = {"Image_path": "./Real/coco/coco2017/val2017/x.jpg"}
    allowed = {"Image_path": "./Diffusion_based/DDPM/train/x.jpg"}

    assert is_forbidden_benchmark_row(dalle)
    assert is_forbidden_benchmark_row(coco)
    assert not is_forbidden_benchmark_row(allowed)


def test_full_relative_path_disambiguates_duplicate_basenames(tmp_path: Path) -> None:
    first = tmp_path / "DALLE" / "Advanced" / "DALLE3" / "dalle3" / "a" / "same.jpg"
    second = tmp_path / "DALLE" / "Advanced" / "DALLE3" / "dalle3" / "b" / "same.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    resolved = resolve_extracted_path(
        tmp_path,
        "./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/b/same.jpg",
    )

    assert resolved == second.resolve()


def test_bk_tree_finds_near_hashes() -> None:
    tree = BKTree()
    tree.add(0b0000)
    tree.add(0b1111)

    assert tree.has_within(0b0001, 1)
    assert not tree.has_within(0b0011, 1)


def _make_manifest(
    tmp_path: Path,
    name: str,
    specifications: list[tuple[int, str, str]],
) -> Path:
    rows = []
    for index, (label, generator, split) in enumerate(specifications):
        path = tmp_path / "images" / f"{name}-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (8 + index % 3, 8 + index % 5),
            color=((index * 37 + len(name)) % 256, label * 127, len(generator) * 11 % 256),
        ).save(path)
        rows.append(
            {
                "path": str(path),
                "label": str(label),
                "id": f"{name}-{index}",
                "generator": generator,
                "source": name,
                "split": split,
                "holdout": "0",
                "sha256": "",
                "phash": "",
            }
        )
    manifest = tmp_path / f"{name}.csv"
    write_manifest(manifest, rows)
    return manifest


def test_compose_v3_enforces_source_quotas_and_generator_balance(
    tmp_path: Path,
) -> None:
    sid = _make_manifest(
        tmp_path,
        "sid",
        [(label, f"sid-{label}", "train") for label in (0, 1) for _ in range(4)],
    )
    cifake = _make_manifest(
        tmp_path,
        "cifake",
        [(label, f"cifake-{label}", "train") for label in (0, 1) for _ in range(4)],
    )
    wildfake_specs = []
    for label, generators in ((0, ("real-a", "real-b")), (1, ("fake-a", "fake-b"))):
        for generator in generators:
            wildfake_specs.extend([(label, generator, "train")] * 3)
    wildfake_specs.extend([(0, "held-real", "val")] * 2)
    wildfake_specs.extend([(1, "held-fake", "val")] * 2)
    wildfake = _make_manifest(tmp_path, "wildfake", wildfake_specs)
    benchmark = _make_manifest(tmp_path, "benchmark", [(0, "benchmark", "test")])

    output = tmp_path / "mixed_v3.csv"
    stats_output = tmp_path / "mixed_v3.stats.json"
    compose_v3_dataset(
        argparse.Namespace(
            sid_manifest=str(sid),
            cifake_manifest=str(cifake),
            wildfake_manifest=str(wildfake),
            benchmark_manifest=str(benchmark),
            benchmark_root=None,
            output=str(output),
            stats_output=str(stats_output),
            sid_train_total=4,
            cifake_train_total=4,
            wildfake_train_total=8,
            wildfake_val_total=4,
            minimum_quota_fraction=0.90,
            phash_distance=-1,
            seed=42,
        )
    )

    rows = read_manifest(output)
    assert len(rows) == 20
    assert Counter((row["source"], row["split"], row["label"]) for row in rows) == {
        ("SID_Set", "train", "0"): 2,
        ("SID_Set", "train", "1"): 2,
        ("CIFAKE", "train", "0"): 2,
        ("CIFAKE", "train", "1"): 2,
        ("WildFake", "train", "0"): 4,
        ("WildFake", "train", "1"): 4,
        ("WildFake", "val", "0"): 2,
        ("WildFake", "val", "1"): 2,
    }
    wildfake_train_counts = Counter(
        row["generator"]
        for row in rows
        if row["source"] == "WildFake" and row["split"] == "train"
    )
    assert set(wildfake_train_counts.values()) == {2}
    stats = json.loads(stats_output.read_text(encoding="utf-8"))
    assert stats["selected_rows"] == 20
    assert stats["requested_train_totals"] == {
        "SID_Set": 4,
        "CIFAKE": 4,
        "WildFake": 8,
    }


def test_balanced_sampler_accepts_small_target_shortfall() -> None:
    rows = [
        {"label": str(label), "generator": f"g{label}-{index % 2}"}
        for label in (0, 1)
        for index in range(4)
    ]

    selected = sample_balanced_by_generator(
        rows, total=10, seed=42, context="small pool", minimum_fraction=0.80
    )

    assert len(selected) == 8
    assert Counter(row["label"] for row in selected) == {"0": 4, "1": 4}


def test_resolve_manifest_paths_uses_benchmark_root(tmp_path: Path) -> None:
    rows = [{"path": "real/example.jpg"}, {"path": str(tmp_path / "absolute.jpg")}]

    resolved = resolve_manifest_paths(rows, str(tmp_path / "benchmark"))

    assert resolved[0]["path"] == str((tmp_path / "benchmark/real/example.jpg").resolve())
    assert resolved[1]["path"] == str((tmp_path / "absolute.jpg").resolve())


def test_prepare_data_parser_exposes_compose_v3_defaults() -> None:
    args = build_parser().parse_args(
        [
            "compose-v3",
            "--sid-manifest",
            "sid.csv",
            "--cifake-manifest",
            "cifake.csv",
            "--wildfake-manifest",
            "wildfake.csv",
            "--benchmark-manifest",
            "benchmark.csv",
            "--output",
            "mixed.csv",
        ]
    )

    assert args.sid_train_total == 40_000
    assert args.cifake_train_total == 40_000
    assert args.wildfake_train_total == 60_000
    assert args.wildfake_val_total == 10_000
    assert args.minimum_quota_fraction == 0.90
    assert args.benchmark_root is None
