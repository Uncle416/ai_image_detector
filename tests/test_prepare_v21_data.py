from pathlib import Path

from prepare_v21_data import BKTree, is_forbidden_benchmark_row, resolve_extracted_path


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
