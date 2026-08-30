import csv
from pathlib import Path

from prepare_wildfake_validation import canonical_key, filter_rows, selected_keys


def test_canonical_key_keeps_dalle_parent_directories() -> None:
    first = canonical_key(
        "/workspace/wildfake_eval/DALLE/Advanced/DALLE3/dalle3/a/same.jpg", 1
    )
    second = canonical_key(
        "./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/b/same.jpg", 1
    )
    assert first != second
    assert first[-1] == second[-1] == "same.jpg"


def test_coco_filter_is_stable_and_limited() -> None:
    rows = [
        {
            "IsFake": "0",
            "Image_path": f"./Real/coco/coco2017/val2017/{name}.jpg",
        }
        for name in ["c", "a", "b"]
    ]
    selected = filter_rows(rows, label=0, selection=None, coco_count=2)
    assert [row["_key"][-1] for row in selected] == ["a.jpg", "b.jpg"]


def test_selection_manifest_uses_full_relative_path(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerow(
            {
                "path": "/old/DALLE/Advanced/DALLE3/dalle3/session-a/repeat.jpg",
                "label": "1",
            }
        )
    keys = selected_keys(manifest)
    assert canonical_key(
        "Diffusion_based/DALLE/Advanced/DALLE3/dalle3/session-a/repeat.jpg", 1
    ) in keys[1]
    assert canonical_key(
        "Diffusion_based/DALLE/Advanced/DALLE3/dalle3/session-b/repeat.jpg", 1
    ) not in keys[1]
