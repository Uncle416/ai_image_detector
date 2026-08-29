from PIL import Image

from dataset import samples_from_imagefolder


def test_imagefolder_limit_is_balanced(tmp_path) -> None:
    for class_name in ("REAL", "FAKE"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(3):
            Image.new("RGB", (8, 8), color=(index, index, index)).save(
                class_dir / f"{index}.png"
            )

    samples = samples_from_imagefolder(
        {"path": str(tmp_path), "max_samples_per_class": 2},
        {"real": 0, "fake": 1},
    )

    assert len(samples) == 4
    assert sum(sample.label == 0 for sample in samples) == 2
    assert sum(sample.label == 1 for sample in samples) == 2
