from PIL import Image

from augmentations import ImagePreprocessor, TexturePatchSampler
from dataset import BinaryImageDataset, ImageSample, samples_from_imagefolder


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


def test_global_local_dataset_returns_paired_patch_tensors(tmp_path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (80, 64), color="gray").save(path)
    preprocessor = ImagePreprocessor(32, [0.0] * 3, [1.0] * 3)
    sampler = TexturePatchSampler(32, rich_patches=2, poor_patches=2)
    dataset = BinaryImageDataset(
        [ImageSample(path=path, label=1, sample_id="sample")],
        preprocessor,
        local_patch_sampler=sampler,
        local_preprocessor=preprocessor,
    )

    item = dataset[0]

    assert item["clean"].shape == (3, 32, 32)
    assert item["clean_local"].shape == (4, 3, 32, 32)
    assert item["augmented_local"].shape == (4, 3, 32, 32)
