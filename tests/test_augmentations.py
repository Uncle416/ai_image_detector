import random
import pickle

import numpy as np
from PIL import Image

from augmentations import (
    CompoundDegradation,
    ImagePreprocessor,
    center_crop_resize,
    deterministic_condition,
    gaussian_blur,
    jpeg_compress,
    resize_roundtrip,
)
import pytest


def test_fixed_degradations_preserve_image_size() -> None:
    image = Image.new("RGB", (80, 64), color=(64, 128, 192))
    outputs = [
        jpeg_compress(image, 30),
        gaussian_blur(image, 2.0),
        resize_roundtrip(image, 0.25),
        center_crop_resize(image, 0.8),
    ]
    assert all(output.size == image.size for output in outputs)


def test_evaluation_noise_is_repeatable_per_sample() -> None:
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    transform = deterministic_condition("noise", 0.05, seed=123)
    first = np.asarray(transform(image, 7))
    second = np.asarray(transform(image, 7))
    different_sample = np.asarray(transform(image, 8))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_sample)


def test_evaluation_condition_is_picklable_for_spawn_workers() -> None:
    transform = deterministic_condition("jpeg", 70, seed=123)
    restored = pickle.loads(pickle.dumps(transform))
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    assert np.array_equal(np.asarray(transform(image, 0)), np.asarray(restored(image, 0)))


def test_late_curriculum_composes_one_to_three_transforms() -> None:
    config = {
        "enabled": True,
        "curriculum": {"early_fraction": 0.25, "middle_fraction": 0.60},
        "jpeg_qualities": [90, 70, 50, 30],
        "blur_sigmas": [0.5, 1.0, 2.0],
        "resize_scales": [0.5, 0.25],
        "noise_sigmas": [0.02, 0.05, 0.10],
        "color_jitter": 0.20,
        "center_crop_fraction": 0.80,
    }
    random.seed(9)
    degradation = CompoundDegradation(config)
    degradation.set_progress(1.0)
    _, applied = degradation(Image.new("RGB", (64, 64), color="gray"))
    assert 1 <= len(applied) <= 3
    assert len({item.name for item in applied}) == len(applied)


def test_center_crop_preprocessor_preserves_square_output() -> None:
    preprocessor = ImagePreprocessor(
        32,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        resize_mode="center_crop",
    )
    image = Image.new("RGB", (80, 40), color="gray")

    assert preprocessor.resize(image).size == (32, 32)
    assert preprocessor(image).shape == (3, 32, 32)


def test_invalid_resize_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="resize_mode"):
        ImagePreprocessor(32, [0.0] * 3, [1.0] * 3, resize_mode="warp")
