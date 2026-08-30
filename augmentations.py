from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class AppliedDegradation:
    name: str
    value: float | int | str


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    down_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    down = image.resize(down_size, Image.Resampling.BICUBIC)
    return down.resize((width, height), Image.Resampling.BICUBIC)


def gaussian_noise(
    image: Image.Image, sigma: float, rng: np.random.Generator | None = None
) -> Image.Image:
    rng = rng or np.random.default_rng()
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
    return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), mode="RGB")


def color_jitter(
    image: Image.Image,
    magnitude: float,
    rng: random.Random | Any = random,
) -> tuple[Image.Image, str]:
    factors = {
        "brightness": rng.uniform(1.0 - magnitude, 1.0 + magnitude),
        "contrast": rng.uniform(1.0 - magnitude, 1.0 + magnitude),
        "saturation": rng.uniform(1.0 - magnitude, 1.0 + magnitude),
    }
    operations: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
        ("brightness", lambda x: ImageEnhance.Brightness(x).enhance(factors["brightness"])),
        ("contrast", lambda x: ImageEnhance.Contrast(x).enhance(factors["contrast"])),
        ("saturation", lambda x: ImageEnhance.Color(x).enhance(factors["saturation"])),
    ]
    rng.shuffle(operations)
    for _, operation in operations:
        image = operation(image)
    value = ",".join(f"{key}={factors[key]:.3f}" for key in sorted(factors))
    return image, value


def center_crop_resize(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


class CompoundDegradation:
    """Challenge-matched degradation composition with an epoch curriculum."""

    names = ("jpeg", "blur", "resize", "noise", "color_jitter", "crop")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.progress = 1.0

    def set_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, float(progress)))

    def _stage(self) -> str:
        curriculum = self.config.get("curriculum", {})
        if self.progress < float(curriculum.get("early_fraction", 0.25)):
            return "early"
        if self.progress < float(curriculum.get("middle_fraction", 0.60)):
            return "middle"
        return "late"

    def _sample_count(self, stage: str) -> int:
        if stage == "early":
            return random.choice([0, 1])
        if stage == "middle":
            return random.choice([1, 2])
        return random.choice([1, 2, 3])

    @staticmethod
    def _severity(values: list[Any], stage: str) -> Any:
        if stage == "early":
            return values[0]
        if stage == "middle":
            cutoff = max(1, (len(values) + 1) // 2)
            return random.choice(values[:cutoff])
        return random.choice(values)

    def __call__(self, image: Image.Image) -> tuple[Image.Image, list[AppliedDegradation]]:
        image = ensure_rgb(image)
        if not self.config.get("enabled", True):
            return image.copy(), []

        stage = self._stage()
        selected = random.sample(self.names, k=self._sample_count(stage))
        applied: list[AppliedDegradation] = []

        for name in selected:
            if name == "jpeg":
                value = self._severity(list(self.config["jpeg_qualities"]), stage)
                image = jpeg_compress(image, int(value))
            elif name == "blur":
                value = self._severity(list(self.config["blur_sigmas"]), stage)
                image = gaussian_blur(image, float(value))
            elif name == "resize":
                value = self._severity(list(self.config["resize_scales"]), stage)
                image = resize_roundtrip(image, float(value))
            elif name == "noise":
                value = self._severity(list(self.config["noise_sigmas"]), stage)
                image = gaussian_noise(image, float(value))
            elif name == "color_jitter":
                max_magnitude = float(self.config["color_jitter"])
                magnitude = max_magnitude * (0.5 if stage == "early" else 1.0)
                image, value = color_jitter(image, magnitude)
            elif name == "crop":
                target = float(self.config["center_crop_fraction"])
                value = 0.90 if stage == "early" else target
                image = center_crop_resize(image, float(value))
            else:  # pragma: no cover - guarded by names
                raise ValueError(f"Unknown degradation: {name}")
            applied.append(AppliedDegradation(name=name, value=value))

        return image, applied


class ImagePreprocessor:
    def __init__(
        self,
        size: int,
        mean: list[float],
        std: list[float],
        resize_mode: str = "stretch",
    ) -> None:
        self.size = int(size)
        self.mean = mean
        self.std = std
        self.resize_mode = str(resize_mode).lower()
        if self.resize_mode not in {"stretch", "center_crop"}:
            raise ValueError("resize_mode must be 'stretch' or 'center_crop'")

    def resize(self, image: Image.Image) -> Image.Image:
        image = ensure_rgb(image)
        if self.resize_mode == "stretch":
            return TF.resize(
                image,
                [self.size, self.size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        resized = TF.resize(
            image,
            self.size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return TF.center_crop(resized, [self.size, self.size])

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.to_tensor(self.resize(image))


class TexturePatchSampler:
    """Select native-resolution patches from texture-rich and texture-poor regions.

    Candidate crops are placed on a regular grid and ranked by the variance of a
    small discrete Laplacian.  Selection is deterministic for a given image.  A
    clean/degraded pair can reuse the clean-image boxes so the consistency loss
    compares the same spatial regions.
    """

    def __init__(
        self,
        patch_size: int = 224,
        rich_patches: int = 2,
        poor_patches: int = 2,
        candidate_grid: int = 4,
        score_size: int = 64,
    ) -> None:
        self.patch_size = int(patch_size)
        self.rich_patches = int(rich_patches)
        self.poor_patches = int(poor_patches)
        self.candidate_grid = int(candidate_grid)
        self.score_size = int(score_size)
        if self.patch_size <= 0:
            raise ValueError("local_patch_size must be positive")
        if self.rich_patches < 0 or self.poor_patches < 0:
            raise ValueError("Texture patch counts cannot be negative")
        if self.rich_patches + self.poor_patches <= 0:
            raise ValueError("At least one local patch is required")
        if self.candidate_grid <= 0:
            raise ValueError("local_candidate_grid must be positive")

    @property
    def num_patches(self) -> int:
        return self.rich_patches + self.poor_patches

    def _ensure_minimum_size(self, image: Image.Image) -> Image.Image:
        image = ensure_rgb(image)
        width, height = image.size
        scale = max(self.patch_size / width, self.patch_size / height, 1.0)
        if scale == 1.0:
            return image
        size = (max(self.patch_size, round(width * scale)), max(self.patch_size, round(height * scale)))
        return image.resize(size, Image.Resampling.BICUBIC)

    def _candidate_boxes(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        width, height = image.size
        max_left = width - self.patch_size
        max_top = height - self.patch_size
        xs = np.linspace(0, max_left, self.candidate_grid).round().astype(int).tolist()
        ys = np.linspace(0, max_top, self.candidate_grid).round().astype(int).tolist()
        return list(
            dict.fromkeys(
                (left, top, left + self.patch_size, top + self.patch_size)
                for top in ys
                for left in xs
            )
        )

    def _texture_score(self, patch: Image.Image) -> float:
        gray = patch.convert("L").resize(
            (self.score_size, self.score_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(gray, dtype=np.float32)
        laplacian = (
            -4.0 * array[1:-1, 1:-1]
            + array[:-2, 1:-1]
            + array[2:, 1:-1]
            + array[1:-1, :-2]
            + array[1:-1, 2:]
        )
        return float(laplacian.var())

    def select_boxes(self, image: Image.Image) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
        image = self._ensure_minimum_size(image)
        candidates = self._candidate_boxes(image)
        ranked = sorted(
            ((self._texture_score(image.crop(box)), index, box) for index, box in enumerate(candidates)),
            key=lambda item: (item[0], item[1]),
        )
        poor = [item[2] for item in ranked[: self.poor_patches]]
        poor_set = set(poor)
        rich = [item[2] for item in reversed(ranked) if item[2] not in poor_set][
            : self.rich_patches
        ]
        selected = rich + poor
        if len(selected) < self.num_patches:
            fallback = [item[2] for item in reversed(ranked)] or candidates
            selected.extend(
                fallback[index % len(fallback)]
                for index in range(self.num_patches - len(selected))
            )
        return image, selected

    def sample(self, image: Image.Image) -> list[Image.Image]:
        image, boxes = self.select_boxes(image)
        return [image.crop(box) for box in boxes]

    def sample_pair(
        self, clean: Image.Image, augmented: Image.Image
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        clean = self._ensure_minimum_size(clean)
        augmented = ensure_rgb(augmented)
        if augmented.size != clean.size:
            augmented = augmented.resize(clean.size, Image.Resampling.BICUBIC)
        clean, boxes = self.select_boxes(clean)
        return (
            [clean.crop(box) for box in boxes],
            [augmented.crop(box) for box in boxes],
        )


def build_model_preprocessors(
    model_config: dict[str, Any],
) -> tuple[ImagePreprocessor, TexturePatchSampler | None, ImagePreprocessor | None]:
    global_preprocessor = ImagePreprocessor(
        model_config["image_size"],
        model_config["image_mean"],
        model_config["image_std"],
        resize_mode=model_config.get("resize_mode", "stretch"),
    )
    if str(model_config.get("architecture", "global")).lower() != "global_local":
        return global_preprocessor, None, None
    sampler = TexturePatchSampler(
        patch_size=int(model_config.get("local_patch_size", 224)),
        rich_patches=int(model_config.get("texture_rich_patches", 2)),
        poor_patches=int(model_config.get("texture_poor_patches", 2)),
        candidate_grid=int(model_config.get("local_candidate_grid", 4)),
        score_size=int(model_config.get("local_texture_score_size", 64)),
    )
    local_preprocessor = ImagePreprocessor(
        sampler.patch_size,
        model_config["image_mean"],
        model_config["image_std"],
        resize_mode="stretch",
    )
    return global_preprocessor, sampler, local_preprocessor


@dataclass(frozen=True)
class DeterministicCondition:
    name: str
    value: float | int | None
    seed: int

    def __call__(self, image: Image.Image, sample_index: int = 0) -> Image.Image:
        image = ensure_rgb(image)
        if self.name == "clean":
            return image.copy()
        if self.name == "jpeg":
            return jpeg_compress(image, int(self.value))
        if self.name == "blur":
            return gaussian_blur(image, float(self.value))
        if self.name == "resize":
            return resize_roundtrip(image, float(self.value))
        if self.name == "noise":
            return gaussian_noise(
                image, float(self.value), np.random.default_rng(self.seed + sample_index)
            )
        if self.name == "color_jitter_minus":
            factor = 1.0 - float(self.value)
            image = ImageEnhance.Brightness(image).enhance(factor)
            image = ImageEnhance.Contrast(image).enhance(factor)
            return ImageEnhance.Color(image).enhance(factor)
        if self.name == "color_jitter_plus":
            factor = 1.0 + float(self.value)
            image = ImageEnhance.Brightness(image).enhance(factor)
            image = ImageEnhance.Contrast(image).enhance(factor)
            return ImageEnhance.Color(image).enhance(factor)
        if self.name == "crop":
            return center_crop_resize(image, float(self.value))
        raise ValueError(f"Unknown evaluation condition: {self.name}")


def deterministic_condition(
    name: str, value: float | int | None, seed: int
) -> DeterministicCondition:
    return DeterministicCondition(name=name, value=value, seed=seed)
