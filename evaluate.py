from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import ImagePreprocessor, deterministic_condition
from dataset import build_dataset
from metrics import binary_metrics
from model import DinoBinaryClassifier
from utils import (
    PrecisionPolicy,
    configure_runtime,
    dump_json,
    get_device,
    load_config,
    resolve_precision,
    runtime_summary,
    seed_everything,
    seed_worker,
)


def condition_grid(config: dict[str, Any]) -> list[tuple[str, str, float | int | None]]:
    augmentation = config["augmentation"]
    conditions: list[tuple[str, str, float | int | None]] = [("clean", "clean", None)]
    conditions.extend(
        (f"jpeg_q{quality}", "jpeg", quality) for quality in augmentation["jpeg_qualities"]
    )
    conditions.extend(
        (f"blur_sigma{sigma:g}", "blur", sigma) for sigma in augmentation["blur_sigmas"]
    )
    conditions.extend(
        (f"resize_{scale:g}x", "resize", scale) for scale in augmentation["resize_scales"]
    )
    conditions.extend(
        (f"noise_sigma{sigma:g}", "noise", sigma) for sigma in augmentation["noise_sigmas"]
    )
    jitter = float(augmentation["color_jitter"])
    conditions.extend(
        [
            (f"color_jitter_minus{jitter:g}", "color_jitter_minus", jitter),
            (f"color_jitter_plus{jitter:g}", "color_jitter_plus", jitter),
        ]
    )
    crop = float(augmentation["center_crop_fraction"])
    conditions.append((f"center_crop_{crop:g}", "crop", crop))
    return conditions


def make_loader(dataset: torch.utils.data.Dataset, config: dict[str, Any], seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=bool(config["data"].get("pin_memory", True)) and torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


@torch.inference_mode()
def score_condition(
    model: DinoBinaryClassifier,
    loader: DataLoader,
    device: torch.device,
    precision: PrecisionPolicy,
    threshold: float,
    description: str,
) -> dict[str, Any]:
    labels: list[float] = []
    probabilities: list[float] = []
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=precision.dtype,
            enabled=precision.autocast_enabled,
        ):
            logits, _ = model(images)
        labels.extend(batch["label"].tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return binary_metrics(labels, probabilities, threshold)


def main(config_path: str, checkpoint_path: str, source_name: str) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    runtime_config = config.get("runtime", {})
    device = get_device(runtime_config.get("device", "auto"))
    configure_runtime(runtime_config, device)
    precision = resolve_precision(runtime_config, device)
    print(f"Runtime: {runtime_summary(device, precision)}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", config)
    model_config = checkpoint_config["model"]
    model = DinoBinaryClassifier(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    if source_name not in config["data"]:
        raise KeyError(f"data.{source_name} is not defined in {config_path}")
    preprocessor = ImagePreprocessor(
        model_config["image_size"], model_config["image_mean"], model_config["image_std"]
    )
    threshold = float(config["evaluation"].get("threshold", 0.5))
    results: list[dict[str, Any]] = []
    for index, (display_name, operation, value) in enumerate(condition_grid(config)):
        transform = deterministic_condition(operation, value, seed + index)
        dataset = build_dataset(
            config["data"][source_name],
            config["data"],
            preprocessor,
            evaluation_transform=transform,
        )
        metrics = score_condition(
            model,
            make_loader(dataset, config, seed),
            device,
            precision,
            threshold,
            display_name,
        )
        result = {"condition": display_name, "operation": operation, "value": value, **metrics}
        results.append(result)
        print(result)

    transformed = [row for row in results if row["condition"] != "clean"]
    summary = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "source": source_name,
        "threshold": threshold,
        "clean_accuracy": results[0]["accuracy"],
        "mean_transformed_accuracy": sum(row["accuracy"] for row in transformed) / len(transformed),
        "worst_condition": min(transformed, key=lambda row: row["accuracy"])["condition"],
        "worst_condition_accuracy": min(row["accuracy"] for row in transformed),
        "conditions": results,
    }

    output_dir = Path(checkpoint_path).expanduser().resolve().parent
    csv_path = output_dir / config["evaluation"].get("output_csv", "robustness.csv")
    json_path = output_dir / config["evaluation"].get("output_json", "robustness.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["condition", "operation", "value", "accuracy", "auroc", "f1", "count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    dump_json(summary, json_path)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(
        "Summary: "
        f"clean={summary['clean_accuracy']:.4f}, "
        f"mean_transformed={summary['mean_transformed_accuracy']:.4f}, "
        f"worst={summary['worst_condition']} ({summary['worst_condition_accuracy']:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clean and transformed robustness")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--source",
        default="val",
        help="Dataset key under config data (for example val or robustness_test)",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.source)
