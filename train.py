from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from augmentations import CompoundDegradation, build_model_preprocessors
from dataset import build_dataset
from losses import paired_classification_consistency_loss
from metrics import binary_metrics
from model import DinoBinaryClassifier
from utils import (
    PrecisionPolicy,
    atomic_torch_save,
    configure_runtime,
    dump_json,
    get_device,
    load_config,
    make_grad_scaler,
    resolve_precision,
    runtime_summary,
    seed_everything,
    seed_worker,
)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_summary(
    dataset: torch.utils.data.Dataset, source_config: dict[str, Any]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(dataset)}
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        summary["labels"] = dict(
            sorted(Counter(str(int(sample.label)) for sample in samples).items())
        )
        summary["generators"] = dict(
            sorted(Counter(str(sample.generator) for sample in samples).items())
        )
    manifest_value = source_config.get("path")
    if manifest_value:
        manifest = Path(manifest_value).expanduser().resolve()
        summary["manifest"] = str(manifest)
        summary["manifest_sha256"] = file_sha256(manifest)
    summary["split"] = source_config.get("split")
    return summary


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    config: dict[str, Any],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    sampling = str(config.get("sampling", "shuffle")).lower()
    if shuffle and sampling == "group_balanced":
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise ValueError("group_balanced sampling requires a manifest/imagefolder dataset")
        group_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
        for sample in samples:
            group_counts[(int(sample.label), str(sample.generator))] += 1
        weights = [
            1.0 / group_counts[(int(sample.label), str(sample.generator))]
            for sample in samples
        ]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    elif sampling not in {"shuffle", "group_balanced"}:
        raise ValueError("data.sampling must be 'shuffle' or 'group_balanced'")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)) and torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def train_one_epoch(
    model: DinoBinaryClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    precision: PrecisionPolicy,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    totals: defaultdict[str, float] = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for batch_index, batch in enumerate(progress):
        clean = batch["clean"].to(device, non_blocking=True)
        augmented = batch["augmented"].to(device, non_blocking=True)
        local_patches = None
        if "clean_local" in batch:
            local_patches = torch.cat(
                [batch["clean_local"], batch["augmented_local"]], dim=0
            ).to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=precision.dtype,
            enabled=precision.autocast_enabled,
        ):
            logits, features = model(
                torch.cat([clean, augmented], dim=0), local_patches=local_patches
            )
            clean_logits, augmented_logits = logits.chunk(2)
            clean_features, augmented_features = features.chunk(2)
            loss, parts = paired_classification_consistency_loss(
                clean_logits,
                augmented_logits,
                clean_features,
                augmented_features,
                labels,
                clean_bce_weight=float(config["loss"].get("clean_bce_weight", 1.0)),
                augmented_bce_weight=float(
                    config["loss"].get("augmented_bce_weight", 1.0)
                ),
                lambda_consistency=float(config["loss"]["lambda_consistency"]),
            )
            scaled_loss = loss / accumulation

        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for key, value in parts.items():
            totals[key] += float(value.item())
        progress.set_postfix(loss=f"{parts['total'].item():.4f}")

    return {key: value / len(loader) for key, value in totals.items()}


@torch.inference_mode()
def validate(
    model: DinoBinaryClassifier,
    loader: DataLoader,
    device: torch.device,
    precision: PrecisionPolicy,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    loss_total = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["clean"].to(device, non_blocking=True)
        local_patches = (
            batch["clean_local"].to(device, non_blocking=True)
            if "clean_local" in batch
            else None
        )
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=precision.dtype,
            enabled=precision.autocast_enabled,
        ):
            logits, _ = model(images, local_patches=local_patches)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss_total += float(loss.item())
        labels.extend(targets.cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    result = binary_metrics(labels, probabilities, threshold)
    result["loss"] = loss_total / len(loader)
    return result


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    runtime_config = config.get("runtime", {})
    device = get_device(runtime_config.get("device", "auto"))
    configure_runtime(runtime_config, device)
    precision = resolve_precision(
        runtime_config,
        device,
        legacy_amp=bool(config["training"].get("amp", True)),
    )
    scaler = make_grad_scaler(precision)
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DinoBinaryClassifier(config["model"])
    preprocessor, local_patch_sampler, local_preprocessor = build_model_preprocessors(
        config["model"]
    )
    train_dataset = build_dataset(
        config["data"]["train"],
        config["data"],
        preprocessor,
        degradation=CompoundDegradation(config["augmentation"]),
        local_patch_sampler=local_patch_sampler,
        local_preprocessor=local_preprocessor,
    )
    val_dataset = build_dataset(
        config["data"]["val"],
        config["data"],
        preprocessor,
        local_patch_sampler=local_patch_sampler,
        local_preprocessor=local_preprocessor,
    )
    train_loader = make_loader(
        train_dataset, int(config["training"]["batch_size"]), config["data"], True, seed
    )
    val_loader = make_loader(
        val_dataset, int(config["evaluation"]["batch_size"]), config["data"], False, seed
    )

    freeze_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
    model.set_backbone_trainable(freeze_epochs == 0)
    model.to(device)
    print(f"Runtime: {runtime_summary(device, precision)}")
    print(f"Parameters: {model.parameter_summary()}")
    print(f"Train/val samples: {len(train_dataset)}/{len(val_dataset)}")
    config_file = Path(config_path).expanduser().resolve()
    run_metadata = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file),
        "model": {
            "architecture": model.architecture,
            **model.parameter_summary(),
        },
        "train_dataset": dataset_summary(train_dataset, config["data"]["train"]),
        "val_dataset": dataset_summary(val_dataset, config["data"]["val"]),
    }
    dump_json(run_metadata, output_dir / "run_metadata.json")
    print(f"Run metadata: {run_metadata}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(config["training"]["backbone_lr"])},
            {"params": model.head.parameters(), "lr": float(config["training"]["head_lr"])},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    updates_per_epoch = math.ceil(
        len(train_loader) / int(config["training"].get("gradient_accumulation_steps", 1))
    )
    total_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = round(total_steps * float(config["training"].get("warmup_fraction", 0.1)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    selection_metric = str(config["training"].get("selection_metric", "accuracy"))
    best_metric = -1.0
    best_accuracy = -1.0
    resume = config["training"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(
            checkpoint.get(
                "best_metric",
                checkpoint.get("best_accuracy", -1.0),
            )
        )
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
        model.set_backbone_trainable(start_epoch >= freeze_epochs)

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs):
        if epoch == freeze_epochs:
            model.set_backbone_trainable(True)
            print("Backbone unfrozen")
        if hasattr(train_dataset, "set_progress"):
            train_dataset.set_progress(epoch / max(1, epochs - 1))
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, precision, device, config
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            precision,
            threshold=float(config["evaluation"].get("threshold", 0.5)),
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)

        metric_value = val_metrics.get(selection_metric)
        if metric_value is None:
            raise ValueError(
                f"training.selection_metric={selection_metric!r} is unavailable in validation metrics"
            )
        metric_value = float(metric_value)
        best_accuracy = max(best_accuracy, float(val_metrics["accuracy"]))
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "selection_metric": selection_metric,
            "best_metric": max(best_metric, metric_value),
            "best_accuracy": best_accuracy,
            "config": config,
            "metadata": run_metadata,
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if metric_value > best_metric:
            best_metric = metric_value
            payload["best_metric"] = best_metric
            atomic_torch_save(payload, output_dir / "best.pt")
        dump_json(history, output_dir / "history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the V0-V3 DINOv2 AIGC detector")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
