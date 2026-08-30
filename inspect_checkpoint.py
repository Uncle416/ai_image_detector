from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def checkpoint_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    return {
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "best_metric": checkpoint.get("best_metric"),
        "best_accuracy": checkpoint.get("best_accuracy"),
        "model_config": config.get("model"),
        "training_config": config.get("training"),
        "train_source_config": config.get("data", {}).get("train"),
        "val_source_config": config.get("data", {}).get("val"),
        "recorded_metadata": checkpoint.get("metadata"),
    }


def main(checkpoint_path: str) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint, got {type(checkpoint).__name__}")
    print(json.dumps(checkpoint_summary(checkpoint), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect trusted detector checkpoint metadata")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)
