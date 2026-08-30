from __future__ import annotations

import argparse

import torch

from model import DinoBinaryClassifier
from utils import configure_runtime, get_device, load_config, resolve_precision, runtime_summary


def main(config_path: str, run_forward: bool) -> None:
    config = load_config(config_path)
    runtime_config = config.get("runtime", {})
    requested = runtime_config.get("device", "cuda")
    if requested == "auto":
        requested = "cuda"
    device = get_device(requested)
    if device.type != "cuda":
        raise RuntimeError("cuda_preflight.py requires a CUDA device")
    configure_runtime(runtime_config, device)
    precision = resolve_precision(runtime_config, device)
    print(runtime_summary(device, precision))

    if run_forward:
        model = DinoBinaryClassifier(config["model"]).to(device).eval()
        size = int(config["model"]["image_size"])
        sample = torch.zeros(1, 3, size, size, device=device)
        local_patches = None
        if model.architecture == "global_local":
            patch_size = int(config["model"].get("local_patch_size", 224))
            patch_count = int(config["model"].get("texture_rich_patches", 2)) + int(
                config["model"].get("texture_poor_patches", 2)
            )
            local_patches = torch.zeros(
                1, patch_count, 3, patch_size, patch_size, device=device
            )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=precision.dtype, enabled=precision.autocast_enabled
        ):
            logits, features = model(sample, local_patches=local_patches)
        print(
            {
                "forward": "ok",
                "logits_shape": tuple(logits.shape),
                "features_shape": tuple(features.shape),
                "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the NVIDIA CUDA runtime")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-forward", action="store_true")
    args = parser.parse_args()
    main(args.config, args.run_forward)
