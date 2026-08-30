from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class DinoBinaryClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config["backbone"])
        if config.get("gradient_checkpointing", True) and hasattr(
            self.backbone, "gradient_checkpointing_enable"
        ):
            self.backbone.gradient_checkpointing_enable()
        patch_size = self.backbone.config.patch_size
        patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
        if int(config["image_size"]) % patch_size != 0:
            raise ValueError(
                f"model.image_size={config['image_size']} must be divisible by patch size {patch_size}"
            )
        self.pooling = config.get("pooling", "cls_mean")
        if self.pooling not in {"cls", "cls_mean"}:
            raise ValueError("model.pooling must be 'cls' or 'cls_mean'")
        self.architecture = str(config.get("architecture", "global")).lower()
        if self.architecture not in {"global", "global_local"}:
            raise ValueError("model.architecture must be 'global' or 'global_local'")
        if self.architecture == "global_local":
            local_patch_size = int(config.get("local_patch_size", 224))
            if local_patch_size % patch_size != 0:
                raise ValueError(
                    f"model.local_patch_size={local_patch_size} must be divisible by "
                    f"patch size {patch_size}"
                )
        self.local_fusion = str(config.get("local_fusion", "concat")).lower()
        if self.local_fusion not in {"concat", "mean"}:
            raise ValueError("model.local_fusion must be 'concat' or 'mean'")
        self.local_forward_chunk_size = int(config.get("local_forward_chunk_size", 0))
        if self.local_forward_chunk_size < 0:
            raise ValueError("model.local_forward_chunk_size cannot be negative")

        hidden_size = int(self.backbone.config.hidden_size)
        encoder_feature_size = hidden_size if self.pooling == "cls" else hidden_size * 2
        feature_size = encoder_feature_size
        if self.architecture == "global_local" and self.local_fusion == "concat":
            feature_size *= 2
        self.encoder_feature_size = encoder_feature_size
        head_hidden = int(config.get("head_hidden_dim", 512))
        self.head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, head_hidden),
            nn.GELU(),
            nn.Dropout(float(config.get("dropout", 0.2))),
            nn.Linear(head_hidden, 1),
        )

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        limit = int(config.get("max_parameters", 2_000_000_000))
        if parameter_count >= limit:
            raise ValueError(
                f"Model has {parameter_count:,} parameters, violating the <{limit:,} limit"
            )

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        cls_feature = hidden[:, 0]
        if self.pooling == "cls":
            return cls_feature
        patch_mean = hidden[:, 1:].mean(dim=1)
        return torch.cat([cls_feature, patch_mean], dim=-1)

    def encode_local(self, local_patches: torch.Tensor) -> torch.Tensor:
        if local_patches.ndim != 5:
            raise ValueError("local_patches must have shape [batch, patches, channels, height, width]")
        batch_size, patch_count = local_patches.shape[:2]
        if patch_count <= 0:
            raise ValueError("local_patches must contain at least one patch")
        flattened = local_patches.flatten(0, 1)
        chunk_size = self.local_forward_chunk_size or flattened.shape[0]
        encoded = torch.cat(
            [self.encode(chunk) for chunk in flattened.split(chunk_size, dim=0)], dim=0
        )
        return encoded.reshape(batch_size, patch_count, -1).mean(dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        local_patches: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_features = self.encode(pixel_values)
        if self.architecture == "global":
            features = global_features
        else:
            if local_patches is None:
                raise ValueError("global_local architecture requires local_patches")
            local_features = self.encode_local(local_patches)
            if self.local_fusion == "concat":
                features = torch.cat([global_features, local_features], dim=-1)
            else:
                features = 0.5 * (global_features + local_features)
        logits = self.head(features).squeeze(-1)
        return logits, features

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
