from types import SimpleNamespace

import pytest
import torch
from torch import nn

import model as model_module


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, patch_size=2)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = True) -> SimpleNamespace:
        pooled = pixel_values.mean(dim=(1, 2, 3), keepdim=False)[:, None, None]
        hidden = pooled.repeat(1, 5, self.config.hidden_size) * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


def v3_config() -> dict[str, object]:
    return {
        "backbone": "fake",
        "image_size": 4,
        "pooling": "cls_mean",
        "architecture": "global_local",
        "local_fusion": "concat",
        "head_hidden_dim": 8,
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "max_parameters": 100_000,
    }


def test_global_local_model_fuses_four_shared_encoder_patches(monkeypatch) -> None:
    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", lambda _: FakeBackbone())
    model = model_module.DinoBinaryClassifier(v3_config())

    logits, features = model(
        torch.ones(2, 3, 4, 4),
        local_patches=torch.ones(2, 4, 3, 2, 2),
    )

    assert logits.shape == (2,)
    assert features.shape == (2, 16)
    assert model.head[1].in_features == 16


def test_global_local_model_requires_local_patches(monkeypatch) -> None:
    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", lambda _: FakeBackbone())
    model = model_module.DinoBinaryClassifier(v3_config())

    with pytest.raises(ValueError, match="requires local_patches"):
        model(torch.ones(1, 3, 4, 4))
