import pytest
import torch

from utils import get_device, make_grad_scaler, resolve_precision, runtime_summary


def test_cpu_runtime_uses_fp32_without_autocast() -> None:
    device = get_device("cpu")
    policy = resolve_precision({"precision": "auto"}, device)

    assert device == torch.device("cpu")
    assert policy.name == "fp32"
    assert not policy.autocast_enabled
    assert not policy.scaler_enabled
    assert runtime_summary(device, policy)["device"] == "cpu"


def test_legacy_amp_switch_can_force_fp32() -> None:
    policy = resolve_precision(
        {"precision": "fp16"}, torch.device("cuda"), legacy_amp=False
    )

    assert policy.name == "fp32"
    assert not policy.autocast_enabled


def test_invalid_precision_is_rejected_for_cuda() -> None:
    with pytest.raises(ValueError, match="runtime.precision"):
        resolve_precision({"precision": "int8"}, torch.device("cuda"))


def test_disabled_scaler_is_safe_without_cuda() -> None:
    policy = resolve_precision({"precision": "auto"}, torch.device("cpu"))
    scaler = make_grad_scaler(policy)

    assert not scaler.is_enabled()
