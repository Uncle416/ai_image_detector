import pytest

from metrics import binary_metrics


def test_binary_metrics_include_class_balanced_diagnostics() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.7], threshold=0.5)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["real_recall"] == pytest.approx(0.5)
    assert metrics["fake_recall"] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
