import torch
import torch.nn.functional as F

from losses import paired_classification_consistency_loss


def test_identical_features_have_zero_consistency() -> None:
    logits = torch.tensor([0.0, 1.0])
    labels = torch.tensor([0.0, 1.0])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    total, parts = paired_classification_consistency_loss(
        logits,
        logits,
        features,
        features,
        labels,
        clean_bce_weight=1.0,
        augmented_bce_weight=1.0,
        lambda_consistency=0.2,
    )
    expected = 2.0 * F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.allclose(parts["consistency"], torch.tensor(0.0))
    assert torch.allclose(total, expected)
