"""Unit tests for the WasteMobileNetV2 model architecture and inference."""

from pathlib import Path
import numpy as np
import pytest

from src.model import TORCH_AVAILABLE, WasteMobileNetV2, create_model

if TORCH_AVAILABLE:
    import torch


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not available.")
def test_model_instantiation():
    """Verifies that the model can be instantiated with custom head without external downloads."""
    model = create_model(num_classes=6, dropout_rate=0.3, pretrained=False)
    assert isinstance(model, WasteMobileNetV2)
    assert model.num_classes == 6
    assert hasattr(model, "backbone")
    assert hasattr(model.backbone, "classifier")
    # Verify final linear layer output features
    final_linear = model.backbone.classifier[-1]
    assert final_linear.out_features == 6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not available.")
def test_model_forward_shape():
    """Verifies that the model accepts (1, 3, 224, 224) and produces raw logits of shape (1, 6)."""
    model = create_model(num_classes=6, pretrained=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not available.")
def test_predict_probabilities_properties():
    """Verifies that predict_probabilities returns normalized probabilities summing to 1.0."""
    model = create_model(num_classes=6, pretrained=False)
    dummy_numpy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    probs = model.predict_probabilities(dummy_numpy)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (6,)
    assert np.isclose(float(probs.sum()), 1.0, atol=1e-5)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not available.")
def test_model_save_and_load_weights(tmp_path: Path):
    """Verifies weights serialization and deserialization."""
    weights_path = tmp_path / "test_weights.pth"
    model1 = create_model(num_classes=6, pretrained=False)
    model1.save_weights(weights_path)
    assert weights_path.exists()

    model2 = create_model(num_classes=6, pretrained=False)
    model2.load_weights(weights_path)

    # Check that parameters match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
