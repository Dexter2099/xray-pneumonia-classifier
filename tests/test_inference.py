import pytest
import torch

from src.inference import predict_image
from src.model import SimpleCNN


def test_predict_image_returns_structured_prediction(tmp_path):
    # create temporary checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "best_model.pth"

    model = SimpleCNN()
    torch.save(model.state_dict(), ckpt_path)

    dummy_img = torch.rand(1, 224, 224)  # single-channel tensor
    config = {"checkpoint_dir": str(ckpt_dir)}

    result = predict_image(dummy_img, config)

    assert set(result) == {"label", "confidence", "probabilities"}
    assert result["label"] in ["NORMAL", "PNEUMONIA"]
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

    probabilities = result["probabilities"]
    assert set(probabilities) == {"NORMAL", "PNEUMONIA"}
    assert all(isinstance(value, float) for value in probabilities.values())
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert probabilities[result["label"]] == pytest.approx(result["confidence"])
    assert sum(probabilities.values()) == pytest.approx(1.0)
