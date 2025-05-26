import pytest
torch = pytest.importorskip("torch")
from src.inference import predict_image
from src.model import SimpleCNN


def test_predict_image_returns_valid_label(tmp_path):
    # create temporary checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "best_model.pth"

    model = SimpleCNN()
    torch.save(model.state_dict(), ckpt_path)

    dummy_img = torch.rand(1, 224, 224)  # single-channel tensor
    config = {"checkpoint_dir": str(ckpt_dir)}

    label = predict_image(dummy_img, config)
    assert label in ["NORMAL", "PNEUMONIA"]
