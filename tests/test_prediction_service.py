import hashlib
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from src.errors import ModelLoadError, PredictionError
from src.inference import PredictionService, predict_image
from src.model import SimpleCNN
from src.settings import InferenceSettings


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(42)
    path = tmp_path_factory.mktemp("model") / "best_model.pth"
    torch.save(SimpleCNN().state_dict(), path)
    return path


def test_model_loads_once_and_repeated_predictions_agree(checkpoint):
    with patch("src.inference.torch.load", wraps=torch.load) as load:
        service = PredictionService(InferenceSettings(checkpoint, "cpu"))
        image = Image.new("L", (224, 224), 80)
        first = service.predict(image)
        second = service.predict(image)
        assert first == second
        assert load.call_count == service.model_load_count == 1
        assert load.call_args.kwargs["weights_only"] is True
        assert service.prediction_count == 2
        assert service.load_seconds > 0


def test_single_and_batch_predictions_agree(checkpoint):
    service = PredictionService(InferenceSettings(checkpoint, "cpu"))
    images = [Image.new("L", (224, 224), value) for value in (0, 80, 255)]
    singles = [service.predict(image) for image in images]
    batch = service.predict_batch(images)
    for single, batched in zip(singles, batch, strict=True):
        assert single.label == batched.label
        assert single.confidence == pytest.approx(batched.confidence, abs=1e-6)


def test_adapter_can_reuse_an_explicit_service(checkpoint):
    service = PredictionService(InferenceSettings(checkpoint, "cpu"))
    result = predict_image(Image.new("L", (224, 224)), {}, service=service)
    assert result == service.predict(Image.new("L", (224, 224))).to_dict()
    assert service.model_load_count == 1


def test_bad_checkpoints_are_actionable(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        PredictionService(InferenceSettings(tmp_path / "missing.pth"))
    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ModelLoadError, match="trusted"):
        PredictionService(InferenceSettings(path))
    torch.save({"wrong": torch.zeros(1)}, path)
    with pytest.raises(ModelLoadError, match="trusted"):
        PredictionService(InferenceSettings(path))


def test_invalid_model_output_and_empty_batches_fail(checkpoint):
    service = PredictionService(InferenceSettings(checkpoint, "cpu"))
    with pytest.raises(PredictionError, match="at least one"):
        service.predict_batch([])
    with patch.object(service.model, "forward", return_value=torch.full((1, 2), float("nan"))):
        with pytest.raises(PredictionError, match="invalid scores"):
            service.predict(Image.new("L", (224, 224)))
    assert service.prediction_count == 0


def test_fingerprint_identifies_loaded_snapshot_when_checkpoint_is_replaced(checkpoint, tmp_path):
    path = tmp_path / "replaceable.pth"
    original = checkpoint.read_bytes()
    path.write_bytes(original)
    real_load = torch.load

    def replace_after_load(*args, **kwargs):
        weights = real_load(*args, **kwargs)
        path.write_bytes(b"a different checkpoint now occupies the same path")
        return weights

    with patch("src.inference.torch.load", side_effect=replace_after_load):
        service = PredictionService(InferenceSettings(path, "cpu"))
    assert service.checkpoint_sha256 == hashlib.sha256(original).hexdigest()
    assert service.checkpoint_sha256 != hashlib.sha256(path.read_bytes()).hexdigest()
    expected = PredictionService(InferenceSettings(checkpoint, "cpu"))
    image = Image.new("L", (224, 224), 80)
    assert service.predict(image) == expected.predict(image)


@pytest.mark.parametrize("operation", ["stack", "transfer", "forward"])
def test_runtime_failures_are_actionable_and_service_can_recover(checkpoint, operation):
    service = PredictionService(InferenceSettings(checkpoint, "cpu"))
    target = {
        "stack": "src.inference.torch.stack",
        "transfer": "torch.Tensor.to",
        "forward": "src.model.SimpleCNN.forward",
    }[operation]
    with patch(target, side_effect=RuntimeError("simulated allocation/device failure")):
        with pytest.raises(PredictionError, match="smaller batch"):
            service.predict(Image.new("L", (224, 224)))
    assert service.prediction_count == 0
    assert service.predict(Image.new("L", (224, 224))).label in {"NORMAL", "PNEUMONIA"}
    assert service.prediction_count == 1
