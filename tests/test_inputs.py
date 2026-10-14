import io

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.errors import ConfigurationError, ImageInputError
from src.inputs import load_image, preprocess_image
from src.settings import InferenceSettings, load_settings


def test_equivalent_image_and_tensor_inputs_match_without_mutation():
    image = Image.fromarray(np.arange(224 * 224, dtype=np.uint8).reshape(224, 224))
    tensor = transforms.ToTensor()(image)
    original = tensor.clone()
    assert torch.equal(preprocess_image(image), preprocess_image(tensor))
    assert torch.equal(tensor, original)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(224, 224),
        torch.zeros(3, 224, 224),
        torch.zeros(1, 32, 32),
        torch.zeros(1, 1, 224, 224),
        torch.zeros(1, 224, 224, dtype=torch.uint8),
        torch.zeros(1, 224, 224, dtype=torch.float64),
        torch.full((1, 224, 224), -0.1),
        torch.full((1, 224, 224), 1.1),
        torch.full((1, 224, 224), float("nan")),
        torch.full((1, 224, 224), float("inf")),
    ],
)
def test_invalid_tensors_fail_at_the_input_boundary(tensor):
    with pytest.raises(ImageInputError):
        preprocess_image(tensor)


def test_files_bytes_and_pil_share_the_same_preprocessing(tmp_path):
    image = Image.new("RGB", (333, 127), (60, 120, 180))
    path = tmp_path / "image.png"
    image.save(path)
    expected = preprocess_image(image)
    assert torch.equal(preprocess_image(path), expected)
    assert torch.equal(preprocess_image(path.read_bytes()), expected)
    assert expected.shape == (1, 224, 224)
    assert expected.dtype == torch.float32


def test_corrupt_input_does_not_prevent_next_valid_image(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ImageInputError, match="corrupt"):
        load_image(corrupt)
    assert load_image(Image.new("L", (12, 12))).size == (12, 12)


def test_missing_unsupported_and_high_bit_depth_images_are_rejected(tmp_path):
    with pytest.raises(ImageInputError, match="not found"):
        load_image(tmp_path / "missing.png")
    encoded = io.BytesIO()
    Image.new("L", (10, 10)).save(encoded, format="GIF")
    with pytest.raises(ImageInputError, match="format"):
        load_image(encoded.getvalue())
    with pytest.raises(ImageInputError, match="16-bit"):
        load_image(Image.new("I;16", (10, 10)))
    with pytest.raises(ImageInputError):
        load_image(None)


def test_image_limits_are_enforced_before_decoding(monkeypatch, tmp_path):
    from src import inputs

    monkeypatch.setattr(inputs, "MAX_IMAGE_BYTES", 3)
    path = tmp_path / "large.png"
    path.write_bytes(b"1234")
    with pytest.raises(ImageInputError, match="file limit"):
        load_image(path)
    with pytest.raises(ImageInputError, match="file limit"):
        load_image(b"1234")
    monkeypatch.setattr(inputs, "MAX_IMAGE_PIXELS", 3)
    with pytest.raises(ImageInputError, match="pixel limit"):
        load_image(Image.new("L", (2, 2)))


def test_relative_checkpoint_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = InferenceSettings.from_mapping({"checkpoint_dir": "outputs/checkpoints"})
    assert settings.checkpoint_path.is_absolute()
    assert settings.checkpoint_path.parent.name == "checkpoints"
    assert settings.checkpoint_path != tmp_path / "outputs/checkpoints/best_model.pth"


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        {},
        {"checkpoint_dir": 123},
        {"checkpoint_dir": ".", "image_size": 128},
        {"checkpoint_dir": ".", "device": "other"},
    ],
)
def test_bad_configuration_fails_clearly(config):
    with pytest.raises(ConfigurationError):
        InferenceSettings.from_mapping(config)


def test_invalid_yaml_fails_clearly(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("invalid: [")
    with pytest.raises(ConfigurationError):
        load_settings(path)
