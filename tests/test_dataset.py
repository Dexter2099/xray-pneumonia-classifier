"""The training loader must obey the checkpoint's input and split contracts."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.dataset import ChestXrayDataset, get_dataloaders
from src.errors import ConfigurationError
from src.inputs import preprocess_image


@pytest.fixture
def dataset_root(tmp_path):
    for split in ("train", "val"):
        for label in ("NORMAL", "PNEUMONIA"):
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            for filename, value in (("b.png", 180), ("a.png", 80)):
                Image.new("RGB", (32, 40), (value, value, value)).save(directory / filename)
    return tmp_path


def test_dataset_has_stable_class_order_and_matches_inference(dataset_root):
    dataset = ChestXrayDataset(dataset_root)
    assert dataset.labels == [0, 0, 1, 1]
    assert [Path(path).name for path in dataset.image_paths] == ["a.png", "b.png", "a.png", "b.png"]
    image, label = dataset[0]
    assert torch.equal(image, preprocess_image(dataset.image_paths[0]))
    assert image.shape == (1, 224, 224) and image.dtype == torch.float32
    assert label.item() == 0 and label.dtype == torch.long


def test_training_loaders_do_not_require_unused_test_split(dataset_root):
    config = {"data_dir": dataset_root, "batch_size": 2, "num_workers": 0}
    train, validation = get_dataloaders(config, splits=("train", "val"))
    assert len(train.dataset) == len(validation.dataset) == 4
    assert not (dataset_root / "test").exists()
    with pytest.raises(ConfigurationError, match="missing"):
        get_dataloaders(config)  # The compatibility API still requests all three splits.


def test_incompatible_size_fails_before_touching_dataset(tmp_path):
    with pytest.raises(ConfigurationError, match="224"):
        ChestXrayDataset(tmp_path / "missing", image_size=128)


def test_corrupt_training_image_uses_shared_actionable_error(dataset_root):
    from src.errors import ImageInputError

    dataset = ChestXrayDataset(dataset_root)
    with open(dataset.image_paths[0], "wb") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ImageInputError, match="corrupt"):
        dataset[0]
