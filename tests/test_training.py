"""Exercise training configuration and an actual tiny CPU optimization loop."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from main import ensure_directories
from src.dataset import ChestXrayDataset
from src.errors import ConfigurationError
from src.settings import InferenceSettings, TrainingSettings
from src.train import build_optimizer, train_model


@pytest.fixture
def training_config(tmp_path):
    root = tmp_path / "data"
    for split in ("train", "val"):
        for label in ("NORMAL", "PNEUMONIA"):
            directory = root / split / label
            directory.mkdir(parents=True)
            Image.new("L", (24, 32), 128).save(directory / "sample.png")
    return {
        "data_dir": root,
        "checkpoint_path": tmp_path / "model.pth",
        "device": "cpu",
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 1,
        "seed": 123,
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "weight_decay": 0.02,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("optimizer", "adma"),
        ("image_size", 128),
        ("batch_size", True),
        ("batch_size", 0),
        ("num_workers", -1),
        ("epochs", 0),
        ("seed", -1),
        ("seed", 2**32),
        ("learning_rate", float("nan")),
        ("learning_rate", 0),
        ("weight_decay", -0.1),
        ("device", "bogus"),
    ],
)
def test_invalid_configuration_fails_before_dataset_or_model(
    training_config, monkeypatch, key, value
):
    def unexpected(*args, **kwargs):
        pytest.fail("Validation must precede data/model work")

    monkeypatch.setattr("src.train.get_dataloaders", unexpected)
    monkeypatch.setattr("src.train.SimpleCNN", unexpected)
    with pytest.raises(ConfigurationError):
        train_model({**training_config, key: value})


@pytest.mark.parametrize(
    "name,optimizer_type", [("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)]
)
def test_optimizer_honors_regularization_for_both_supported_choices(
    training_config, name, optimizer_type
):
    settings = replace(TrainingSettings.from_mapping(training_config), optimizer=name)
    optimizer = build_optimizer(nn.Linear(1, 2), settings)
    assert isinstance(optimizer, optimizer_type)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.02


class TinyTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, inputs):
        return self.linear(inputs.mean(dim=(1, 2, 3)).unsqueeze(1))


def test_seeded_training_is_repeatable_without_test_data(training_config, monkeypatch):
    # Run real optimization on a tiny substitute: never retrain/overwrite the demo checkpoint.
    monkeypatch.setattr("src.train.SimpleCNN", TinyTrainingModel)
    snapshots, random_values = [], []
    for _ in range(2):
        train_model(training_config)
        snapshots.append(torch.load(training_config["checkpoint_path"], weights_only=True))
        random_values.append((random.random(), np.random.random()))
    assert random_values[0] == random_values[1]
    assert all(torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0])
    assert not (training_config["data_dir"] / "test").exists()


def test_cli_dataset_and_inference_paths_share_repo_root(tmp_path, monkeypatch):
    repository, elsewhere = tmp_path / "repository", tmp_path / "elsewhere"
    repository.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr("src.settings.ROOT", repository)
    monkeypatch.chdir(elsewhere)
    for label in ("NORMAL", "PNEUMONIA"):
        (repository / "data/train" / label).mkdir(parents=True)
    config = {
        "checkpoint_dir": "outputs/checkpoints",
        "visualizations_dir": "outputs/visualizations",
    }
    ensure_directories(config)
    assert InferenceSettings.from_mapping(config).checkpoint_path.parent.is_dir()
    assert (repository / "outputs/visualizations").is_dir()
    assert not (elsewhere / "outputs").exists()
    assert len(ChestXrayDataset("data")) == 0
