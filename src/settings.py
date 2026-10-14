"""Validated application settings; every entry point uses repository-relative paths."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.contracts import IMAGE_SIZE
from src.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: object, name: str = "Path") -> Path:
    """Give every entry point the same meaning for relative configuration paths."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError(f"{name} must be a non-empty filesystem path.")
    return (ROOT / Path(value).expanduser()).resolve()


def validated_integer(value: int, name: str, minimum: int = 1) -> int:
    # bool is an int subclass, but accepting True as a batch size hides YAML mistakes.
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}.")
    return value


def _validated_number(value: float, name: str, *, allow_zero: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number.")
    if value < 0 or (value == 0 and not allow_zero):
        raise ConfigurationError(f"{name} must be {'non-negative' if allow_zero else 'positive'}.")
    return float(value)


@dataclass(frozen=True)
class InferenceSettings:
    checkpoint_path: Path
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", resolve_project_path(self.checkpoint_path))
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("Device must be auto, cpu, or cuda.")

    @classmethod
    def from_mapping(cls, config: Mapping) -> "InferenceSettings":
        if not isinstance(config, Mapping):
            raise ConfigurationError("Configuration must be a mapping.")
        if config.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
            raise ConfigurationError("This CNN checkpoint requires image_size: 224.")
        checkpoint = config.get("checkpoint_path")
        directory = config.get("checkpoint_dir")
        if checkpoint is None and directory is None:
            raise ConfigurationError("Set checkpoint_path or checkpoint_dir.")
        if isinstance(checkpoint, (str, Path)) and str(checkpoint).strip():
            path = Path(checkpoint)
        elif checkpoint is None and isinstance(directory, (str, Path)) and str(directory).strip():
            path = Path(directory) / "best_model.pth"
        else:
            raise ConfigurationError("Checkpoint location must be a filesystem path.")
        return cls(path, config.get("device", "auto"))


@dataclass(frozen=True)
class TrainingSettings:
    """Reject ambiguous training choices before opening data or allocating a model."""

    inference: InferenceSettings
    data_dir: Path
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    seed: int

    @classmethod
    def from_mapping(cls, config: Mapping) -> "TrainingSettings":
        inference = InferenceSettings.from_mapping(config)
        optimizer = config.get("optimizer", "adam")
        if optimizer not in ("adam", "sgd"):
            raise ConfigurationError("Optimizer must be adam or sgd; no implicit fallback is used.")
        seed = validated_integer(config.get("seed", 42), "seed", 0)
        if seed >= 2**32:
            raise ConfigurationError("seed must be less than 2**32 for NumPy compatibility.")
        return cls(
            inference,
            resolve_project_path(config.get("data_dir"), "data_dir"),
            validated_integer(config.get("batch_size", 32), "batch_size"),
            validated_integer(config.get("num_workers", 4), "num_workers", 0),
            validated_integer(config.get("epochs", 10), "epochs"),
            _validated_number(config.get("learning_rate", 0.001), "learning_rate"),
            _validated_number(config.get("weight_decay", 0.0001), "weight_decay", allow_zero=True),
            optimizer,
            seed,
        )


def load_config(path: Path = ROOT / "src/config.yaml") -> dict:
    try:
        with resolve_project_path(path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Cannot read configuration. Check the YAML file.") from exc
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a mapping.")
    return config


def load_settings(path: Path = ROOT / "src/config.yaml") -> InferenceSettings:
    return InferenceSettings.from_mapping(load_config(path))
