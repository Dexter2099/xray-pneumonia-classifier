"""Validated inference settings; relative paths are anchored to the repository."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InferenceSettings:
    checkpoint_path: Path
    device: str = "auto"

    def __post_init__(self) -> None:
        path = Path(self.checkpoint_path).expanduser()
        object.__setattr__(self, "checkpoint_path", (ROOT / path).resolve())
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("Device must be auto, cpu, or cuda.")

    @classmethod
    def from_mapping(cls, config: Mapping) -> "InferenceSettings":
        if not isinstance(config, Mapping):
            raise ConfigurationError("Configuration must be a mapping.")
        if config.get("image_size", 224) != 224:
            raise ConfigurationError("This CNN checkpoint requires image_size: 224.")
        checkpoint = config.get("checkpoint_path")
        directory = config.get("checkpoint_dir")
        if checkpoint is None and directory is None:
            raise ConfigurationError("Set checkpoint_path or checkpoint_dir.")
        try:
            path = Path(checkpoint) if checkpoint is not None else Path(directory) / "best_model.pth"
        except TypeError as exc:
            raise ConfigurationError("Checkpoint location must be a filesystem path.") from exc
        return cls(path, config.get("device", "auto"))


def load_settings(path: Path = ROOT / "src/config.yaml") -> InferenceSettings:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Cannot read configuration. Check the YAML file.") from exc
    return InferenceSettings.from_mapping(config)
