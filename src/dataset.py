"""Dataset indexing and loading use the same input contract as prediction."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.contracts import IMAGE_SIZE, LABELS
from src.errors import ConfigurationError
from src.inputs import preprocess_image
from src.settings import resolve_project_path, validated_integer


class ChestXrayDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str = "train", image_size: int = IMAGE_SIZE):
        # Three pooling layers feed a fixed-size dense layer; resizing to a different
        # dimension cannot be supported by this checkpoint merely by changing YAML.
        if image_size != IMAGE_SIZE:
            raise ConfigurationError("This CNN requires image_size: 224.")
        if split not in {"train", "val", "test"}:
            raise ConfigurationError("Dataset split must be train, val, or test.")
        self.image_paths: list[str] = []
        self.labels: list[int] = []
        split_dir = resolve_project_path(root_dir, "data_dir") / split
        for label_index, label in enumerate(LABELS):
            label_dir = split_dir / label
            if not label_dir.is_dir():
                raise ConfigurationError(f"Dataset directory is missing: {label_dir}")
            # A seeded shuffle is only reproducible when the underlying index order is stable.
            for path in sorted(label_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.image_paths.append(str(path))
                    self.labels.append(label_index)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep decoding, grayscale conversion and normalisation identical to inference.
        # preprocess_image also closes its file before returning the detached tensor.
        image = preprocess_image(self.image_paths[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


def get_dataloaders(
    config: Mapping,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, ...]:
    """Return loaders in requested order; the default retains the original three-loader API."""
    batch_size = validated_integer(config.get("batch_size", 32), "batch_size")
    num_workers = validated_integer(config.get("num_workers", 4), "num_workers", 0)
    if not splits:
        raise ConfigurationError("Request at least one dataset split.")
    loaders = []
    data_dir = resolve_project_path(config.get("data_dir"), "data_dir")
    for split in splits:
        dataset = ChestXrayDataset(data_dir, split, config.get("image_size", IMAGE_SIZE))
        if not len(dataset):
            raise ConfigurationError(f"The {split} dataset is empty.")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=num_workers,
                generator=generator,
            )
        )
    return tuple(loaders)
