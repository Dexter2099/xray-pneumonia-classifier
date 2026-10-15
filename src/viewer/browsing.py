"""Index image folders without decoding pixels or touching the model."""

import os
from dataclasses import dataclass
from pathlib import Path

from src.errors import ImageInputError


@dataclass(frozen=True)
class ImageFolder:
    folder: Path
    images: tuple[Path, ...]


def scan_folder(path: str) -> ImageFolder:
    """Run in the worker; include subfolders in a stable, case-insensitive order."""

    def unreadable(error: OSError) -> None:
        raise error

    try:
        folder = Path(path).resolve()
        if not folder.is_dir():
            raise ImageInputError("Image folder is missing. Choose another folder.")
        images = []
        for directory, _, filenames in os.walk(folder, onerror=unreadable):
            for name in filenames:
                image = Path(directory) / name
                if image.suffix.lower() in {".png", ".jpg", ".jpeg"} and image.is_file():
                    images.append(image)
        images.sort(key=lambda image: (image.relative_to(folder).as_posix().casefold(), str(image)))
    except OSError as exc:
        raise ImageInputError("Cannot read the image folder. Choose another folder.") from exc
    if not images:
        raise ImageInputError("No PNG or JPEG images in this folder or its subfolders.")
    return ImageFolder(folder, tuple(images))
