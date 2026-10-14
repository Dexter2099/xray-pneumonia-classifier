"""One input contract shared by prediction and explanation.

The initial draft was produced locally with Qwen and reviewed before integration.
"""

import io
import warnings
from pathlib import Path
from typing import TypeAlias

import torch
from PIL import Image
from torchvision import transforms

from src.errors import ImageInputError

IMAGE_SIZE = 224
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ImageSource: TypeAlias = str | Path | bytes | Image.Image
ModelInput: TypeAlias = ImageSource | torch.Tensor
_ALLOWED_MODES = {"L", "RGB", "RGBA", "1", "P"}
_NORMALIZE = transforms.Normalize([0.5], [0.5])
_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)


def _decode(image: Image.Image) -> Image.Image:
    if image.mode not in _ALLOWED_MODES:
        raise ImageInputError(
            "Use an 8-bit grayscale or RGB image; 16-bit and float images are unsupported."
        )
    if image.width <= 0 or image.height <= 0:
        raise ImageInputError("Image dimensions must be positive.")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageInputError("Image exceeds the 20-million-pixel limit. Resize it before opening.")
    image.load()  # Decode here, while the file is open, so truncation cannot escape validation.
    return image.convert("L").copy()


def load_image(source: ImageSource) -> Image.Image:
    """Decode PNG/JPEG into independent grayscale pixels without closing caller images."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if isinstance(source, Image.Image):
                return _decode(source)
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.is_file():
                    raise ImageInputError("Image file was not found. Choose another PNG or JPEG.")
                if path.stat().st_size > MAX_IMAGE_BYTES:
                    raise ImageInputError("Image exceeds the 20 MiB file limit.")
                opened = Image.open(path)
            elif isinstance(source, bytes):
                if len(source) > MAX_IMAGE_BYTES:
                    raise ImageInputError("Image exceeds the 20 MiB file limit.")
                opened = Image.open(io.BytesIO(source))
            else:
                raise ImageInputError("Choose a PNG/JPEG path, encoded bytes, or a PIL image.")
            with opened as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ImageInputError("Unsupported image format. Choose PNG or JPEG.")
                return _decode(image)
    except ImageInputError:
        raise
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageInputError(
            "Cannot decode this image. It may be corrupt or truncated; choose another file."
        ) from exc


def preprocess_image(source: ModelInput) -> torch.Tensor:
    """Return float32 (1,224,224) in [-1,1]; raw tensors must already be resized and in [0,1]."""
    if isinstance(source, torch.Tensor):
        if source.layout != torch.strided or source.device.type not in {"cpu", "cuda"}:
            raise ImageInputError("Tensor must be a dense CPU or CUDA tensor.")
        if source.dtype != torch.float32 or tuple(source.shape) != (
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ImageInputError("Tensor must be float32 with shape (1, 224, 224).")
        tensor = source.detach().cpu().clone()
        if not torch.isfinite(tensor).all():
            raise ImageInputError("Tensor contains NaN or infinite values.")
        if (tensor < 0).any() or (tensor > 1).any():
            raise ImageInputError("Tensor values must be in [0, 1] before normalisation.")
        return _NORMALIZE(tensor).contiguous()
    return _IMAGE_TRANSFORM(load_image(source)).contiguous()
