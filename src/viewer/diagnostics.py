"""Small detached diagnostics; local Qwen drafted this module before review/correction."""

from dataclasses import dataclass

import torch
from PIL import Image

from src.inputs import preprocess_image


@dataclass(frozen=True)
class InputInspection:
    source_size: tuple[int, int]
    grayscale_preview: Image.Image
    resized_preview: Image.Image
    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    std: float
    histogram: tuple[int, ...]


@dataclass(frozen=True)
class TimelineEvent:
    request_id: int
    stage: str
    outcome: str
    elapsed_seconds: float
    worker_thread_id: int
    detail: str


def inspect_input(image: Image.Image) -> InputInspection:
    """Preview the shared preprocessing contract, separately from the inference call.

    The argument is already decoded grayscale, so original colour is not available.
    No tensor/graph is retained or substituted into prediction. Reverse normalisation
    is for the resized DISPLAY preview only.
    """
    tensor = preprocess_image(image).detach().cpu()
    preview = image.copy()
    preview.thumbnail((224, 224))
    display = ((tensor[0] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    counts, _ = torch.histogram(tensor, bins=16, range=(-1.0, 1.0))
    # Describe this complete image population, rather than estimating a sample variance.
    return InputInspection(
        image.size,
        preview,
        Image.fromarray(display.numpy()).copy(),
        tuple(tensor.shape),
        str(tensor.dtype),
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.mean()),
        float(tensor.std(correction=0)),
        tuple(int(value) for value in counts.tolist()),
    )
