from dataclasses import fields

import torch
from PIL import Image

from src.viewer.diagnostics import inspect_input


def test_inspector_shows_known_pixels_without_mutating_or_retaining_tensor():
    image = Image.new("L", (224, 224), 0)
    image.paste(255, (112, 0, 224, 224))
    original = image.tobytes()
    inspection = inspect_input(image)
    assert inspection.source_size == (224, 224)
    assert inspection.shape == (1, 224, 224)
    assert inspection.dtype == "torch.float32"
    assert (inspection.minimum, inspection.maximum, inspection.mean, inspection.std) == (
        -1,
        1,
        0,
        1,
    )
    assert inspection.histogram == (25088, *([0] * 14), 25088)
    assert inspection.resized_preview.tobytes() == original
    assert image.tobytes() == original
    assert all(
        not isinstance(getattr(inspection, field.name), torch.Tensor)
        for field in fields(inspection)
    )


def test_inspector_bounds_previews_and_uses_population_std_for_constant_image():
    inspection = inspect_input(Image.new("L", (1200, 600), 255))
    assert inspection.source_size == (1200, 600)
    assert inspection.grayscale_preview.size == (224, 112)
    assert inspection.resized_preview.size == (224, 224)
    assert inspection.minimum == inspection.maximum == inspection.mean == 1
    assert inspection.std == 0
    assert sum(inspection.histogram) == 224 * 224
