from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.errors import ExplanationError
from src.gradcam import GradCAM, GradCamExplainer, normalize_cam


def test_normalize_cam_returns_zeros_when_cam_has_no_signal():
    cam = np.zeros((8, 8), dtype=np.float32)

    normalized = normalize_cam(cam)

    assert normalized.shape == cam.shape
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized == 0.0)


@pytest.mark.parametrize("cam", [np.array([]), np.array([[np.nan]]), np.array([[np.inf]])])
def test_unusable_heatmaps_are_explicit(cam):
    with pytest.raises(ExplanationError):
        normalize_cam(cam)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1)

    def forward(self, tensor):
        return self.conv(tensor).mean(dim=(2, 3))


def test_repeated_explanations_release_hooks_tensors_and_gradients():
    model = TinyModel()
    tensor = torch.rand(1, 1, 12, 12)
    before = model(tensor).detach().clone()
    explainer = GradCAM(model, model.conv)
    for _ in range(4):
        explainer.generate(tensor)
        assert len(model.conv._forward_hooks) == len(model.conv._backward_hooks) == 0
        assert explainer.activations is explainer.gradients is None
        assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(before, model(tensor).detach())


def test_failure_also_releases_hooks():
    model = TinyModel()
    with pytest.raises(ExplanationError, match="class index"):
        GradCAM(model, model.conv).generate(torch.rand(1, 1, 12, 12), class_idx=9)
    assert not model.conv._forward_hooks
    assert not model.conv._backward_hooks


def test_overlay_uses_rgb_and_reports_zero_signal():
    from threading import RLock
    from types import SimpleNamespace

    import cv2
    from PIL import Image

    service = SimpleNamespace(device="cpu", lock=RLock(), model=SimpleNamespace(conv3=object()))
    image = Image.new("L", (224, 224), 60)
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[:112] = 1
    with (
        patch.object(GradCAM, "generate", return_value=cam),
        patch.object(GradCAM, "__init__", return_value=None),
    ):
        overlay = np.asarray(GradCamExplainer(service).explain(image))
    expected_bgr = cv2.addWeighted(
        np.full((224, 224, 3), 60, dtype=np.uint8),
        0.6,
        cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
        0.4,
        0,
    )
    assert np.array_equal(overlay, cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB))
    with (
        patch.object(GradCAM, "generate", return_value=np.zeros_like(cam)),
        patch.object(GradCAM, "__init__", return_value=None),
    ):
        with pytest.raises(ExplanationError, match="No usable"):
            GradCamExplainer(service).explain(image)
