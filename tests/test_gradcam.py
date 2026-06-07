import numpy as np

from src.gradcam import normalize_cam


def test_normalize_cam_returns_zeros_when_cam_has_no_signal():
    cam = np.zeros((8, 8), dtype=np.float32)

    normalized = normalize_cam(cam)

    assert normalized.shape == cam.shape
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized == 0.0)
