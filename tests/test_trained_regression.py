"""Golden results captured from fe9c3d2 BEFORE changing inference."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.inference import predict_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.trained
def test_trained_checkpoint_preserves_reference_predictions():
    baseline = json.loads((ROOT / "tests/fixtures/trained_baseline.json").read_text())
    checkpoint = ROOT / "outputs/checkpoints/best_model.pth"
    assert checkpoint.is_file(), "The explicitly requested trained checkpoint is missing"
    with checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == baseline["checkpoint_sha256"]
    for case in baseline["cases"]:
        path = ROOT / case["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
        with Image.open(path) as image:
            actual = predict_image(image, {"checkpoint_dir": str(checkpoint.parent)})
        assert actual["label"] == case["prediction"]["label"]
        for label, expected in case["prediction"]["probabilities"].items():
            assert actual["probabilities"][label] == pytest.approx(
                expected,
                abs=baseline["absolute_tolerance"],
                rel=baseline["relative_tolerance"],
            )
