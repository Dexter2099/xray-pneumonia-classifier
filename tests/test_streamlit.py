"""The optional web consumer must respect device selection and model reuse."""

import importlib
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit", reason="Install requirements.txt for the optional web UI")

cached_prediction_service = importlib.import_module("app").cached_prediction_service


def test_cache_reuses_same_model_but_separates_device_and_checkpoint_identity(tmp_path):
    cached_prediction_service.clear()
    try:
        with patch(
            "src.inference.PredictionService", side_effect=lambda settings: object()
        ) as build:
            path = str(tmp_path / "model.pth")
            first = cached_prediction_service(path, 100, 1, "cpu")
            assert cached_prediction_service(path, 100, 1, "cpu") is first
            assert build.call_args.args[0].device == "cpu"
            assert cached_prediction_service(path, 100, 1, "cuda") is not first
            assert build.call_args.args[0].device == "cuda"
            assert cached_prediction_service(path, 100, 2, "cpu") is not first
            assert build.call_count == 3
    finally:
        cached_prediction_service.clear()
