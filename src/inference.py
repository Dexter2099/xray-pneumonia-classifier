"""Reusable, UI-independent prediction service and legacy one-shot adapter."""

import hashlib
import io
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Protocol

import torch

from src.errors import ConfigurationError, ModelLoadError, PredictionError
from src.inputs import ModelInput, preprocess_image
from src.model import SimpleCNN
from src.settings import InferenceSettings

LABELS = ("NORMAL", "PNEUMONIA")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    normal_probability: float
    pneumonia_probability: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": {
                "NORMAL": self.normal_probability,
                "PNEUMONIA": self.pneumonia_probability,
            },
        }


class Predictor(Protocol):
    """Small prediction dependency; test doubles implement this contract too."""

    def predict(self, source: ModelInput) -> Prediction: ...


class PredictionService:
    """Load once per service. Serialize use when sharing a model with Grad-CAM."""

    def __init__(self, settings: InferenceSettings) -> None:
        started = perf_counter()
        self.settings = settings
        self.lock = RLock()
        self.prediction_count = 0
        self.model_load_count = 0
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is unavailable. Select cpu or auto.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device if settings.device == "auto" else settings.device)
        if not settings.checkpoint_path.is_file():
            raise ModelLoadError(
                "Model checkpoint was not found. Configure best_model.pth and reload the model."
            )
        try:
            # Hash and deserialize one snapshot: a concurrent file replacement must not
            # make the displayed fingerprint identify weights this service never loaded.
            with io.BytesIO(settings.checkpoint_path.read_bytes()) as snapshot:
                self.checkpoint_sha256 = hashlib.sha256(snapshot.getbuffer()).hexdigest()
                weights = torch.load(snapshot, map_location="cpu", weights_only=True)
            self.model = SimpleCNN()
            self.model.load_state_dict(weights, strict=True)
            self.model.to(self.device).eval()
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            EOFError,
            pickle.UnpicklingError,
        ) as exc:
            raise ModelLoadError(
                "Cannot load this checkpoint. Use the trusted SimpleCNN state dictionary."
            ) from exc
        self.model_load_count = 1
        self.load_seconds = perf_counter() - started

    def predict(self, source: ModelInput) -> Prediction:
        return self.predict_batch([source])[0]

    def predict_batch(self, sources: Sequence[ModelInput]) -> list[Prediction]:
        if not sources:
            raise PredictionError("Provide at least one image for prediction.")
        with self.lock, torch.inference_mode():
            try:
                # Allocation and device transfer can fail before the forward pass.
                # Keep them inside the same recoverable prediction-error boundary.
                tensors = torch.stack([preprocess_image(source) for source in sources]).to(
                    self.device
                )
                outputs = self.model(tensors)
                if (
                    outputs.shape != (len(sources), len(LABELS))
                    or not torch.isfinite(outputs).all()
                ):
                    raise PredictionError(
                        "Model produced invalid scores. Reload the trusted checkpoint."
                    )
                probabilities = torch.softmax(outputs, dim=1).cpu()
            except RuntimeError as exc:
                raise PredictionError(
                    "Prediction failed. Try a smaller batch or use the CPU device."
                ) from exc
            results = []
            for row in probabilities:
                index = int(row.argmax().item())
                results.append(
                    Prediction(LABELS[index], float(row[index]), float(row[0]), float(row[1]))
                )
            self.prediction_count += len(results)
            return results


def predict_image(
    img: ModelInput, config: Mapping, *, service: PredictionService | None = None
) -> dict:
    """Compatibility adapter. Repeated callers should pass a long-lived service."""
    engine = service or PredictionService(InferenceSettings.from_mapping(config))
    return engine.predict(img).to_dict()
