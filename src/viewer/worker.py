"""A single worker owns model lifetime; no GUI objects cross this boundary."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

from src.errors import XrayError
from src.gradcam import Explainer, GradCamExplainer
from src.inference import Prediction, PredictionService, Predictor
from src.inputs import load_image
from src.settings import InferenceSettings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisBackend:
    predictor: Predictor
    explainer: Explainer
    checkpoint_sha256: str
    load_seconds: float
    device: str


def create_backend(settings: InferenceSettings) -> AnalysisBackend:
    service = PredictionService(settings)
    return AnalysisBackend(
        service,
        GradCamExplainer(service),
        service.checkpoint_sha256,
        service.load_seconds,
        str(service.device),
    )


@dataclass(frozen=True)
class AnalysisResult:
    request_id: int
    prediction: Prediction
    overlay: Image.Image | None
    warning: str
    prediction_seconds: float
    explanation_seconds: float
    model_loads: int
    predictions: int


class AnalysisWorker(QObject):
    ready = Signal(object)
    image_loaded = Signal(int, object)
    completed = Signal(object)
    failed = Signal(int, str)
    stage = Signal(int, str)

    def __init__(
        self,
        settings: InferenceSettings,
        factory: Callable[[InferenceSettings], AnalysisBackend] = create_backend,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.factory = factory
        self.backend: AnalysisBackend | None = None
        self.model_loads = 0
        self.predictions = 0

    @Slot()
    def initialize(self) -> None:
        self.backend = None
        try:
            self.backend = self.factory(self.settings)
            self.model_loads += 1
            self.ready.emit(self.backend)
        except XrayError as exc:
            self.failed.emit(0, str(exc))
        except Exception:
            LOGGER.exception("Unexpected model initialization failure")
            self.failed.emit(
                0, "Unexpected model error. Check the configuration and reload the model."
            )

    @Slot(int, str)
    def open_image(self, request_id: int, path: str) -> None:
        try:
            self.image_loaded.emit(request_id, load_image(path))
        except XrayError as exc:
            self.failed.emit(request_id, str(exc))
        except Exception:
            LOGGER.exception("Unexpected image decoding failure")
            self.failed.emit(request_id, "Unexpected image error. Choose another PNG or JPEG.")

    @Slot(int, object)
    def analyze(self, request_id: int, image: Image.Image) -> None:
        if self.backend is None:
            self.failed.emit(
                request_id, "Model is unavailable. Reload the model before predicting."
            )
            return
        try:
            started = perf_counter()
            prediction = self.backend.predictor.predict(image)
            prediction_seconds = perf_counter() - started
            self.predictions += 1
            self.stage.emit(request_id, "Generating Grad-CAM explanation…")
            started = perf_counter()
            overlay, warning = None, ""
            try:
                overlay = self.backend.explainer.explain(image)
            except XrayError as exc:
                warning = str(exc)
            except Exception:
                LOGGER.exception("Unexpected explanation failure")
                warning = "Explanation failed. The prediction remains available; try another image."
            self.completed.emit(
                AnalysisResult(
                    request_id,
                    prediction,
                    overlay,
                    warning,
                    prediction_seconds,
                    perf_counter() - started,
                    self.model_loads,
                    self.predictions,
                )
            )
        except XrayError as exc:
            self.failed.emit(request_id, str(exc))
        except Exception:
            LOGGER.exception("Unexpected prediction failure")
            self.failed.emit(
                request_id, "Unexpected prediction error. Reload the model and try again."
            )
