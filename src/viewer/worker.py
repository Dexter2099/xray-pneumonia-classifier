"""A single worker owns model lifetime; no GUI objects cross this boundary."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import get_ident
from time import perf_counter

from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

from src.errors import XrayError
from src.gradcam import Explainer, GradCamExplainer
from src.inference import PredictionService, Predictor
from src.inputs import load_image
from src.settings import InferenceSettings
from src.viewer.browsing import scan_folder
from src.viewer.diagnostics import TimelineEvent, inspect_input
from src.viewer.evidence import VerificationEvidence, load_verification
from src.viewer.messages import AnalysisResult as AnalysisResult
from src.viewer.messages import ModelReady

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


class AnalysisWorker(QObject):
    ready = Signal(object)
    image_loaded = Signal(int, object)
    folder_loaded = Signal(int, object)
    completed = Signal(object)
    failed = Signal(int, str)
    stage = Signal(int, str)
    inspection_ready = Signal(int, object)
    timeline_event = Signal(object)
    verification_loaded = Signal(int, object)

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

    @contextmanager
    def measured(self, request_id: int, stage: str, detail: str) -> Iterator[None]:
        """Pair Running with Complete/Failed while preserving the operation's exception."""
        started = perf_counter()
        thread_id = get_ident()
        self.timeline_event.emit(TimelineEvent(request_id, stage, "Running", 0, thread_id, detail))
        try:
            yield
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, XrayError)
                else "Operation failed; see application error."
            )
            self.timeline_event.emit(
                TimelineEvent(
                    request_id, stage, "Failed", perf_counter() - started, thread_id, message
                )
            )
            raise
        else:
            self.timeline_event.emit(
                TimelineEvent(
                    request_id, stage, "Complete", perf_counter() - started, thread_id, detail
                )
            )

    @Slot(int)
    def read_verification(self, request_id: int = 0) -> None:
        try:
            with self.measured(
                request_id,
                "Verification report",
                "load_verification: read saved JSON; no tests executed",
            ):
                evidence = load_verification(self.backend.checkpoint_sha256 if self.backend else "")
        except Exception:
            LOGGER.exception("Unexpected verification report failure")
            evidence = VerificationEvidence(
                "Unavailable",
                "No usable evidence",
                "Cannot read verification evidence. Prediction remains available.",
            )
        self.verification_loaded.emit(request_id, evidence)

    @Slot()
    def initialize(self) -> None:
        self.backend = None
        try:
            with self.measured(
                0,
                "Model load",
                "create_backend → PredictionService: load trusted weights; evaluation mode",
            ):
                self.backend = self.factory(self.settings)
            self.model_loads += 1
            self.read_verification()
            # A queued signal transfers metadata, not permission for the GUI to touch
            # the shared model or read mutable worker counters across threads.
            self.ready.emit(
                ModelReady(
                    self.backend.checkpoint_sha256,
                    self.backend.load_seconds,
                    self.backend.device,
                    self.model_loads,
                    self.predictions,
                )
            )
        except XrayError as exc:
            self.failed.emit(0, str(exc))
        except Exception:
            LOGGER.exception("Unexpected model initialization failure")
            self.failed.emit(
                0, "Unexpected model error. Check the configuration and reload the model."
            )

    @Slot(int, str)
    def open_folder(self, request_id: int, path: str) -> None:
        try:
            with self.measured(
                request_id, "Folder scan", "scan_folder: index PNG/JPEG paths; no prediction"
            ):
                folder = scan_folder(path)
            self.folder_loaded.emit(request_id, folder)
        except XrayError as exc:
            self.failed.emit(request_id, str(exc))
        except Exception:
            LOGGER.exception("Unexpected image folder failure")
            self.failed.emit(request_id, "Cannot open the image folder. Choose another folder.")

    @Slot(int, str)
    def open_image(self, request_id: int, path: str) -> None:
        try:
            with self.measured(
                request_id,
                "Decode / validate",
                "load_image: bounded PNG/JPEG decoding to 8-bit grayscale",
            ):
                image = load_image(path)
            try:
                with self.measured(
                    request_id,
                    "Input inspection",
                    "inspect_input → preprocess_image: preview resize and normalisation; separate from prediction",
                ):
                    inspection = inspect_input(image)
                self.inspection_ready.emit(request_id, inspection)
            except Exception:
                # Diagnostics explain a valid image; their failure must not reject it.
                LOGGER.exception("Input inspection failed; decoded image remains available")
                self.inspection_ready.emit(request_id, None)
            self.image_loaded.emit(request_id, image)
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
            with self.measured(
                request_id,
                "Prediction",
                "PredictionService.predict: shared preprocessing + forward + softmax; evaluation mode, gradients disabled",
            ):
                prediction = self.backend.predictor.predict(image)
            prediction_seconds = perf_counter() - started
            self.predictions += 1
            self.stage.emit(request_id, "Generating Grad-CAM explanation…")
            started = perf_counter()
            overlay, warning = None, ""
            # Explanation is optional enrichment. Its failure must not erase an
            # already completed prediction or increment the prediction count again.
            try:
                with self.measured(
                    request_id,
                    "Grad-CAM",
                    "GradCamExplainer.explain: separate gradient-enabled forward/backward; scoped hooks cleaned afterwards",
                ):
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
