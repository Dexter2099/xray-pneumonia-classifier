"""Application state and queued communication; model work stays off the GUI thread."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent

from src.settings import InferenceSettings
from src.viewer.window import ViewerWindow
from src.viewer.worker import AnalysisBackend, AnalysisResult, AnalysisWorker, create_backend


class ViewerController(QObject):
    initialize_requested = Signal()
    image_requested = Signal(int, str)
    analysis_requested = Signal(int, object)

    def __init__(
        self,
        window: ViewerWindow,
        settings: InferenceSettings,
        factory: Callable[[InferenceSettings], AnalysisBackend] = create_backend,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.image: Image.Image | None = None
        self.request_id = 0
        self.busy = True
        self.model_ready = False
        self.closing = False
        self.worker_thread = QThread(self)
        self.worker_thread.setObjectName("xray-analysis-worker")
        self.worker = AnalysisWorker(settings, factory)
        self.worker.moveToThread(self.worker_thread)
        self.initialize_requested.connect(self.worker.initialize)
        self.image_requested.connect(self.worker.open_image)
        self.analysis_requested.connect(self.worker.analyze)
        self.worker.ready.connect(self._ready)
        self.worker.image_loaded.connect(self._image_loaded)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.stage.connect(self._stage)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.window.close)
        window.open_requested.connect(self.open_image)
        window.predict_requested.connect(self.predict)
        window.retry_requested.connect(self.reload_model)
        window.closing.connect(self.close)
        self._set_busy(True, "Loading model...")
        self.worker_thread.start()
        self.initialize_requested.emit()

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.window.open_button.setEnabled(not busy and not self.closing)
        self.window.retry_button.setEnabled(not busy and not self.closing)
        self.window.predict_button.setEnabled(
            not busy and self.model_ready and self.image is not None and not self.closing
        )
        self.window.status_label.setText(status)
        if self.closing and not busy:
            self.worker_thread.quit()

    @Slot(object)
    def _ready(self, backend: AnalysisBackend) -> None:
        self.model_ready = True
        self.window.session_label.setText(
            f"Model loads: {self.worker.model_loads}  |  Predictions: {self.worker.predictions}  |  Device: {backend.device}"
        )
        self.window.timing_label.setText(f"Model startup: {backend.load_seconds * 1000:.1f} ms")
        self.window.set_message(f"Checkpoint SHA-256: {backend.checkpoint_sha256[:16]}...")
        self._set_busy(False, "Ready - open an image")

    @Slot(str)
    def open_image(self, path: str) -> None:
        if self.busy or self.closing:
            return
        self.request_id += 1
        self.image = None
        self.window.set_preview(None)
        self.window.clear_result()
        self._set_busy(True, f"Opening {Path(path).name}...")
        self.image_requested.emit(self.request_id, path)

    @Slot(int, object)
    def _image_loaded(self, request_id: int, image: Image.Image) -> None:
        if request_id != self.request_id:
            return
        self.image = image
        self.window.set_preview(image)
        self.window.set_message("Image validated. Ready for prediction.")
        self._set_busy(False, "Ready")

    @Slot()
    def predict(self) -> None:
        if self.busy or self.closing or not self.model_ready or self.image is None:
            return
        self.request_id += 1
        self.window.clear_result()
        self._set_busy(True, "Running prediction...")
        self.analysis_requested.emit(self.request_id, self.image)

    @Slot(object)
    def _completed(self, result: AnalysisResult) -> None:
        if result.request_id != self.request_id:
            return
        prediction = result.prediction
        self.window.prediction_label.setText(f"Prediction: {prediction.label}")
        self.window.probabilities_label.setText(
            f"Model probabilities - NORMAL: {prediction.normal_probability:.2%}  |  PNEUMONIA: {prediction.pneumonia_probability:.2%}"
        )
        self.window.timing_label.setText(
            f"Prediction: {result.prediction_seconds * 1000:.1f} ms  |  Grad-CAM: {result.explanation_seconds * 1000:.1f} ms"
        )
        self.window.session_label.setText(
            f"Model loads: {result.model_loads}  |  Predictions: {result.predictions}"
        )
        self.window.set_overlay(result.overlay)
        self.window.set_message(
            result.warning
            or "Highlighted regions influenced this model output; they are not a diagnosis.",
            bool(result.warning),
        )
        self._set_busy(False, "Complete - prediction available" if result.warning else "Complete")

    @Slot(int, str)
    def _failed(self, request_id: int, message: str) -> None:
        if request_id not in (0, self.request_id):
            return
        if request_id == 0:
            self.model_ready = False
        self.window.set_message(message, True)
        self._set_busy(False, "Error - choose another image or reload the model")

    @Slot(int, str)
    def _stage(self, request_id: int, message: str) -> None:
        if request_id == self.request_id:
            self.window.status_label.setText(message)

    @Slot()
    def reload_model(self) -> None:
        if self.busy or self.closing:
            return
        self.model_ready = False
        self.window.clear_result()
        self._set_busy(True, "Loading model...")
        self.initialize_requested.emit()

    @Slot(object)
    def close(self, event: QCloseEvent) -> None:
        if self.worker_thread.isRunning():
            event.ignore()
            self.closing = True
            self._set_busy(self.busy, "Finishing current operation before closing...")
