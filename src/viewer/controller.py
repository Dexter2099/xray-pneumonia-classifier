"""Application state and queued communication; model work stays off the GUI thread."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QObject, QSettings, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent

from src.settings import InferenceSettings
from src.viewer.browsing import ImageFolder
from src.viewer.diagnostics import InputInspection, TimelineEvent
from src.viewer.evidence import VerificationEvidence
from src.viewer.messages import AnalysisResult, ModelReady
from src.viewer.window import ViewerWindow
from src.viewer.worker import AnalysisBackend, AnalysisWorker, create_backend


class ViewerController(QObject):
    initialize_requested = Signal()
    image_requested = Signal(int, str)
    folder_requested = Signal(int, str)
    analysis_requested = Signal(int, object)
    verification_requested = Signal(int)

    def __init__(
        self,
        window: ViewerWindow,
        settings: InferenceSettings,
        factory: Callable[[InferenceSettings], AnalysisBackend] = create_backend,
        *,
        startup_folder: Path | None = None,
        preferences: QSettings | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.image: Image.Image | None = None
        self.request_id = 0
        self.busy = True
        self.model_ready = False
        self.closing = False
        self.startup_folder = startup_folder
        self.preferences = preferences
        self.folder: Path | None = None
        self.image_paths: tuple[Path, ...] = ()
        self.image_index = -1
        self.worker_thread = QThread(self)
        self.worker_thread.setObjectName("xray-analysis-worker")
        self.worker = AnalysisWorker(settings, factory)
        self.worker.moveToThread(self.worker_thread)
        self.initialize_requested.connect(self.worker.initialize)
        self.image_requested.connect(self.worker.open_image)
        self.folder_requested.connect(self.worker.open_folder)
        self.analysis_requested.connect(self.worker.analyze)
        self.verification_requested.connect(self.worker.read_verification)
        self.worker.ready.connect(self._ready)
        self.worker.image_loaded.connect(self._image_loaded)
        self.worker.folder_loaded.connect(self._folder_loaded)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.stage.connect(self._stage)
        self.worker.inspection_ready.connect(self._inspection_ready)
        self.worker.timeline_event.connect(self._timeline_event)
        self.worker.verification_loaded.connect(self._verification_loaded)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.window.close)
        window.open_requested.connect(self.open_image)
        window.folder_requested.connect(self.open_folder)
        window.next_requested.connect(self.next_image)
        window.predict_requested.connect(self.predict)
        window.retry_requested.connect(self.reload_model)
        window.closing.connect(self.close)
        window.hood.refresh_requested.connect(self.refresh_verification)
        self._set_busy(True, "Loading model...")
        self.worker_thread.start()
        self.initialize_requested.emit()

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        available = not busy and not self.closing
        self.window.set_actions_enabled(
            available=available,
            can_predict=available and self.model_ready and self.image is not None,
        )
        self.window.set_status(status)
        if self.closing and not busy:
            self.worker_thread.quit()

    @Slot(object)
    def _ready(self, ready: ModelReady) -> None:
        self.model_ready = True
        self.window.show_model_ready(ready)
        self._set_busy(False, "Ready - open an image")
        if self.startup_folder is not None:
            folder, self.startup_folder = self.startup_folder, None
            self.open_folder(str(folder))

    def _clear_input(self) -> None:
        self.image = None
        self.window.set_preview(None)
        self.window.hood.set_inspection(None)
        self.window.clear_result()

    def _show_browser(self, single_image: Path | None = None) -> None:
        current = self.image_paths[self.image_index] if self.image_index >= 0 else single_image
        self.window.show_browser(self.folder, current, self.image_index + 1, len(self.image_paths))

    @Slot(str)
    def open_folder(self, path: str) -> None:
        if self.busy or self.closing:
            return
        self.startup_folder = None
        self.request_id += 1
        self.folder = Path(path)
        self.image_paths = ()
        self.image_index = -1
        self._clear_input()
        self._show_browser()
        self._set_busy(True, "Finding images in folder...")
        self.folder_requested.emit(self.request_id, path)

    @Slot(int, object)
    def _folder_loaded(self, request_id: int, folder: ImageFolder) -> None:
        if request_id != self.request_id:
            return
        if self.closing:
            self._set_busy(False, "Folder scan finished")
            return
        self.folder, self.image_paths = folder.folder, folder.images
        self.image_index = 0
        if self.preferences is not None:
            self.preferences.setValue("browse/folder", str(folder.folder))
            self.preferences.sync()
        self._set_busy(False, "Folder loaded")
        self.open_image(str(self.image_paths[0]))

    @Slot()
    def next_image(self) -> None:
        if self.busy or self.closing or self.image_index + 1 >= len(self.image_paths):
            return
        self.open_image(str(self.image_paths[self.image_index + 1]))

    @Slot(str)
    def open_image(self, path: str) -> None:
        if self.busy or self.closing:
            return
        self.startup_folder = None
        self.request_id += 1
        selected = Path(path).resolve()
        if selected in self.image_paths:
            self.image_index = self.image_paths.index(selected)
        else:
            # A manually opened file outside the sequence must not show its old counter.
            self.folder, self.image_paths, self.image_index = None, (), -1
        # Invalidate old pixels/results before decoding: a failed open must never leave
        # the previous image's result displayed beside the new filename.
        self._clear_input()
        self._show_browser(selected)
        self._set_busy(True, f"Opening {Path(path).name}...")
        self.image_requested.emit(self.request_id, path)

    @Slot(int, object)
    def _image_loaded(self, request_id: int, image: Image.Image) -> None:
        # Queued results can outlive the request that produced them.
        if request_id != self.request_id:
            return
        self.image = image
        self.window.set_preview(image)
        self.window.set_message("Image validated. Ready for prediction.")
        self._set_busy(False, "Ready")

    @Slot(int, object)
    def _inspection_ready(self, request_id: int, inspection: InputInspection | None) -> None:
        if request_id == self.request_id:
            self.window.hood.set_inspection(inspection)

    @Slot(object)
    def _timeline_event(self, event: TimelineEvent) -> None:
        if event.request_id in (0, self.request_id):
            self.window.hood.append_event(event)

    @Slot()
    def refresh_verification(self) -> None:
        if self.busy or self.closing:
            return
        self.request_id += 1
        self._set_busy(True, "Reading saved verification report...")
        self.verification_requested.emit(self.request_id)

    @Slot(int, object)
    def _verification_loaded(self, request_id: int, evidence: VerificationEvidence) -> None:
        if request_id in (0, self.request_id):
            self.window.hood.set_verification(evidence)
            if request_id:
                self._set_busy(False, "Saved verification report loaded")

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
        self.window.show_analysis_result(result)
        self._set_busy(False, "Complete - prediction available" if result.warning else "Complete")

    @Slot(int, str)
    def _failed(self, request_id: int, message: str) -> None:
        if request_id not in (0, self.request_id):
            return
        if request_id == 0:
            self.model_ready = False
            self.window.hood.set_verification(
                VerificationEvidence(
                    "Unavailable",
                    "No loaded model",
                    "Model loading failed. Reload the trusted checkpoint before comparing evidence.",
                )
            )
        self.window.set_message(message, True)
        self._set_busy(False, "Error - choose another image or reload the model")

    @Slot(int, str)
    def _stage(self, request_id: int, message: str) -> None:
        if request_id == self.request_id:
            self.window.set_status(message)

    @Slot()
    def reload_model(self) -> None:
        if self.busy or self.closing:
            return
        self.model_ready = False
        self.window.clear_result()
        self.window.hood.set_verification(
            VerificationEvidence(
                "Unavailable",
                "Model reloading",
                "The previous checkpoint comparison is cleared until loading completes.",
            )
        )
        self._set_busy(True, "Loading model...")
        self.initialize_requested.emit()

    @Slot(object)
    def close(self, event: QCloseEvent) -> None:
        if self.worker_thread.isRunning():
            # Forcibly terminating a running PyTorch call can leave native resources
            # inconsistent. Finish it, then let thread.finished close the window.
            event.ignore()
            self.closing = True
            self._set_busy(self.busy, "Finishing current operation before closing...")
