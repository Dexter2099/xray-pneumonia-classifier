import threading
import time

import pytest
from PIL import Image
from PySide6.QtCore import QTimer

from src.errors import ExplanationError, ModelLoadError
from src.inference import Prediction
from src.settings import InferenceSettings
from src.viewer.controller import ViewerController
from src.viewer.window import ViewerWindow
from src.viewer.worker import AnalysisBackend


class FakePredictor:
    def __init__(self):
        self.calls = 0
        self.thread_ids = []

    def predict(self, source):
        self.thread_ids.append(threading.get_ident())
        self.calls += 1
        time.sleep(0.08)
        return Prediction("NORMAL", 0.8, 0.8, 0.2)


class FakeExplainer:
    fail = False

    def explain(self, image):
        if self.fail:
            raise ExplanationError("No usable heatmap signal.")
        return Image.new("RGB", (224, 224), (255, 60, 10))


@pytest.fixture
def viewer(qtbot, tmp_path):
    predictor, explainer = FakePredictor(), FakeExplainer()
    calls = []

    def factory(settings):
        calls.append(threading.get_ident())
        return AnalysisBackend(predictor, explainer, "f" * 64, 0.01, "cpu")

    window = ViewerWindow()
    controller = ViewerController(window, InferenceSettings(tmp_path / "unused.pth"), factory)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: not controller.busy, timeout=10000)
    yield window, controller, predictor, explainer, calls
    window.close()
    qtbot.waitUntil(lambda: not controller.worker_thread.isRunning(), timeout=10000)


def open_valid(qtbot, tmp_path, controller):
    path = tmp_path / "valid.png"
    Image.new("L", (320, 200), 80).save(path)
    controller.open_image(str(path))
    qtbot.waitUntil(lambda: not controller.busy)
    return path


def test_valid_image_prediction_explanation_and_model_reuse(viewer, qtbot, tmp_path):
    window, controller, predictor, _, loads = viewer
    open_valid(qtbot, tmp_path, controller)
    assert window.predict_button.isEnabled()
    assert not window.image_label.pixmap().isNull()
    for _ in range(2):
        controller.predict()
        assert not window.open_button.isEnabled()
        qtbot.waitUntil(lambda: not controller.busy)
    assert predictor.calls == 2
    assert len(loads) == 1
    assert "Model loads: 1" in window.session_label.text()
    assert "Predictions: 2" in window.session_label.text()
    assert not window.overlay_label.pixmap().isNull()
    assert all(thread_id != threading.get_ident() for thread_id in loads + predictor.thread_ids)


def test_corrupt_image_clears_results_then_recovers(viewer, qtbot, tmp_path):
    window, controller, *_ = viewer
    open_valid(qtbot, tmp_path, controller)
    controller.predict()
    qtbot.waitUntil(lambda: not controller.busy)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    controller.open_image(str(corrupt))
    qtbot.waitUntil(lambda: not controller.busy)
    assert "corrupt" in window.message_label.text()
    assert controller.image is None
    assert not window.predict_button.isEnabled()
    assert window._overlay is None
    assert window.prediction_label.text() == "Prediction: -"
    open_valid(qtbot, tmp_path, controller)
    controller.predict()
    qtbot.waitUntil(lambda: not controller.busy)
    assert "NORMAL" in window.prediction_label.text()


def test_explanation_failure_retains_prediction(viewer, qtbot, tmp_path):
    window, controller, _, explainer, _ = viewer
    explainer.fail = True
    open_valid(qtbot, tmp_path, controller)
    controller.predict()
    qtbot.waitUntil(lambda: not controller.busy)
    assert "NORMAL" in window.prediction_label.text()
    assert "No usable" in window.message_label.text()
    assert window._overlay is None
    assert window.predict_button.isEnabled()


def test_gui_event_loop_runs_and_conflicting_requests_are_rejected(viewer, qtbot, tmp_path):
    window, controller, predictor, *_ = viewer
    open_valid(qtbot, tmp_path, controller)
    ticks = []
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    controller.predict()
    controller.predict()
    controller.open_image(str(tmp_path / "missing.png"))
    qtbot.waitUntil(lambda: not controller.busy)
    timer.stop()
    assert predictor.calls == 1
    assert len(ticks) >= 2
    assert controller.image is not None


def test_close_during_processing_waits_for_worker(viewer, qtbot, tmp_path):
    window, controller, *_ = viewer
    open_valid(qtbot, tmp_path, controller)
    controller.predict()
    window.close()
    assert controller.closing
    qtbot.waitUntil(lambda: not controller.worker_thread.isRunning(), timeout=10000)
    qtbot.waitUntil(lambda: not window.isVisible())


def test_startup_error_can_be_retried(qtbot, tmp_path):
    attempts = []

    def factory(settings):
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelLoadError("Checkpoint missing")
        return AnalysisBackend(FakePredictor(), FakeExplainer(), "f" * 64, 0.01, "cpu")

    window = ViewerWindow()
    controller = ViewerController(window, InferenceSettings(tmp_path / "unused"), factory)
    qtbot.addWidget(window)
    window.show()
    try:
        qtbot.waitUntil(lambda: not controller.busy)
        assert not controller.model_ready
        assert "Checkpoint missing" in window.message_label.text()
        controller.reload_model()
        qtbot.waitUntil(lambda: not controller.busy)
        assert controller.model_ready
    finally:
        window.close()
        qtbot.waitUntil(lambda: not controller.worker_thread.isRunning(), timeout=10000)
