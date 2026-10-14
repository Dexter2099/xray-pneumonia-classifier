import threading
import time

import pytest
from PIL import Image
from PySide6.QtCore import Qt, QTimer

from src.errors import ExplanationError, ModelLoadError
from src.inference import Prediction
from src.settings import InferenceSettings
from src.viewer.controller import ViewerController
from src.viewer.diagnostics import TimelineEvent
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
    events = timeline(window)
    assert [event.stage for event in events if event.request_id > 0] == [
        "Decode / validate",
        "Input inspection",
        "Prediction",
        "Grad-CAM",
        "Prediction",
        "Grad-CAM",
    ]
    assert all(event.outcome == "Complete" for event in events)
    assert all(event.worker_thread_id != threading.get_ident() for event in events)
    assert all(event.elapsed_seconds > 0 for event in events)


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
    assert window.hood.histogram.counts == ()
    assert window.hood.grayscale_label.pixmap().isNull()
    assert timeline(window)[-1].stage == "Decode / validate"
    assert timeline(window)[-1].outcome == "Failed"
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
    assert [(event.stage, event.outcome) for event in timeline(window)[-2:]] == [
        ("Prediction", "Complete"),
        ("Grad-CAM", "Failed"),
    ]


def timeline(window):
    return [
        window.hood.timeline_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        for row in range(window.hood.timeline_table.rowCount())
    ]


def test_optional_panel_and_inspector_do_not_trigger_predictions(viewer, qtbot, tmp_path):
    window, controller, predictor, _, loads = viewer
    assert window.hood_dock.isHidden()
    qtbot.mouseClick(window.hood_button, Qt.MouseButton.LeftButton)
    assert window.hood_dock.isVisible()
    open_valid(qtbot, tmp_path, controller)
    assert "320 × 200" in window.hood.input_summary.text()
    assert "(1, 1, 224, 224)" in window.hood.input_summary.text()
    assert sum(window.hood.histogram.counts) == 224 * 224
    assert not window.hood.resized_label.pixmap().isNull()
    for index in range(3):
        window.hood.tabs.setCurrentIndex(index)
    controller.refresh_verification()
    qtbot.waitUntil(lambda: not controller.busy)
    assert "Different active checkpoint" in window.hood.verification_status.text()
    assert window.hood.verification_table.rowCount() == 3
    window.hood_dock.close()
    assert not window.hood_button.isChecked()
    assert predictor.calls == 0 and len(loads) == 1


def test_text_size_changes_all_panels_without_changing_analysis(viewer, qtbot, tmp_path):
    window, controller, predictor, _, loads = viewer
    open_valid(qtbot, tmp_path, controller)
    controller.predict()
    qtbot.waitUntil(lambda: not controller.busy)
    prediction = window.prediction_label.text()
    original_preview = window._preview.cacheKey()
    original_overlay = window._overlay.cacheKey()
    window.hood_button.setChecked(True)
    for mode, points in (("Large", 15), ("Normal", 12)):
        window.text_size_combo.setCurrentText(mode)
        qtbot.waitUntil(
            lambda points=points: window.hood.timeline_table.font().pointSize() == points
        )
        assert window.open_button.font().pointSize() == points
        assert window.hood.input_summary.font().pointSize() == points
        assert window.hood.verification_detail.font().pointSize() == points
        assert window.prediction_label.font().pointSize() > points
        for width, height in ((1024, 720), (1640, 1000)):
            window.resize(width, height)
            qtbot.waitUntil(lambda width=width: window.width() == width)
            assert window.open_button.font().pointSize() == points
            pixmap = window.image_label.pixmap()
            assert pixmap.width() <= window.image_label.width()
            assert pixmap.height() <= window.image_label.height()
            assert pixmap.width() / pixmap.height() == pytest.approx(320 / 200, abs=0.02)
        assert window.prediction_label.text() == prediction
        assert window._preview.cacheKey() == original_preview
        assert window._overlay.cacheKey() == original_overlay
    assert predictor.calls == 1 and len(loads) == 1
    window.hood_dock.setFloating(True)
    window.text_size_combo.setCurrentText("Large")
    qtbot.waitUntil(lambda: window.hood.verification_table.font().pointSize() == 15)
    assert (
        window.hood.timeline_table.rowHeight(0) >= window.hood.timeline_table.fontMetrics().height()
    )


def test_diagnostic_failure_does_not_block_valid_input_or_prediction(
    viewer, qtbot, tmp_path, monkeypatch
):
    window, controller, predictor, *_ = viewer

    def fail(image):
        raise RuntimeError("Simulated inspector failure")

    monkeypatch.setattr("src.viewer.worker.inspect_input", fail)
    open_valid(qtbot, tmp_path, controller)
    assert controller.image is not None
    assert timeline(window)[-1].outcome == "Failed"
    controller.predict()
    qtbot.waitUntil(lambda: not controller.busy)
    assert predictor.calls == 1
    assert "NORMAL" in window.prediction_label.text()


def test_verification_failure_does_not_strand_busy_window(viewer, qtbot, monkeypatch):
    window, controller, *_ = viewer

    def fail(checkpoint):
        raise RuntimeError("Simulated report reader failure")

    monkeypatch.setattr("src.viewer.worker.load_verification", fail)
    controller.refresh_verification()
    qtbot.waitUntil(lambda: not controller.busy)
    assert "Unavailable" in window.hood.verification_status.text()
    assert window.open_button.isEnabled()
    assert timeline(window)[-1].outcome == "Failed"


def test_failed_model_reload_clears_previous_checkpoint_comparison(viewer, qtbot):
    window, controller, *_ = viewer
    assert window.hood.verification_table.rowCount() == 3

    def fail(settings):
        raise ModelLoadError("Simulated changed/missing checkpoint")

    controller.worker.factory = fail
    controller.reload_model()
    assert window.hood.verification_table.rowCount() == 0
    assert "Model reloading" in window.hood.verification_status.text()
    qtbot.waitUntil(lambda: not controller.busy)
    assert not controller.model_ready
    assert "No loaded model" in window.hood.verification_status.text()
    assert window.hood.verification_table.rowCount() == 0
    assert timeline(window)[-1].stage == "Model load"
    assert timeline(window)[-1].outcome == "Failed"


def test_timeline_is_bounded_and_stale_events_are_ignored(viewer):
    window, controller, *_ = viewer
    window.hood._clear_timeline()
    controller._timeline_event(TimelineEvent(999, "Stale", "Complete", 0.01, 42, "ignore"))
    assert timeline(window) == []
    for index in range(110):
        window.hood.append_event(TimelineEvent(index, "Stage", "Complete", 0.01, 42, "detail"))
    assert len(timeline(window)) == 100
    assert timeline(window)[0].request_id == 10


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
