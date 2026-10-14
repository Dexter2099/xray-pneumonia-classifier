"""Exercise the real desktop worker, checkpoint, corrupt-input recovery and model reuse."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PySide6.QtCore import QEventLoop
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from src.settings import ROOT, load_settings
from src.viewer.controller import ViewerController
from src.viewer.window import ViewerWindow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo-output/viewer")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    app = QApplication([])
    # Qt's Windows offscreen plugin does not discover system fonts automatically.
    if not QFontDatabase.families() and os.name == "nt":
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    window = ViewerWindow()
    controller = ViewerController(window, load_settings())
    results = []
    controller.worker.completed.connect(results.append)
    window.show()

    def wait(predicate):
        deadline = monotonic() + 30
        while not predicate():
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            if monotonic() > deadline:
                raise TimeoutError("Desktop operation did not complete within 30 seconds")

    try:
        wait(lambda: not controller.busy)
        if not controller.model_ready:
            raise RuntimeError(window.message_label.text())
        if not window.hood_dock.isHidden():
            raise RuntimeError("Diagnostics must be optional and hidden initially")
        window.resize(1360, 940)
        sample = ROOT / "demo-assets/sample-xrays/pneumonia-high-confidence.jpeg"
        controller.open_image(str(sample))
        wait(lambda: not controller.busy)
        for _ in range(2):
            if results:
                window.text_size_combo.setCurrentText("Large")
            controller.predict()
            wait(lambda: not controller.busy)
        if window._overlay is None or "PNEUMONIA" not in window.prediction_label.text():
            raise RuntimeError("The real prediction or Grad-CAM overlay is missing")
        app.processEvents()
        for mode in ("Normal", "Large"):
            window.text_size_combo.setCurrentText(mode)
            app.processEvents()
            window.grab().save(str(args.output / f"prediction-{mode.lower()}.png"))
        window.grab().save(str(args.output / "prediction.png"))
        window.showMaximized()
        app.processEvents()
        window.grab().save(str(args.output / "prediction-maximized-large.png"))
        window.showNormal()
        window.resize(1640, 1000)
        window.hood_button.setChecked(True)
        for mode in ("Normal", "Large"):
            window.text_size_combo.setCurrentText(mode)
            for index, name in enumerate(("input-inspector", "execution-timeline", "verification")):
                window.hood.tabs.setCurrentIndex(index)
                app.processEvents()
                window.grab().save(str(args.output / f"{name}-{mode.lower()}.png"))
                window.grab().save(str(args.output / f"{name}.png"))
        window.showMaximized()
        window.hood.tabs.setCurrentIndex(2)
        app.processEvents()
        window.grab().save(str(args.output / "verification-maximized-large.png"))
        window.showNormal()
        window.resize(1640, 1000)
        if sum(window.hood.histogram.counts) != 224 * 224:
            raise RuntimeError("Input inspector did not show the canonical tensor histogram")
        if window.hood.verification_table.rowCount() != 3:
            raise RuntimeError("Verification panel did not load the three baseline cases")
        window.hood.tabs.setCurrentIndex(0)
        corrupt = args.output / "corrupt-example.png"
        corrupt.write_bytes(b"Deliberately corrupt interview demonstration input")
        controller.open_image(str(corrupt))
        wait(lambda: not controller.busy)
        if controller.image is not None or "corrupt" not in window.message_label.text():
            raise RuntimeError("Corrupt input was not handled cleanly")
        if window.hood.histogram.counts:
            raise RuntimeError("Corrupt input left stale inspection data")
        for mode in ("Normal", "Large"):
            window.text_size_combo.setCurrentText(mode)
            app.processEvents()
            window.grab().save(str(args.output / f"corrupt-input-{mode.lower()}.png"))
        window.grab().save(str(args.output / "corrupt-input.png"))
        controller.open_image(str(sample))
        wait(lambda: not controller.busy)
        controller.predict()
        wait(lambda: not controller.busy)
        if controller.worker.model_loads != 1 or controller.worker.predictions != 3:
            raise RuntimeError("The desktop model was unexpectedly reloaded")
        if len(results) != 3 or not all(
            result.prediction == results[0].prediction for result in results
        ):
            raise RuntimeError("Predictions changed with diagnostics displayed or after recovery")
        window.grab().save(str(args.output / "recovered.png"))
        report = {
            "passed": True,
            "real_checkpoint": True,
            "predictions": 3,
            "model_loads": 1,
            "corrupt_image_recovery": True,
            "explanation_displayed": window._overlay is not None,
            "input_inspector_displayed": True,
            "execution_stages_displayed": window.hood.timeline_table.rowCount(),
            "verification_cases_displayed": window.hood.verification_table.rowCount(),
            "diagnostics_preserve_predictions": True,
            "normal_large_text_preserve_predictions": True,
        }
        (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        window.close()
        wait(lambda: not controller.worker_thread.isRunning())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
