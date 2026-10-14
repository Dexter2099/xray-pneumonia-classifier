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
        sample = ROOT / "demo-assets/sample-xrays/pneumonia-high-confidence.jpeg"
        controller.open_image(str(sample))
        wait(lambda: not controller.busy)
        for _ in range(2):
            controller.predict()
            wait(lambda: not controller.busy)
        if window._overlay is None or "PNEUMONIA" not in window.prediction_label.text():
            raise RuntimeError("The real prediction or Grad-CAM overlay is missing")
        app.processEvents()
        window.grab().save(str(args.output / "prediction.png"))
        corrupt = args.output / "corrupt-example.png"
        corrupt.write_bytes(b"Deliberately corrupt interview demonstration input")
        controller.open_image(str(corrupt))
        wait(lambda: not controller.busy)
        if controller.image is not None or "corrupt" not in window.message_label.text():
            raise RuntimeError("Corrupt input was not handled cleanly")
        window.grab().save(str(args.output / "corrupt-input.png"))
        controller.open_image(str(sample))
        wait(lambda: not controller.busy)
        controller.predict()
        wait(lambda: not controller.busy)
        if controller.worker.model_loads != 1 or controller.worker.predictions != 3:
            raise RuntimeError("The desktop model was unexpectedly reloaded")
        window.grab().save(str(args.output / "recovered.png"))
        report = {
            "passed": True,
            "real_checkpoint": True,
            "predictions": 3,
            "model_loads": 1,
            "corrupt_image_recovery": True,
            "explanation_displayed": window._overlay is not None,
        }
        (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        window.close()
        wait(lambda: not controller.worker_thread.isRunning())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
