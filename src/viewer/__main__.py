"""Launch with python -m src.viewer; the working directory does not affect data paths."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from src.errors import XrayError
from src.settings import ROOT, load_settings
from src.viewer.controller import ViewerController
from src.viewer.window import ViewerWindow


def main() -> int:
    parser = argparse.ArgumentParser(
        description="X-ray desktop viewer (educational portfolio demo)"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "src/config.yaml")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.WARNING)
    try:
        settings = load_settings(args.config)
    except XrayError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    app = QApplication(sys.argv[:1])
    window = ViewerWindow()
    preferences_path = ROOT / "outputs/viewer-preferences.ini"
    preferences_path.parent.mkdir(parents=True, exist_ok=True)
    preferences = QSettings(str(preferences_path), QSettings.Format.IniFormat)
    default_folder = ROOT / "data/raw/chest_xray/test"
    startup_folder = Path(str(preferences.value("browse/folder", str(default_folder))))
    controller = ViewerController(
        window, settings, startup_folder=startup_folder, preferences=preferences
    )
    window.show()
    exit_code = app.exec()
    controller.worker_thread.wait()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
