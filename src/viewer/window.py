"""Render analysis snapshots and emit user intent; all widgets stay on the GUI thread."""

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.viewer.messages import AnalysisResult, ModelReady
from src.viewer.panels import UnderTheHood


class ViewerWindow(QMainWindow):
    open_requested = Signal(str)
    folder_requested = Signal(str)
    next_requested = Signal()
    predict_requested = Signal()
    retry_requested = Signal()
    closing = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Analysis Viewer")
        available = self.screen().availableGeometry()
        self.resize(min(1360, available.width() - 64), min(940, available.height() - 64))
        self._preview: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self._browse_folder: Path | None = None
        self._has_next = False

        central_widget = QWidget()
        central_widget.setObjectName("viewer_content")
        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.main_scroll.setWidget(central_widget)
        self.setCentralWidget(self.main_scroll)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title_label = QLabel("X-ray analysis viewer")
        title_label.setObjectName("title")
        subtitle_label = QLabel("PyTorch inference and Grad-CAM")
        subtitle_label.setObjectName("subtitle")
        edu_label = QLabel("Educational demonstration only. Not for clinical diagnosis.")
        edu_label.setObjectName("edu")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addWidget(edu_label)
        main_layout.addLayout(header_layout)

        browser_layout = QHBoxLayout()
        self.folder_button = QPushButton("Choose folder")
        self.folder_button.clicked.connect(self._on_folder_clicked)
        self.next_button = QPushButton("Next image")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.next_requested.emit)
        self.image_position_label = QLabel("Choose a folder to browse images")
        self.image_position_label.setWordWrap(True)
        self.image_position_label.setTextFormat(Qt.TextFormat.PlainText)
        browser_layout.addWidget(self.folder_button)
        browser_layout.addWidget(self.next_button)
        browser_layout.addWidget(self.image_position_label, 1)
        main_layout.addLayout(browser_layout)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)
        self.open_button = QPushButton("Open image")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self._on_open_clicked)
        self.predict_button = QPushButton("Predict and explain")
        self.predict_button.setObjectName("primary")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict_requested.emit)
        self.retry_button = QPushButton("Reload model")
        self.retry_button.setObjectName("secondary")
        self.retry_button.clicked.connect(self.retry_requested.emit)
        toolbar_layout.addWidget(self.open_button)
        toolbar_layout.addWidget(self.predict_button)
        toolbar_layout.addWidget(self.retry_button)
        self.hood_button = QPushButton("Under the hood")
        self.hood_button.setCheckable(True)
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)
        tools_layout = QHBoxLayout()
        tools_layout.addWidget(self.hood_button)
        tools_layout.addStretch()
        text_size_label = QLabel("Text size:")
        self.text_size_combo = QComboBox()
        self.text_size_combo.addItems(["Normal", "Large"])
        self.text_size_combo.setAccessibleName("Text size")
        self.text_size_combo.setToolTip("Change text throughout the viewer and diagnostic panels")
        text_size_label.setBuddy(self.text_size_combo)
        tools_layout.addWidget(text_size_label)
        tools_layout.addWidget(self.text_size_combo)
        main_layout.addLayout(tools_layout)

        images_frame = QFrame()
        images_frame.setObjectName("panel")
        images_layout = QHBoxLayout(images_frame)
        images_layout.setSpacing(20)
        images_layout.setContentsMargins(15, 15, 15, 15)

        left_container = QVBoxLayout()
        left_title = QLabel("Original image")
        left_title.setObjectName("panel_title")
        self.image_label = QLabel()
        self.image_label.setMinimumSize(260, 200)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setWordWrap(True)
        self.image_label.installEventFilter(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setObjectName("image_box")
        left_container.addWidget(left_title)
        left_container.addWidget(self.image_label, 1)

        right_container = QVBoxLayout()
        right_title = QLabel("Grad-CAM explanation")
        right_title.setObjectName("panel_title")
        self.overlay_label = QLabel()
        self.overlay_label.setMinimumSize(260, 200)
        self.overlay_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.overlay_label.setWordWrap(True)
        self.overlay_label.installEventFilter(self)
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setObjectName("image_box")
        right_container.addWidget(right_title)
        right_container.addWidget(self.overlay_label, 1)

        images_layout.addLayout(left_container, 1)
        images_layout.addLayout(right_container, 1)
        main_layout.addWidget(images_frame, 1)

        results_frame = QFrame()
        results_frame.setObjectName("panel")
        results_layout = QVBoxLayout(results_frame)
        results_layout.setSpacing(8)
        results_layout.setContentsMargins(15, 15, 15, 15)

        self.prediction_label = QLabel("Prediction: -")
        self.prediction_label.setWordWrap(True)
        self.prediction_label.setTextFormat(Qt.TextFormat.PlainText)
        self.prediction_label.setObjectName("prediction_text")

        self.probabilities_label = QLabel("Probabilities: -")
        self.probabilities_label.setWordWrap(True)
        self.probabilities_label.setTextFormat(Qt.TextFormat.PlainText)
        self.probabilities_label.setObjectName("result_text")

        self.timing_label = QLabel("Timing: -")
        self.timing_label.setWordWrap(True)
        self.timing_label.setTextFormat(Qt.TextFormat.PlainText)
        self.timing_label.setObjectName("result_text")

        self.session_label = QLabel("Session: -")
        self.session_label.setWordWrap(True)
        self.session_label.setTextFormat(Qt.TextFormat.PlainText)
        self.session_label.setObjectName("session_text")

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setObjectName("message_text")

        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setObjectName("status_text")

        results_layout.addWidget(self.prediction_label)
        results_layout.addWidget(self.probabilities_label)
        results_layout.addWidget(self.timing_label)
        results_layout.addWidget(self.session_label)
        results_layout.addWidget(self.message_label)
        results_layout.addWidget(self.status_label)
        main_layout.addWidget(results_frame)

        self.hood = UnderTheHood()
        self.hood_dock = QDockWidget("Under the hood", self)
        self.hood_dock.setObjectName("under_the_hood")
        self.hood_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.hood_dock.setWidget(self.hood)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.hood_dock)
        self.hood_dock.hide()
        self.hood_button.toggled.connect(self._toggle_hood)
        self.hood_dock.visibilityChanged.connect(self.hood_button.setChecked)

        self.text_size_combo.currentIndexChanged.connect(self._apply_text_size)
        self._apply_text_size(0)
        self.set_preview(None)
        self.set_overlay(None)

    def _apply_text_size(self, index: int) -> None:
        large = index == 1
        font = QFont(self.font())
        font.setPointSize(15 if large else 12)
        self.setFont(font)
        self.hood.setFont(font)
        self.setStyleSheet(self._build_stylesheet(large))
        self.hood.refresh_text_layout()
        if self.hood_dock.isVisible() and not self.hood_dock.isFloating():
            self._size_hood()

    def _toggle_hood(self, visible: bool) -> None:
        self.hood_dock.setVisible(visible)
        if visible:
            self._size_hood()

    def _size_hood(self) -> None:
        content = self.main_scroll.widget()
        if content is None:
            return
        desired = max(560, self.hood.verification_table.horizontalHeader().length() + 60)
        available = max(480, self.width() - content.minimumSizeHint().width())
        self.resizeDocks([self.hood_dock], [min(desired, available)], Qt.Orientation.Horizontal)

    def _build_stylesheet(self, large: bool = False):
        # Local Qwen drafted the font roles; review kept the existing palette and button hierarchy.
        body, heading, panel, small = (15, 24, 16, 14) if large else (12, 20, 13, 11)
        return f"""
            QMainWindow, QScrollArea {{ background-color: #111827; }}
            QWidget {{ color: #e2e8f0; font-family: 'Segoe UI', sans-serif; font-size: {body}pt; }}
            QWidget#viewer_content, QWidget#inspector_content {{ background-color: #111827; }}
            QDockWidget {{ color: #f3f4f6; background-color: #1f2937; }}
            QDockWidget::title {{ padding: 8px; background-color: #1f2937; }}
            QTabWidget::pane {{ border: 1px solid #374151; background-color: #1f2937; }}
            QTabBar::tab {{ background: #111827; padding: 8px; }}
            QTabBar::tab:selected {{ color: #2dd4bf; background: #1f2937; }}
            QTableWidget, QPlainTextEdit {{ background: #111827; color: #e2e8f0; border: 1px solid #374151; selection-background-color: #374151; }}
            QHeaderView::section {{ background: #374151; color: #f3f4f6; padding: 6px; }}
            QFrame#panel {{ background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; }}
            QLabel#title, QLabel#prediction_text {{ font-size: {heading}pt; font-weight: bold; color: #f3f4f6; }}
            QLabel#subtitle, QLabel#message_text, QLabel#status_text {{ color: #cbd5e1; }}
            QLabel#edu {{ font-size: {small}pt; color: #cbd5e1; }}
            QLabel#panel_title {{ font-size: {panel}pt; font-weight: bold; color: #e5e7eb; margin-bottom: 5px; }}
            QLabel#image_box {{ background-color: #111827; border: 1px solid #374151; border-radius: 4px; }}
            QLabel#session_text {{ font-weight: bold; }}
            QPushButton, QComboBox {{ background-color: #1f2937; color: #f3f4f6; border: 1px solid #64748b; border-radius: 6px; padding: 8px 16px; }}
            QPushButton:hover {{ background-color: #374151; }}
            QPushButton:pressed {{ background-color: #111827; }}
            QPushButton:checked {{ border-color: #2dd4bf; }}
            QPushButton:disabled {{ color: #94a3b8; background-color: #374151; border-color: #374151; }}
            QPushButton#primary {{ background-color: #2dd4bf; color: #111827; font-weight: bold; border: none; }}
            QPushButton#primary:hover {{ background-color: #14b8a6; }}
            QPushButton#primary:disabled {{ background-color: #374151; color: #94a3b8; }}
            QPushButton#secondary {{ background-color: transparent; border: 1px solid #2dd4bf; color: #2dd4bf; }}
            QPushButton#secondary:hover {{ background-color: #2dd4bf; color: #111827; }}
            QComboBox QAbstractItemView {{ background: #1f2937; color: #f3f4f6; selection-background-color: #374151; }}
        """

    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(self._browse_folder or ""), "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.open_requested.emit(path)

    def _on_folder_clicked(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose image folder", str(self._browse_folder or "")
        )
        if path:
            self.folder_requested.emit(path)

    def show_browser(
        self, folder: Path | None, current: Path | None, position: int, count: int
    ) -> None:
        self._browse_folder = folder
        self._has_next = 0 < position < count
        if folder is not None and current is not None:
            name = current.relative_to(folder).as_posix()
            self.image_position_label.setText(f"Image {position} of {count} · {name}")
        elif current is not None:
            self.image_position_label.setText(f"Single image · {current.name}")
        else:
            self.image_position_label.setText("No images selected · Choose a folder")
        self.image_position_label.setToolTip(str(folder or current or ""))

    def _pil_to_pixmap(self, image: Image.Image) -> QPixmap:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        data = rgb_image.tobytes()
        qimage = QImage(data, width, height, width * 3, QImage.Format.Format_RGB888)
        # QImage can borrow Python's byte buffer; own the pixels before that buffer dies.
        return QPixmap.fromImage(qimage.copy())

    def _scale_pixmap(self, pixmap: QPixmap, target_widget: QLabel) -> QPixmap:
        if not pixmap or pixmap.isNull():
            return pixmap
        target_size = target_widget.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return pixmap
        return pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def set_preview(self, image: Image.Image | None):
        if image is None:
            self._preview = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Open a PNG or JPEG image")
            return
        self._preview = self._pil_to_pixmap(image)
        # Retain full-resolution pixels so repeated resizes never rescale a scaled copy.
        self.image_label.setPixmap(self._scale_pixmap(self._preview, self.image_label))

    def set_overlay(self, image: Image.Image | None):
        if image is None:
            self._overlay = None
            self.overlay_label.setPixmap(QPixmap())
            self.overlay_label.setText("Run prediction to generate an explanation")
            return
        self._overlay = self._pil_to_pixmap(image)
        self.overlay_label.setPixmap(self._scale_pixmap(self._overlay, self.overlay_label))

    def clear_result(self):
        self.prediction_label.setText("Prediction: -")
        self.probabilities_label.setText("Probabilities: -")
        self.timing_label.setText("Timing: -")
        self.session_label.setText("Session: -")
        self.message_label.setText("")
        self.status_label.setText("Status: Ready")
        self.set_overlay(None)

    def set_actions_enabled(self, *, available: bool, can_predict: bool) -> None:
        self.open_button.setEnabled(available)
        self.folder_button.setEnabled(available)
        self.next_button.setEnabled(available and self._has_next)
        self.retry_button.setEnabled(available)
        self.hood.refresh_button.setEnabled(available)
        self.predict_button.setEnabled(can_predict)

    def set_status(self, status: str) -> None:
        self.status_label.setText(status)

    def show_model_ready(self, ready: ModelReady) -> None:
        self.session_label.setText(
            f"Model loads: {ready.model_loads}  |  Predictions: {ready.predictions}  |  Device: {ready.device}"
        )
        self.timing_label.setText(f"Model startup: {ready.load_seconds * 1000:.1f} ms")
        self.set_message(f"Checkpoint SHA-256: {ready.checkpoint_sha256[:16]}...")

    def show_analysis_result(self, result: AnalysisResult) -> None:
        prediction = result.prediction
        self.prediction_label.setText(f"Prediction: {prediction.label}")
        self.probabilities_label.setText(
            f"Model probabilities - NORMAL: {prediction.normal_probability:.2%}  |  PNEUMONIA: {prediction.pneumonia_probability:.2%}"
        )
        self.timing_label.setText(
            f"Prediction: {result.prediction_seconds * 1000:.1f} ms  |  Grad-CAM: {result.explanation_seconds * 1000:.1f} ms"
        )
        self.session_label.setText(
            f"Model loads: {result.model_loads}  |  Predictions: {result.predictions}"
        )
        self.set_overlay(result.overlay)
        self.set_message(
            result.warning
            or "Highlighted regions influenced this model output; they are not a diagnosis.",
            bool(result.warning),
        )

    def set_message(self, message: str, error: bool = False):
        self.message_label.setText(message)
        if error:
            self.message_label.setStyleSheet("color: #fca5a5;")
        else:
            self.message_label.setStyleSheet("color: #cbd5e1;")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize:
            if watched is self.image_label and self._preview is not None:
                self.image_label.setPixmap(self._scale_pixmap(self._preview, self.image_label))
            elif watched is self.overlay_label and self._overlay is not None:
                self.overlay_label.setPixmap(self._scale_pixmap(self._overlay, self.overlay_label))
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (
            hasattr(self, "hood_dock")
            and self.hood_dock.isVisible()
            and not self.hood_dock.isFloating()
        ):
            self._size_hood()

    def closeEvent(self, event: QCloseEvent):
        self.closing.emit(event)
        if event.isAccepted():
            super().closeEvent(event)
