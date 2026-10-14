from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ViewerWindow(QMainWindow):
    open_requested = Signal(str)
    predict_requested = Signal()
    retry_requested = Signal()
    closing = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Analysis Viewer")
        self.resize(1120, 760)
        self._preview: QPixmap | None = None
        self._overlay: QPixmap | None = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
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
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        images_frame = QFrame()
        images_frame.setObjectName("panel")
        images_layout = QHBoxLayout(images_frame)
        images_layout.setSpacing(20)
        images_layout.setContentsMargins(15, 15, 15, 15)

        left_container = QVBoxLayout()
        left_title = QLabel("Original image")
        left_title.setObjectName("panel_title")
        self.image_label = QLabel()
        self.image_label.setMinimumSize(360, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setObjectName("image_box")
        left_container.addWidget(left_title)
        left_container.addWidget(self.image_label)

        right_container = QVBoxLayout()
        right_title = QLabel("Grad-CAM explanation")
        right_title.setObjectName("panel_title")
        self.overlay_label = QLabel()
        self.overlay_label.setMinimumSize(360, 300)
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setObjectName("image_box")
        right_container.addWidget(right_title)
        right_container.addWidget(self.overlay_label)

        images_layout.addLayout(left_container, 1)
        images_layout.addLayout(right_container, 1)
        main_layout.addWidget(images_frame)

        results_frame = QFrame()
        results_frame.setObjectName("panel")
        results_layout = QVBoxLayout(results_frame)
        results_layout.setSpacing(8)
        results_layout.setContentsMargins(15, 15, 15, 15)

        self.prediction_label = QLabel("Prediction: -")
        self.prediction_label.setWordWrap(True)
        self.prediction_label.setTextFormat(Qt.TextFormat.PlainText)
        self.prediction_label.setObjectName("result_text")

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
        self.session_label.setObjectName("result_text")

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

        self.setStyleSheet(self._build_stylesheet())
        self.set_preview(None)
        self.set_overlay(None)

    def _build_stylesheet(self):
        return """
            QMainWindow { background-color: #111827; }
            QWidget { color: #f3f4f6; font-family: 'Segoe UI', sans-serif; }
            QFrame#panel { background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; }
            QLabel#title { font-size: 20px; font-weight: bold; color: #f3f4f6; }
            QLabel#subtitle { font-size: 13px; color: #9ca3af; }
            QLabel#edu { font-size: 11px; color: #6b7280; font-style: italic; }
            QLabel#panel_title { font-size: 14px; font-weight: bold; color: #e5e7eb; margin-bottom: 5px; }
            QLabel#image_box { background-color: #111827; border: 1px solid #374151; border-radius: 4px; }
            QLabel#result_text { font-size: 13px; color: #d1d5db; }
            QLabel#message_text { font-size: 13px; color: #9ca3af; }
            QLabel#status_text { font-size: 12px; color: #6b7280; }
            QPushButton { background-color: #1f2937; color: #f3f4f6; border: 1px solid #374151; border-radius: 6px; padding: 8px 16px; font-size: 13px; }
            QPushButton:hover { background-color: #374151; }
            QPushButton:pressed { background-color: #111827; }
            QPushButton#primary { background-color: #2dd4bf; color: #111827; font-weight: bold; border: none; }
            QPushButton#primary:hover { background-color: #14b8a6; }
            QPushButton#primary:disabled { background-color: #374151; color: #6b7280; }
            QPushButton#secondary { background-color: transparent; border: 1px solid #2dd4bf; color: #2dd4bf; }
            QPushButton#secondary:hover { background-color: #2dd4bf; color: #111827; }
        """

    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.open_requested.emit(path)

    def _pil_to_pixmap(self, image: Image.Image) -> QPixmap:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        data = rgb_image.tobytes()
        qimage = QImage(data, width, height, width * 3, QImage.Format.Format_RGB888)
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

    def set_message(self, message: str, error: bool = False):
        self.message_label.setText(message)
        if error:
            self.message_label.setStyleSheet("color: #ef4444;")
        else:
            self.message_label.setStyleSheet("color: #9ca3af;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._preview is not None:
            self.image_label.setPixmap(self._scale_pixmap(self._preview, self.image_label))
        if self._overlay is not None:
            self.overlay_label.setPixmap(self._scale_pixmap(self._overlay, self.overlay_label))

    def closeEvent(self, event: QCloseEvent):
        self.closing.emit(event)
        if event.isAccepted():
            super().closeEvent(event)
