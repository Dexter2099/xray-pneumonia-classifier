"""Optional presentation of detached input diagnostics and saved verification evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from src.viewer.diagnostics import InputInspection, TimelineEvent
    from src.viewer.evidence import VerificationEvidence

MAX_TIMELINE_ROWS = 100


def text_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def table(headers: list[str]) -> QTableWidget:
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.verticalHeader().hide()
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    widget.horizontalHeader().setDefaultSectionSize(150)
    widget.horizontalHeader().setStretchLastSection(True)
    widget.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    return widget


class Histogram(QWidget):
    """Sixteen fixed bins of normalised tensor values, painted without a plotting backend."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: tuple[int, ...] = ()
        self.setMinimumHeight(90)
        self.setMaximumHeight(120)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111827"))
        if self.counts:
            maximum = max(self.counts) or 1
            width = self.width() / len(self.counts)
            for index, count in enumerate(self.counts):
                height = int((self.height() - 8) * count / maximum)
                painter.fillRect(
                    int(index * width) + 1,
                    self.height() - height,
                    max(1, int(width) - 2),
                    height,
                    QColor("#2dd4bf"),
                )
        painter.end()


class UnderTheHood(QWidget):
    refresh_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        input_page = QWidget()
        input_page.setObjectName("inspector_content")
        input_layout = QVBoxLayout(input_page)
        input_layout.addWidget(
            text_label(
                "PNG/JPEG decoding produces 8-bit grayscale. This inspector previews the same "
                "preprocessing function used by prediction; it does not intercept its tensor."
            )
        )
        previews = QHBoxLayout()
        self.grayscale_label = QLabel()
        self.resized_label = QLabel()
        for title, label in (
            ("Decoded grayscale", self.grayscale_label),
            ("Resized 224 × 224", self.resized_label),
        ):
            column = QVBoxLayout()
            column.addWidget(text_label(title))
            label.setFixedSize(200, 170)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(label)
            previews.addLayout(column)
        input_layout.addLayout(previews)
        self.input_summary = text_label()
        input_layout.addWidget(self.input_summary)
        self.histogram = Histogram()
        input_layout.addWidget(
            text_label("Normalised values: 16 bins from −1 (left) to +1 (right)")
        )
        input_layout.addWidget(self.histogram)
        input_layout.addWidget(
            text_label(
                "Resize → float values [0,1] → (value − 0.5) / 0.5. "
                "The resized preview reverses normalisation for display only."
            )
        )
        input_layout.addStretch()
        self.input_scroll = QScrollArea()
        self.input_scroll.setWidgetResizable(True)
        self.input_scroll.setWidget(input_page)
        self.tabs.addTab(self.input_scroll, "Input inspector")

        timeline_page = QWidget()
        timeline_layout = QVBoxLayout(timeline_page)
        timeline_layout.addWidget(
            text_label(
                "Real worker events; request 0 is model/session setup. Durations include each "
                "stage's work and signal emission, not a profiler or the benchmark. Last 100 stages."
            )
        )
        self.timeline_table = table(["Request", "Stage", "Outcome", "ms", "Worker thread"])
        self.timeline_table.setColumnWidth(1, 220)
        self.timeline_table.horizontalHeader().sectionResized.connect(
            self.timeline_table.resizeRowsToContents
        )
        timeline_layout.addWidget(self.timeline_table)
        self.timeline_detail = text_label("Select a stage to see its responsibility.")
        timeline_layout.addWidget(self.timeline_detail)
        self.timeline_table.itemSelectionChanged.connect(self._select_stage)
        self.clear_timeline_button = QPushButton("Clear displayed timeline")
        self.clear_timeline_button.clicked.connect(self._clear_timeline)
        timeline_layout.addWidget(self.clear_timeline_button)
        self.tabs.addTab(timeline_page, "Execution timeline")

        verification_page = QWidget()
        verification_layout = QVBoxLayout(verification_page)
        self.verification_status = text_label("No evidence loaded")
        verification_layout.addWidget(self.verification_status)
        self.verification_detail = QPlainTextEdit()
        self.verification_detail.setReadOnly(True)
        self.verification_detail.setMinimumHeight(160)
        self.verification_detail.setMaximumHeight(260)
        verification_layout.addWidget(self.verification_detail)
        verification_layout.addWidget(text_label("N = NORMAL · P = PNEUMONIA"))
        self.verification_table = table(["Case", "Expected", "Recorded", "Max |Δ|", "Result"])
        self.verification_table.horizontalHeader().setStretchLastSection(False)
        self.verification_table.horizontalHeader().sectionResized.connect(
            self.verification_table.resizeRowsToContents
        )
        verification_layout.addWidget(self.verification_table)
        verification_layout.addWidget(
            text_label(
                "Generate fresh local evidence in a terminal:\npython -m scripts.verify_regression\n"
                "Refresh reads the report; it does not run tests."
            )
        )
        self.refresh_button = QPushButton("Refresh saved report")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        verification_layout.addWidget(self.refresh_button)
        self.tabs.addTab(verification_page, "Verification")
        self.set_inspection(None)

    def refresh_text_layout(self) -> None:
        """Keep rows readable when the inherited text size changes."""
        for widget in (self.timeline_table, self.verification_table):
            widget.ensurePolished()
            height = widget.fontMetrics().height() + 14
            widget.verticalHeader().setMinimumSectionSize(height)
            widget.verticalHeader().setDefaultSectionSize(height)
            widget.resizeRowsToContents()
        self.timeline_table.resizeColumnsToContents()
        self._size_verification_columns()

    def _size_verification_columns(self) -> None:
        self.verification_table.resizeColumnsToContents()
        self.verification_table.setColumnWidth(
            0, self.verification_table.fontMetrics().height() * 6
        )
        self.verification_table.resizeRowsToContents()

    @staticmethod
    def _preview(label: QLabel, image: Image.Image | None) -> None:
        if image is None:
            label.clear()
            return
        rgb = image.convert("RGB")
        qimage = QImage(
            rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage.copy())
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_inspection(self, inspection: InputInspection | None) -> None:
        self._preview(self.grayscale_label, inspection.grayscale_preview if inspection else None)
        self._preview(self.resized_label, inspection.resized_preview if inspection else None)
        self.histogram.counts = inspection.histogram if inspection else ()
        self.histogram.update()
        if inspection is None:
            self.input_summary.setText("No inspection available. Open a valid PNG or JPEG.")
            return
        self.input_summary.setText(
            f"Source dimensions: {inspection.source_size[0]} × {inspection.source_size[1]}\n"
            f"Tensor: {inspection.dtype}, shape {inspection.shape}; inference adds batch dimension → (1, 1, 224, 224)\n"
            f"Min {inspection.minimum:.5f} | Max {inspection.maximum:.5f}\n"
            f"Mean {inspection.mean:.5f} | Population std {inspection.std:.5f}\n"
            "Contract: one channel, 224 × 224, finite float32, normalised once to [−1,1]."
        )

    def append_event(self, event: TimelineEvent) -> None:
        """Update the matching Running row, retaining at most 100 operation records."""
        widget = self.timeline_table
        row = widget.rowCount()
        # Pair by request AND stage: a later completion must not overwrite unrelated work.
        if event.outcome != "Running":
            for candidate in range(widget.rowCount() - 1, -1, -1):
                item = widget.item(candidate, 0)
                if item is None:
                    continue
                previous = item.data(Qt.ItemDataRole.UserRole)
                if (previous.request_id, previous.stage, previous.outcome) == (
                    event.request_id,
                    event.stage,
                    "Running",
                ):
                    row = candidate
                    break
        if row == widget.rowCount():
            if row >= MAX_TIMELINE_ROWS:
                widget.removeRow(0)
                row -= 1
            widget.insertRow(row)
        values = [
            str(event.request_id),
            event.stage,
            event.outcome,
            "—" if event.outcome == "Running" else f"{event.elapsed_seconds * 1000:.2f}",
            str(event.worker_thread_id),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.ItemDataRole.UserRole, event)
            item.setToolTip(f"{value}\n{event.detail}")
            widget.setItem(row, column, item)
        if widget.rowCount() <= 2:
            widget.resizeColumnsToContents()
        widget.scrollToBottom()

    def _select_stage(self) -> None:
        row = self.timeline_table.currentRow()
        item = self.timeline_table.item(row, 0)
        if item is not None:
            event = item.data(Qt.ItemDataRole.UserRole)
            self.timeline_detail.setText(f"{event.stage}: {event.detail}")

    def _clear_timeline(self) -> None:
        self.timeline_table.setRowCount(0)
        self.timeline_detail.setText("Select a stage to see its responsibility.")

    def set_verification(self, evidence: VerificationEvidence) -> None:
        self.verification_status.setText(f"{evidence.status} · {evidence.provenance}")
        self.verification_detail.setPlainText(evidence.detail)
        self.verification_table.setRowCount(len(evidence.cases))
        for row, case in enumerate(evidence.cases):
            values = [
                case.name.rsplit(".", 1)[0].replace("-", " "),
                f"N {case.expected[0]:.6g}\nP {case.expected[1]:.6g}",
                f"N {case.actual[0]:.6g}\nP {case.actual[1]:.6g}",
                f"{case.error:.3g}",
                "PASS" if case.passed else "FAIL",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    f"{case.name}\nExpected {case.expected_label}: {case.expected}\n"
                    f"Recorded actual {case.actual_label}: {case.actual}"
                )
                self.verification_table.setItem(row, column, item)
        self._size_verification_columns()
