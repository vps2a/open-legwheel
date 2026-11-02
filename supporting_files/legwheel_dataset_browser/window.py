"""Main window for packet inspection and manual terrain classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .classification import TERRAIN_CLASSES
from .model import DatasetFormatError, LegWheelDataset, PacketData
from .widgets import (
    ImageCanvas,
    MetadataTable,
    PointMapCanvas,
    SlopeSummary,
    TelemetryPanel,
)


class DatasetBrowserWindow(QMainWindow):
    """Resizable packet browser with explicitly enabled classification controls."""

    SETTINGS_LAST_DATASET = "last_dataset_directory"

    def __init__(self, initial_dataset: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("LegWheel Dataset Browser")
        self.resize(1920, 1080)
        self.setMinimumSize(1280, 720)

        self.settings = QSettings()
        self.dataset: LegWheelDataset | None = None
        self.packet: PacketData | None = None
        self.packet_index = 0
        self.fixed_depth_limits_m = (0.0, 1.0)

        self._build_interface()
        self._connect_navigation()

        candidate = initial_dataset
        if candidate is None:
            saved = self.settings.value(self.SETTINGS_LAST_DATASET, "", type=str)
            if saved:
                candidate = Path(saved)
        if candidate is not None and candidate.is_dir():
            if not self.open_dataset(candidate, show_error=False):
                QTimer.singleShot(0, self.choose_dataset)
        else:
            QTimer.singleShot(0, self.choose_dataset)

    def _build_interface(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.identity_label = QLabel("No dataset selected")
        identity_font = self.identity_label.font()
        identity_font.setBold(True)
        identity_font.setPointSize(identity_font.pointSize() + 3)
        self.identity_label.setFont(identity_font)
        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.change_dataset_button = QPushButton("Change dataset…")
        self.change_dataset_button.clicked.connect(self.choose_dataset)
        header.addWidget(self.identity_label)
        header.addWidget(self.path_label, 1)
        header.addWidget(self.change_dataset_button)
        outer.addLayout(header)

        self.slope_summary = SlopeSummary()
        outer.addWidget(self.slope_summary)

        classification_controls = QHBoxLayout()
        self.class_buttons = []
        for class_id, description in enumerate(TERRAIN_CLASSES):
            button = QPushButton(f"{class_id} - {description}")
            button.setCheckable(True)
            button.setEnabled(False)
            button.setToolTip("Classify the highlighted terrain patch and advance.")
            # Keep an existing selection visible even when editing is disabled.
            button.setStyleSheet(
                "QPushButton:checked, QPushButton:checked:disabled {"
                " background-color: #c9e7ff; color: #16324f;"
                " border: 1px solid #6badd6; border-radius: 3px; padding: 5px; }"
            )
            button.clicked.connect(
                lambda _checked=False, selected=class_id: self.classify_packet(selected)
            )
            self.class_buttons.append(button)
            classification_controls.addWidget(button)
        self.manual_mode_checkbox = QCheckBox("Manual Classification Mode")
        self.manual_mode_checkbox.setEnabled(False)
        self.manual_mode_checkbox.toggled.connect(self._toggle_manual_mode)
        classification_controls.addWidget(self.manual_mode_checkbox)
        classification_controls.addStretch(1)
        outer.addLayout(classification_controls)

        body = QSplitter(Qt.Orientation.Horizontal)
        self.left_telemetry = TelemetryPanel("motor")
        self.right_telemetry = TelemetryPanel("imu")
        self.left_telemetry.currentChanged.connect(self.right_telemetry.setCurrentIndex)
        self.right_telemetry.currentChanged.connect(self.left_telemetry.setCurrentIndex)

        self.central_tabs = QTabWidget()
        self.central_tabs.addTab(self._build_images_tab(), "Images and patches")
        self.point_map_canvas = PointMapCanvas()
        self.central_tabs.addTab(self.point_map_canvas, "3D diagnostic")
        self.metadata_table = MetadataTable()
        self.central_tabs.addTab(self.metadata_table, "Metadata")

        body.addWidget(self.left_telemetry)
        body.addWidget(self.central_tabs)
        body.addWidget(self.right_telemetry)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 5)
        body.setStretchFactor(2, 2)
        body.setSizes((350, 1050, 350))
        outer.addWidget(body, 1)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("← Previous packet")
        self.next_button = QPushButton("Next packet →")
        self.packet_position_label = QLabel("Packet 0 of 0")
        self.packet_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        navigation.addWidget(self.previous_button)
        navigation.addStretch(1)
        navigation.addWidget(self.packet_position_label)
        navigation.addStretch(1)
        navigation.addWidget(self.next_button)
        outer.addLayout(navigation)

        # Keep direct packet selection in the same persistent bottom area as
        # sequential navigation, so it remains available on every tab.
        jump_navigation = QHBoxLayout()
        self.jump_packet_input = QLineEdit()
        self.jump_packet_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.jump_packet_input.setFixedWidth(90)
        self.jump_packet_input.setPlaceholderText("Packet no.")
        self.jump_packet_input.setToolTip(
            "Enter the packet's 1-based position in the sorted dataset list."
        )
        # Hint that this is a numeric field without blocking invalid text.
        # Allowing every submission through lets both Enter and the button
        # produce the same helpful inline validation message.
        self.jump_packet_input.setInputMethodHints(
            Qt.InputMethodHint.ImhDigitsOnly
        )
        self.jump_packet_input.setEnabled(False)
        self.jump_packet_button = QPushButton("Jump to packet")
        self.jump_packet_button.setEnabled(False)
        self.jump_packet_error_label = QLabel("")
        self.jump_packet_error_label.setStyleSheet("color: #c62828;")
        self.jump_packet_error_label.setVisible(False)

        jump_navigation.addStretch(1)
        jump_navigation.addWidget(QLabel("Go to:"))
        jump_navigation.addWidget(self.jump_packet_input)
        jump_navigation.addWidget(self.jump_packet_button)
        jump_navigation.addWidget(self.jump_packet_error_label)
        jump_navigation.addStretch(1)
        outer.addLayout(jump_navigation)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Select a configuration-hash dataset directory.")

    def _build_images_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.crop_checkbox = QCheckBox("Show cropped masked patches")
        self.auto_depth_checkbox = QCheckBox("Auto-scale depth for current packet")
        controls.addWidget(self.crop_checkbox)
        controls.addWidget(self.auto_depth_checkbox)
        controls.addStretch(1)
        layout.addLayout(controls)

        images = QSplitter(Qt.Orientation.Horizontal)
        self.rgb_canvas = ImageCanvas()
        self.depth_canvas = ImageCanvas()
        images.addWidget(self.rgb_canvas)
        images.addWidget(self.depth_canvas)
        images.setSizes((700, 700))
        layout.addWidget(images, 1)
        return tab

    def _connect_navigation(self) -> None:
        self.previous_button.clicked.connect(self.previous_packet)
        self.next_button.clicked.connect(self.next_packet)
        self.jump_packet_button.clicked.connect(self.jump_to_packet)
        self.jump_packet_input.returnPressed.connect(self.jump_to_packet)
        self.jump_packet_input.textEdited.connect(self._clear_jump_error)
        self.crop_checkbox.toggled.connect(self._refresh_images)
        self.auto_depth_checkbox.toggled.connect(self._refresh_images)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(
            self.previous_packet
        )
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            self.next_packet
        )
        self.classification_shortcuts = []
        for class_id in range(len(TERRAIN_CLASSES)):
            shortcut = QShortcut(QKeySequence(str(class_id)), self)
            # Holding a number must not classify a succession of packets.
            shortcut.setAutoRepeat(False)
            shortcut.setEnabled(False)
            shortcut.activated.connect(
                lambda selected=class_id: self.classify_packet(selected)
            )
            self.classification_shortcuts.append(shortcut)
        QApplication.instance().focusChanged.connect(self._refresh_classification_shortcuts)

    def _refresh_classification_shortcuts(self, *_args) -> None:
        """Leave number keys available for typing into the Go to field."""
        enabled = (
            self.manual_mode_checkbox.isChecked()
            and not isinstance(QApplication.focusWidget(), QLineEdit)
        )
        for shortcut in self.classification_shortcuts:
            shortcut.setEnabled(enabled)

    def _toggle_manual_mode(self, enabled: bool) -> None:
        """Confirm once when entering manual mode; toggling never writes data."""
        if enabled:
            answer = QMessageBox.warning(
                self,
                "Manual classification",
                "Selecting a class will immediately save it and advance to the "
                "next packet. Any previous classification will be manually "
                "overwritten.\n\nEnable Manual Classification Mode?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Ok:
                self.manual_mode_checkbox.blockSignals(True)
                self.manual_mode_checkbox.setChecked(False)
                self.manual_mode_checkbox.blockSignals(False)
        self._refresh_classification_controls()

    def _refresh_classification_controls(self) -> None:
        selected = self.packet.metadata.get("class_id") if self.packet else None
        enabled = self.packet is not None and self.manual_mode_checkbox.isChecked()
        for class_id, button in enumerate(self.class_buttons):
            button.setChecked(selected == class_id)
            button.setEnabled(enabled)
        self._refresh_classification_shortcuts()

    def classify_packet(self, class_id: int) -> None:
        """Save immediately, then advance only after a successful write."""
        if self.dataset is None or self.packet is None:
            return
        if not self.manual_mode_checkbox.isChecked():
            return
        try:
            self.dataset.classify_packet(self.packet_index, class_id)
        except (DatasetFormatError, OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot save classification", str(error))
            self.statusBar().showMessage(f"Classification save failed: {error}")
            self._refresh_classification_controls()
            return

        packet_id = self.packet.metadata["packet_id"]
        # Reopening the final packet refreshes its highlight without moving past
        # the dataset. Every earlier selection advances to the next packet.
        self.show_packet(min(self.packet_index + 1, len(self.dataset) - 1))
        self.statusBar().showMessage(
            f"Saved {packet_id}: {class_id} - {TERRAIN_CLASSES[class_id]} (manual)",
            8000,
        )

    def choose_dataset(self) -> None:
        """Ask for a configuration-hash directory and retain the last choice."""
        start = self.settings.value(self.SETTINGS_LAST_DATASET, "", type=str)
        if not start or not Path(start).is_dir():
            start = str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select derived dataset configuration-hash directory",
            start,
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.open_dataset(Path(selected), show_error=True)

    def open_dataset(self, directory: Path, *, show_error: bool = True) -> bool:
        """Upgrade legacy classification fields, then display the dataset."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dataset = LegWheelDataset.open(directory)
            depth_limits = dataset.global_depth_limits_m()
            first_packet = dataset.load_packet(0)
        except (DatasetFormatError, OSError, ValueError) as error:
            if show_error:
                QMessageBox.critical(self, "Cannot open dataset", str(error))
            self.statusBar().showMessage(f"Could not open dataset: {error}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

        self.dataset = dataset
        # Manual mode is deliberately not persisted in application settings.
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.setEnabled(True)
        self.fixed_depth_limits_m = depth_limits
        self.packet_index = 0
        self.jump_packet_input.setEnabled(True)
        self.jump_packet_button.setEnabled(True)
        self.jump_packet_input.clear()
        self.jump_packet_input.setPlaceholderText(f"1–{len(dataset)}")
        self._clear_jump_error()
        self.settings.setValue(self.SETTINGS_LAST_DATASET, str(dataset.root))
        self.identity_label.setText(
            f"Experiment {dataset.experiment_id}  ·  Run {dataset.run_number}"
        )
        self.path_label.setText(f"Dataset: {dataset.root.name}")
        self.path_label.setToolTip(str(dataset.root))
        self._show_loaded_packet(first_packet)
        self.statusBar().showMessage(
            f"Loaded {len(dataset)} packets from {dataset.root}", 8000
        )
        return True

    def show_packet(self, index: int) -> bool:
        """Load a zero-based packet index, returning whether it succeeded."""
        if self.dataset is None or not 0 <= index < len(self.dataset):
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            packet = self.dataset.load_packet(index)
        except (DatasetFormatError, OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot load packet", str(error))
            self.statusBar().showMessage(f"Packet load failed: {error}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.packet_index = index
        self._show_loaded_packet(packet)
        self._clear_jump_error()
        return True

    def jump_to_packet(self) -> None:
        """Jump to a 1-based position in the dataset's sorted packet list."""
        if self.dataset is None:
            self._show_jump_error("Select a dataset before choosing a packet.")
            return

        entered_text = self.jump_packet_input.text().strip()
        try:
            packet_number = int(entered_text)
        except ValueError:
            self._show_jump_error(
                f"Enter a whole packet number from 1 to {len(self.dataset)}."
            )
            return

        if not 1 <= packet_number <= len(self.dataset):
            self._show_jump_error(
                f"Packet number must be from 1 to {len(self.dataset)}."
            )
            return

        # The model is already chronologically sorted. Convert the user's
        # 1-based packet number to the model's zero-based list index.
        if self.show_packet(packet_number - 1):
            self.jump_packet_input.clear()

    def _show_jump_error(self, message: str) -> None:
        """Display a non-blocking validation message beside the jump control."""
        self.jump_packet_error_label.setText(message)
        self.jump_packet_error_label.setVisible(True)

    def _clear_jump_error(self, _text: str = "") -> None:
        """Clear stale validation feedback after editing or successful navigation."""
        self.jump_packet_error_label.clear()
        self.jump_packet_error_label.setVisible(False)

    def _show_loaded_packet(self, packet: PacketData) -> None:
        self.packet = packet
        self.slope_summary.show_packet(packet)
        self.left_telemetry.show_packet(packet)
        self.right_telemetry.show_packet(packet)
        self.point_map_canvas.show_packet(packet)
        self.metadata_table.show_packet(packet)
        self._refresh_classification_controls()
        self._refresh_images()

        assert self.dataset is not None
        self.packet_position_label.setText(
            f"Packet {self.packet_index + 1} of {len(self.dataset)}  ·  "
            f"{packet.metadata.get('packet_id', 'unknown')}"
        )
        self.previous_button.setEnabled(self.packet_index > 0)
        self.next_button.setEnabled(self.packet_index + 1 < len(self.dataset))

    def _refresh_images(self) -> None:
        if self.packet is None or self.dataset is None:
            return
        cropped = self.crop_checkbox.isChecked()
        rgb_polygon = self.packet.geometry.get("rgb_patch_polygon_pixels")
        depth_polygon = self.packet.geometry.get("depth_patch_polygon_pixels")
        self.rgb_canvas.show_rgb(
            self.packet.rgb,
            self.packet.rgb_mask,
            rgb_polygon,
            cropped=cropped,
        )

        depth_m = self.dataset.depth_to_metres(self.packet.depth)
        limits = self.fixed_depth_limits_m
        if self.auto_depth_checkbox.isChecked():
            valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
            if valid.size:
                lower = float(np.min(valid))
                upper = float(np.max(valid))
                if lower < upper:
                    limits = (lower, upper)
        self.depth_canvas.show_depth(
            depth_m,
            self.packet.depth_mask,
            depth_polygon,
            limits,
            cropped=cropped,
        )

    def previous_packet(self) -> None:
        self.show_packet(self.packet_index - 1)

    def next_packet(self) -> None:
        self.show_packet(self.packet_index + 1)
