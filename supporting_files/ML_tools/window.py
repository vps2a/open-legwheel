"""Desktop exploration of already-classified exported telemetry packets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis import (
    CLASS_COLORS,
    CLASS_NAMES,
    LoadCancelled,
    TelemetryDataset,
    load_dataset,
)
from .plot_export import DEFAULT_EXPORT_DPI, MAX_EXPORT_DPI, MIN_EXPORT_DPI
from .plots import PlotPanel, PlotReport, channel_title
from .qt_helpers import DecimalSpinBox as StdMultiplierSpinBox
from .training_page import TrainingPage


class DatasetLoader(QThread):
    """Load numerical columns and cache both STD windows off the GUI thread."""

    loaded = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path

    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise LoadCancelled()

    def run(self):
        try:
            dataset = load_dataset(self.path, self.progress.emit, self._check_cancel)
            self._check_cancel()
            self.loaded.emit(dataset)
        except LoadCancelled:
            pass
        except Exception as error:  # noqa: BLE001 -- report failures at the worker boundary.
            self.failed.emit(str(error))


class VisualiserWindow(QMainWindow):
    def __init__(self, initial_dataset: Path | None = None):
        super().__init__()
        self.setWindowTitle("LegWheel telemetry classification visualiser")
        self.resize(1600, 1050)
        self.setMinimumSize(1180, 800)
        self.dataset: TelemetryDataset | None = None
        self.loader: DatasetLoader | None = None
        self.terrain_items = {}
        self.dirty_tabs = {0, 1, 2}
        self.reports: dict[int, PlotReport] = {}
        self._closing = False
        self._build_interface()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(80)
        self.refresh_timer.timeout.connect(self.refresh)
        self._connect_controls()
        if initial_dataset is not None:
            QTimer.singleShot(0, lambda: self.open_dataset(initial_dataset))

    def _build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        header = QHBoxLayout()
        self.open_button = QPushButton("Open dataset folder…")
        self.open_file_button = QPushButton("Open Parquet file…")
        self.path_label = QLabel("Choose an exported Parquet dataset to begin.")
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.path_label.setWordWrap(True)
        self.open_button.clicked.connect(self.choose_dataset)
        self.open_file_button.clicked.connect(self.choose_parquet)
        header.addWidget(self.open_button)
        header.addWidget(self.open_file_button)
        header.addWidget(self.path_label, 1)
        header.addWidget(QLabel("Export DPI"))
        self.export_dpi_spin = QSpinBox()
        self.export_dpi_spin.setRange(MIN_EXPORT_DPI, MAX_EXPORT_DPI)
        self.export_dpi_spin.setValue(DEFAULT_EXPORT_DPI)
        self.export_dpi_spin.setSingleStep(50)
        self.export_dpi_spin.setKeyboardTracking(False)
        self.export_dpi_spin.setToolTip(
            "Resolution for all saved plots, including experiment figures. "
            "Higher DPI creates sharper, larger images without resizing the screen plots."
        )
        header.addWidget(self.export_dpi_spin)
        outer.addLayout(header)

        # Dataset loading is shared; plot visibility and training assignments
        # are deliberately separate so hiding a class cannot alter an experiment.
        self.pages = QTabWidget()
        exploration = QWidget()
        exploration_layout = QVBoxLayout(exploration)
        exploration_layout.setContentsMargins(0, 0, 0, 0)
        self.pages.addTab(exploration, "Data exploration")
        self.training_page = TrainingPage()
        self.pages.addTab(self.training_page, "Model training and comparison")
        self.training_page.dataset_requested.connect(self.open_dataset)
        self.training_page.busy_changed.connect(self._training_busy_changed)
        outer.addWidget(self.pages, 1)

        self.controls = QWidget()
        filters = QHBoxLayout(self.controls)
        filters.setContentsMargins(0, 0, 0, 0)
        terrain_box = QGroupBox("Terrain setups and lighting — shared by all tabs")
        terrain_layout = QVBoxLayout(terrain_box)
        self.terrain_tree = QTreeWidget()
        self.terrain_tree.setHeaderLabels(
            ("Physical setup / terrain and light", "Labelled")
        )
        self.terrain_tree.header().setStretchLastSection(False)
        self.terrain_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.terrain_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.terrain_tree.setMinimumWidth(320)
        terrain_layout.addWidget(self.terrain_tree)
        terrain_buttons = QHBoxLayout()
        self.all_terrains_button = QPushButton("Select all")
        self.no_terrains_button = QPushButton("Deselect all")
        terrain_buttons.addWidget(self.all_terrains_button)
        terrain_buttons.addWidget(self.no_terrains_button)
        terrain_layout.addLayout(terrain_buttons)
        filters.addWidget(terrain_box, 2)

        settings_box = QGroupBox("Classes and plot controls")
        settings = QGridLayout(settings_box)
        self.class_checks = []
        for class_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            checkbox = QCheckBox(f"{class_id}  {name}")
            checkbox.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            checkbox.setChecked(True)
            self.class_checks.append(checkbox)
            settings.addWidget(checkbox, 0, class_id)

        self.window_combo = QComboBox()
        self.window_combo.addItem("Spectral window", "spectral")
        self.window_combo.addItem("Local patch window", "local")
        self.window_combo.setToolTip(
            "Changes within-packet STD only. Band powers always use saved spectral windows."
        )
        self.histogram_mode = QComboBox()
        self.histogram_mode.addItem("Probability density", True)
        self.histogram_mode.addItem("Packet count", False)
        settings.addWidget(QLabel("STD window"), 1, 0)
        settings.addWidget(self.window_combo, 1, 1)
        settings.addWidget(QLabel("Histogram"), 1, 2)
        settings.addWidget(self.histogram_mode, 1, 3)

        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 200)
        self.bins_spin.setValue(30)
        self.bins_spin.setKeyboardTracking(False)
        self.bins_spin.setToolTip(
            "Identical bin edges for all selected classes within each channel."
        )
        self.uncertainty_spin = StdMultiplierSpinBox()
        self.uncertainty_spin.setRange(0.0, 3.0)
        self.uncertainty_spin.setDecimals(2)
        self.uncertainty_spin.setSingleStep(0.25)
        self.uncertainty_spin.setValue(2.0)
        self.uncertainty_spin.setSuffix(" × STD")
        self.uncertainty_spin.setKeyboardTracking(False)
        self.uncertainty_spin.setToolTip(
            "Type 0–3 and press Enter. Bars show mean ± this many between-packet sample standard deviations."
        )
        settings.addWidget(QLabel("Histogram bins"), 2, 0)
        settings.addWidget(self.bins_spin, 2, 1)
        settings.addWidget(QLabel("Band-power error bars"), 2, 2)
        settings.addWidget(self.uncertainty_spin, 2, 3)

        self.scale_combo = QComboBox()
        self.scale_combo.addItem("Linear", False)
        self.scale_combo.addItem("Logarithmic", True)
        settings.addWidget(QLabel("Value scale"), 3, 0)
        settings.addWidget(self.scale_combo, 3, 1)
        hint = QLabel(
            "Log: histogram x-axis; band-power y-axis. Click a legend to toggle its class."
        )
        hint.setWordWrap(True)
        settings.addWidget(hint, 3, 2, 1, 2)
        filters.addWidget(settings_box, 5)
        self.controls.setMaximumHeight(215)
        exploration_layout.addWidget(self.controls)

        body = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.panels = [PlotPanel(kind) for kind in ("std", "motor", "imu")]
        for panel, title in zip(
            self.panels,
            ("Motor standard deviations", "Motor band powers", "IMU band powers"),
        ):
            self.tabs.addTab(panel, title)
            panel.class_toggled.connect(self.toggle_class)
        body.addWidget(self.tabs)

        summary_box = QGroupBox("Current selection")
        summary_layout = QVBoxLayout(summary_box)
        self.count_label = QLabel("0 packets")
        font = self.count_label.font()
        font.setPointSize(font.pointSize() + 7)
        font.setBold(True)
        self.count_label.setFont(font)
        summary_layout.addWidget(self.count_label)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(
            "Open an exported dataset.\n\nAll four labelled classes are enabled initially."
        )
        summary_layout.addWidget(self.summary, 1)
        summary_box.setMinimumWidth(270)
        body.addWidget(summary_box)
        body.setStretchFactor(0, 5)
        body.setStretchFactor(1, 1)
        body.setSizes((1240, 310))
        exploration_layout.addWidget(body, 1)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.cancel_button = QPushButton("Cancel loading")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_load)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage("Ready. Source datasets are read-only.")

    def _connect_controls(self):
        self.export_dpi_spin.valueChanged.connect(self._set_export_dpi)
        self._set_export_dpi(self.export_dpi_spin.value())
        for checkbox in self.class_checks:
            checkbox.toggled.connect(self.schedule_refresh)
        self.terrain_tree.itemChanged.connect(self.schedule_refresh)
        for combo in (self.window_combo, self.histogram_mode, self.scale_combo):
            combo.currentIndexChanged.connect(self.schedule_refresh)
        for spin in (self.bins_spin, self.uncertainty_spin):
            spin.valueChanged.connect(self.schedule_refresh)
        self.tabs.currentChanged.connect(self.refresh)
        self.all_terrains_button.clicked.connect(lambda: self.set_all_terrains(True))
        self.no_terrains_button.clicked.connect(lambda: self.set_all_terrains(False))

    def _set_export_dpi(self, dpi: int):
        for panel in self.panels:
            panel.toolbar.export_dpi = dpi
        self.training_page.set_export_dpi(dpi)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose exported dataset folder",
            str(self.dataset.root) if self.dataset else "",
        )
        if path:
            self.open_dataset(Path(path))

    def choose_parquet(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an exported packet Parquet file",
            str(self.dataset.root) if self.dataset else "",
            "Parquet files (*.parquet)",
        )
        if path:
            self.open_dataset(Path(path))

    def open_dataset(self, path: Path):
        if self.loader is not None or self.training_page.is_busy:
            return
        self.loader = DatasetLoader(path, self)
        self.loader.loaded.connect(self.set_dataset)
        self.loader.failed.connect(self._load_failed)
        self.loader.progress.connect(self._load_progress)
        self.loader.finished.connect(self._load_finished)
        self.open_button.setEnabled(False)
        self.open_file_button.setEnabled(False)
        self.controls.setEnabled(False)
        self.training_page.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.statusBar().showMessage(f"Loading numerical packet features from {path}…")
        self.loader.start()

    def _load_progress(self, current: int, total: int):
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    def cancel_load(self):
        if self.loader:
            # A cancelled "Open experiment" must not resume on a later load.
            self.training_page.pending_experiment = None
            self.loader.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Cancelling dataset load…")

    def _load_failed(self, message: str):
        self.training_page.pending_experiment = None
        if not self._closing:
            QMessageBox.critical(self, "Cannot open dataset", message)

    def _load_finished(self):
        loader, self.loader = self.loader, None
        loader.deleteLater()
        self.open_button.setEnabled(True)
        self.open_file_button.setEnabled(True)
        self.controls.setEnabled(True)
        self.training_page.setEnabled(True)
        self.progress.hide()
        self.cancel_button.hide()
        self.statusBar().showMessage("Ready. Source datasets are read-only.")
        if self._closing:
            self.close()

    def _training_busy_changed(self, busy: bool):
        self.open_button.setEnabled(not busy and self.loader is None)
        self.open_file_button.setEnabled(not busy and self.loader is None)
        if self._closing and not busy and self.loader is None:
            self.close()

    def set_dataset(self, dataset: TelemetryDataset):
        if self._closing:
            return
        self.dataset = dataset
        self.path_label.setText(f"{dataset.root.name}\n{dataset.root}")
        self.path_label.setToolTip(str(dataset.root))
        self.terrain_tree.blockSignals(True)
        self.terrain_tree.clear()
        self.terrain_items.clear()
        totals = Counter(
            key
            for key, label in zip(dataset.terrain_keys, dataset.class_ids)
            if label in range(4)
        )
        for setup in sorted({key[0] for key in dataset.terrain_keys}):
            keys = sorted({key for key in dataset.terrain_keys if key[0] == setup})
            parent = QTreeWidgetItem(
                self.terrain_tree, (setup, str(sum(totals[key] for key in keys)))
            )
            parent.setFlags(
                parent.flags()
                | Qt.ItemFlag.ItemIsAutoTristate
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            for key in keys:
                child = QTreeWidgetItem(parent, (key[1], str(totals[key])))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
                self.terrain_items[key] = child
            parent.setExpanded(True)
        self.terrain_tree.blockSignals(False)
        for checkbox in self.class_checks:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self.dirty_tabs = {0, 1, 2}
        self.reports.clear()
        self.refresh()
        self.training_page.set_dataset(dataset)

    def set_all_terrains(self, selected: bool):
        self.terrain_tree.blockSignals(True)
        for child in self.terrain_items.values():
            child.setCheckState(
                0, Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
            )
        self.terrain_tree.blockSignals(False)
        self.schedule_refresh()

    def toggle_class(self, class_id: int):
        self.class_checks[class_id].toggle()

    def selected_classes(self) -> set[int]:
        return {
            class_id
            for class_id, checkbox in enumerate(self.class_checks)
            if checkbox.isChecked()
        }

    def selected_terrains(self):
        return {
            key
            for key, item in self.terrain_items.items()
            if item.checkState(0) == Qt.CheckState.Checked
        }

    def schedule_refresh(self, *_args):
        self.dirty_tabs = {0, 1, 2}
        self.refresh_timer.start()

    def refresh(self, *_args):
        if self.dataset is None:
            return
        # A tab change can render immediately while a debounced filter refresh
        # is pending; the other tabs remain dirty until they are visited.
        self.refresh_timer.stop()
        dataset = self.dataset
        classes, terrains = self.selected_classes(), self.selected_terrains()
        selection = dataset.selection(classes, terrains)
        index = self.tabs.currentIndex()
        if index in self.dirty_tabs:
            self.reports[index] = self.panels[index].render(
                dataset,
                selection,
                classes,
                window=self.window_combo.currentData(),
                bins=self.bins_spin.value(),
                density=self.histogram_mode.currentData(),
                multiplier=self.uncertainty_spin.value(),
                logarithmic=self.scale_combo.currentData(),
            )
            self.dirty_tabs.discard(index)
        self._update_summary(selection, classes, terrains, self.reports[index])

    def _update_summary(self, selection, classes, terrains, report):
        dataset = self.dataset
        terrain_mask = dataset.terrain_selection(terrains)
        count = int(selection.sum())
        self.count_label.setText(f"{count:,} packets")
        selected_runs = {
            source for source, keep in zip(dataset.source_ids, selection) if keep
        }
        selected_setups = {
            key[0] for key, keep in zip(dataset.terrain_keys, selection) if keep
        }
        lines = [
            f"Loaded: {len(dataset.packet_ids):,} packets",
            f"Labelled: {int(np.isin(dataset.class_ids, range(4)).sum()):,}",
            f"Selected runs: {len(selected_runs)}",
            f"Selected physical setups: {len(selected_setups)}",
            "",
            "Classes: selected / in checked terrains",
        ]
        for class_id, name in enumerate(CLASS_NAMES):
            available = int(
                np.count_nonzero(terrain_mask & (dataset.class_ids == class_id))
            )
            lines.append(
                f"{class_id} {name}: {available if class_id in classes else 0} / {available}"
            )
        lines += [
            "",
            f"Unlabelled in checked terrains: {int(np.count_nonzero(terrain_mask & (dataset.class_ids == -1)))}",
            f"Unsupported labels: {int(np.count_nonzero(terrain_mask & (dataset.class_ids == -2)))}",
            "",
            "Measurements used in this tab:",
        ]
        for name, numbers in zip(
            self.panels[self.tabs.currentIndex()].channels, report.counts
        ):
            lines.append(f"{channel_title(name)}: {numbers}")
        if report.omitted_nonpositive:
            noun = "STD values" if self.tabs.currentIndex() == 0 else "class/band means"
            lines += [
                "",
                f"Log view omits {report.omitted_nonpositive} nonpositive {noun}.",
            ]
        if report.clipped_error_bars:
            lines += [
                "",
                f"{report.clipped_error_bars} lower error limits cross zero.",
                "▼ marks truncation at the positive plot floor.",
            ]
        lines += [
            "",
            "Equal weight per packet.",
            "Overlapping packets are retained.",
            "Unlabelled packets are excluded.",
            "",
            "Within-window STD: population (ddof=0).",
            "Band-power bars: mean ± "
            + f"{self.uncertainty_spin.value():g} × sample STD (ddof=1).",
            "Bars describe spread, not confidence intervals.",
            "A single value has no sample-STD bar.",
            "",
            "Band powers are read as exported.",
            "Changing STD window does not change powers.",
        ]
        if dataset.export_version:
            lines.append(f"Export version: {dataset.export_version}")
        if dataset.export_version == "1.1":
            lines += [
                "",
                "Older odd-length FFT spectra may underweight",
                "the highest bin. No correction is applied here.",
            ]
        if dataset.issues:
            lines += ["", "Input diagnostics:", *dataset.issues]
        self.summary.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        if self.loader is not None or self.training_page.is_busy:
            self._closing = True
            self.cancel_load()
            self.training_page.cancel_for_close()
            event.ignore()
            return
        super().closeEvent(event)
