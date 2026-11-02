"""Packet selection and a small export form; calculations live outside Qt."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .calculations import frequency_bands
from .export import (
    ExportCancelled,
    ExportOptions,
    ExportResult,
    export_dataset,
    png_image,
)
from .sources import CLASS_NAMES, Packet, SourceDataset, discover_sources

PACKET_ROLE = Qt.ItemDataRole.UserRole


class Task(QThread):
    """Run discovery/export off the GUI thread with cooperative cancellation."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, action: Callable, parent=None):
        super().__init__(parent)
        self.action = action

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise ExportCancelled()

    def run(self):
        try:
            self.completed.emit(self.action(self.progress.emit, self.check_cancel))
        except ExportCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 -- Qt thread boundary must report library failures.
            # Report unexpected input/library failures too, without leaving the
            # window disabled or letting an exception escape a Qt worker.
            self.failed.emit(str(error))


class ExporterWindow(QMainWindow):
    def __init__(self, directory: Path | None = None):
        super().__init__()
        self.setWindowTitle("LegWheel dataset exporter")
        self.resize(1250, 850)
        self.sources: dict[Path, SourceDataset] = {}
        self.packet_items: list[QTreeWidgetItem] = []
        self._size_estimates: dict[tuple, int] = {}
        self.task: Task | None = None
        self._close_when_done = False
        # Keep group checkboxes current immediately, but debounce the more
        # expensive review/size summary when selecting individual packets.
        self.summary_timer = QTimer(self)
        self.summary_timer.setSingleShot(True)
        self.summary_timer.setInterval(80)
        self.summary_timer.timeout.connect(self.update_summary)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        self.controls = QWidget()
        layout = QVBoxLayout(self.controls)
        layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.controls, 1)

        source_bar = QHBoxLayout()
        self.add_button = QPushButton("Add source folder…")
        self.add_button.clicked.connect(self.choose_source)
        source_bar.addWidget(self.add_button)
        self.clear_button = QPushButton("Clear sources")
        self.clear_button.clicked.connect(self.clear_sources)
        source_bar.addWidget(self.clear_button)
        hint = QLabel(
            "Choose recordings, an experiment/run folder, or one derived dataset. Source files stay unchanged."
        )
        hint.setWordWrap(True)
        source_bar.addWidget(hint, 1)
        layout.addLayout(source_bar)

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Filter experiment, run, processing ID, packet ID or status…"
        )
        self.search.textChanged.connect(self.apply_filters)
        filters.addWidget(self.search, 1)
        self.class_filter = QComboBox()
        for title, value in [
            ("All classes", "all"),
            ("Unlabelled", "unlabelled"),
            *[(f"{i}: {name}", i) for i, name in enumerate(CLASS_NAMES)],
        ]:
            self.class_filter.addItem(title, value)
        self.class_filter.currentIndexChanged.connect(self.apply_filters)
        filters.addWidget(self.class_filter)
        self.select_button = QPushButton("Select visible")
        self.select_button.clicked.connect(lambda: self.select_visible(True))
        filters.addWidget(self.select_button)
        self.deselect_button = QPushButton("Deselect visible")
        self.deselect_button.clicked.connect(lambda: self.select_visible(False))
        filters.addWidget(self.deselect_button)
        layout.addLayout(filters)

        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ("Experiment / run / processing version / packet", "Class", "Method", "Status")
        )
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self.selection_changed)
        self.tree.currentItemChanged.connect(self.show_preview)
        splitter.addWidget(self.tree)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        self.preview = QLabel("Select a packet to preview its RGB patch.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(220, 140)
        self.preview.setMaximumHeight(220)
        self.preview.setWordWrap(True)
        panel_layout.addWidget(self.preview)
        self.packet_details = QLabel("")
        self.packet_details.setWordWrap(True)
        panel_layout.addWidget(self.packet_details)
        form = QFormLayout()
        self.name = QLineEdit("terrain_dataset")
        form.addRow("Dataset name", self.name)
        destination_bar = QHBoxLayout()
        self.destination = QLineEdit()
        self.destination.setPlaceholderText("Choose an existing output folder")
        destination_bar.addWidget(self.destination)
        self.destination_button = QPushButton("Choose…")
        self.destination_button.clicked.connect(self.choose_destination)
        destination_bar.addWidget(self.destination_button)
        form.addRow("Save location", destination_bar)
        self.band_width = QDoubleSpinBox()
        self.band_width.setRange(0.1, 10000)
        self.band_width.setDecimals(2)
        self.band_width.setSuffix(" Hz")
        self.band_width.setValue(10.0)
        self.band_width.setToolTip(
            "Uniform bands; absolute PSD integration with the zero-frequency bin excluded."
        )
        self.band_width.valueChanged.connect(self.update_summary)
        form.addRow("Band width", self.band_width)
        panel_layout.addLayout(form)
        self.band_preview = QLabel("Bands will be shown after selecting packets.")
        self.band_preview.setWordWrap(True)
        panel_layout.addWidget(self.band_preview)
        self.full_frames = QCheckBox("Include full RGB and depth frames")
        self.point_maps = QCheckBox("Include numerical point maps")
        for checkbox in (self.full_frames, self.point_maps):
            checkbox.toggled.connect(self.update_summary)
            panel_layout.addWidget(checkbox)
        panel_layout.addWidget(
            QLabel("Splits remain unassigned. Every selected packet is retained.")
        )
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        panel_layout.addWidget(self.summary, 1)
        self.export_button = QPushButton("Export selected packets")
        self.export_button.clicked.connect(self.start_export)
        panel_layout.addWidget(self.export_button)
        splitter.addWidget(panel)
        splitter.setSizes([780, 470])

        progress_bar = QHBoxLayout()
        self.progress = QProgressBar()
        progress_bar.addWidget(self.progress, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)
        progress_bar.addWidget(self.cancel_button)
        outer.addLayout(progress_bar)
        self.status = QLabel("Add a source folder to begin.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.status)
        self.update_summary()
        if directory is not None:
            QTimer.singleShot(0, lambda: self.scan(directory))

    def choose_source(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose recordings or a derived dataset")
        if directory:
            self.scan(Path(directory))

    def choose_destination(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Choose export destination", self.destination.text()
        )
        if directory:
            self.destination.setText(directory)

    def scan(self, directory: Path):
        self.status.setText(f"Inspecting {directory}")
        self.progress.setRange(0, 0)
        self.start_task(lambda progress, cancel: discover_sources(directory, cancel))

    def start_task(self, action):
        if self.task is not None:
            return
        self.controls.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.task = Task(action, self)
        self.task.progress.connect(self.report_progress)
        self.task.completed.connect(self.task_completed)
        self.task.failed.connect(self.task_failed)
        self.task.cancelled.connect(
            lambda: self.status.setText("Cancelled. Incomplete export files were removed.")
        )
        self.task.finished.connect(self.task_finished)
        self.task.start()

    def report_progress(self, percent, message):
        self.progress.setRange(0, 100)
        self.progress.setValue(percent)
        self.status.setText(message)

    def task_completed(self, result):
        if isinstance(result, ExportResult):
            lines = [f"Exported {result.packet_count} packets to:\n{result.directory}\n"]
            for run in result.runs:
                lines.append(
                    f"{run['experiment_id']}/run_{run['run_number']}: {run['selected_packet_count']} selected; "
                    f"maximum non-overlapping: {run['maximum_temporal_nonoverlapping_count']} temporal, "
                    f"{run['maximum_spatial_nonoverlapping_count']} spatial."
                )
            self.summary.setPlainText("\n".join(lines))
            self.status.setText(f"Export complete: {result.directory}")
            return
        sources, warnings = result
        selected = {packet.export_id for packet in self.selected_packets()}
        for source in sources:
            self.sources[source.root] = source
        self.populate_tree(selected)
        self.status.setText(
            f"Loaded {len(sources)} source datasets; {len(warnings)} discovery warnings."
        )
        if warnings:
            message = QMessageBox(self)
            message.setWindowTitle("Some source folders could not be loaded")
            message.setText(
                "Usable datasets are available. Expand details to inspect skipped folders."
            )
            message.setDetailedText("\n".join(warnings))
            message.exec()

    def task_failed(self, message):
        self.status.setText(f"Could not complete the operation: {message}")
        if not self._close_when_done:
            QMessageBox.critical(self, "Operation could not complete", message)

    def task_finished(self):
        task = self.task
        self.task = None
        task.deleteLater()
        self.controls.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 100)
        if self._close_when_done:
            self.close()

    def cancel_task(self):
        if self.task is not None:
            self.task.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling and cleaning up temporary output…")

    def clear_sources(self):
        self.sources.clear()
        self.populate_tree(set())

    def populate_tree(self, selected: set[str]):
        self._size_estimates.clear()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.packet_items.clear()
        experiments, runs = {}, {}
        for source in sorted(
            self.sources.values(),
            key=lambda item: (
                item.experiment_id,
                item.run_number,
                item.processing_id,
                str(item.root),
            ),
        ):
            if source.experiment_id not in experiments:
                experiments[source.experiment_id] = self.group_item(self.tree, source.experiment_id)
            if source.run_key not in runs:
                runs[source.run_key] = self.group_item(
                    experiments[source.experiment_id], f"Run {source.run_number}"
                )
            title = f"{source.processing_id} ({len(source.packets)} packets)"
            source_item = self.group_item(runs[source.run_key], title)
            source_item.setToolTip(0, str(source.root))
            for packet in source.packets:
                terrain_class = (
                    "Unlabelled"
                    if packet.label is None
                    else f"{packet.label}: {CLASS_NAMES[packet.label]}"
                    if packet.label in range(4)
                    else "Invalid label"
                )
                item = QTreeWidgetItem(
                    source_item,
                    [
                        f"{packet.number}: {packet.packet_id}",
                        terrain_class,
                        str(packet.features.get("classification_method") or "—"),
                        "Blocked" if packet.problem else "Ready",
                    ],
                )
                item.setData(0, PACKET_ROLE, packet)
                item.setToolTip(0, packet.packet_id)
                item.setToolTip(3, packet.problem or "FFT trajectory coverage available")
                if not packet.problem:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if packet.export_id in selected
                        else Qt.CheckState.Unchecked,
                    )
                else:
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.packet_items.append(item)
        self.refresh_group_states()
        self.tree.expandToDepth(2)
        self.tree.blockSignals(False)
        self.apply_filters()

    @staticmethod
    def group_item(parent, title):
        item = QTreeWidgetItem(parent, [title])
        # Automatic tri-state propagation cannot handle blocked children with
        # no checkbox. Manage group states using only exportable descendants.
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setToolTip(
            3,
            "Select all exportable packets in this group, including filtered packets. Blocked packets are skipped.",
        )
        item.setCheckState(0, Qt.CheckState.Unchecked)
        return item

    def selection_changed(self, item, column):
        """Apply group toggles once, leaving blocked packet checkboxes absent."""
        if column != 0:
            return

        def set_packets(node, state):
            packet = node.data(0, PACKET_ROLE)
            if isinstance(packet, Packet):
                if not packet.problem:
                    node.setCheckState(0, state)
                return
            for index in range(node.childCount()):
                set_packets(node.child(index), state)

        self.tree.blockSignals(True)
        try:
            if not isinstance(item.data(0, PACKET_ROLE), Packet):
                set_packets(item, item.checkState(0))
            self.refresh_group_states()
        finally:
            self.tree.blockSignals(False)
        self.summary_timer.start()

    def refresh_group_states(self):
        """Reflect eligible descendants only; callers temporarily block signals."""

        def count_packets(item):
            packet = item.data(0, PACKET_ROLE)
            if isinstance(packet, Packet):
                if packet.problem:
                    return 0, 0
                return 1, int(item.checkState(0) == Qt.CheckState.Checked)

            eligible = checked = 0
            for index in range(item.childCount()):
                child_eligible, child_checked = count_packets(item.child(index))
                eligible += child_eligible
                checked += child_checked
            if eligible:
                state = (
                    Qt.CheckState.Checked
                    if checked == eligible
                    else Qt.CheckState.PartiallyChecked
                    if checked
                    else Qt.CheckState.Unchecked
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, state)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setData(0, Qt.ItemDataRole.CheckStateRole, None)
            return eligible, checked

        for index in range(self.tree.topLevelItemCount()):
            count_packets(self.tree.topLevelItem(index))

    def apply_filters(self):
        search = self.search.text().strip().casefold()
        category = self.class_filter.currentData()
        for item in self.packet_items:
            packet = item.data(0, PACKET_ROLE)
            haystack = f"{packet.source.experiment_id} run_{packet.source.run_number} {packet.source.processing_id} {packet.packet_id} {packet.problem} {item.text(3)}"
            matches = (
                category == "all"
                or category == packet.label
                or (category == "unlabelled" and packet.label is None)
            )
            item.setHidden(not matches or search not in haystack.casefold())

        def hide_empty(item):
            if item.childCount():
                for index in range(item.childCount()):
                    hide_empty(item.child(index))
                item.setHidden(all(item.child(i).isHidden() for i in range(item.childCount())))

        for index in range(self.tree.topLevelItemCount()):
            hide_empty(self.tree.topLevelItem(index))
        self.update_summary()

    def select_visible(self, checked):
        self.tree.blockSignals(True)
        for item in self.packet_items:
            packet = item.data(0, PACKET_ROLE)
            if not item.isHidden() and not packet.problem:
                item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.refresh_group_states()
        self.tree.blockSignals(False)
        self.update_summary()

    def selected_packets(self) -> list[Packet]:
        return [
            item.data(0, PACKET_ROLE)
            for item in self.packet_items
            if item.checkState(0) == Qt.CheckState.Checked and not item.data(0, PACKET_ROLE).problem
        ]

    def update_summary(self, *args):
        self.summary_timer.stop()
        selected = self.selected_packets()
        counts = Counter(
            "Unlabelled" if packet.label is None else CLASS_NAMES[packet.label]
            for packet in selected
        )
        run_sources = {}
        for packet in selected:
            run_sources.setdefault(packet.source.run_key, set()).add(packet.source.root)
        conflicts = any(len(roots) > 1 for roots in run_sources.values())
        lines = [f"{len(selected)} selected packets from {len(run_sources)} runs."]
        lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))
        blocked = sum(bool(item.data(0, PACKET_ROLE).problem) for item in self.packet_items)
        if blocked:
            lines.append(f"\n{blocked} source packets blocked. Select a packet for the reason.")
        if any(packet.source.terrain[0] is None for packet in selected):
            lines.append(
                "\nSome packets have no terrain/light ID; their grouping fields will be null."
            )
        if conflicts:
            lines.append("\nChoose only one processing version/source per experiment and run.")
        # This is an input-size estimate, not a promise about PNG/Parquet ratios.
        size = 0
        for packet in selected:
            key = (
                packet.source.root,
                packet.number,
                self.full_frames.isChecked(),
                self.point_maps.isChecked(),
            )
            if key in self._size_estimates:
                size += self._size_estimates[key]
                continue
            packet_size = 0
            paths = [
                *packet.directory.glob("stage2/*"),
                packet.directory / "telemetry.npz",
                packet.directory / "geometry.npz",
            ]
            for path in paths:
                if path.name == "point_map.npz" and not self.point_maps.isChecked():
                    continue
                if path.is_file():
                    packet_size += path.stat().st_size
            if self.full_frames.isChecked():
                packet_size += sum(
                    path.stat().st_size
                    for path in (packet.directory / "rgb.npy", packet.directory / "depth.npy")
                    if path.is_file()
                )
            self._size_estimates[key] = packet_size
            size += packet_size
        if selected:
            lines.append(
                f"\nApproximately {size / 1024**2:.1f} MiB of source data; exported size depends on compression."
            )
            maximum = max(packet.source.sample_rate_hz / 2 for packet in selected)
            bands = frequency_bands(self.band_width.value(), maximum)
            text = ", ".join(f"{low:g}–{high:g}" for low, high in bands[:8])
            self.band_preview.setText(
                f"{len(bands)} bands: {text}{' …' if len(bands) > 8 else ''} Hz. DC excluded."
            )
        else:
            self.band_preview.setText("Select packets to preview frequency bands.")
        self.summary.setPlainText("\n".join(lines))
        self.export_button.setEnabled(bool(selected) and not conflicts)

    def show_preview(self, item, previous=None):
        packet = item.data(0, PACKET_ROLE) if item else None
        if not isinstance(packet, Packet):
            self.preview.clear()
            self.packet_details.clear()
            return
        self.packet_details.setText(
            packet.problem
            or (
                f"Packet {packet.number} · {packet.source.experiment_id}/run_{packet.source.run_number}\n"
                f"Terrain/light: {packet.source.terrain[0] or 'unavailable'}"
            )
        )
        try:
            image = QImage.fromData(png_image(packet.directory / "stage2/rgb_patch.npy")["bytes"])
            self.preview.setPixmap(
                QPixmap.fromImage(image).scaled(320, 200, Qt.AspectRatioMode.KeepAspectRatio)
            )
        except (OSError, ValueError):
            self.preview.setText("RGB patch preview unavailable.")

    def start_export(self):
        selected = self.selected_packets()
        destination = Path(self.destination.text().strip())
        if not self.destination.text().strip() or not destination.is_dir():
            QMessageBox.warning(
                self, "Choose a saving location", "Choose an existing output directory first."
            )
            return
        options = ExportOptions(
            self.band_width.value(), self.full_frames.isChecked(), self.point_maps.isChecked()
        )
        name = self.name.text()
        self.start_task(
            lambda progress, cancel: export_dataset(
                selected, destination, name, options, progress, cancel
            )
        )

    def closeEvent(self, event: QCloseEvent):
        if self.task is not None:
            self._close_when_done = True
            self.cancel_task()
            event.ignore()
        else:
            event.accept()
