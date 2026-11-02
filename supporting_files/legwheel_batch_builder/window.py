"""A small wizard for selecting experiments, runs and shared processing settings."""

from collections import Counter
from dataclasses import fields
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from legwheel_dataset.config import ProcessingConfig
from legwheel_dataset.config_creator import FIELD_HELP
from legwheel_dataset.terrain import review_terrain_ids

from .model import (
    BuildResult, ExperimentInfo, RunInfo, discover_experiments, inspect_run, mark_run_valid,
)
from .runner import BatchRunner
from .terrain_ids import TerrainIdsDialog


SETTING_LABELS = {
    "rgb_depth_max_delta_ms": "RGB/depth maximum separation (ms)",
    "spectral_window_ms": "Spectral window (ms)",
    "telemetry_resample_hz": "Telemetry sampling rate (Hz)",
    "maximum_interpolation_gap_ms": "Maximum interpolation gap (ms)",
    "maximum_pitch_uncertainty_deg": "Pitch uncertainty diagnostic threshold (degrees)",
    "alpha_off_rad": "Patch centre angular offset (rad)",
    "alpha_sp_rad": "Patch angular width (rad)",
    "grouser_effective_height_m": "Effective grouser height (m)",
    "minimum_target_torque_fraction": "Minimum target torque fraction",
    "source_rate_tolerance_fraction": "Source rate tolerance fraction",
    "track_point_count": "Track boundary point count",
    "depth_scale_m": "Depth scale (m per integer unit)",
    "minimum_plane_points": "Minimum plane points",
    "plane_ransac_iterations": "Plane RANSAC iterations",
    "plane_inlier_threshold_m": "Plane inlier threshold (m)",
    "minimum_spectral_valid_fraction": "Minimum valid spectral fraction",
}


def label(text: str, *, bold: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setWordWrap(True)
    font = widget.font()
    font.setBold(bold)
    widget.setFont(font)
    return widget


class RunSelectionPage(QWidget):
    """One experiment's run checklist, including immediate validity editing."""

    def __init__(self, experiment: ExperimentInfo, selected: set[Path]) -> None:
        super().__init__()
        self.experiment = experiment
        layout = QVBoxLayout(self)
        layout.addWidget(label(f"Select runs — {experiment.directory.name}", bold=True))
        layout.addWidget(label(
            f"Experiment ID: {experiment.experiment_id}. Invalid and unreviewed runs "
            "can be selected. Runs with missing inputs cannot be selected."
        ))
        controls = QHBoxLayout()
        select_all, clear_all = QPushButton("Select all available"), QPushButton("Clear all")
        select_all.clicked.connect(lambda: self._select_all(True))
        clear_all.clicked.connect(lambda: self._select_all(False))
        controls.addWidget(select_all)
        controls.addWidget(clear_all)
        controls.addStretch()
        layout.addLayout(controls)
        self.table = QTableWidget(len(experiment.runs), 5)
        self.table.setHorizontalHeaderLabels(("Include", "Run", "Datasets", "Warnings", "Validity"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for row, run in enumerate(experiment.runs):
            self._render_row(row, run.directory in selected)
        layout.addWidget(self.table, 1)
        self.feedback = label("")
        layout.addWidget(self.feedback)

    def _render_row(self, row: int, selected: bool) -> None:
        run = self.experiment.runs[row]
        checkbox = QTableWidgetItem()
        checkbox.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        checkbox.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
        if run.problems:
            checkbox.setFlags(Qt.ItemFlag.NoItemFlags)
            checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.table.setItem(row, 0, checkbox)
        self.table.setItem(row, 1, QTableWidgetItem(run.directory.name))
        self.table.setItem(row, 2, QTableWidgetItem(run.counts.label()))
        warnings = "; ".join([*run.warnings, *run.problems]) or "None"
        item = QTableWidgetItem(warnings)
        item.setToolTip(warnings)
        self.table.setItem(row, 3, item)
        button = QPushButton("Valid" if run.valid is True else "Mark valid")
        button.setEnabled(run.valid is not True and (run.directory / "run_config.yaml").is_file())
        button.setToolTip("Immediately save run_valid: true in this run's run_config.yaml.")
        button.clicked.connect(lambda _checked=False, index=row: self._mark_valid(index))
        self.table.setCellWidget(row, 4, button)
        self.table.setRowHeight(row, 64)

    def _select_all(self, checked: bool) -> None:
        for row, run in enumerate(self.experiment.runs):
            if not run.problems:
                self.table.item(row, 0).setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                )

    def _mark_valid(self, row: int) -> None:
        selected = self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        try:
            run = mark_run_valid(self.experiment.runs[row])
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot update run validity", str(error))
            return
        self.experiment.runs[row] = run
        self._render_row(row, selected)
        self.feedback.setText(f"Saved {run.directory.name}: run_valid: true.")

    def selected_runs(self) -> list[RunInfo]:
        return [
            run for row, run in enumerate(self.experiment.runs)
            if not run.problems and self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        ]


class SettingsPage(QWidget):
    """Edit the actual ProcessingConfig defaults, without a second schema."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(label("Shared processing settings", bold=True))
        self.summary = label("")
        layout.addWidget(self.summary)
        layout.addWidget(label(
            "Start batch saves this configuration as postprocess.yaml in every selected run, "
            "replacing existing settings. Analysis uses each run's recorded experiment snapshot. "
            "Hover over a setting for its explanation."
        ))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.inputs: dict[str, QLineEdit] = {}
        defaults = ProcessingConfig()
        for setting in fields(defaults):
            if setting.name == "schema_version":
                continue
            editor = QLineEdit(str(getattr(defaults, setting.name)))
            editor.setMaximumWidth(230)
            editor.setToolTip(setting.name + "\n" + " ".join(FIELD_HELP[setting.name]))
            self.inputs[setting.name] = editor
            form.addRow(SETTING_LABELS[setting.name], editor)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, 1)
        restore = QPushButton("Restore defaults")
        restore.clicked.connect(self.restore_defaults)
        layout.addWidget(restore)

    def restore_defaults(self) -> None:
        defaults = ProcessingConfig()
        for name, editor in self.inputs.items():
            editor.setText(str(getattr(defaults, name)))

    def configuration(self) -> ProcessingConfig:
        defaults, values = ProcessingConfig(), {}
        for name, editor in self.inputs.items():
            try:
                convert = int if isinstance(getattr(defaults, name), int) else float
                values[name] = convert(editor.text().strip())
            except ValueError as error:
                editor.setFocus()
                raise ValueError(f"Enter a valid number for {SETTING_LABELS[name]}.") from error
        config = ProcessingConfig(**values)
        config.validate()
        return config


class BatchBuilderWindow(QMainWindow):
    """Navigate selection pages, then show live progress and a per-run summary."""

    def __init__(self, directory: Path | None = None, ros_python: str = "/usr/bin/python3") -> None:
        super().__init__()
        self.setWindowTitle("LegWheel Batch Dataset Builder")
        self.resize(1200, 820)
        self.setMinimumSize(1000, 680)
        self.root: Path | None = None
        self.run_pages: list[RunSelectionPage] = []
        self._close_after_cancel = False
        self.runner = BatchRunner(ros_python, self)
        self.runner.progress.connect(self._show_progress)
        self.runner.result_ready.connect(self._show_result)
        self.runner.run_started.connect(self._show_started)
        self.runner.log.connect(self._append_log)
        self.runner.finished.connect(self._batch_finished)
        central = QWidget()
        outer = QVBoxLayout(central)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.experiments_page = self._build_experiments_page()
        self.settings_page = SettingsPage()
        self.progress_page = self._build_progress_page()
        for page in (self.experiments_page, self.settings_page, self.progress_page):
            self.stack.addWidget(page)
        navigation = QHBoxLayout()
        self.back_button, self.next_button = QPushButton("Back"), QPushButton("Next")
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        navigation.addWidget(self.back_button)
        self.step_label = label("")
        navigation.addWidget(self.step_label, 1)
        navigation.addWidget(self.next_button)
        outer.addLayout(navigation)
        self.feedback = label("")
        outer.addWidget(self.feedback)
        self.setCentralWidget(central)
        self.flow = [self.experiments_page, self.settings_page]
        self._show_step(0)
        if directory is not None:
            self.load_directory(directory)

    def _build_experiments_page(self) -> QWidget:
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        layout.addWidget(label("Select experiments", bold=True))
        layout.addWidget(label("Choose the directory whose immediate subfolders contain experiments."))
        choose = QPushButton("Choose experiments directory…")
        choose.clicked.connect(self._choose_directory)
        layout.addWidget(choose)
        self.directory_label = label("No directory selected")
        layout.addWidget(self.directory_label)
        controls = QHBoxLayout()
        for text, checked in (("Select all available", True), ("Clear all", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _clicked=False, value=checked: self._select_experiments(value))
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        self.experiment_list = QListWidget()
        layout.addWidget(self.experiment_list, 1)
        self.discovery_feedback = label("")
        layout.addWidget(self.discovery_feedback)
        return page

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose experiments directory", str(self.root or Path.home()),
        )
        if selected:
            self.load_directory(Path(selected))

    def load_directory(self, directory: Path) -> None:
        try:
            experiments, warnings = discover_experiments(directory.expanduser().resolve())
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot inspect experiments", str(error))
            return
        self.root = directory.expanduser().resolve()
        self.directory_label.setText(str(self.root))
        self.experiment_list.clear()
        for experiment in experiments:
            item = QListWidgetItem(
                f"{experiment.directory.name}  ·  {experiment.experiment_id}  ·  "
                f"{len(experiment.runs)} runs"
            )
            item.setData(Qt.ItemDataRole.UserRole, experiment)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            if not experiment.runs:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setText(item.text() + "  (no recorded runs)")
            elif all(run.problems for run in experiment.runs):
                item.setText(item.text() + "  (inputs missing; open to inspect)")
            self.experiment_list.addItem(item)
        message = (
            f"Found {len(experiments)} experiment folders." if experiments
            else "No experiment configurations found."
        )
        if warnings:
            message += f" {len(warnings)} discovery warnings (hover to read)."
        self.discovery_feedback.setText(message)
        self.discovery_feedback.setToolTip("\n".join(warnings))

    def _select_experiments(self, checked: bool) -> None:
        for index in range(self.experiment_list.count()):
            item = self.experiment_list.item(index)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _show_step(self, index: int) -> None:
        self.step = index
        self.stack.setCurrentWidget(self.flow[index])
        self.back_button.setEnabled(index > 0)
        self.next_button.setText("Start batch" if self.flow[index] is self.settings_page else "Next")
        self.step_label.setText(f"Step {index + 1} of {len(self.flow)}")
        self.feedback.clear()

    def _back(self) -> None:
        if self.step > 0:
            self._show_step(self.step - 1)

    def _next(self) -> None:
        page = self.flow[self.step]
        if page is self.experiments_page:
            experiments = [
                self.experiment_list.item(index).data(Qt.ItemDataRole.UserRole)
                for index in range(self.experiment_list.count())
                if self.experiment_list.item(index).checkState() == Qt.CheckState.Checked
            ]
            if not experiments:
                self.feedback.setText("Select at least one experiment.")
                return
            if not self._resolve_terrain_ids(experiments):
                return
            selected = {run.directory for old in self.run_pages for run in old.selected_runs()}
            for old in self.run_pages:
                self.stack.removeWidget(old)
                old.deleteLater()
            self.run_pages = [RunSelectionPage(experiment, selected) for experiment in experiments]
            for run_page in self.run_pages:
                self.stack.addWidget(run_page)
            self.flow = [self.experiments_page, *self.run_pages, self.settings_page]
        elif isinstance(page, RunSelectionPage) and not page.selected_runs():
            self.feedback.setText("Select at least one run, or go back and remove this experiment.")
            return
        elif page is self.settings_page:
            self._start_batch()
            return
        self._show_step(self.step + 1)
        if self.flow[self.step] is self.settings_page:
            runs = [run for run_page in self.run_pages for run in run_page.selected_runs()]
            replacements = sum((run.directory / "postprocess.yaml").exists() for run in runs)
            experiment_word = "experiment" if len(self.run_pages) == 1 else "experiments"
            run_word = "run" if len(runs) == 1 else "runs"
            self.settings_page.summary.setText(
                f"{len(runs)} selected {run_word} across {len(self.run_pages)} {experiment_word}. "
                f"{replacements} existing postprocess.yaml files will be replaced."
            )

    def _resolve_terrain_ids(self, experiments: list[ExperimentInfo]) -> bool:
        """Resolve experiment-wide metadata before any runs are chosen."""
        try:
            reviews = [review_terrain_ids(experiment.directory) for experiment in experiments]
            if any(not review.documents for review in reviews):
                raise ValueError("An experiment's metadata is no longer available. Reload the directory.")
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot review terrain and lighting IDs", str(error))
            return False
        missing = [review for review in reviews if review.needs_resolution]
        if missing:
            dialog = TerrainIdsDialog(missing, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
        # Saving IDs may mark datasets in any run as requiring regeneration.
        for experiment in experiments:
            experiment.runs = [
                inspect_run(run.directory, experiment.directory.name) for run in experiment.runs
            ]
        return True

    def _build_progress_page(self) -> QWidget:
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        self.batch_status = label("Batch progress", bold=True)
        layout.addWidget(self.batch_status)
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 1000)
        layout.addWidget(self.overall_progress)
        self.eta_label = label("Estimated time remaining: Estimating…")
        layout.addWidget(self.eta_label)
        self.current_run_label = label("", bold=True)
        layout.addWidget(self.current_run_label)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 1000)
        layout.addWidget(self.run_progress)
        self.phase_label = label("")
        layout.addWidget(self.phase_label)
        self.summary_table = QTableWidget(0, 4)
        self.summary_table.setHorizontalHeaderLabels(("Run", "Result", "Details", "Output dataset"))
        self.summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.summary_table, 1)
        show_log = QCheckBox("Show processing log")
        layout.addWidget(show_log)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMaximumHeight(150)
        self.log_view.hide()
        show_log.toggled.connect(self.log_view.setVisible)
        layout.addWidget(self.log_view)
        buttons = QHBoxLayout()
        self.cancel_button, self.new_batch_button = QPushButton("Cancel batch"), QPushButton("New batch")
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.new_batch_button.clicked.connect(self._new_batch)
        self.new_batch_button.setEnabled(False)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch()
        buttons.addWidget(self.new_batch_button)
        layout.addLayout(buttons)
        return page

    def _start_batch(self) -> None:
        try:
            config = self.settings_page.configuration()
        except ValueError as error:
            self.feedback.setText(str(error))
            return
        runs = [run for page in self.run_pages for run in page.selected_runs()]
        self._summary_rows = {run.directory: row for row, run in enumerate(runs)}
        self.summary_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            for column, text in enumerate((run.label, "Pending", "", "")):
                self.summary_table.setItem(row, column, QTableWidgetItem(text))
        self.log_view.clear()
        self.batch_status.setText("Batch in progress")
        self.cancel_button.setEnabled(True)
        self.new_batch_button.setEnabled(False)
        self.stack.setCurrentWidget(self.progress_page)
        self.back_button.hide()
        self.next_button.hide()
        self.step_label.clear()
        self.feedback.clear()
        self.runner.start(runs, config)

    def _show_progress(self, overall: float, current: float, run: str, phase: str, eta: str) -> None:
        self.overall_progress.setValue(round(overall * 1000))
        self.run_progress.setValue(round(current * 1000))
        self.current_run_label.setText(run)
        self.phase_label.setText(phase)
        self.eta_label.setText(f"Estimated time remaining: {eta}")

    def _show_result(self, result: BuildResult) -> None:
        row = self._summary_rows[result.run.directory]
        for column, text in enumerate((
            result.status.replace("_", " ").title(), result.message, str(result.output or ""),
        ), start=1):
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            self.summary_table.setItem(row, column, item)

    def _show_started(self, run: RunInfo) -> None:
        row = self._summary_rows[run.directory]
        self.summary_table.item(row, 1).setText("In progress")

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _batch_finished(self, results: list[BuildResult], cancelled: bool) -> None:
        counts = Counter(result.status for result in results)
        title = "Batch cancelled" if cancelled else "Batch finished"
        self.batch_status.setText(title + " — " + ", ".join(
            f"{count} {status.replace('_', ' ')}" for status, count in counts.items()
        ))
        self.cancel_button.setEnabled(False)
        self.new_batch_button.setEnabled(True)
        if self._close_after_cancel:
            self.close()

    def _new_batch(self) -> None:
        if self.root is not None:
            self.load_directory(self.root)
        self.back_button.show()
        self.next_button.show()
        self._show_step(0)

    def closeEvent(self, event) -> None:
        if self.runner.running:
            event.ignore()
            self._close_after_cancel = True
            self.runner.cancel()
        else:
            event.accept()
