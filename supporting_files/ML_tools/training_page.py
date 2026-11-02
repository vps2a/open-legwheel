"""Coordinate training controls, worker lifecycle and experiment save/reopen."""

from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .experiments import load_experiment, read_experiment_manifest, save_experiment
from .features import (
    PARTITIONS,
    ConfigurationError,
    ExperimentConfig,
    prepare_data,
    require_trainable,
)
from .plot_export import DEFAULT_EXPORT_DPI
from .training import TrainingCancelled, evaluate_test, train_experiment
from .training_controls import FeaturePanel, SplitPanel
from .training_results import ResultsPanel


class TrainingTask(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action

    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise TrainingCancelled()

    def run(self):
        try:
            result = self.action(
                progress=self.progress.emit, check_cancel=self._check_cancel
            )
            self._check_cancel()
            self.completed.emit(result)
        except TrainingCancelled:
            pass
        except Exception as error:  # noqa: BLE001 -- report failures at the worker boundary.
            self.failed.emit(str(error))


class TrainingPage(QWidget):
    dataset_requested = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.experiment = None
        self.task = None
        self.pending_experiment = None
        self.restoring = False
        self.stale = False
        self.closing = False
        self.export_dpi = DEFAULT_EXPORT_DPI
        self._build_interface()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh_summary)
        self.split.changed.connect(self.settings_changed)
        self.features.changed.connect(self.settings_changed)
        self.refresh_summary()

    def _build_interface(self):
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.open_button = QPushButton("Open experiment…")
        self.open_button.setToolTip(
            "Open your own saved experiment or another trusted source; joblib files are Python objects."
        )
        self.save_button = QPushButton("Save experiment…")
        self.train_button = QPushButton("Train all three models")
        self.test_button = QPushButton("Evaluate reserved test set")
        self.cancel_button = QPushButton("Cancel after current model")
        self.cancel_button.hide()
        for button in (
            self.open_button,
            self.save_button,
            self.train_button,
            self.test_button,
            self.cancel_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.status = QLabel("Assign training and validation runs to begin.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        body = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.split = SplitPanel()
        self.features = FeaturePanel()
        self.results = ResultsPanel()
        for widget, title in (
            (self.split, "1. Data split"),
            (self.features, "2. Features and models"),
            (self.results, "3. Performance"),
        ):
            self.tabs.addTab(widget, title)
        body.addWidget(self.tabs)
        summary_box = QGroupBox("Training selection")
        summary_layout = QVBoxLayout(summary_box)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        summary_layout.addWidget(self.summary)
        summary_box.setMinimumWidth(255)
        body.addWidget(summary_box)
        body.setStretchFactor(0, 5)
        body.setStretchFactor(1, 1)
        body.setSizes((1220, 300))
        layout.addWidget(body, 1)
        self.train_button.clicked.connect(self.train)
        self.test_button.clicked.connect(self.test)
        self.cancel_button.clicked.connect(self.cancel)
        self.save_button.clicked.connect(self.choose_save)
        self.open_button.clicked.connect(self.choose_open)

    @property
    def is_busy(self):
        return self.task is not None

    def set_export_dpi(self, dpi: int):
        # Export resolution is a presentation setting, not a model setting:
        # changing it must not mark an otherwise valid experiment as outdated.
        self.export_dpi = dpi
        self.results.toolbar.export_dpi = dpi

    def set_dataset(self, dataset):
        self.restoring = True
        self.dataset = dataset
        self.experiment = None
        self.stale = False
        self.results.clear()
        self.split.set_dataset(dataset)
        self.features.set_dataset(dataset)
        self.restoring = False
        self.refresh_summary()
        if self.pending_experiment is not None:
            pending, self.pending_experiment = self.pending_experiment, None
            self.restore_experiment(pending)

    def config(self):
        return ExperimentConfig(
            self.features.settings(),
            self.split.assignments(),
            self.split.mode_combo.currentData(),
            self.features.linear_c.value(),
            self.features.rbf_c.value(),
            "scale"
            if self.features.gamma_auto.isChecked()
            else self.features.gamma.value(),
        )

    def settings_changed(self):
        if self.restoring:
            return
        if self.experiment is not None:
            self.stale = True
            self.results.set_stale(True)
            self.status.setText(
                "Settings changed. Existing results are outdated; train again to compare this configuration."
            )
        self.refresh_timer.start()

    def refresh_summary(self):
        self.refresh_timer.stop()
        ready = False
        self.save_button.setEnabled(
            self.experiment is not None and not self.stale and not self.is_busy
        )
        self.test_button.setEnabled(False)
        if self.dataset is None:
            self.summary.setPlainText(
                "Open an exported dataset first.\n\nTraining uses only classes 0 (firm) and 1 (loose)."
            )
            self.train_button.setEnabled(False)
            return
        lines = [
            "Positive class: 1 — Loose",
            "Class 0: Firm",
            "",
            f"Selected features: {len(self.features.settings().features)}",
        ]
        try:
            config = self.config()
            data = prepare_data(self.dataset, config)
            for role in PARTITIONS:
                assigned, mask = data.assigned[role], data.masks[role]
                missing = int(np.count_nonzero(assigned & data.missing_features))
                transition = int(
                    np.count_nonzero(
                        assigned & ~data.missing_features & data.transition_excluded
                    )
                )
                labels = self.dataset.class_ids[mask]
                runs = {
                    source
                    for source, keep in zip(self.dataset.source_ids, mask)
                    if keep
                }
                setups = {
                    key[0] for key, keep in zip(self.dataset.terrain_keys, mask) if keep
                }
                lines += [
                    "",
                    f"{role.title()}: {int(mask.sum())} eligible packets",
                    f"Firm: {int((labels == 0).sum())}  |  Loose: {int((labels == 1).sum())}",
                    f"Runs: {len(runs)}  |  Setups: {len(setups)}",
                    f"Incomplete features excluded: {missing}",
                    f"Transition-adjacent excluded: {transition}",
                ]
            try:
                require_trainable(self.dataset, data)
                ready = True
            except ConfigurationError as error:
                lines += ["", str(error)]
            if self.experiment is not None and not self.stale:
                reserved = np.any(self.experiment.prepared.masks["testing"])
                self.test_button.setEnabled(
                    reserved
                    and "testing" not in self.experiment.evaluations
                    and not self.is_busy
                )
        except (ValueError, KeyError) as error:
            lines += ["", str(error)]
        lines += [
            "",
            "All models use identical eligible packets.",
            "Scaler and log floors are fitted on training only.",
            "STD and speed use the selected window.",
            "Transition annotations are checked within each run.",
            "Unlabelled/transition classes are not targets.",
            "",
            "Test results stay hidden until explicit evaluation.",
        ]
        self.summary.setPlainText("\n".join(lines))
        self.train_button.setEnabled(ready and not self.is_busy)

    def train(self):
        if self.is_busy or self.dataset is None:
            return
        self._start_task(partial(train_experiment, self.dataset, self.config()))

    def test(self):
        if self.experiment is None or self.stale or self.is_busy:
            return
        self._start_task(partial(evaluate_test, self.experiment))

    def _start_task(self, action):
        self.task = TrainingTask(action, self)
        self.task.completed.connect(self._completed)
        self.task.failed.connect(self._failed)
        self.task.progress.connect(self.status.setText)
        self.task.finished.connect(self._finished)
        self.split.setEnabled(False)
        self.features.setEnabled(False)
        self.open_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.busy_changed.emit(True)
        self.refresh_summary()
        self.task.start()

    def _completed(self, experiment):
        if self.closing or (
            self.task is not None and self.task.isInterruptionRequested()
        ):
            return
        self.experiment = experiment
        self.stale = False
        self.results.set_experiment(experiment)
        if "testing" in experiment.evaluations:
            self.results.partition.setCurrentIndex(
                self.results.partition.findData("testing")
            )
        self.tabs.setCurrentWidget(self.results)
        self.status.setText(
            "Comparison complete. Models were fitted on training packets only; no refitting was performed."
        )

    def _failed(self, message):
        self.status.setText(message)
        if not self.closing:
            QMessageBox.critical(self, "Model comparison failed", message)

    def _finished(self):
        task, self.task = self.task, None
        task.deleteLater()
        self.split.setEnabled(True)
        self.features.setEnabled(True)
        self.open_button.setEnabled(True)
        self.cancel_button.hide()
        if task.isInterruptionRequested():
            self.status.setText(
                "Cancelled. No partial experiment replaced the previous results."
            )
        self.refresh_summary()
        self.busy_changed.emit(False)

    def cancel(self):
        if self.task:
            self.task.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current model operation…")

    def cancel_for_close(self):
        self.closing = True
        self.cancel()

    def choose_save(self):
        if self.experiment is None or self.stale or self.is_busy:
            return
        destination = QFileDialog.getExistingDirectory(
            self, "Choose a parent folder for this saved experiment"
        )
        if not destination:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = save_experiment(
                self.experiment,
                Path(destination),
                export_dpi=self.export_dpi,
            )
            self.status.setText(f"Saved exact evaluated models and results to {path}")
        except Exception as error:  # noqa: BLE001 -- report file/serialization errors at the GUI boundary.
            QMessageBox.critical(self, "Cannot save experiment", str(error))
        finally:
            QApplication.restoreOverrideCursor()

    def choose_open(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open a saved experiment from a trusted source"
        )
        if path:
            self.open_experiment(Path(path))

    def open_experiment(self, path: Path):
        try:
            root, manifest = read_experiment_manifest(path)
            if (
                self.dataset is not None
                and self.dataset.fingerprint == manifest["dataset_fingerprint"]
            ):
                self.restore_experiment(root)
                return
            dataset_path = Path(manifest["dataset_path"])
            if not (dataset_path / "packets").is_dir():
                selected = QFileDialog.getExistingDirectory(
                    self, "Locate the original exported dataset for this experiment"
                )
                if not selected:
                    return
                dataset_path = Path(selected)
            self.pending_experiment = root
            self.dataset_requested.emit(dataset_path)
        except Exception as error:  # noqa: BLE001 -- show malformed/incompatible saved files in the UI.
            QMessageBox.critical(self, "Cannot open experiment", str(error))

    def restore_experiment(self, path: Path):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            experiment = load_experiment(path, self.dataset)
            self.restoring = True
            self.split.restore(experiment.config)
            self.features.restore(experiment.config)
            self.experiment = experiment
            self.stale = False
            self.results.set_experiment(experiment)
            self.tabs.setCurrentWidget(self.results)
            self.status.setText(
                f"Restored saved models, settings and results from {path}"
            )
        except Exception as error:  # noqa: BLE001 -- loading a model must not terminate the viewer.
            QMessageBox.critical(self, "Cannot restore experiment", str(error))
        finally:
            self.restoring = False
            QApplication.restoreOverrideCursor()
            self.refresh_summary()
