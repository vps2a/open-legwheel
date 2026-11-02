"""Metric tables, confusion matrices and ROC curves for one frozen experiment."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .evaluation_plots import comparison_figure
from .qt_helpers import ExportNavigationToolbar
from .training import METRIC_NAMES, MODEL_NAMES, per_run_metrics


def format_metric(value, name):
    if value is None:
        return "N/A"
    return f"{value:.3f}" if name == "roc_auc" else f"{value:.1%}"


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.ResizeToContents
    )
    table.horizontalHeader().setStretchLastSection(True)
    return table


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.experiment = None
        layout = QVBoxLayout(self)
        self.notice = QLabel("Train models to compare their validation performance.")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        row = QHBoxLayout()
        row.addWidget(QLabel("Evaluated partition"))
        self.partition = QComboBox()
        row.addWidget(self.partition)
        row.addWidget(
            QLabel(
                "Positive class: 1 — Loose. Confusion cells show counts and percentages within each actual class."
            ),
            1,
        )
        layout.addLayout(row)
        self.metrics = _table(["Model", "Packets", *METRIC_NAMES.values()])
        self.metrics.setMinimumHeight(115)
        self.metrics.setMaximumHeight(145)
        layout.addWidget(self.metrics)
        splitter = QSplitter(Qt.Orientation.Vertical)
        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Preserve readable matrices/axis labels when the window is compact.
        # Scrolling is preferable to shrinking the scientific figure to illegibility.
        self.canvas.setMinimumSize(600, 440)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.canvas)
        self.toolbar = ExportNavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(scroll, 1)
        splitter.addWidget(plots)
        self.run_metrics = _table(["Model", "Run", "Packets", *METRIC_NAMES.values()])
        self.run_metrics.setMinimumHeight(90)
        splitter.addWidget(self.run_metrics)
        splitter.setSizes((550, 130))
        layout.addWidget(splitter, 1)
        self.partition.currentIndexChanged.connect(self.refresh)
        self.clear()

    def clear(self):
        self.experiment = None
        self.partition.clear()
        self.metrics.setRowCount(0)
        self.run_metrics.setRowCount(0)
        self.figure.clear()
        self.figure.text(
            0.5, 0.5, "No trained experiment", ha="center", va="center", color="#666666"
        )
        self.canvas.draw_idle()
        self.set_stale(False)

    def set_stale(self, stale: bool):
        if stale:
            self.notice.setText(
                "OUTDATED RESULTS — settings or split changed. Train again before saving or evaluating the test set."
            )
            self.notice.setStyleSheet("color: #a65b00; font-weight: bold;")
        else:
            self.notice.setText(
                "Results use the saved training-only preprocessing and fixed decision thresholds. N/A means the metric is undefined."
                if self.experiment
                else "Train models to compare their validation performance."
            )
            self.notice.setStyleSheet("")

    def set_experiment(self, experiment):
        previous = self.partition.currentData()
        self.experiment = experiment
        self.partition.blockSignals(True)
        self.partition.clear()
        for role in ("validation", "training", "testing"):
            if role in experiment.evaluations:
                self.partition.addItem(role.title(), role)
        index = self.partition.findData(previous)
        self.partition.setCurrentIndex(max(index, 0))
        self.partition.blockSignals(False)
        self.set_stale(False)
        self.refresh()

    def refresh(self, *_args):
        if self.experiment is None or self.partition.currentData() is None:
            return
        role = self.partition.currentData()
        results = self.experiment.evaluations[role]
        self.metrics.setRowCount(len(MODEL_NAMES))
        for row, (name, result) in enumerate(results.items()):
            values = [
                MODEL_NAMES[name],
                str(len(result.indices)),
                *(format_metric(result.metrics[key], key) for key in METRIC_NAMES),
            ]
            for column, value in enumerate(values):
                self.metrics.setItem(row, column, QTableWidgetItem(value))
        self.metrics.setFixedHeight(
            self.metrics.horizontalHeader().height()
            + self.metrics.verticalHeader().length()
            + 2 * self.metrics.frameWidth()
            + 4
        )
        rows = list(per_run_metrics(self.experiment, role))
        self.run_metrics.setRowCount(len(rows))
        for row, result in enumerate(rows):
            values = [
                MODEL_NAMES[result["model"]],
                result["run"],
                str(result["packets"]),
                *(format_metric(result[key], key) for key in METRIC_NAMES),
            ]
            for column, value in enumerate(values):
                self.run_metrics.setItem(row, column, QTableWidgetItem(value))
        comparison_figure(self.experiment, role, self.figure)
        self.toolbar.update()
        self.canvas.draw_idle()
