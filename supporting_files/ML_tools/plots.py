"""Six-channel plots; all aggregation is delegated to the analysis module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .analysis import (
    BAND_CHANNELS,
    CLASS_COLORS,
    CLASS_NAMES,
    IMU_CHANNELS,
    MOTOR_CHANNELS,
    TelemetryDataset,
    histogram_edges,
    summarize_band_power,
)
from .qt_helpers import ExportNavigationToolbar


def channel_title(name: str) -> str:
    if name.startswith("imu."):
        _, quantity, axis = name.split(".")
        return f"{axis.upper()} {'angular velocity' if quantity == 'angular_velocity' else 'linear acceleration'}"
    joint, quantity = name.split(".")
    return f"{joint.removesuffix('_joint').title()} {quantity}"


def channel_unit(name: str) -> str:
    if "linear_acceleration" in name:
        return "m/s²"
    return "N·m" if name.endswith("effort") else "rad/s"


@dataclass(frozen=True)
class PlotReport:
    counts: tuple[str, ...]
    omitted_nonpositive: int = 0
    clipped_error_bars: int = 0


class PlotPanel(QWidget):
    """One tab with a 3×2 plot grid and clickable, synchronized legends."""

    class_toggled = Signal(int)

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.channels = IMU_CHANNELS if kind == "imu" else MOTOR_CHANNELS
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = ExportNavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.legend_classes = {}
        self.canvas.mpl_connect("pick_event", self._pick_class)
        self.show_empty("Open an exported dataset to begin.")

    def _pick_class(self, event):
        class_id = self.legend_classes.get(event.artist)
        if class_id is not None:
            self.class_toggled.emit(class_id)

    def show_empty(self, message: str):
        self.figure.clear()
        self.legend_classes.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center", color="#666666")
        self.canvas.draw_idle()

    def _legend(self, axis, classes: set[int], counts: dict[int, str]):
        handles = [
            Line2D([], [], color=color, marker="o", markersize=3, linewidth=1.5,
                   alpha=1.0 if class_id in classes else 0.25,
                   label=f"{class_id} {name}" + (f" ({counts[class_id]})" if class_id in counts else ""))
            for class_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
        ]
        legend = axis.legend(handles=handles, fontsize=6.5, ncol=2, loc="upper right")
        for class_id, (handle, text) in enumerate(zip(legend.legend_handles, legend.get_texts())):
            for artist in (handle, text):
                artist.set_picker(5)
                self.legend_classes[artist] = class_id
            if class_id not in classes:
                text.set_alpha(0.35)

    def render(
        self, dataset: TelemetryDataset, selection: np.ndarray, classes: set[int],
        *, window: str, bins: int, density: bool, multiplier: float, logarithmic: bool,
    ) -> PlotReport:
        self.figure.clear()
        self.legend_classes.clear()
        axes = self.figure.subplots(3, 2).ravel()
        reports = []
        omitted = clipped = 0
        for channel_index, (name, axis) in enumerate(zip(self.channels, axes)):
            axis.set_title(channel_title(name), fontsize=10, loc="left")
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.18)
            if self.kind == "std":
                count, dropped = self._histogram(
                    axis, dataset, selection, classes, channel_index,
                    window, bins, density, logarithmic,
                )
                reports.append(count)
                omitted += dropped
            else:
                count, dropped, truncated = self._band_power(
                    axis, dataset, selection, classes, name, multiplier, logarithmic,
                )
                reports.append(count)
                omitted += dropped
                clipped += truncated
        self.toolbar.update()
        self.canvas.draw_idle()
        return PlotReport(tuple(reports), omitted, clipped)

    def _histogram(self, axis, dataset, selection, classes, column, window, bins, density, logarithmic):
        values = dataset.standard_deviations[window][:, column]
        valid = selection & np.isfinite(values)
        displayed = valid & ((values > 0) if logarithmic else (values >= 0))
        edges = histogram_edges(values[displayed], bins, logarithmic)
        counts = {}
        for class_id in sorted(classes):
            samples = values[displayed & (dataset.class_ids == class_id)]
            counts[class_id] = f"n={len(samples)}"
            if samples.size and edges.size:
                heights, _ = np.histogram(samples, bins=edges, density=density)
                axis.stairs(heights, edges, color=CLASS_COLORS[class_id], fill=True, alpha=0.18)
                axis.stairs(heights, edges, color=CLASS_COLORS[class_id], linewidth=1.4)
        if not edges.size:
            axis.text(0.5, 0.45, "No positive values" if logarithmic and np.any(valid)
                      else "No valid selected values", transform=axis.transAxes,
                      ha="center", fontsize=9, color="#666666")
        elif logarithmic:
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlim(edges[0], edges[-1])
        axis.set_xlabel(f"Within-window STD [{channel_unit(self.channels[column])}]", fontsize=8)
        axis.set_ylabel("Probability density" if density else "Packet count", fontsize=8)
        axis.set_ylim(bottom=0)
        self._legend(axis, classes, counts)
        available, shown = int(valid.sum()), int(displayed.sum())
        return f"{available} valid / {shown} shown", available - shown

    def _band_power(self, axis, dataset, selection, classes, name, multiplier, logarithmic):
        column = BAND_CHANNELS.index(name)
        values = dataset.band_powers[:, :, column]
        centers = dataset.band_edges_hz.mean(axis=1)
        summaries = {
            class_id: summarize_band_power(values[selection & (dataset.class_ids == class_id)])
            for class_id in sorted(classes)
        }
        # A lower SD limit <= 0 cannot exist on a log axis. Draw it down to a
        # documented positive floor and mark it, preserving the raw statistics.
        positive_limits = []
        for summary in summaries.values():
            lower = summary.mean - multiplier * summary.std
            positive_limits.extend(summary.mean[np.isfinite(summary.mean) & (summary.mean > 0)])
            positive_limits.extend(lower[np.isfinite(lower) & (lower > 0)])
        floor = max(min(positive_limits) * 0.5, np.finfo(float).tiny) if positive_limits else None
        counts = {}
        omitted = clipped = 0
        for class_id, summary in summaries.items():
            minimum, maximum = int(summary.count.min()), int(summary.count.max())
            counts[class_id] = f"n={minimum}" if minimum == maximum else f"n={minimum}–{maximum}"
            good = np.isfinite(summary.mean)
            if logarithmic:
                omitted += int(np.count_nonzero(good & (summary.mean <= 0)))
                good &= summary.mean > 0
            if not np.any(good):
                continue
            y = np.where(good, summary.mean, np.nan)
            color = CLASS_COLORS[class_id]
            axis.plot(centers, y, "-o", color=color, linewidth=1.3, markersize=3, alpha=0.85)
            with_error = good & np.isfinite(summary.std)
            if multiplier > 0 and np.any(with_error):
                x, means = centers[with_error], summary.mean[with_error]
                error = multiplier * summary.std[with_error]
                lower_error = error.copy()
                truncated = (means - error <= 0) if logarithmic else np.zeros(len(means), bool)
                if np.any(truncated):
                    lower_error[truncated] = means[truncated] - floor
                    axis.plot(x[truncated], np.full(np.count_nonzero(truncated), floor),
                              "v", color=color, markersize=4)
                    clipped += int(np.count_nonzero(truncated))
                axis.errorbar(x, means, yerr=np.vstack((lower_error, error)),
                              fmt="none", color=color, capsize=3, alpha=0.65, linewidth=1)
        if logarithmic and floor is not None:
            axis.set_yscale("log")
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.set_ylim(bottom=floor * 0.8)
        if not any(np.any(np.isfinite(s.mean) & ((s.mean > 0) if logarithmic else True))
                   for s in summaries.values()):
            axis.text(0.5, 0.45, "No positive values" if logarithmic else "No valid selected values",
                      transform=axis.transAxes, ha="center", fontsize=9, color="#666666")
        if clipped:
            axis.text(0.02, 0.02, "▼ lower SD limit ≤ 0", transform=axis.transAxes,
                      fontsize=6.5, color="#666666")
        axis.set_xticks(centers, [f"{a:g}–{b:g}" for a, b in dataset.band_edges_hz])
        axis.set_xlabel("Frequency band [Hz]", fontsize=8)
        axis.set_ylabel(f"Mean band power [({channel_unit(name)})²]", fontsize=8)
        self._legend(axis, classes, counts)
        count = (np.isfinite(values[selection]) & (values[selection] >= 0)).sum(axis=0)
        low, high = int(count.min()), int(count.max())
        report = f"{low} per band" if low == high else f"{low}–{high} per band"
        return report, omitted, clipped
