"""Run assignments and explicit feature/model controls for the training page."""

from collections import defaultdict

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis import MOTOR_CHANNELS, TelemetryDataset
from .features import (
    ROLES,
    ExperimentConfig,
    FeatureSettings,
    available_features,
)
from .plots import channel_title
from .qt_helpers import DecimalSpinBox


def _role_combo() -> QComboBox:
    combo = QComboBox()
    for role in ROLES:
        combo.addItem(role.title(), role)
    return combo


class SplitPanel(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.run_combos = {}
        self.setup_combos = {}
        self.setup_sources = defaultdict(list)
        self.run_counts = {}
        self.updating = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Evaluation mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("New setup — keep physical terrains together", "setup")
        self.mode_combo.addItem(
            "New run — allow familiar terrain across partitions", "run"
        )
        row.addWidget(self.mode_combo, 1)
        layout.addLayout(row)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            (
                "Physical setup / experiment / run",
                "Terrain and light",
                "Firm",
                "Loose",
                "Partition",
            )
        )
        self.tree.setColumnWidth(0, 290)
        self.tree.setColumnWidth(1, 150)
        self.tree.setColumnWidth(2, 65)
        self.tree.setColumnWidth(3, 65)
        layout.addWidget(self.tree, 1)
        note = QLabel(
            "Counts above are labelled packets before feature/transition filtering. "
            "The summary shows the eligible counts. Assign complete runs; test evaluation is optional and separate."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def set_dataset(self, dataset: TelemetryDataset):
        self.dataset = dataset
        self.updating = True
        self.tree.clear()
        self.run_combos.clear()
        self.setup_combos.clear()
        self.setup_sources.clear()
        self.run_counts.clear()
        ids = np.asarray(dataset.source_ids)
        parents = {}
        for source in sorted(set(ids), key=lambda key: dataset.run_names[key]):
            indices = np.flatnonzero(ids == source)
            setup = dataset.terrain_keys[indices[0]][0]
            if setup not in parents:
                parent = QTreeWidgetItem(self.tree, (setup, "", "", "", ""))
                parent.setExpanded(True)
                parents[setup] = parent
                combo = _role_combo()
                combo.addItem("Mixed (runs differ)", "mixed")
                combo.model().item(combo.count() - 1).setEnabled(False)
                self.tree.setItemWidget(parent, 4, combo)
                self.setup_combos[setup] = combo
                combo.currentIndexChanged.connect(
                    lambda _index, key=setup: self._setup_changed(key)
                )
            labels = dataset.class_ids[indices]
            firm, loose = int((labels == 0).sum()), int((labels == 1).sum())
            self.run_counts[source] = firm + loose
            lighting = ", ".join(sorted({dataset.terrain_keys[i][1] for i in indices}))
            child = QTreeWidgetItem(
                parents[setup],
                (dataset.run_names[source], lighting, str(firm), str(loose), ""),
            )
            combo = _role_combo()
            self.tree.setItemWidget(child, 4, combo)
            self.run_combos[source] = combo
            self.setup_sources[setup].append(source)
            combo.currentIndexChanged.connect(
                lambda _index, key=setup: self._run_changed(key)
            )
        self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.updating = False
        self._mode_changed()

    def _setup_changed(self, setup):
        if self.updating:
            return
        role = self.setup_combos[setup].currentData()
        if role == "mixed":
            return
        self.updating = True
        for source in self.setup_sources[setup]:
            combo = self.run_combos[source]
            combo.setCurrentIndex(
                combo.findData(role if self.run_counts[source] else "excluded")
            )
        self.updating = False
        self.changed.emit()

    def _run_changed(self, setup):
        if self.updating:
            return
        self._sync_parent(setup)
        self.changed.emit()

    def _sync_parent(self, setup):
        roles = {
            self.run_combos[source].currentData()
            for source in self.setup_sources[setup]
            if self.run_counts[source]
        }
        role = (
            next(iter(roles)) if len(roles) == 1 else "mixed" if roles else "excluded"
        )
        combo = self.setup_combos[setup]
        combo.blockSignals(True)
        combo.setCurrentIndex(combo.findData(role))
        combo.blockSignals(False)

    def _mode_changed(self):
        by_run = self.mode_combo.currentData() == "run"
        self.explanation.setText(
            "New-run evaluation: different recordings of the same ground may be in training and validation."
            if by_run
            else "New-setup evaluation: all included runs of a physical setup stay in one partition, including different lighting. "
            "Switching to this mode clears conflicting assignments within a setup."
        )
        for setup in self.setup_sources:
            if not by_run and self.setup_combos[setup].currentData() == "mixed":
                self.setup_combos[setup].setCurrentIndex(0)
            for source in self.setup_sources[setup]:
                self.run_combos[source].setEnabled(
                    by_run and self.run_counts[source] > 0
                )
        self.changed.emit()

    def assignments(self) -> dict[str, str]:
        return {
            source: combo.currentData() for source, combo in self.run_combos.items()
        }

    def restore(self, config: ExperimentConfig):
        self.updating = True
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(config.split_mode))
        for source, combo in self.run_combos.items():
            combo.setCurrentIndex(
                combo.findData(config.run_roles.get(source, "excluded"))
            )
        for setup in self.setup_sources:
            self._sync_parent(setup)
        self.updating = False
        self._mode_changed()


class FeaturePanel(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.feature_items = {}
        self.updating = False
        layout = QVBoxLayout(self)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Feature preset"))
        self.preset = QComboBox()
        for title, value in (
            ("Motor standard deviations only", "std"),
            ("Motor band powers only", "power"),
            ("STD + motor powers", "both"),
            ("STD + motor powers + wheel speed", "speed"),
            ("Compact: STD + motor powers + speed + slopes", "compact"),
            ("Compact + all IMU powers", "imu"),
            ("Custom selection", "custom"),
        ):
            self.preset.addItem(title, value)
        self.preset.setCurrentIndex(4)
        preset_row.addWidget(self.preset, 1)
        self.count_label = QLabel("0 features")
        preset_row.addWidget(self.count_label)
        layout.addLayout(preset_row)
        body = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ("Selected features — expand channels to choose individual bands",)
        )
        body.addWidget(self.tree)
        options = QWidget()
        form = QFormLayout(options)
        self.window_combo = QComboBox()
        self.window_combo.addItem("Spectral window", "spectral")
        self.window_combo.addItem("Local patch window", "local")
        form.addRow("Window for STD + speed", self.window_combo)
        self.log_check = QCheckBox("Log-transform band powers")
        self.log_check.setChecked(True)
        self.log_check.setToolTip(
            "Numerical log10 transformation, independent of the exploration plot scale."
        )
        form.addRow(self.log_check)
        self.transition_check = QCheckBox(
            "Exclude windows touching labelled transitions"
        )
        self.transition_check.setChecked(True)
        form.addRow(self.transition_check)
        self.padding = DecimalSpinBox()
        self.padding.setDecimals(3)
        self.padding.setRange(0, 60)
        self.padding.setSuffix(" s")
        form.addRow("Extra transition padding", self.padding)
        note = QLabel(
            "Standardization is always fitted on training packets only. Incomplete feature rows are excluded. Each packet has equal weight."
        )
        note.setWordWrap(True)
        form.addRow(note)

        model_box = QGroupBox("Model settings")
        model_form = QFormLayout(model_box)
        fixed = QLabel(
            "Logistic regression: fixed L2 regularization, C = 1. No tuning controls."
        )
        fixed.setWordWrap(True)
        model_form.addRow(fixed)
        self.linear_c = self._positive_number(1, 0.0001, 10000, 4)
        self.rbf_c = self._positive_number(1, 0.0001, 10000, 4)
        self.gamma = self._positive_number(0.1, 0.000001, 10000, 6)
        self.gamma_auto = QCheckBox("Automatic gamma")
        self.gamma_auto.setChecked(True)
        self.gamma.setEnabled(False)
        model_form.addRow("Linear SVM C", self.linear_c)
        model_form.addRow("RBF SVM C", self.rbf_c)
        model_form.addRow(self.gamma_auto)
        model_form.addRow("RBF gamma", self.gamma)
        explanation = QLabel(
            "C penalizes margin violations; the margin is learned. Larger gamma gives more local influence and permits finer boundary detail."
        )
        explanation.setWordWrap(True)
        model_form.addRow(explanation)
        form.addRow(model_box)
        body.addWidget(options)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        layout.addWidget(body, 1)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.tree.itemChanged.connect(self._tree_changed)
        self.window_combo.currentIndexChanged.connect(self._emit_changed)
        for checkbox in (self.log_check, self.transition_check):
            checkbox.toggled.connect(self._emit_changed)
        for spin in (self.padding, self.linear_c, self.rbf_c, self.gamma):
            spin.valueChanged.connect(self._emit_changed)
        self.gamma_auto.toggled.connect(self._gamma_changed)

    @staticmethod
    def _positive_number(value, minimum, maximum, decimals):
        spin = DecimalSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _emit_changed(self, *_args):
        if not self.updating:
            self.changed.emit()

    def _gamma_changed(self, checked):
        self.gamma.setEnabled(not checked)
        self._emit_changed()

    @staticmethod
    def _parent(parent, title):
        item = QTreeWidgetItem(parent, (title,))
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsAutoTristate
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        return item

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.updating = True
        self.tree.clear()
        self.feature_items.clear()
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData("compact"))
        self.preset.blockSignals(False)
        groups = {
            kind: self._parent(self.tree, title)
            for kind, title in (
                ("std", "Motor standard deviations"),
                ("motor", "Motor band powers"),
                ("imu", "IMU band powers"),
                ("speed", "Wheel speed"),
                ("slope", "Terrain slopes"),
            )
        }
        channels = {}
        for feature in available_features(dataset):
            if feature.kind == "power":
                group = "motor" if feature.channel in MOTOR_CHANNELS else "imu"
                if feature.channel not in channels:
                    channels[feature.channel] = self._parent(
                        groups[group], channel_title(feature.channel)
                    )
                parent = channels[feature.channel]
                lower, upper = dataset.band_edges_hz[feature.band]
                title = f"{lower:g}–{upper:g} Hz"
            else:
                parent = groups[feature.kind]
                title = (
                    channel_title(feature.channel)
                    if feature.kind == "std"
                    else "Mean absolute wheel angular velocity"
                    if feature.kind == "speed"
                    else f"{feature.channel.title()}-travel slope"
                )
            item = QTreeWidgetItem(parent, (title,))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            self.feature_items[feature] = item
        for group in groups.values():
            group.setExpanded(True)
        self.updating = False
        self._apply_preset()

    def _apply_preset(self, *_args):
        preset = self.preset.currentData()
        if self.updating or self.dataset is None or preset == "custom":
            return
        self.updating = True
        for feature, item in self.feature_items.items():
            is_motor_power = (
                feature.kind == "power" and feature.channel in MOTOR_CHANNELS
            )
            selected = (
                feature.kind == "std"
                and preset != "power"
                or is_motor_power
                and preset != "std"
                or feature.kind == "speed"
                and preset in ("speed", "compact", "imu")
                or feature.kind == "slope"
                and preset in ("compact", "imu")
                or feature.kind == "power"
                and feature.channel not in MOTOR_CHANNELS
                and preset == "imu"
            )
            item.setCheckState(
                0, Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
            )
        self.updating = False
        self._selection_changed()

    def _tree_changed(self, *_args):
        if self.updating:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData("custom"))
        self.preset.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self):
        self.count_label.setText(f"{len(self.settings().features)} features")
        self.changed.emit()

    def settings(self) -> FeatureSettings:
        return FeatureSettings(
            tuple(
                feature
                for feature, item in self.feature_items.items()
                if item.checkState(0) == Qt.CheckState.Checked
            ),
            self.window_combo.currentData(),
            self.log_check.isChecked(),
            self.transition_check.isChecked(),
            self.padding.value(),
        )

    def restore(self, config: ExperimentConfig):
        self.updating = True
        selected = set(config.features.features)
        for feature, item in self.feature_items.items():
            item.setCheckState(
                0,
                Qt.CheckState.Checked
                if feature in selected
                else Qt.CheckState.Unchecked,
            )
        self.preset.setCurrentIndex(self.preset.findData("custom"))
        self.window_combo.setCurrentIndex(
            self.window_combo.findData(config.features.window)
        )
        self.log_check.setChecked(config.features.log_powers)
        self.transition_check.setChecked(config.features.exclude_transitions)
        self.padding.setValue(config.features.transition_margin_s)
        self.linear_c.setValue(config.linear_c)
        self.rbf_c.setValue(config.rbf_c)
        automatic = config.rbf_gamma == "scale"
        self.gamma_auto.setChecked(automatic)
        self.gamma.setEnabled(not automatic)
        if not automatic:
            self.gamma.setValue(float(config.rbf_gamma))
        self.updating = False
        self._selection_changed()
