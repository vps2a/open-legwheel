"""Resolve missing/conflicting setup IDs before the batch run-selection pages."""

from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel, QMessageBox,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from legwheel_dataset.config_creator import write_processing_config
from legwheel_dataset.terrain import TerrainIdReview, review_terrain_ids
from legwheel_experiments.terrain import validate_terrain_id


def save_terrain_ids(selections: list[tuple[TerrainIdReview, str]]) -> None:
    """Validate the complete change first, and restore prior files on a write error."""
    updates: dict[Path, str] = {}
    for review, terrain_id in selections:
        # Do not overwrite metadata edited by another program while this dialog
        # was open. Reopen the dialog to review any newly discovered conflicts.
        if review_terrain_ids(review.directory).documents != review.documents:
            raise ValueError(f"Metadata changed in {review.directory}. Reopen the ID dialog.")
        updates.update(review.updates(terrain_id))
    originals = {path: path.read_bytes().decode("utf-8") for path in updates}
    written = []
    try:
        for path, text in updates.items():
            write_processing_config(path, text)
            written.append(path)
    except OSError as error:
        failures = []
        for path in reversed(written):
            try:
                write_processing_config(path, originals[path])
            except OSError:
                failures.append(str(path))
        if failures:
            raise OSError("Could not restore these metadata files: " + ", ".join(failures)) from error
        raise


class TerrainIdsDialog(QDialog):
    """One editable ID per experiment, with exact regeneration warnings."""

    def __init__(self, reviews: list[TerrainIdReview], parent=None) -> None:
        super().__init__(parent)
        self.reviews = reviews
        self.setWindowTitle("Terrain and lighting setup IDs")
        self.resize(1000, 580)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Add missing IDs or resolve conflicting values. IDs are case-sensitive; use "
            "letters, digits, underscores and hyphens, with at least one letter or digit. "
            "Multiple experiments may share an ID. Save IDs updates each experiment's "
            "configuration, all run snapshots and all dataset provenance files immediately, "
            "including runs not selected for this batch."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.table = QTableWidget(len(reviews), 3)
        self.table.setHorizontalHeaderLabels(("Experiment", "terrain_and_light_ID", "Existing values"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.inputs: list[QComboBox] = []
        for row, review in enumerate(reviews):
            self.table.setItem(row, 0, QTableWidgetItem(review.directory.name))
            editor = QComboBox()
            editor.setEditable(True)
            editor.addItems(review.known_ids)
            # A conflict requires an explicit choice; do not pick a winner.
            if len(review.known_ids) != 1:
                editor.setCurrentIndex(-1)
            editor.lineEdit().setPlaceholderText("Enter or choose an ID")
            self.inputs.append(editor)
            self.table.setCellWidget(row, 1, editor)
            values = [value for document in review.documents for value in document.values()]
            invalid = 0
            for value in values:
                try:
                    validate_terrain_id(value)
                except ValueError:
                    invalid += 1
            details = "; ".join(review.known_ids) or "No ID recorded"
            if invalid:
                value_word = "value" if invalid == 1 else "values"
                details += f"; {invalid} missing/invalid {value_word}"
            item = QTableWidgetItem(details)
            item.setToolTip("\n".join(
                f"{document.path.relative_to(review.directory)}: {document.values()!r}"
                for document in review.documents
            ))
            self.table.setItem(row, 2, item)
            self.table.setRowHeight(row, 64)
        layout.addWidget(self.table, 1)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        warning_font = self.warning.font()
        warning_font.setBold(True)
        self.warning.setFont(warning_font)
        layout.addWidget(self.warning)
        self.affected_paths = QPlainTextEdit()
        self.affected_paths.setReadOnly(True)
        self.affected_paths.setMaximumHeight(150)
        layout.addWidget(self.affected_paths)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
        )
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("Save IDs")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for editor in self.inputs:
            editor.currentTextChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _selections(self) -> list[tuple[TerrainIdReview, str]]:
        return [
            (review, validate_terrain_id(editor.currentText().strip()))
            for review, editor in zip(self.reviews, self.inputs)
        ]

    def _refresh_preview(self) -> None:
        try:
            selections = self._selections()
        except ValueError as error:
            self.validation.setText(str(error))
            self.save_button.setEnabled(False)
            self.warning.clear()
            self.affected_paths.clear()
            return
        self.validation.clear()
        self.save_button.setEnabled(True)
        affected = [
            path for review, terrain_id in selections
            for path in review.affected_datasets(terrain_id)
        ]
        if affected:
            dataset_word = "dataset" if len(affected) == 1 else "datasets"
            self.warning.setText(
                f"Warning: {len(affected)} existing {dataset_word} will need to be regenerated "
                "after these IDs are saved. Their files and classifications will be preserved. "
                "Selected runs will be rebuilt when the batch starts."
            )
        else:
            self.warning.setText("No existing datasets need metadata changes.")
        self.affected_paths.setPlainText("\n".join(str(path) for path in affected))
        self.affected_paths.setVisible(bool(affected))

    def _save(self) -> None:
        try:
            save_terrain_ids(self._selections())
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot save terrain and lighting IDs", str(error))
            return
        self.accept()
