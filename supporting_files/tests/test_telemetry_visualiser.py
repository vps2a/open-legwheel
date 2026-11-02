"""Numerical meaning, exported-data loading and linked GUI filter tests."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ML_tools.analysis import (
    BAND_CHANNELS, MOTOR_CHANNELS, DatasetFormatError, LoadCancelled,
    histogram_edges, load_dataset, summarize_band_power, window_standard_deviations,
)
from ML_tools.window import VisualiserWindow


def make_export(root: Path) -> Path:
    """Unequal run sizes, reordered channels, every class, null and unknown labels."""
    (root / "packets").mkdir(parents=True)
    rows = []
    for index, label in enumerate((0, 0, 0, 0, 1, 2, 3, None, 9)):
        names = list(BAND_CHANNELS)
        if index % 2:
            names.reverse()
        # Class-zero powers 1,1,1,9 have packet mean=3, sample STD=4.
        power = 9.0 if index == 3 else 1.0
        rows.append({
            "export_id": f"packet_{index}", "source_id": "run_a" if index < 3 else "run_b",
            "class_id": label, "terrain_setup_id": "KK_01" if index < 4 else "KK_02",
            "terrain_and_light_ID": "KK_01_L1" if index < 3 else "KK_01_L2" if index == 3 else "KK_02_L1",
            "channel_names": names,
            "spectral_values": [[float(sample * (BAND_CHANNELS.index(name) + 1)) for name in names]
                                for sample in range(4)],
            "spectral_valid": [[True] * len(names) for _ in range(4)],
            "local_values": [[float(sample * (BAND_CHANNELS.index(name) + 1)) for name in names]
                             for sample in (0, 2)],
            "local_valid": [[True] * len(names) for _ in range(2)],
            "band_edges_hz": [[0.0, 10.0], [10.0, 20.0]],
            "band_power": [[power * (BAND_CHANNELS.index(name) + 1) * band for name in names]
                           for band in (1, 2)],
            "unused_image_bytes": b"Image payload must not be requested by the loader.",
        })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table.slice(0, 4), root / "packets/part-00000.parquet")
    pq.write_table(table.slice(4), root / "packets/part-00001.parquet")
    pq.write_table(pa.table({"source_id": ["run_a", "run_b"]}), root / "runs.parquet")
    (root / "export_manifest.json").write_text(json.dumps({"export_version": "1.1"}))
    return root


def digests(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class NumericalTests(unittest.TestCase):
    def test_population_std_uses_mask_finite_values_and_at_least_two_samples(self):
        names = list(MOTOR_CHANNELS)
        values = np.tile([0.0, 2.0, 1000.0, np.nan], (6, 1)).T
        valid = np.ones(values.shape, dtype=bool)
        valid[2] = False
        valid[1:, 1] = False
        result = window_standard_deviations(values, valid, names)
        self.assertEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[2:], 1.0)
        self.assertTrue(np.isnan(window_standard_deviations([[1]], [[True]], names)).all())

    def test_equal_packet_weight_and_sample_std_not_equal_run_weight(self):
        summary = summarize_band_power(np.array([[1, 2], [1, 2], [1, 2], [9, 18]], float))
        np.testing.assert_allclose(summary.mean, [3, 6])
        np.testing.assert_allclose(summary.std, [4, 8])
        np.testing.assert_equal(summary.count, [4, 4])

    def test_missing_values_are_not_zero_and_one_value_has_no_error_bar(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            summary = summarize_band_power(np.array([[0, 4, np.nan], [2, np.nan, -1]]))
            empty = summarize_band_power(np.empty((0, 3)))
        np.testing.assert_allclose(summary.mean[:2], [1, 4])
        self.assertTrue(np.isnan(summary.mean[2]))
        np.testing.assert_equal(summary.count, [2, 1, 0])
        self.assertAlmostEqual(summary.std[0], np.sqrt(2))
        self.assertTrue(np.isnan(summary.std[1:]).all())
        self.assertTrue(np.isnan(empty.mean).all())

    def test_common_histogram_edges_and_log_zero_handling(self):
        edges = histogram_edges(np.array([0, 1, 100, np.nan]), 2, True)
        np.testing.assert_allclose(edges, [1, 10, 100])
        self.assertEqual(histogram_edges(np.array([0]), 10, True).size, 0)
        edges = histogram_edges(np.array([0, 0]), 10, False)
        self.assertEqual(edges[0], 0)
        self.assertGreater(edges[-1], 0)


class ExportLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_export(Path(self.temporary.name) / "export")

    def tearDown(self):
        self.temporary.cleanup()

    def test_load_all_shards_by_file_or_folder_without_source_writes(self):
        before = digests(self.root)
        for path in (self.root, self.root / "packets/part-00001.parquet", self.root / "runs.parquet"):
            dataset = load_dataset(path)
            self.assertEqual(len(dataset.packet_ids), 9)
            np.testing.assert_allclose(dataset.standard_deviations["spectral"][:, 0], np.sqrt(1.25))
            np.testing.assert_allclose(dataset.standard_deviations["local"][:, 0], 1.0)
            self.assertEqual(dataset.band_powers.shape, (9, 2, 12))
            np.testing.assert_allclose(dataset.band_powers[1, 0], np.arange(1, 13))
            self.assertEqual(dataset.export_version, "1.1")
        self.assertEqual(before, digests(self.root))

    def test_class_zero_and_physical_setup_lighting_selection(self):
        dataset = load_dataset(self.root)
        mask = dataset.selection({0}, {("KK_01", "KK_01_L1")})
        np.testing.assert_equal(np.flatnonzero(mask), [0, 1, 2])
        mask = dataset.selection(set(range(4)), set(dataset.terrain_keys))
        self.assertEqual(mask.sum(), 7)
        self.assertEqual(dataset.class_ids[-2], -1)
        self.assertEqual(dataset.class_ids[-1], -2)

    def test_different_band_layouts_fail_instead_of_comparing_unrelated_buckets(self):
        path = self.root / "packets/part-00001.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["band_edges_hz"] = [[0, 5], [5, 20]]
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(DatasetFormatError, "different frequency bands"):
            load_dataset(self.root)

    def test_duplicate_packets_fail_instead_of_double_counting(self):
        path = self.root / "packets/part-00001.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["export_id"] = "packet_0"
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(DatasetFormatError, "duplicate export_id"):
            load_dataset(self.root)

    def test_numeric_columns_only_and_cancellable(self):
        actual = pq.ParquetFile
        requested = []

        class CheckedParquet:
            def __init__(self, path):
                self.parquet = actual(path)
                self.metadata = self.parquet.metadata
                self.schema_arrow = self.parquet.schema_arrow

            def iter_batches(self, **kwargs):
                requested.extend(kwargs["columns"])
                return self.parquet.iter_batches(**kwargs)

        with patch("ML_tools.analysis.pq.ParquetFile", CheckedParquet):
            load_dataset(self.root)
        self.assertNotIn("unused_image_bytes", requested)
        with self.assertRaises(LoadCancelled):
            load_dataset(self.root, check_cancel=lambda: (_ for _ in ()).throw(LoadCancelled()))


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_export(Path(self.temporary.name) / "export")
        self.window = VisualiserWindow()
        self.window.set_dataset(load_dataset(self.root))

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.temporary.cleanup()

    def test_filters_and_legend_toggles_apply_to_every_tab(self):
        self.assertEqual(self.window.count_label.text(), "7 packets")
        panel = self.window.panels[0]
        artist = next(artist for artist, label in panel.legend_classes.items() if label == 0)
        panel._pick_class(SimpleNamespace(artist=artist))
        self.window.refresh()
        self.assertFalse(self.window.class_checks[0].isChecked())
        self.assertEqual(self.window.count_label.text(), "3 packets")
        self.window.tabs.setCurrentIndex(1)
        self.assertEqual(self.window.count_label.text(), "3 packets")
        self.window.terrain_tree.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)
        self.window.refresh()
        self.assertEqual(self.window.count_label.text(), "0 packets")
        self.window.set_all_terrains(True)
        self.window.refresh()
        self.assertEqual(self.window.count_label.text(), "3 packets")

    def test_window_histogram_controls_and_typed_uncertainty_limit(self):
        self.window.window_combo.setCurrentIndex(1)
        self.window.histogram_mode.setCurrentIndex(1)
        self.window.bins_spin.setValue(12)
        self.window.refresh()
        axis = self.window.panels[0].figure.axes[0]
        self.assertEqual(axis.get_ylabel(), "Packet count")
        self.assertIn("local", self.window.window_combo.currentData())
        spin = self.window.uncertainty_spin
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "2.75")
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(spin.value(), 2.75)
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "1,5")
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(spin.value(), 1.5)
        self.assertEqual(spin.maximum(), 3.0)
        self.assertEqual(spin.minimum(), 0.0)

    def test_log_band_error_crossing_zero_is_marked_and_single_packet_is_valid(self):
        self.window.scale_combo.setCurrentIndex(1)
        self.window.tabs.setCurrentIndex(1)
        self.window.refresh()
        self.assertGreater(self.window.reports[1].clipped_error_bars, 0)
        self.assertIn("cross zero", self.window.summary.toPlainText())
        self.assertEqual(self.window.panels[1].figure.axes[0].get_yscale(), "log")
        self.window.tabs.setCurrentIndex(2)
        self.assertEqual(len(self.window.panels[2].figure.axes), 6)
        self.window.uncertainty_spin.setValue(0)
        self.window.refresh()
        self.assertEqual(self.window.reports[2].clipped_error_bars, 0)

    def test_all_tabs_handle_no_classes_and_no_terrain_without_warnings(self):
        for checkbox in self.window.class_checks:
            checkbox.setChecked(False)
        self.window.scale_combo.setCurrentIndex(1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for index in range(3):
                self.window.tabs.setCurrentIndex(index)
                self.window.refresh()
                self.window.panels[index].canvas.draw()
        self.assertEqual(self.window.count_label.text(), "0 packets")

    def test_background_load_finishes_and_keeps_source_files_unchanged(self):
        before = digests(self.root)
        self.window.open_dataset(self.root)
        deadline = time.monotonic() + 10
        while self.window.loader is not None and time.monotonic() < deadline:
            self.application.processEvents()
            QTest.qWait(10)
        self.assertIsNone(self.window.loader)
        self.assertTrue(self.window.open_button.isEnabled())
        self.assertEqual(self.window.count_label.text(), "7 packets")
        self.assertEqual(digests(self.root), before)


if __name__ == "__main__":
    unittest.main()
