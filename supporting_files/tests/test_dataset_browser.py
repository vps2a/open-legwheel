import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from legwheel_dataset_browser.model import (  # noqa: E402
    DatasetFormatError,
    LegWheelDataset,
    crop_to_mask,
)
from legwheel_dataset_browser.formatting import metadata_rows  # noqa: E402
from legwheel_dataset_browser.window import DatasetBrowserWindow  # noqa: E402
from legwheel_dataset_browser.classification import CLASSIFICATION_FIELDS  # noqa: E402
from tests.synthetic_dataset import (  # noqa: E402
    CHANNEL_NAMES,
    create_synthetic_dataset,
)


def dataset_contents(root):
    """Capture source bytes to detect accidental writes and partial saves."""
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class DatasetModelTests(unittest.TestCase):
    def test_open_upgrades_every_legacy_packet_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            dataset = LegWheelDataset.open(root)
            for index in range(len(dataset)):
                packet = dataset.load_packet(index)
                for name in CLASSIFICATION_FIELDS:
                    self.assertIn(name, packet.metadata)
                    self.assertIsNone(packet.metadata[name])
                    if packet.features is not None:
                        self.assertIsNone(packet.features[name])
            aggregate = json.loads((root / "features.jsonl").read_text())
            self.assertTrue(all(aggregate[name] is None for name in CLASSIFICATION_FIELDS))
            before = dataset_contents(root)
            LegWheelDataset.open(root)
            self.assertEqual(dataset_contents(root), before)

    def test_manual_class_zero_replaces_automatic_label_in_all_three_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            metadata_path = root / "packets/packet_000000/metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.update(class_id=3, classification_method="automatic", model_id="model-v1")
            metadata_path.write_text(json.dumps(metadata))
            dataset = LegWheelDataset.open(root)
            self.assertEqual(dataset.load_packet(0).features["model_id"], "model-v1")

            dataset.classify_packet(0, 0)

            packet = dataset.load_packet(0)
            aggregate = json.loads((root / "features.jsonl").read_text())
            for record in (packet.metadata, packet.features, aggregate):
                self.assertEqual(record["class_id"], 0)
                self.assertEqual(record["classification_method"], "manual")
                self.assertIsNone(record["model_id"])
            self.assertEqual(packet.features["along_slope_deg"], 4.25)

    def test_stage1_classification_only_updates_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            dataset = LegWheelDataset.open(root)
            before = (root / "features.jsonl").read_bytes()
            dataset.classify_packet(1, 2)
            packet = dataset.load_packet(1)
            self.assertEqual(packet.metadata["class_id"], 2)
            self.assertEqual(packet.metadata["classification_method"], "manual")
            self.assertFalse((packet.packet_directory / "stage2").exists())
            self.assertEqual((root / "features.jsonl").read_bytes(), before)

    def test_failed_aggregate_replacement_restores_both_packet_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            dataset = LegWheelDataset.open(root)
            before = dataset_contents(root)
            replace = os.replace

            def fail_aggregate(source, destination):
                if destination == dataset.root / "features.jsonl":
                    raise OSError("Synthetic write failure")
                replace(source, destination)

            with patch("legwheel_dataset_browser.model.os.replace", side_effect=fail_aggregate):
                with self.assertRaises(OSError):
                    dataset.classify_packet(0, 1)
            self.assertEqual(dataset_contents(root), before)

    def test_invalid_feature_index_does_not_partially_upgrade_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            (root / "features.jsonl").write_text("invalid json\n")
            before = dataset_contents(root)
            with self.assertRaises(DatasetFormatError):
                LegWheelDataset.open(root)
            self.assertEqual(dataset_contents(root), before)

    def test_imu_uncertainty_is_visible_for_stage1_and_stage2_packets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            dataset = LegWheelDataset.open(root)
            processed = dataset.load_packet(0)
            processed.features["IMU uncertainty"] = 4.25
            self.assertEqual(
                dict(metadata_rows(processed))["IMU uncertainty"], "4.250 ° (2σ)"
            )
            stage1 = dataset.load_packet(1)
            stage1.metadata["imu_uncertainty_deg"] = 8.0
            self.assertEqual(
                dict(metadata_rows(stage1))["IMU uncertainty"], "8.000 ° (2σ)"
            )

    def test_loads_manifest_chronologically_and_restores_packet_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            dataset = LegWheelDataset.open(root)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.experiment_id, "synthetic_experiment")
            self.assertEqual(dataset.run_number, 4)
            self.assertEqual(dataset.records[0]["image_time_ns"], 2_000_000_000)

            packet = dataset.load_packet(0)
            self.assertEqual(packet.channel_names, CHANNEL_NAMES)
            self.assertEqual(packet.rgb_mask.shape, packet.rgb.shape[:2])
            self.assertTrue(packet.stage2_available)
            self.assertEqual(packet.stage2_arrays["spectral_power"].shape, (51, 15))
            self.assertEqual(packet.point_map["points_base_m"].shape, (4, 3))

    def test_stage1_only_packet_remains_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            packet = LegWheelDataset.open(root).load_packet(1)

            self.assertFalse(packet.stage2_available)
            self.assertIsNone(packet.features)
            self.assertEqual(packet.stage2_arrays, {})
            self.assertEqual(packet.point_map, {})

    def test_depth_limits_are_fixed_across_dataset_and_ignore_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            dataset = LegWheelDataset.open(root)

            self.assertEqual(dataset.global_depth_limits_m(), (1.0, 2.0))

    def test_crop_returns_patch_and_full_image_origin(self):
        image = [[row * 10 + column for column in range(5)] for row in range(4)]
        import numpy as np

        image = np.asarray(image)
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:5] = True

        crop, crop_mask, origin = crop_to_mask(image, mask)

        self.assertEqual(crop.shape, (2, 3))
        self.assertTrue(crop_mask.all())
        self.assertEqual(origin, (2, 1))

    def test_rejects_directory_without_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(DatasetFormatError):
                LegWheelDataset.open(temporary)


class DatasetWindowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QCoreApplication.setOrganizationName("LegWheel Browser Tests")
        QCoreApplication.setApplicationName("LegWheel Browser Tests")
        cls.application = QApplication.instance() or QApplication([])
        cls.settings_directory = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope,
            cls.settings_directory.name,
        )

    @classmethod
    def tearDownClass(cls):
        cls.settings_directory.cleanup()

    def test_buttons_save_each_packet_and_manual_mode_carries_to_last_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            window = DatasetBrowserWindow(root)
            self.addCleanup(window.close)
            self.assertFalse(window.manual_mode_checkbox.isChecked())
            self.assertTrue(all(not button.isEnabled() for button in window.class_buttons))
            before = dataset_contents(root)
            with patch.object(
                QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok,
            ) as warning:
                window.manual_mode_checkbox.click()
                self.assertEqual(dataset_contents(root), before)
                window.class_buttons[0].click()
                self.assertEqual(window.packet_index, 1)
                self.assertTrue(window.manual_mode_checkbox.isChecked())
                self.assertEqual(window.dataset.load_packet(0).metadata["class_id"], 0)
                window.class_buttons[3].click()
                self.assertEqual(window.packet_index, 1)
                self.assertEqual(window.packet.metadata["class_id"], 3)
                warning.assert_called_once()

            window.manual_mode_checkbox.click()
            self.assertTrue(window.class_buttons[3].isChecked())
            self.assertFalse(window.class_buttons[3].isEnabled())
            reopened = DatasetBrowserWindow(root)
            self.addCleanup(reopened.close)
            self.assertFalse(reopened.manual_mode_checkbox.isChecked())

    def test_cancel_keeps_manual_mode_off_and_dataset_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            window = DatasetBrowserWindow(root)
            self.addCleanup(window.close)
            before = dataset_contents(root)
            with patch.object(
                QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel,
            ):
                window.manual_mode_checkbox.click()
            self.assertFalse(window.manual_mode_checkbox.isChecked())
            self.assertTrue(all(not button.isEnabled() for button in window.class_buttons))
            self.assertEqual(dataset_contents(root), before)

    def test_number_shortcuts_classify_but_leave_go_to_typing_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            window = DatasetBrowserWindow(root)
            self.addCleanup(window.close)
            window.show()
            window.activateWindow()
            self.application.processEvents()
            with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok):
                window.manual_mode_checkbox.click()
            window.jump_packet_input.setFocus()
            self.application.processEvents()
            before = dataset_contents(root)
            QTest.keyClicks(window.jump_packet_input, "1032")
            self.assertEqual(window.jump_packet_input.text(), "1032")
            self.assertEqual(window.packet_index, 0)
            self.assertEqual(dataset_contents(root), before)

            window.manual_mode_checkbox.setFocus()
            self.application.processEvents()
            QTest.keyClick(window, Qt.Key.Key_2)
            self.assertEqual(window.packet_index, 1)
            self.assertEqual(window.dataset.load_packet(0).metadata["class_id"], 2)
            QTest.keyClick(window, Qt.Key.Key_1)
            self.assertEqual(window.packet_index, 1)
            self.assertEqual(window.packet.metadata["class_id"], 1)

    def test_save_failure_does_not_advance_or_change_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            window = DatasetBrowserWindow(root)
            self.addCleanup(window.close)
            with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok):
                window.manual_mode_checkbox.click()
            with (
                patch.object(window.dataset, "classify_packet", side_effect=OSError("Read only")),
                patch.object(QMessageBox, "critical") as error,
            ):
                window.class_buttons[1].click()
            error.assert_called_once()
            self.assertEqual(window.packet_index, 0)
            self.assertTrue(all(not button.isChecked() for button in window.class_buttons))

    def test_opening_another_dataset_resets_manual_mode_and_displays_saved_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            other = create_synthetic_dataset(Path(temporary) / "def456")
            LegWheelDataset.open(other).classify_packet(0, 0)
            window = DatasetBrowserWindow(root)
            self.addCleanup(window.close)
            with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok):
                window.manual_mode_checkbox.click()
            self.assertTrue(window.open_dataset(other))
            self.assertFalse(window.manual_mode_checkbox.isChecked())
            self.assertTrue(window.class_buttons[0].isChecked())
            self.assertFalse(window.class_buttons[0].isEnabled())

    def test_window_loads_and_navigates_without_writing_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            manifest_before = (root / "manifest.jsonl").read_bytes()
            window = DatasetBrowserWindow(root)

            self.assertIsNotNone(window.dataset)
            self.assertEqual(window.packet_index, 0)
            self.assertFalse(window.previous_button.isEnabled())
            self.assertTrue(window.next_button.isEnabled())
            self.assertIn("4.250", window.slope_summary.along_label.text())

            window.next_packet()

            self.assertEqual(window.packet_index, 1)
            self.assertTrue(window.previous_button.isEnabled())
            self.assertFalse(window.next_button.isEnabled())
            self.assertIn("Unavailable", window.slope_summary.along_label.text())
            self.assertEqual((root / "manifest.jsonl").read_bytes(), manifest_before)
            window.close()

    def test_window_jumps_to_one_based_sorted_packet_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            window = DatasetBrowserWindow(root)

            # Clicking the button interprets 2 as the second item in the
            # model's already-sorted packet list, not as a stored packet ID.
            window.jump_packet_input.setText("2")
            window.jump_packet_button.click()
            self.assertEqual(window.packet_index, 1)
            self.assertEqual(window.jump_packet_input.text(), "")
            self.assertEqual(window.jump_packet_error_label.text(), "")

            # Pressing Enter uses exactly the same navigation path.
            window.jump_packet_input.setText("1")
            window.jump_packet_input.returnPressed.emit()
            self.assertEqual(window.packet_index, 0)
            self.assertEqual(window.jump_packet_input.text(), "")
            window.close()

    def test_invalid_packet_jump_keeps_current_packet_and_shows_inline_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_synthetic_dataset(Path(temporary) / "abc123")
            window = DatasetBrowserWindow(root)

            for invalid_value in ("", "0", "3", "not-a-number"):
                window.jump_packet_input.setText(invalid_value)
                window.jump_to_packet()
                self.assertEqual(window.packet_index, 0)
                self.assertNotEqual(window.jump_packet_error_label.text(), "")

            window.close()


if __name__ == "__main__":
    unittest.main()
