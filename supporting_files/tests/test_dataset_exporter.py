"""Scientific calculations, portable Parquet exports, and GUI selection tests."""

import itertools
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from legwheel_dataset_exporter.calculations import (
    TAU,
    WindowCoverage,
    analyze_overlaps,
    band_power,
    frequency_bands,
    spatial_overlap,
    window_coverage,
)
from legwheel_dataset_exporter.export import (
    ExportCancelled,
    ExportOptions,
    export_dataset,
)
from legwheel_dataset_exporter.sources import (
    discover_sources,
    load_source,
    terrain_parts,
)
from legwheel_dataset_exporter.window import ExporterWindow
from tests.synthetic_dataset import CHANNEL_NAMES

EPOCH = 1_792_061_700_000_000_000


def write_json(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def make_source(root: Path, *, run=1, count=4) -> Path:
    """Build valid processed packets without importing any ROS/source packages."""
    root.mkdir(parents=True)
    (root / "packets").mkdir()
    write_json(
        root / "provenance.json",
        {
            "stage": "stage2_complete",
            "dataset_identifier": "configuration_1",
            "terrain_and_light_ID": "KK_01_L1",
            "run_directory": "/machine/local/run",
            "experiment_config": {"environment": {"terrain_and_light_ID": "KK_01_L1"}},
            "run_config": {"torque": 3},
            "processing_config": {"telemetry_resample_hz": 100, "depth_scale_m": 0.001},
        },
    )
    ticks = EPOCH + np.arange(101, dtype=np.int64) * 100_000_000
    np.savez_compressed(
        root / "packets/run_motion_analysis.npz",
        tick_times_ns=ticks,
        theta_y_motion_at_ticks_rad=np.arange(101) / 10,
    )
    records, features = [], []
    for index in range(count):
        packet_id = f"packet_{index}"
        directory = root / "packets" / packet_id
        stage2 = directory / "stage2"
        stage2.mkdir(parents=True)
        contact = EPOCH + 1_000_000_000 + index * 500_000_000
        records.append(
            {
                "packet_id": packet_id,
                "packet_path": f"packets/{packet_id}",
                "image_time_ns": contact - 500_000_000,
            }
        )
        classification = {
            "class_id": 0 if index == 0 else None,
            "classification_method": "manual" if index == 0 else None,
            "model_id": None,
        }
        write_json(
            directory / "metadata.json",
            {
                "packet_id": packet_id,
                "experiment_id": "experiment_1",
                "run_number": run,
                "contact_time_ns": contact,
                "image_time_ns": contact - 500_000_000,
                "lap_index": 0,
                "telemetry_channel_names": CHANNEL_NAMES,
                **classification,
            },
        )
        feature = {
            "packet_id": packet_id,
            "valid": True,
            "along_slope_deg": 3.5,
            "rgb_crop_origin_uv": [1, 2],
            "depth_crop_origin_uv": [1, 2],
            "custom_future_feature": 42,
            **classification,
        }
        features.append(feature)
        write_json(stage2 / "features.json", feature)
        times = np.arange(-50, 51, dtype=np.int64) * 10_000_000
        values = np.column_stack([np.sin(np.arange(101) + channel) for channel in range(15)])
        np.savez_compressed(
            directory / "telemetry.npz",
            local_values=values[:5],
            local_valid=np.ones((5, 15), dtype=bool),
            local_relative_times_ns=times[:5],
            spectral_values=values,
            spectral_valid=np.ones((101, 15), dtype=bool),
            spectral_relative_times_ns=times,
        )
        np.savez_compressed(
            stage2 / "features.npz",
            frequencies_hz=np.fft.rfftfreq(101, d=0.01),
            spectral_power=np.ones((51, 15)),
            dominant_frequency_hz=np.ones((2, 15)),
            dominant_power=np.ones((2, 15)),
        )
        np.savez_compressed(
            directory / "geometry.npz", rgb_patch_polygon_pixels=np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        rgb = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        depth = np.arange(20, dtype=np.uint16).reshape(4, 5) * 3000
        depth[0, 0] = 65535
        np.save(stage2 / "rgb_patch.npy", rgb)
        np.save(stage2 / "depth_patch.npy", depth)
        np.save(stage2 / "rgb_patch_mask.npy", depth != 0)
        np.save(stage2 / "depth_patch_mask.npy", depth != 0)
        np.save(directory / "rgb.npy", rgb)
        np.save(directory / "depth.npy", depth)
        np.savez_compressed(
            stage2 / "point_map.npz",
            points_base_m=np.ones((2, 3)),
            pixel_uv=np.ones((2, 2)),
            depth_m=np.ones(2),
            ransac_inlier_mask=np.array([True, False]),
        )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in reversed(records)) + "\n"
    )
    (root / "features.jsonl").write_text(
        "\n".join(json.dumps(record) for record in features) + "\n"
    )
    return root


def contents(root):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


class CalculationTests(unittest.TestCase):
    def test_band_power_excludes_dc_and_assigns_edges_once(self):
        frequencies = np.arange(51, dtype=float)
        power = np.full((51, 2), 2.0)
        power[0] = 1000
        power[:, 1] = np.nan
        result = band_power(frequencies, power, frequency_bands(10, 50), 50)
        np.testing.assert_allclose(result[:, 0], [18, 20, 20, 20, 22])
        self.assertEqual(result[:, 0].sum(), 100)
        self.assertTrue(np.isnan(result[:, 1]).all())

    def test_sinusoid_power_and_unsupported_bands(self):
        signal = np.sin(2 * np.pi * 7 * np.arange(100) / 100)
        psd = abs(np.fft.rfft(signal))[:, None] ** 2 / (100 * 100)
        psd[1:-1] *= 2
        result = band_power(np.fft.rfftfreq(100, 0.01), psd, frequency_bands(10, 50), 50)
        self.assertAlmostEqual(result[0, 0], 0.5)
        self.assertAlmostEqual(result[1:, 0].sum(), 0)
        clipped = frequency_bands(10, 25)
        np.testing.assert_array_equal(clipped[-1], [20, 25])
        lower_rate = band_power(
            np.arange(26, dtype=float), np.ones((26, 1)), frequency_bands(10, 50), 25
        )
        self.assertTrue(np.isnan(lower_rate[2:]).all())
        for invalid in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                frequency_bands(invalid, 50)

    def test_coverage_integer_precision_and_conservative_bounds(self):
        times = EPOCH + np.array([0, 100, 200, 300], dtype=np.int64)
        coverage = window_coverage(
            EPOCH, np.array([51, 249], dtype=np.int64), times, np.array([0.0, 1.0, 2.0, 3.0]), 100
        )
        self.assertAlmostEqual(coverage.start_angle_rad, 0.51)
        self.assertAlmostEqual(coverage.end_angle_rad, 2.49)
        self.assertEqual((coverage.lower_angle_rad, coverage.upper_angle_rad), (0.0, 3.0))
        reverse = window_coverage(
            EPOCH, np.array([100, 200]), times, np.array([3.0, 2.0, 1.0, 0.0]), 100
        )
        self.assertEqual((reverse.lower_angle_rad, reverse.upper_angle_rad), (1.0, 2.0))

    def test_coverage_blocks_range_gaps_and_reversals(self):
        times = np.array([0, 100, 200, 300], dtype=np.int64)
        for relative, angles, gap in (
            ([-1, 50], [0.0, 1.0, 2.0, 3.0], 100),
            ([0, 301], [0.0, 1.0, 2.0, 3.0], 100),
            ([0, 300], [0.0, 1.0, 2.0, 3.0], 99),
            ([0, 300], [0.0, 1.0, 0.0, 2.0], 100),
        ):
            with self.assertRaises(ValueError):
                window_coverage(0, np.array(relative), times, np.array(angles), gap)

    def test_temporal_chain_counts_and_circular_laps(self):
        windows = [
            WindowCoverage(start, end, low, high, low, high)
            for start, end, low, high in (
                (0, 10, 6.0, 6.5),
                (10, 20, 0.1, 0.4),
                (20, 30, 2.0, 2.5),
                (40, 50, 2.0 + TAU, 2.5 + TAU),
            )
        ]
        temporal, spatial, time_count, space_count = analyze_overlaps(windows)
        self.assertEqual(temporal, [[1], [0, 2], [1], []])
        self.assertEqual(spatial, [[1], [0], [3], [2]])
        self.assertEqual(time_count, 3)
        self.assertEqual(space_count, 2)

    def test_maximum_circular_count_matches_exhaustive_search(self):
        rng = np.random.default_rng(101)
        for _ in range(35):
            starts = rng.uniform(-TAU, 2 * TAU, 8)
            widths = rng.uniform(0, TAU + 0.5, 8)
            windows = [
                WindowCoverage(i, i + 1, start, start + width, start, start + width)
                for i, (start, width) in enumerate(zip(starts, widths))
            ]
            brute = 0
            for mask in range(1 << len(windows)):
                subset = [window for index, window in enumerate(windows) if mask & (1 << index)]
                if all(not spatial_overlap(a, b) for a, b in itertools.combinations(subset, 2)):
                    brute = max(brute, len(subset))
            self.assertEqual(analyze_overlaps(windows)[3], brute)


class ExportTests(unittest.TestCase):
    def test_roundtrip_optional_images_labels_masks_and_source_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_source(base / "source")
            before = contents(root)
            source = load_source(root)
            self.assertFalse(any(packet.problem for packet in source.packets))
            output = export_dataset(
                source.packets,
                base,
                "terrain",
                ExportOptions(include_full_frames=True, include_point_maps=True),
            )
            self.assertEqual(contents(root), before)
            dataset = Dataset.from_parquet(
                str(output.directory / "packets/part-00000.parquet"), cache_dir=str(base / "cache")
            )
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.features["class_id"].names[0], "Firm Terrain")
            first = dataset[0]
            self.assertEqual(first["class_id"], 0)
            self.assertIsNone(dataset[1]["class_id"])
            self.assertIsNone(first["split"])
            self.assertEqual(first["terrain_setup_id"], "KK_01")
            self.assertEqual(first["lighting_id"], "L1")
            self.assertEqual(first["contact_time_ns"], EPOCH + 1_000_000_000)
            np.testing.assert_array_equal(
                np.asarray(first["depth_patch"]),
                np.load(root / "packets/packet_0/stage2/depth_patch.npy"),
            )
            self.assertEqual(np.asarray(first["depth_frame"])[0, 0], 65535)
            self.assertEqual(np.asarray(first["depth_patch_mask"])[0, 0], 255)
            self.assertEqual(first["point_map"]["ransac_inlier_mask"], [True, False])
            self.assertEqual(np.asarray(first["band_power"]).shape, (5, 15))
            self.assertEqual(first["temporal_overlap_packet_numbers"], [2, 3])
            self.assertNotIn(first["export_id"], first["temporal_overlap_ids"])
            self.assertEqual(json.loads(first["features_json"])["custom_future_feature"], 42)
            run = pq.read_table(output.directory / "runs.parquet").to_pylist()[0]
            self.assertEqual(run["maximum_temporal_nonoverlapping_count"], 2)
            self.assertNotIn("/machine/local", run["provenance_json"])
            self.assertIn("split: unassigned", (output.directory / "README.md").read_text())

    def test_hash_is_order_path_and_name_independent_but_selection_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = load_source(make_source(base / "source"))
            first = export_dataset(source.packets, base, "first")
            second = export_dataset(list(reversed(source.packets)), base, "second")
            digest = lambda output: json.loads(
                (output.directory / "export_manifest.json").read_text()
            )["sha256"]
            self.assertEqual(digest(first), digest(second))
            subset = export_dataset(source.packets[::2], base, "subset")
            self.assertNotEqual(digest(first), digest(subset))
            rows = pq.read_table(subset.directory / "packets").to_pylist()
            self.assertEqual([row["packet_number"] for row in rows], [1, 3])
            self.assertEqual(rows[0]["temporal_overlap_packet_numbers"], [3])
            self.assertNotIn("rgb_frame", rows[0])
            self.assertNotIn("point_map", rows[0])
            changed = export_dataset(source.packets, base, "bands", ExportOptions(band_width_hz=5))
            self.assertNotEqual(digest(first), digest(changed))
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_dataset(source.packets, base, "first")
            self.assertFalse(list(base.glob(".export-*")))

    def test_fresh_labels_and_configuration_change_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = load_source(make_source(base / "source"))
            first = export_dataset(source.packets, base, "first")
            for relative in (
                "packets/packet_0/metadata.json",
                "packets/packet_0/stage2/features.json",
            ):
                path = source.root / relative
                document = json.loads(path.read_text())
                document["class_id"] = 2
                write_json(path, document)
            path = source.root / "features.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["class_id"] = 2
            path.write_text("\n".join(json.dumps(record) for record in records))
            second = export_dataset(source.packets, base, "second")
            self.assertNotEqual(
                first.directory.name.split("_")[-1], second.directory.name.split("_")[-1]
            )
            self.assertEqual(pq.read_table(second.directory / "packets")["class_id"][0].as_py(), 2)

            provenance_path = source.root / "provenance.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["run_config"]["torque"] = 5
            write_json(provenance_path, provenance)
            third = export_dataset(source.packets, base, "third")
            self.assertNotEqual(
                second.directory.name.split("_")[-1], third.directory.name.split("_")[-1]
            )

    def test_multiple_runs_stay_separate_and_hub_card_loads_unassigned_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = load_source(make_source(base / "first", run=1))
            second = load_source(make_source(base / "second", run=2))
            result = export_dataset(first.packets + second.packets, base, "combined")
            self.assertEqual(len(result.runs), 2)
            rows = pq.read_table(result.directory / "packets").to_pylist()
            ids = {row["export_id"]: row["source_id"] for row in rows}
            self.assertEqual(len(ids), 8)
            for row in rows:
                for reference in row["temporal_overlap_ids"] + row["spatial_overlap_ids"]:
                    self.assertEqual(ids[reference], row["source_id"])
            dataset = load_dataset(
                str(result.directory), "packets", split="unassigned", cache_dir=str(base / "cache")
            )
            self.assertEqual(len(dataset), 8)
            runs = load_dataset(
                str(result.directory), "runs", split="unassigned", cache_dir=str(base / "cache")
            )
            self.assertEqual(len(runs), 2)

    def test_sharding_and_moving_sources_preserve_content_identity(self):
        import shutil

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = load_source(make_source(base / "source"))
            first = export_dataset(source.packets, base, "first")
            shutil.copytree(source.root, base / "moved_source")
            moved = load_source(base / "moved_source")
            other_destination = base / "exports"
            other_destination.mkdir()
            with (
                patch("legwheel_dataset_exporter.export.ROWS_PER_GROUP", 1),
                patch("legwheel_dataset_exporter.export.SHARD_TARGET_BYTES", 1),
            ):
                second = export_dataset(moved.packets, other_destination, "second")
            self.assertEqual(len(list(second.directory.glob("packets/*.parquet"))), 4)
            self.assertEqual(
                first.directory.name.split("_")[-1], second.directory.name.split("_")[-1]
            )
            self.assertEqual(pq.read_table(second.directory / "packets").num_rows, 4)

    def test_block_missing_coverage_and_conflicting_source_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = load_source(make_source(base / "first"))
            other = load_source(make_source(base / "other"))
            with self.assertRaisesRegex(ValueError, "one source processing version"):
                export_dataset([source.packets[0], other.packets[1]], base, "bad")
            (source.root / "packets/run_motion_analysis.npz").unlink()
            loaded = load_source(source.root)
            self.assertTrue(all(packet.problem for packet in loaded.packets))
            with self.assertRaisesRegex(ValueError, "Cannot map FFT positions"):
                export_dataset(source.packets, base, "bad")

    def test_event_interval_limit_is_two_seconds_independent_of_pitch_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = make_source(Path(temporary) / "source")
            provenance_path = root / "provenance.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["pitch_estimation"] = {"maximum_encoder_gap_s": 0.25}
            write_json(provenance_path, provenance)

            def save_movement_events(interval_ns):
                np.savez_compressed(
                    root / "packets/run_motion_analysis.npz",
                    tick_times_ns=EPOCH + np.arange(4, dtype=np.int64) * interval_ns,
                    theta_y_motion_at_ticks_rad=np.arange(4, dtype=float) * 0.1,
                )

            # Slow, continuous motion can produce sparse count-change events
            # even though the encoder publishes unchanged readings frequently.
            save_movement_events(2_000_000_000)
            source = load_source(root)
            self.assertEqual(source.maximum_encoder_gap_ns, 2_000_000_000)
            self.assertFalse(any(packet.problem for packet in source.packets))

            # A larger pitch threshold must not relax the exporter's own limit.
            provenance["pitch_estimation"]["maximum_encoder_gap_s"] = 10.0
            write_json(provenance_path, provenance)
            save_movement_events(2_000_000_001)
            source = load_source(root)
            self.assertEqual(source.maximum_encoder_gap_ns, 2_000_000_000)
            self.assertTrue(
                all("encoder movement events" in packet.problem for packet in source.packets)
            )
            self.assertTrue(all("limit is 2 s" in packet.problem for packet in source.packets))

    def test_cancellation_and_corrupt_data_leave_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = load_source(make_source(base / "source"))
            state = {"cancel": False}

            def progress(percent, message):
                if percent >= 20:
                    state["cancel"] = True

            def cancel():
                if state["cancel"]:
                    raise ExportCancelled()

            with self.assertRaises(ExportCancelled):
                export_dataset(
                    source.packets, base, "cancelled", progress=progress, check_cancel=cancel
                )
            self.assertFalse(list(base.glob(".export-*")))
            np.save(
                source.root / "packets/packet_0/stage2/depth_patch.npy",
                np.ones((4, 5), dtype=np.uint8),
            )
            with self.assertRaisesRegex(ValueError, "16-bit depth"):
                export_dataset(source.packets, base, "bad")
            self.assertFalse(list(base.glob(".export-*")))
            self.assertEqual(sorted(path.name for path in base.iterdir()), ["source"])

    def test_discovery_and_terrain_parsing_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_source(base / "experiment/run_1/derived/configuration")
            before = contents(root)
            sources, warnings = discover_sources(base)
            self.assertEqual(len(sources), 1)
            self.assertEqual(warnings, [])
            self.assertEqual(contents(root), before)
            self.assertEqual(
                terrain_parts({"terrain_and_light_ID": "KK_01_L2"}),
                ("KK_01_L2", "KK", "KK_01", "L2"),
            )


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def wait_for_task(self, window):
        deadline = time.monotonic() + 10
        while window.task is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.assertIsNone(window.task, "Background operation did not finish.")

    def test_run_checkbox_selects_exportable_packets_when_one_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = make_source(Path(temporary) / "source")
            (root / "packets/packet_1/stage2/rgb_patch.npy").unlink()
            source = load_source(root)
            window = ExporterWindow()
            try:
                window.sources = {source.root: source}
                window.populate_tree(set())
                experiment = window.tree.topLevelItem(0)
                run = experiment.child(0)
                version = run.child(0)
                blocked = window.packet_items[1]
                self.assertTrue(blocked.data(0, Qt.ItemDataRole.UserRole).problem)

                # Qt's automatic tri-state logic previously hid this checkbox
                # because the blocked child intentionally had no check state.
                self.assertIsNotNone(run.data(0, Qt.ItemDataRole.CheckStateRole))
                run.setCheckState(0, Qt.CheckState.Checked)
                self.assertEqual([packet.number for packet in window.selected_packets()], [1, 3, 4])
                for group in (version, run, experiment):
                    self.assertEqual(group.checkState(0), Qt.CheckState.Checked)
                self.assertNotEqual(blocked.checkState(0), Qt.CheckState.Checked)

                # Individual changes are reflected by every ancestor, counting
                # only eligible packets when deciding the partial/full state.
                window.packet_items[0].setCheckState(0, Qt.CheckState.Unchecked)
                for group in (version, run, experiment):
                    self.assertEqual(group.checkState(0), Qt.CheckState.PartiallyChecked)
                run.setCheckState(0, Qt.CheckState.Unchecked)
                self.assertEqual(window.selected_packets(), [])
                experiment.setCheckState(0, Qt.CheckState.Checked)
                self.assertEqual(len(window.selected_packets()), 3)
                version.setCheckState(0, Qt.CheckState.Unchecked)
                self.assertEqual(window.selected_packets(), [])
            finally:
                window.close()

    def test_filtered_bulk_selection_and_fully_blocked_runs_have_consistent_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            mixed_root = make_source(base / "mixed", run=1)
            (mixed_root / "packets/packet_1/stage2/rgb_patch.npy").unlink()
            blocked_root = make_source(base / "blocked", run=2)
            for path in blocked_root.glob("packets/*/stage2/rgb_patch.npy"):
                path.unlink()
            mixed, blocked = load_source(mixed_root), load_source(blocked_root)
            window = ExporterWindow()
            try:
                window.sources = {mixed.root: mixed, blocked.root: blocked}
                window.populate_tree(set())
                experiment = window.tree.topLevelItem(0)
                mixed_run, blocked_run = experiment.child(0), experiment.child(1)
                self.assertIsNone(blocked_run.data(0, Qt.ItemDataRole.CheckStateRole))
                self.assertFalse(blocked_run.flags() & Qt.ItemFlag.ItemIsUserCheckable)

                window.class_filter.setCurrentIndex(1)  # Unlabelled only.
                window.select_visible(True)
                self.assertEqual([packet.number for packet in window.selected_packets()], [3, 4])
                self.assertEqual(mixed_run.checkState(0), Qt.CheckState.PartiallyChecked)
                window.select_visible(False)
                self.assertEqual(mixed_run.checkState(0), Qt.CheckState.Unchecked)

                # Whole-run selection includes hidden eligible packets, whereas
                # Select visible respects the filter. Neither includes blockers.
                mixed_run.setCheckState(0, Qt.CheckState.Checked)
                self.assertEqual([packet.number for packet in window.selected_packets()], [1, 3, 4])
                self.assertEqual(experiment.checkState(0), Qt.CheckState.Checked)
                window.class_filter.setCurrentIndex(0)
                self.assertEqual(len(window.selected_packets()), 3)
                self.assertTrue(all(not packet.problem for packet in window.selected_packets()))
            finally:
                window.close()

    def test_gui_discovers_and_exports_through_background_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_source(base / "source")
            before = contents(root)
            window = ExporterWindow()
            try:
                with patch("legwheel_dataset_exporter.window.QMessageBox.critical") as errors:
                    window.scan(root)
                    self.wait_for_task(window)
                    window.select_visible(True)
                    window.name.setText("gui_export")
                    window.destination.setText(str(base))
                    window.start_export()
                    self.wait_for_task(window)
                    errors.assert_not_called()
                self.assertIn("Export complete", window.status.text())
                output = next(base.glob("gui_export_*"))
                self.assertEqual(pq.read_table(output / "packets").num_rows, 4)
                self.assertEqual(contents(root), before)
            finally:
                window.close()
                if window.task is not None:
                    self.wait_for_task(window)

    def test_filter_keeps_selection_and_conflicting_versions_block_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = load_source(make_source(base / "first"))
            second = load_source(make_source(base / "second"))
            window = ExporterWindow()
            try:
                window.sources = {source.root: source, second.root: second}
                window.populate_tree(set())
                self.assertEqual(window.band_width.value(), 10)
                self.assertFalse(window.full_frames.isChecked())
                self.assertFalse(window.point_maps.isChecked())
                item = window.packet_items[0]
                item.setCheckState(0, Qt.CheckState.Checked)
                self.assertEqual(len(window.selected_packets()), 1)
                window.class_filter.setCurrentIndex(1)  # Unlabelled only.
                self.assertTrue(item.isHidden())
                self.assertEqual(len(window.selected_packets()), 1)
                window.class_filter.setCurrentIndex(0)
                window.select_visible(True)
                self.assertFalse(window.export_button.isEnabled())
                self.assertIn("one processing version", window.summary.toPlainText())
                window.select_visible(False)
                self.assertEqual(window.selected_packets(), [])
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
