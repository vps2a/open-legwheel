"""Scientific split/metric invariants and the complete experiment lifecycle."""

import csv
import json
import os
import shutil
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ML_tools.analysis import BAND_CHANNELS, MOTOR_CHANNELS, load_dataset
from ML_tools.experiments import (
    load_experiment,
    save_experiment,
)
from ML_tools.features import (
    ConfigurationError,
    ExperimentConfig,
    FeatureSettings,
    FeatureSpec,
    available_features,
    feature_matrix,
    prepare_data,
    require_trainable,
    transition_exclusions,
)
from ML_tools.training import (
    PowerLogTransformer,
    TrainingCancelled,
    classification_metrics,
    evaluate_test,
    train_experiment,
)
from ML_tools.window import VisualiserWindow
from tests.test_telemetry_visualiser import digests

EPOCH = 1_790_000_000_000_000_000


def make_training_export(root: Path) -> Path:
    """Three physical setups, two runs each, pure and transition-adjacent rows."""
    (root / "packets").mkdir(parents=True)
    rows = []
    for run in range(6):
        setup = f"KK_{run // 2 + 1:02d}"
        origin = EPOCH + run * 100_000_000_000
        for packet, (label, seconds) in enumerate(
            zip((0, 0, 1, 1, 2, 3, None), (1, 2, 4, 5, 2, 8, 6))
        ):
            amplitude = (3 if label == 0 else 0.3) + run * 0.01
            contact = origin + seconds * 1_000_000_000
            values = np.array(
                [
                    [sample * amplitude * (column + 1) for column in range(12)]
                    for sample in range(4)
                ]
            )
            rows.append(
                {
                    "export_id": f"run_{run}:packet_{packet}",
                    "source_id": f"run_{run}",
                    "experiment_id": f"experiment_{run // 2}",
                    "run_number": str(run),
                    "class_id": label,
                    "terrain_setup_id": setup,
                    "terrain_and_light_ID": f"{setup}_L{run % 2 + 1}",
                    "channel_names": list(BAND_CHANNELS),
                    "local_values": values[:2].tolist(),
                    "local_valid": np.ones((2, 12), bool).tolist(),
                    "spectral_values": values.tolist(),
                    "spectral_valid": np.ones((4, 12), bool).tolist(),
                    "band_edges_hz": [[0.0, 10.0], [10.0, 20.0]],
                    "band_power": [
                        [amplitude**2 * (column + 1) * band for column in range(12)]
                        for band in (1, 2)
                    ],
                    "along_slope_rad": 0.1 if label == 0 else 0.0,
                    "cross_slope_rad": 0.01,
                    "fft_start_time_ns": contact - 500_000_000,
                    "fft_end_time_ns": contact + 500_000_000,
                    "metadata_json": json.dumps(
                        {
                            "local_start_time_ns": contact - 200_000_000,
                            "local_end_time_ns": contact + 200_000_000,
                        }
                    ),
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), root / "packets/part-00000.parquet")
    (root / "export_manifest.json").write_text(json.dumps({"export_version": "1.1"}))
    return root


def compact_settings(dataset, **kwargs):
    specs = tuple(
        feature
        for feature in available_features(dataset)
        if feature.kind != "power" or feature.channel in MOTOR_CHANNELS
    )
    return FeatureSettings(specs, **kwargs)


def setup_config(dataset):
    return ExperimentConfig(
        compact_settings(dataset),
        {
            source: ("training", "validation", "testing")[
                int(source.split("_")[1]) // 2
            ]
            for source in set(dataset.source_ids)
        },
    )


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_training_export(Path(self.temporary.name) / "export")
        self.dataset = load_dataset(self.root)
        self.config = setup_config(self.dataset)

    def tearDown(self):
        self.temporary.cleanup()

    def test_feature_order_window_and_common_eligibility(self):
        matrix = feature_matrix(self.dataset, self.config.features)
        self.assertEqual(matrix.shape, (42, 21))
        local = feature_matrix(
            self.dataset, replace(self.config.features, window="local")
        )
        self.assertLess(local[0, 0], matrix[0, 0])
        data = prepare_data(self.dataset, self.config)
        self.assertEqual(
            [
                int(data.masks[role].sum())
                for role in ("training", "validation", "testing")
            ],
            [6, 6, 6],
        )
        self.assertEqual(int(data.transition_excluded.sum()), 6)
        require_trainable(self.dataset, data)

    def test_new_setup_prevents_shared_ground_but_new_run_allows_it(self):
        roles = {source: "excluded" for source in self.config.run_roles}
        roles.update(run_0="training", run_1="validation")
        config = replace(self.config, run_roles=roles)
        with self.assertRaisesRegex(ConfigurationError, "multiple partitions"):
            prepare_data(self.dataset, config)
        data = prepare_data(self.dataset, replace(config, split_mode="run"))
        require_trainable(self.dataset, data)
        self.assertEqual(int(data.masks["training"].sum()), 3)

    def test_transition_closed_bounds_padding_and_integer_nanoseconds(self):
        fft = self.dataset.fft_bounds_ns.copy()
        local = self.dataset.local_bounds_ns.copy()
        # One nanosecond matters at a timestamp far beyond float64 precision.
        end = int(fft[0, 1])
        local[4] = [end + 1, end + 2]
        dataset = replace(self.dataset, local_bounds_ns=local)
        self.assertFalse(transition_exclusions(dataset, {"run_0"}, 0)[0])
        self.assertTrue(transition_exclusions(dataset, {"run_0"}, 1e-9)[0])
        local[4] = [end, end + 2]
        self.assertTrue(
            transition_exclusions(
                replace(dataset, local_bounds_ns=local), {"run_0"}, 0
            )[0]
        )

    def test_transition_annotations_do_not_leak_across_runs(self):
        local = self.dataset.local_bounds_ns.copy()
        local[4] = self.dataset.fft_bounds_ns[
            7
        ]  # run_0 annotation at run_1's clock time
        dataset = replace(self.dataset, local_bounds_ns=local)
        exclusions = transition_exclusions(dataset, {"run_0", "run_1"}, 0)
        self.assertFalse(exclusions[7])

    def test_missing_transition_timing_is_explicit_and_filter_can_be_disabled(self):
        local = self.dataset.local_bounds_ns.copy()
        local[4] = [-1, -1]
        dataset = replace(self.dataset, local_bounds_ns=local)
        with self.assertRaisesRegex(ConfigurationError, "timing is unavailable"):
            prepare_data(dataset, self.config)
        config = replace(
            self.config,
            features=replace(self.config.features, exclude_transitions=False),
        )
        self.assertEqual(int(prepare_data(dataset, config).masks["training"].sum()), 8)

    def test_incomplete_features_drop_rows_and_one_class_training_is_rejected(self):
        slopes = self.dataset.slopes.copy()
        slopes[self.dataset.class_ids == 0, 0] = np.nan
        data = prepare_data(replace(self.dataset, slopes=slopes), self.config)
        self.assertGreater(int(data.missing_features.sum()), 0)
        with self.assertRaisesRegex(ConfigurationError, "eligible firm and.*loose"):
            require_trainable(self.dataset, data)

    def test_training_only_log_floor_and_standardization(self):
        model = PowerLogTransformer((0,)).fit(np.array([[1.0, 4], [10, 8], [0, 12]]))
        np.testing.assert_allclose(model.floors_, [0.01])
        np.testing.assert_allclose(
            model.transform([[0, 2], [1000, 5]]), [[-2, 2], [3, 5]]
        )
        experiment = train_experiment(self.dataset, self.config)
        powers = self.dataset.band_powers.copy()
        powers[~experiment.prepared.masks["training"]] *= 100_000
        changed = train_experiment(
            replace(self.dataset, band_powers=powers), self.config
        )
        for name in experiment.models:
            original = experiment.models[name].pipeline
            altered = changed.models[name].pipeline
            np.testing.assert_array_equal(
                original["power_log"].floors_, altered["power_log"].floors_
            )
            np.testing.assert_array_equal(
                original["standardize"].mean_, altered["standardize"].mean_
            )
            np.testing.assert_array_equal(
                original["standardize"].scale_, altered["standardize"].scale_
            )

    def test_metrics_positive_class_and_undefined_rates(self):
        values = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [-2, 1, -1, 2])
        for key in ("accuracy", "tpr", "fpr", "balanced_accuracy", "precision", "f1"):
            self.assertEqual(values[key], 0.5)
        self.assertEqual(values["roc_auc"], 0.75)
        single = classification_metrics([0, 0], [0, 0], [-2, -1])
        self.assertIsNone(single["tpr"])
        self.assertIsNone(single["precision"])
        self.assertIsNone(single["roc_auc"])

    def test_reserved_test_is_not_evaluated_until_explicit_action_and_never_refitted(
        self,
    ):
        from ML_tools.training import evaluate_model

        with patch(
            "ML_tools.training.evaluate_model", wraps=evaluate_model
        ) as evaluate:
            experiment = train_experiment(self.dataset, self.config)
        self.assertEqual(
            {call.args[3] for call in evaluate.call_args_list},
            {"training", "validation"},
        )
        self.assertNotIn("testing", experiment.evaluations)
        with patch(
            "sklearn.pipeline.Pipeline.fit",
            side_effect=AssertionError("Unexpected refit"),
        ):
            tested = evaluate_test(experiment)
        self.assertIn("testing", tested.evaluations)
        self.assertNotIn("testing", experiment.evaluations)
        self.assertIs(tested.models, experiment.models)

    def test_cancellation_discards_an_incomplete_training_operation(self):
        with self.assertRaises(TrainingCancelled):
            train_experiment(
                self.dataset,
                self.config,
                check_cancel=lambda: (_ for _ in ()).throw(TrainingCancelled()),
            )

    def test_save_reload_exact_models_without_fit_and_without_test_predictions(self):
        before = digests(self.root)
        experiment = train_experiment(self.dataset, self.config)
        folder = save_experiment(experiment, Path(self.temporary.name))
        with (folder / "predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["partition"] for row in rows}, {"training", "validation"})
        with patch(
            "sklearn.pipeline.Pipeline.fit",
            side_effect=AssertionError("Unexpected refit"),
        ):
            restored = load_experiment(folder, self.dataset)
        self.assertEqual(restored.config, experiment.config)
        for role in experiment.evaluations:
            for name, original in experiment.evaluations[role].items():
                actual = restored.evaluations[role][name]
                np.testing.assert_array_equal(actual.predictions, original.predictions)
                np.testing.assert_array_equal(actual.scores, original.scores)
                self.assertEqual(actual.metrics, original.metrics)
        self.assertEqual(digests(self.root), before)
        self.assertTrue((folder / "figures/validation.png").is_file())

    def test_test_results_save_and_reopen_without_new_training(self):
        experiment = evaluate_test(train_experiment(self.dataset, self.config))
        folder = save_experiment(experiment, Path(self.temporary.name))
        restored = load_experiment(folder, self.dataset)
        self.assertIn("testing", restored.evaluations)
        self.assertEqual(
            experiment.evaluations["testing"]["rbf_svm"].metrics,
            restored.evaluations["testing"]["rbf_svm"].metrics,
        )

    def test_moved_dataset_matches_but_changed_labels_do_not(self):
        experiment = train_experiment(self.dataset, self.config)
        folder = save_experiment(experiment, Path(self.temporary.name))
        moved = Path(self.temporary.name) / "moved_export"
        shutil.copytree(self.root, moved)
        self.assertEqual(load_dataset(moved).fingerprint, self.dataset.fingerprint)
        load_experiment(folder, load_dataset(moved))
        table_path = moved / "packets/part-00000.parquet"
        rows = pq.read_table(table_path).to_pylist()
        rows[0]["class_id"] = 1
        pq.write_table(pa.Table.from_pylist(rows), table_path)
        with self.assertRaisesRegex(ConfigurationError, "does not match"):
            load_experiment(folder, load_dataset(moved))

    def test_checksum_failure_precedes_model_deserialization_and_failed_save_cleans_up(
        self,
    ):
        experiment = train_experiment(self.dataset, self.config)
        folder = save_experiment(experiment, Path(self.temporary.name))
        with (folder / "linear_svm.joblib").open("ab") as stream:
            stream.write(b"changed")
        with (
            patch(
                "ML_tools.experiments.joblib.load",
                side_effect=AssertionError("Should not deserialize"),
            ),
            self.assertRaisesRegex(ValueError, "missing or changed"),
        ):
            load_experiment(folder, self.dataset)
        before = set(Path(self.temporary.name).iterdir())
        with (
            patch("ML_tools.experiments.joblib.dump", side_effect=OSError("Disk full")),
            self.assertRaisesRegex(OSError, "Disk full"),
        ):
            save_experiment(experiment, Path(self.temporary.name))
        self.assertEqual(before, set(Path(self.temporary.name).iterdir()))


class TrainingGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_training_export(Path(self.temporary.name) / "export")
        self.dataset = load_dataset(self.root)
        self.window = VisualiserWindow()
        self.window.set_dataset(self.dataset)
        self.page = self.window.training_page
        self.window.pages.setCurrentWidget(self.page)
        self.page.split.restore(setup_config(self.dataset))
        self.page.refresh_summary()

    def tearDown(self):
        if self.page.is_busy:
            self.page.cancel()
            self.wait_for_task()
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.temporary.cleanup()

    def wait_for_task(self):
        deadline = time.monotonic() + 15
        while self.page.is_busy and time.monotonic() < deadline:
            self.application.processEvents()
            QTest.qWait(10)
        self.assertFalse(self.page.is_busy)

    def test_whole_setup_assignments_and_mixed_run_mode(self):
        self.assertTrue(self.page.train_button.isEnabled())
        self.assertEqual(len(self.page.config().features.features), 21)
        self.page.split.mode_combo.setCurrentIndex(1)
        self.page.split.run_combos["run_1"].setCurrentIndex(2)  # validation
        self.assertEqual(self.page.split.setup_combos["KK_01"].currentData(), "mixed")
        self.page.split.mode_combo.setCurrentIndex(0)
        self.assertEqual(self.page.split.run_combos["run_0"].currentData(), "excluded")
        self.assertEqual(self.page.split.run_combos["run_1"].currentData(), "excluded")

    def test_feature_presets_individual_bands_and_display_filters_are_independent(self):
        features = self.page.features
        features.preset.setCurrentIndex(features.preset.findData("std"))
        self.assertEqual(len(features.settings().features), 6)
        feature = FeatureSpec("power", MOTOR_CHANNELS[0], 0)
        features.feature_items[feature].setCheckState(0, Qt.CheckState.Checked)
        self.assertEqual(len(features.settings().features), 7)
        self.assertEqual(features.preset.currentData(), "custom")
        for checkbox in self.window.class_checks:
            checkbox.setChecked(False)
        self.window.set_all_terrains(False)
        self.page.refresh_summary()
        self.assertTrue(self.page.train_button.isEnabled())
        self.assertIn("Training: 6 eligible", self.page.summary.toPlainText())

    def test_worker_results_staleness_and_explicit_test(self):
        self.page.train()
        self.wait_for_task()
        self.assertIsNotNone(self.page.experiment)
        self.assertEqual(self.page.results.metrics.rowCount(), 3)
        self.assertTrue(self.page.save_button.isEnabled())
        self.assertNotIn("testing", self.page.experiment.evaluations)
        self.page.test()
        self.wait_for_task()
        self.assertIn("testing", self.page.experiment.evaluations)
        self.assertEqual(self.page.results.partition.currentData(), "testing")
        self.page.features.linear_c.setValue(2)
        self.page.refresh_summary()
        self.assertTrue(self.page.stale)
        self.assertFalse(self.page.save_button.isEnabled())
        self.assertFalse(self.page.test_button.isEnabled())

    def test_cancelled_dataset_load_clears_pending_experiment_restore(self):
        self.page.pending_experiment = (
            Path(self.temporary.name) / "cancelled_experiment"
        )
        with patch("ML_tools.training_page.load_experiment") as restore:
            self.window.open_dataset(self.root)
            self.window.cancel_load()
            deadline = time.monotonic() + 10
            while self.window.loader is not None and time.monotonic() < deadline:
                self.application.processEvents()
                QTest.qWait(10)
            self.assertIsNone(self.window.loader)
            self.assertIsNone(self.page.pending_experiment)
            restore.assert_not_called()

    def test_reopen_restores_controls_results_and_moved_dataset_prompt(self):
        config = replace(
            setup_config(self.dataset),
            split_mode="run",
            linear_c=2.0,
            rbf_c=3.0,
            rbf_gamma=0.2,
        )
        experiment = train_experiment(self.dataset, config)
        folder = save_experiment(experiment, Path(self.temporary.name))
        self.page.restore_experiment(folder)
        self.assertEqual(self.page.config(), config)
        self.assertEqual(self.page.results.metrics.rowCount(), 3)
        self.assertFalse(self.page.stale)
        moved = Path(self.temporary.name) / "moved_export"
        self.root.rename(moved)
        self.page.dataset = None
        with patch(
            "ML_tools.training_page.QFileDialog.getExistingDirectory",
            return_value=str(moved),
        ) as locate:
            self.page.open_experiment(folder)
            deadline = time.monotonic() + 15
            while self.window.loader is not None and time.monotonic() < deadline:
                self.application.processEvents()
                QTest.qWait(10)
        self.assertTrue(locate.called)
        self.assertIsNotNone(self.page.experiment)
        self.assertEqual(self.page.dataset.root, moved.resolve())
        self.assertEqual(self.page.config(), config)


if __name__ == "__main__":
    unittest.main()
