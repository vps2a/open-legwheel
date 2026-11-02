"""Terrain ID resolution, experiment-wide propagation and safe dataset regeneration."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# These fixtures expose the source packages without requiring a ROS installation.
from tests import test_batch_dataset_builder as fixtures  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402
import yaml  # noqa: E402

from legwheel_dataset.config import ProcessingConfig  # noqa: E402
from legwheel_dataset.config_creator import (  # noqa: E402
    render_processing_config, write_processing_config,
)
from legwheel_dataset.processor import (  # noqa: E402
    dataset_output_directory, extract_features, isolate_packets,
)
from legwheel_dataset.terrain import review_terrain_ids  # noqa: E402
from legwheel_batch_builder.model import (  # noqa: E402
    count_datasets, dataset_requires_regeneration, inspect_run,
)
from legwheel_batch_builder.terrain_ids import TerrainIdsDialog, save_terrain_ids  # noqa: E402
from legwheel_batch_builder.window import BatchBuilderWindow  # noqa: E402
from legwheel_batch_builder.worker import build_run  # noqa: E402


def file_contents(directory):
    return {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def change_yaml_id(path, terrain_id):
    value = yaml.safe_load(path.read_text())
    if terrain_id is None:
        value.setdefault("environment", {}).pop("terrain_and_light_ID", None)
    else:
        value.setdefault("environment", {})["terrain_and_light_ID"] = terrain_id
    fixtures.write_yaml(path, value)


class TerrainIdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = self.root / "experiment_1"
        self.runs = [fixtures.create_run(self.experiment, number) for number in (1, 2)]
        self.config_path = self.experiment / "arbitrary_experiment_name.yml"
        fixtures.write_yaml(self.config_path, fixtures.EXPERIMENT)

    def remove_ids(self):
        change_yaml_id(self.config_path, None)
        for path in self.experiment.glob("run_*/experiment_config_run_*_snapshot.yaml"):
            change_yaml_id(path, None)

    def test_propagation_updates_unselected_runs_and_preserves_dataset_files_and_identity(self):
        self.remove_ids()
        outputs = [run / "derived" / "old_hash" for run in self.runs]
        for output in outputs:
            fixtures.create_output(output)
            provenance = json.loads((output / "provenance.json").read_text())
            provenance.pop("terrain_and_light_ID")
            provenance["experiment_config"]["environment"].pop("terrain_and_light_ID")
            provenance["dataset_identifier"] = "old_hash"
            (output / "provenance.json").write_text(json.dumps(provenance))
            (output / "packets" / "classification.json").write_text('{"class_id": 0}')
            (output / "features.jsonl").write_text('{"class_id": 0, "valid": true}\n')
        before = file_contents(self.experiment)
        review = review_terrain_ids(self.experiment)
        self.assertTrue(review.needs_resolution)
        self.assertEqual(review.affected_datasets("Terrain_A-01"), outputs)

        save_terrain_ids([(review, "Terrain_A-01")])

        after = file_contents(self.experiment)
        self.assertEqual(set(before), set(after))
        for path in before:
            if (
                path.name not in {"provenance.json", self.config_path.name}
                and "snapshot.yaml" not in path.name
            ):
                self.assertEqual(before[path], after[path])
        self.assertFalse(review_terrain_ids(self.experiment).needs_resolution)
        for run, output in zip(self.runs, outputs):
            snapshot = run / f"experiment_config_{run.name}_snapshot.yaml"
            saved = yaml.safe_load(snapshot.read_text())
            self.assertEqual(saved["environment"]["terrain_and_light_ID"], "Terrain_A-01")
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["terrain_and_light_ID"], "Terrain_A-01")
            self.assertEqual(provenance["dataset_identifier"], "old_hash")
            self.assertTrue(provenance["requires_regeneration"])
            self.assertEqual(count_datasets(run).completed, 1)
            warnings = inspect_run(run, "experiment_1").warnings
            self.assertTrue(any("regeneration" in warning for warning in warnings))
        stable = file_contents(self.experiment)
        save_terrain_ids([(review_terrain_ids(self.experiment), "Terrain_A-01")])
        self.assertEqual(file_contents(self.experiment), stable)

    def test_case_conflicts_require_an_explicit_choice_and_warning_lists_affected_datasets(self):
        change_yaml_id(self.config_path, "tl001")
        output = self.runs[1] / "derived" / "old_hash"
        fixtures.create_output(output)
        review = review_terrain_ids(self.experiment)
        self.assertEqual(review.known_ids, ["TL001", "tl001"])
        dialog = TerrainIdsDialog([review])
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.inputs[0].currentText(), "")
        self.assertFalse(dialog.save_button.isEnabled())
        dialog.inputs[0].setCurrentText("tl001")
        self.assertTrue(dialog.save_button.isEnabled())
        self.assertIn("need to be regenerated", dialog.warning.text())
        self.assertIn(str(output), dialog.affected_paths.toPlainText())
        dialog.inputs[0].setCurrentText("invalid value")
        self.assertFalse(dialog.save_button.isEnabled())
        dialog.inputs[0].setCurrentText("tl001")
        dialog.save_button.click()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(review_terrain_ids(self.experiment).known_ids, ["tl001"])

    def test_shared_ids_are_allowed_across_experiments(self):
        self.remove_ids()
        other = self.root / "experiment_2"
        fixtures.create_run(other)
        fixtures.write_yaml(other / "experiment.yaml", fixtures.EXPERIMENT)
        reviews = [review_terrain_ids(path) for path in (self.experiment, other)]
        save_terrain_ids([(review, "Shared_001-A") for review in reviews])
        for directory in (self.experiment, other):
            review = review_terrain_ids(directory)
            self.assertFalse(review.needs_resolution)
            self.assertEqual(review.known_ids, ["Shared_001-A"])

    def test_write_failure_restores_every_previously_written_file(self):
        self.remove_ids()
        review = review_terrain_ids(self.experiment)
        before = file_contents(self.experiment)
        calls = 0

        def fail_second(path, text):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Synthetic write failure")
            write_processing_config(path, text)

        with patch(
            "legwheel_batch_builder.terrain_ids.write_processing_config", side_effect=fail_second,
        ):
            with self.assertRaises(OSError):
                save_terrain_ids([(review, "New_ID")])
        self.assertEqual(file_contents(self.experiment), before)

    def test_dialog_cancellation_leaves_metadata_and_selection_step_unchanged(self):
        self.remove_ids()
        window = BatchBuilderWindow(self.root)
        self.addCleanup(window.close)
        before = file_contents(self.experiment)
        window._select_experiments(True)
        with patch.object(TerrainIdsDialog, "exec", return_value=QDialog.DialogCode.Rejected) as show:
            window.next_button.click()
        show.assert_called_once()
        self.assertIs(window.stack.currentWidget(), window.experiments_page)
        self.assertEqual(window.run_pages, [])
        self.assertEqual(file_contents(self.experiment), before)

    def test_wizard_saves_ids_before_showing_runs_and_includes_unselected_runs(self):
        self.remove_ids()
        window = BatchBuilderWindow(self.root)
        self.addCleanup(window.close)
        window._select_experiments(True)

        def enter_id(dialog):
            dialog.inputs[0].setCurrentText("Setup_A-01")
            dialog.save_button.click()
            return dialog.result()

        with patch.object(TerrainIdsDialog, "exec", new=enter_id):
            window.next_button.click()
        self.assertIs(window.stack.currentWidget(), window.run_pages[0])
        self.assertEqual(window.run_pages[0].selected_runs(), [])
        self.assertEqual(review_terrain_ids(self.experiment).known_ids, ["Setup_A-01"])

    def test_external_metadata_changes_are_not_overwritten(self):
        review = review_terrain_ids(self.experiment)
        config = yaml.safe_load(self.config_path.read_text())
        config["environment"]["notes"] = "An external edit"
        fixtures.write_yaml(self.config_path, config)
        before = file_contents(self.experiment)
        with self.assertRaisesRegex(ValueError, "Metadata changed"):
            save_terrain_ids([(review, "Changed_ID")])
        self.assertEqual(file_contents(self.experiment), before)

    def test_individual_builds_reject_missing_and_conflicting_ids_before_analysis(self):
        snapshot = self.runs[0] / "experiment_config_run_1_snapshot.yaml"
        change_yaml_id(snapshot, None)
        with patch("legwheel_dataset.processor.index_bag") as index:
            with self.assertRaisesRegex(ValueError, "batch dataset builder"):
                isolate_packets(self.runs[0], ProcessingConfig())
        index.assert_not_called()
        self.assertFalse((self.runs[0] / "derived").exists())
        change_yaml_id(snapshot, "Conflicting_ID")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            dataset_output_directory(self.runs[0], ProcessingConfig())

    def test_id_changes_participate_in_dataset_identity(self):
        before = dataset_output_directory(self.runs[0], ProcessingConfig())
        save_terrain_ids([(review_terrain_ids(self.experiment), "tl001")])
        after = dataset_output_directory(self.runs[0], ProcessingConfig())
        self.assertNotEqual(before, after)

    def test_stage2_cannot_clear_a_regeneration_warning(self):
        output = self.runs[0] / "derived" / "old_hash"
        fixtures.create_output(output, stage="stage1_complete")
        save_terrain_ids([(review_terrain_ids(self.experiment), "Changed_ID")])
        before = file_contents(output)
        with self.assertRaisesRegex(ValueError, "requires regeneration"):
            extract_features(output)
        self.assertEqual(file_contents(output), before)

    def test_worker_rebuilds_a_flagged_completed_dataset_even_when_its_hash_is_unchanged(self):
        run = self.runs[0]
        config_path = run / "postprocess.yaml"
        config_path.write_text(render_processing_config(ProcessingConfig(), "2026-10-30T00:00:00+00:00"))
        output = dataset_output_directory(run, ProcessingConfig())
        fixtures.create_output(output)
        provenance = json.loads((output / "provenance.json").read_text())
        provenance["requires_regeneration"] = True
        (output / "provenance.json").write_text(json.dumps(provenance))
        (output / "packets/keep_classification.json").write_text('{"class_id": 0}')

        def isolate(*args, output_directory, **kwargs):
            fixtures.create_output(output_directory)

        with (
            patch("legwheel_dataset.processor.isolate_packets", side_effect=isolate) as build,
            patch("legwheel_dataset.processor.extract_features"), redirect_stdout(StringIO()),
        ):
            self.assertEqual(build_run(run, config_path), 0)
        build.assert_called_once()
        archived = list(output.parent.glob(output.name + "_attempt_*"))
        self.assertEqual(len(archived), 1)
        preserved = archived[0] / "packets/keep_classification.json"
        self.assertEqual(preserved.read_text(), '{"class_id": 0}')
        self.assertFalse(dataset_requires_regeneration(output))


if __name__ == "__main__":
    unittest.main()
