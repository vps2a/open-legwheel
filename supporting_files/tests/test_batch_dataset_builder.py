"""Batch discovery, wizard and real-process orchestration checks without ROS."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The source packages are pure Python until a bag is opened. Exposing them here
# keeps the supporting-tools tests runnable without building/sourcing ROS.
REPOSITORY = Path(__file__).resolve().parents[2]
for package in ("legwheel_dataset", "legwheel_encoder", "legwheel_kinematics", "legwheel_experiments"):
    sys.path.insert(0, str(REPOSITORY / "src" / package))

import yaml  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from legwheel_dataset.config import ProcessingConfig  # noqa: E402
from legwheel_dataset.config_creator import render_processing_config  # noqa: E402
from legwheel_dataset.processor import dataset_output_directory  # noqa: E402
from legwheel_batch_builder.model import (  # noqa: E402
    ATTEMPT_STATUS_FILE, EVENT_PREFIX, DatasetCounts, archive_incomplete_output,
    count_datasets, dataset_state, discover_experiments, inspect_run, mark_run_valid,
)
from legwheel_batch_builder.runner import BatchRunner  # noqa: E402
from legwheel_batch_builder.window import BatchBuilderWindow, RunSelectionPage  # noqa: E402
from legwheel_batch_builder.worker import build_run  # noqa: E402


EXPERIMENT = {
    "experiment_id": "terrain_test", "rig_config": {}, "leg_config": {},
    "wheel_config": {}, "electronics_hardware": {},
    "environment": {"terrain_and_light_ID": "TL001"},
}


def write_yaml(path, contents):
    path.write_text(yaml.safe_dump(contents, sort_keys=False), encoding="utf-8")


def create_run(experiment, number=1, valid=None):
    run = experiment / f"run_{number}"
    run.mkdir(parents=True)
    write_yaml(run / "run_config.yaml", {
        "run_length_loops": 2, "run_valid": valid, "notes": "Preserve these notes",
    })
    write_yaml(run / f"experiment_config_run_{number}_snapshot.yaml", EXPERIMENT)
    write_yaml(run / f"camera_driver_config_run_{number}_snapshot.yaml", {
        "depth_registration": True, "align_target_stream": "COLOR",
    })
    bag = run / "rosbag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    (bag / "recording.mcap").touch()
    return run


def create_output(output, stage="stage2_complete"):
    output.mkdir(parents=True)
    (output / "packets").mkdir()
    (output / "manifest.jsonl").write_text("")
    (output / "provenance.json").write_text(json.dumps({
        "stage": stage, "terrain_and_light_ID": "TL001", "experiment_config": EXPERIMENT,
    }))
    if stage == "stage2_complete":
        (output / "features.jsonl").write_text("")


class BatchModelTests(unittest.TestCase):
    def test_discovery_uses_contents_and_immediate_children_and_sorts_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment_a"
            create_run(experiment, 10, False)
            create_run(experiment, 2)
            write_yaml(experiment / "an_arbitrary_filename.yml", EXPERIMENT)
            nested = root / "group" / "experiment_nested"
            nested.mkdir(parents=True)
            write_yaml(nested / "experiment.yaml", EXPERIMENT)
            unrelated = root / "unrelated"
            unrelated.mkdir()
            write_yaml(unrelated / "experiment.yaml", {"a": 1})
            experiments, warnings = discover_experiments(root)
            self.assertFalse(warnings)
            self.assertEqual([item.directory.name for item in experiments], ["experiment_a"])
            self.assertEqual([run.number for run in experiments[0].runs], [2, 10])
            self.assertEqual(experiments[0].runs[1].warnings, ["Marked invalid"])
            self.assertFalse(experiments[0].runs[1].problems)

    def test_missing_inputs_stay_visible_and_mark_valid_preserves_other_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            directory = create_run(experiment, valid=False)
            (directory / "rosbag" / "recording.mcap").unlink()
            run = inspect_run(directory, experiment.name)
            self.assertTrue(run.problems)
            run = mark_run_valid(run)
            self.assertIs(run.valid, True)
            self.assertFalse(run.warnings)
            self.assertTrue(run.problems)
            saved = yaml.safe_load((directory / "run_config.yaml").read_text())
            self.assertIs(saved["run_valid"], True)
            self.assertEqual(saved["notes"], "Preserve these notes")
            del saved["run_valid"]
            write_yaml(directory / "run_config.yaml", saved)
            mark_run_valid(run)
            self.assertIs(yaml.safe_load((directory / "run_config.yaml").read_text())["run_valid"], True)

    def test_counts_are_disjoint_and_extra_labels_are_conditional(self):
        self.assertEqual(DatasetCounts(2).label(), "Completed Datasets: 2")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            create_output(run / "derived" / "completed")
            create_output(run / "derived" / "partial", "stage1_complete")
            failed = run / "derived" / "failed"
            create_output(failed, "stage1_complete")
            (failed / "packets" / "analysis_report.json").write_text(json.dumps({
                "status": "failed_pitch_estimation",
            }))
            cancelled = run / "derived" / "cancelled"
            cancelled.mkdir()
            (cancelled / ATTEMPT_STATUS_FILE).write_text('{"status": "cancelled"}')
            (run / "derived" / "unrelated_directory").mkdir()
            counts = count_datasets(run)
            self.assertEqual((counts.completed, counts.partial, counts.failed), (1, 1, 2))
            self.assertIn("Partial Datasets: 1", counts.label())
            self.assertIn("Failed Attempts: 2", counts.label())

    def test_archive_preserves_files_and_refuses_completed_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "hash"
            create_output(output, "stage1_complete")
            (output / "classified_metadata.json").write_text('{"class_id": 0}')
            archived = archive_incomplete_output(output)
            self.assertFalse(output.exists())
            self.assertEqual((archived / "classified_metadata.json").read_text(), '{"class_id": 0}')
            create_output(output)
            with self.assertRaises(ValueError):
                archive_incomplete_output(output)
            self.assertTrue(output.exists())


class BatchWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = create_run(Path(self.temporary.name) / "experiment")
        self.config = ProcessingConfig()
        self.config_path = self.run / "postprocess.yaml"
        self.config_path.write_text(render_processing_config(self.config, "2026-10-30T10:00:00+00:00"))
        self.output = dataset_output_directory(self.run, self.config)

    def test_complete_matching_dataset_is_skipped_without_running_analysis(self):
        create_output(self.output)
        stdout = StringIO()
        with patch("legwheel_dataset.processor.isolate_packets") as isolate, redirect_stdout(stdout):
            result = build_run(self.run, self.config_path)
        self.assertEqual(result, 0)
        isolate.assert_not_called()
        self.assertIn('"status": "skipped"', stdout.getvalue())

    def test_retry_preserves_old_attempt_and_uses_the_production_progress_callbacks(self):
        create_output(self.output, "stage1_complete")
        (self.output / "keep.txt").write_text("Previous classification stays here")

        def isolate(run, config, *, output_directory, progress):
            self.assertEqual(run, self.run)
            self.assertEqual(output_directory, self.output)
            create_output(output_directory, "stage1_complete")
            progress(1.0, "Packets complete")

        def extract(output, config, *, progress):
            (output / "provenance.json").write_text('{"stage": "stage2_complete"}')
            (output / "features.jsonl").write_text("")
            progress(1.0, "Features complete")

        stdout = StringIO()
        with (
            patch("legwheel_dataset.processor.isolate_packets", side_effect=isolate),
            patch("legwheel_dataset.processor.extract_features", side_effect=extract),
            redirect_stdout(stdout),
        ):
            self.assertEqual(build_run(self.run, self.config_path), 0)
        archived = list(self.output.parent.glob(self.output.name + "_attempt_*"))
        self.assertEqual(len(archived), 1)
        self.assertEqual((archived[0] / "keep.txt").read_text(), "Previous classification stays here")
        self.assertEqual(dataset_state(self.output), "completed")
        events = [json.loads(line[len(EVENT_PREFIX):]) for line in stdout.getvalue().splitlines()]
        fractions = [event["fraction"] for event in events if event["type"] == "progress"]
        self.assertEqual(fractions, [0.0, 0.65, 0.99, 1.0])

    def test_stage2_failure_is_counted_as_failed_even_with_stage1_provenance(self):
        def isolate(*args, **kwargs):
            create_output(self.output, "stage1_complete")

        with (
            patch("legwheel_dataset.processor.isolate_packets", side_effect=isolate),
            patch(
                "legwheel_dataset.processor.extract_features", side_effect=ValueError("Plane failure"),
            ),
            redirect_stdout(StringIO()), patch("traceback.print_exc"),
        ):
            self.assertEqual(build_run(self.run, self.config_path), 1)
        self.assertEqual(dataset_state(self.output), "failed")

    def test_early_isolation_failure_is_preserved_and_counted_as_an_attempt(self):
        with (
            patch(
                "legwheel_dataset.processor.isolate_packets", side_effect=ValueError("Invalid geometry"),
            ),
            redirect_stdout(StringIO()), patch("traceback.print_exc"),
        ):
            self.assertEqual(build_run(self.run, self.config_path), 1)
        self.assertTrue((self.output / ATTEMPT_STATUS_FILE).is_file())
        self.assertEqual(count_datasets(self.run).failed, 1)

    def test_output_identity_uses_the_run_snapshot_instead_of_experiment_level_yaml(self):
        write_yaml(self.run.parent / "experiment.yaml", {**EXPERIMENT, "experiment_id": "edited"})
        self.assertEqual(dataset_output_directory(self.run, self.config), self.output)
        snapshot = self.run / "experiment_config_run_1_snapshot.yaml"
        write_yaml(snapshot, {**EXPERIMENT, "experiment_id": "different_snapshot"})
        self.assertNotEqual(dataset_output_directory(self.run, self.config), self.output)


class BatchGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Real child processes need the environment a sourced ROS workspace
        # would provide; the sys.path changes above only affect this process.
        source_paths = [str(REPOSITORY / "src" / name) for name in (
            "legwheel_dataset", "legwheel_encoder", "legwheel_kinematics", "legwheel_experiments",
        )]
        inherited = os.environ.get("PYTHONPATH", "")
        environment = patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join(source_paths + [inherited])})
        environment.start()
        self.addCleanup(environment.stop)
        self.experiment = self.root / "experiment_a"
        self.directories = [create_run(self.experiment, number, False) for number in (1, 2, 3)]
        write_yaml(self.experiment / "custom_name.yaml", EXPERIMENT)

    def wait_until(self, condition, timeout=8):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.application.processEvents()
            QTest.qWait(5)
        self.assertTrue(condition(), "Timed out waiting for the batch process")

    def runner(self, outcomes):
        runner = BatchRunner(sys.executable)
        stub = Path(__file__).with_name("batch_worker_stub.py")
        runner._worker_arguments = lambda run: [
            "-u", str(stub), str(run.directory), outcomes[run.number],
        ]

        def cleanup():
            if runner.running:
                runner.cancel()
                self.wait_until(lambda: not runner.running)

        self.addCleanup(cleanup)
        return runner

    def test_runner_continues_after_failure_and_reports_skipped_runs(self):
        runner = self.runner({1: "completed", 2: "failed", 3: "skipped"})
        runs = [inspect_run(path, self.experiment.name) for path in self.directories]
        progress = []
        runner.progress.connect(lambda overall, *args: progress.append(overall))
        runner.start(runs, ProcessingConfig(spectral_window_ms=750.0))
        self.wait_until(lambda: not runner.running)
        self.assertEqual(
            [result.status for result in runner.results], ["completed", "failed", "skipped"],
        )
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(progress, sorted(progress))
        for directory in self.directories:
            saved = ProcessingConfig.from_yaml(directory / "postprocess.yaml")
            self.assertEqual(saved.spectral_window_ms, 750.0)
        self.assertEqual(count_datasets(self.directories[1]).failed, 1)

    def test_cancel_kills_current_process_preserves_output_and_does_not_start_next_run(self):
        runner = self.runner({1: "hang", 2: "completed", 3: "completed"})
        runs = [inspect_run(path, self.experiment.name) for path in self.directories]
        progress = []
        estimates = []
        runner.progress.connect(lambda overall, *args: progress.append(overall))
        runner.progress.connect(lambda *args: estimates.append(args[-1]))
        runner.start(runs, ProcessingConfig())
        self.wait_until(lambda: runner._fraction == 0.25)
        self.wait_until(lambda: any(estimate != "Estimating…" for estimate in estimates))
        started = time.monotonic()
        runner.cancel()
        self.wait_until(lambda: not runner.running, timeout=3)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(
            [result.status for result in runner.results], ["cancelled", "not_started", "not_started"],
        )
        self.assertLess(progress[-1], 1.0)
        self.assertEqual(count_datasets(self.directories[0]).failed, 1)
        self.assertFalse((self.directories[1] / "derived").exists())
        self.assertFalse((self.directories[0] / "derived/stub_dataset/should_not_exist").exists())

    def test_unavailable_interpreter_reports_failures_without_stalling(self):
        runner = self.runner({1: "completed", 2: "completed", 3: "completed"})
        runner.ros_python = str(self.root / "missing_python")
        runner.start(
            [inspect_run(path, self.experiment.name) for path in self.directories], ProcessingConfig(),
        )
        self.wait_until(lambda: not runner.running)
        self.assertEqual([result.status for result in runner.results], ["failed"] * 3)

    def test_cancellation_after_stage2_finished_does_not_mark_completed_data_failed(self):
        runner = self.runner({1: "late_cancel", 2: "completed", 3: "completed"})
        runs = [inspect_run(path, self.experiment.name) for path in self.directories]
        runner.start(runs, ProcessingConfig())
        self.wait_until(lambda: runner._fraction == 0.99)
        runner.cancel()
        self.wait_until(lambda: not runner.running)
        self.assertEqual(runner.results[0].status, "completed")
        counts = count_datasets(self.directories[0])
        self.assertEqual((counts.completed, counts.failed), (1, 0))
        self.assertEqual(runner.results[1].status, "not_started")

    def test_configuration_write_failure_does_not_prevent_other_runs(self):
        (self.directories[1] / "postprocess.yaml").mkdir()
        runner = self.runner({1: "completed", 2: "completed", 3: "completed"})
        runner.start(
            [inspect_run(path, self.experiment.name) for path in self.directories],
            ProcessingConfig(),
        )
        self.wait_until(lambda: not runner.running)
        statuses = {result.run.number: result.status for result in runner.results}
        self.assertEqual(statuses, {1: "completed", 2: "failed", 3: "completed"})
        self.assertTrue((self.directories[1] / "postprocess.yaml").is_dir())

    def test_wizard_builds_selected_runs_and_displays_the_final_summary(self):
        create_output(self.directories[2] / "derived/stub_dataset")
        window = BatchBuilderWindow(self.root, ros_python=sys.executable)
        self.addCleanup(window.close)
        outcomes = {1: "completed", 2: "failed", 3: "skipped"}
        stub = Path(__file__).with_name("batch_worker_stub.py")
        window.runner._worker_arguments = lambda run: [
            "-u", str(stub), str(run.directory), outcomes[run.number],
        ]
        window._select_experiments(True)
        window.next_button.click()
        window.run_pages[0]._select_all(True)
        window.next_button.click()
        window.settings_page.inputs["plane_ransac_iterations"].setText("35")
        window.next_button.click()
        self.wait_until(lambda: not window.runner.running)
        self.assertIs(window.stack.currentWidget(), window.progress_page)
        statuses = [window.summary_table.item(row, 1).text() for row in range(3)]
        self.assertEqual(statuses, ["Completed", "Failed", "Skipped"])
        self.assertEqual(window.overall_progress.value(), 1000)
        self.assertEqual(window.run_progress.value(), 1000)
        self.assertIn("1 skipped", window.batch_status.text())
        self.assertFalse(window.cancel_button.isEnabled())
        self.assertTrue(window.new_batch_button.isEnabled())
        for directory in self.directories:
            saved = ProcessingConfig.from_yaml(directory / "postprocess.yaml")
            self.assertEqual(saved.plane_ransac_iterations, 35)

    def test_run_page_edits_validity_immediately_and_disables_only_missing_inputs(self):
        (self.directories[1] / "rosbag/recording.mcap").unlink()
        experiments, _ = discover_experiments(self.root)
        page = RunSelectionPage(experiments[0], set())
        self.addCleanup(page.close)
        page._select_all(True)
        self.assertEqual([run.number for run in page.selected_runs()], [1, 3])
        page.table.cellWidget(0, 4).click()
        saved = yaml.safe_load((self.directories[0] / "run_config.yaml").read_text())
        self.assertIs(saved["run_valid"], True)
        self.assertEqual(page.table.item(0, 3).text(), "None")
        self.assertTrue(page.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEnabled)
        self.assertFalse(page.table.item(1, 0).flags() & Qt.ItemFlag.ItemIsEnabled)

    def test_settings_validate_values_and_wizard_uses_one_page_per_experiment(self):
        other = self.root / "experiment_b"
        create_run(other)
        write_yaml(other / "experiment.yml", EXPERIMENT)
        window = BatchBuilderWindow(self.root)
        self.addCleanup(window.close)
        window._select_experiments(True)
        window._next()
        self.assertEqual(len(window.run_pages), 2)
        window.run_pages[0]._select_all(True)
        window._next()
        self.assertIs(window.stack.currentWidget(), window.run_pages[1])
        window.run_pages[1]._select_all(True)
        window._next()
        self.assertIs(window.stack.currentWidget(), window.settings_page)
        self.assertEqual(window.settings_page.configuration(), ProcessingConfig())
        self.assertIn("4 selected runs", window.settings_page.summary.text())
        self.assertFalse(any((path / "postprocess.yaml").exists() for path in self.directories))
        window.settings_page.inputs["alpha_sp_rad"].setText("-1")
        window._next()
        self.assertFalse(window.runner.running)
        self.assertIn("alpha_sp_rad", window.feedback.text())


if __name__ == "__main__":
    unittest.main()
