"""Exercise complete-build progress without requiring a ROS installation."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from legwheel_dataset import bag_reader
from legwheel_dataset.cli import build_dataset_main
from legwheel_dataset.progress import TerminalProgress, progress_items


class ProgressTest(unittest.TestCase):
    def test_complete_build_uses_one_monotonic_bar_and_keeps_result_on_stdout(self):
        def isolate(*args, progress, **kwargs):
            progress(0.0, "Reading bag")
            progress(0.5, "Isolating packets")
            progress(1.0, "Packet isolation complete")
            return Path("/run/derived/hash")

        def extract(*args, progress):
            progress(0.0, "Extracting packet features")
            progress(1.0, "Feature extraction complete")

        stdout, stderr = StringIO(), StringIO()
        with (
            patch("legwheel_dataset.cli.isolate_packets", side_effect=isolate),
            patch("legwheel_dataset.cli.extract_features", side_effect=extract),
            redirect_stdout(stdout), redirect_stderr(stderr),
        ):
            build_dataset_main(["--run-directory", "/recording/run_1"])
        lines = stderr.getvalue().splitlines()
        percentages = [int(line.split("] ")[1].split("%")[0]) for line in lines]
        self.assertEqual(percentages[0], 0)
        self.assertEqual(percentages[-1], 100)
        self.assertEqual(percentages, sorted(percentages))
        self.assertEqual(percentages.count(100), 1)
        self.assertIn("Complete dataset written to /run/derived/hash", stdout.getvalue())
        self.assertNotIn("%", stdout.getvalue())

    def test_failed_build_does_not_claim_completion(self):
        def fail(*args, progress, **kwargs):
            progress(0.5, "Estimating IMU pitch")
            raise RuntimeError("Synthetic processing failure")

        stderr = StringIO()
        with (
            patch("legwheel_dataset.cli.isolate_packets", side_effect=fail),
            patch("legwheel_dataset.cli.extract_features") as extract,
            redirect_stderr(stderr), self.assertRaises(RuntimeError),
        ):
            build_dataset_main(["--run-directory", "/recording/run_1"])
        self.assertNotIn("100%", stderr.getvalue())
        extract.assert_not_called()

    def test_rejected_candidates_and_empty_stages_still_complete_their_work(self):
        events = []
        report = lambda fraction, label: events.append(fraction)
        for item in progress_items([1, 2, 3], report, 0.0, 0.5, "Candidates"):
            if item != 2:
                continue
        list(progress_items([], report, 0.5, 1.0, "Empty stage"))
        self.assertEqual(events[0], 0.0)
        self.assertEqual(events[-1], 1.0)
        self.assertEqual(events, sorted(events))

    def test_terminal_bar_does_not_move_backwards(self):
        output = StringIO()
        with TerminalProgress("run_1", output) as progress:
            progress.update(0.8, "Later phase")
            progress.update(0.5, "Earlier report")
        self.assertIn("80% Earlier report", output.getvalue())

    def test_interactive_bar_stays_on_one_line_in_a_narrow_terminal(self):
        class TerminalOutput(StringIO):
            def isatty(self):
                return True

        output = TerminalOutput()
        with patch(
            "legwheel_dataset.progress.shutil.get_terminal_size",
            return_value=os.terminal_size((60, 24)),
        ):
            with TerminalProgress("run_1", output) as progress:
                progress.update(0.5, "Estimating IMU pitch: stationary calibration")
        self.assertEqual(output.getvalue().count("\n"), 1)
        for line in output.getvalue().split("\r"):
            self.assertLessEqual(len(line.rstrip("\n")), 59)
        self.assertIn("100%", output.getvalue())

    def test_bag_index_reports_work_even_when_messages_are_not_image_candidates(self):
        rows = [
            (bag_reader.REQUESTED_TORQUE_TOPIC, SimpleNamespace(data=2.0), index + 1)
            for index in range(240)
        ]
        reader = SimpleNamespace(
            get_all_topics_and_types=lambda: [
                SimpleNamespace(name=topic, type="unused") for topic in bag_reader.INDEX_TOPICS
            ],
            get_metadata=lambda: SimpleNamespace(message_count=240),
            has_next=lambda: bool(rows),
            read_next=lambda: rows.pop(0),
        )
        events = []
        with (
            patch.object(bag_reader, "_open_reader", return_value=reader),
            patch.object(bag_reader, "_deserialize", side_effect=lambda message, _: message),
        ):
            index = bag_reader.index_bag(
                Path("unused"), progress=lambda fraction, label: events.append(fraction),
            )
        self.assertEqual(events, [0.0, 100 / 240, 200 / 240, 1.0])
        self.assertEqual(len(index.streams["torque_command"].times_ns), 240)


if __name__ == "__main__":
    unittest.main()
