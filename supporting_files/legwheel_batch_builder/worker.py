"""Process exactly one run under the sourced ROS Python interpreter.

The GUI owns this process directly. Killing it therefore stops the active
analysis immediately, without leaving a ros2 wrapper or a worker thread alive.
"""

import argparse
import json
from pathlib import Path
import time
import traceback

from .model import (
    EVENT_PREFIX, archive_incomplete_output, dataset_requires_regeneration,
    dataset_state, write_attempt_status,
)


def emit_event(kind: str, **values) -> None:
    """Use a distinct prefix so ROS diagnostic output cannot become GUI events."""
    print(EVENT_PREFIX + json.dumps({"type": kind, **values}), flush=True)


def build_run(run_directory: Path, config_path: Path) -> int:
    """Reuse both production stages and report an explicit final result."""
    output = None
    owns_output = False
    try:
        # Keep imports inside the error handler so missing ROS packages are
        # reported in the batch summary instead of silently closing the worker.
        from legwheel_dataset.config import ProcessingConfig
        from legwheel_dataset.processor import (
            dataset_output_directory, extract_features, isolate_packets,
        )
        from legwheel_dataset.progress import progress_range

        config = ProcessingConfig.from_yaml(config_path)
        output = dataset_output_directory(run_directory, config)
        if output.exists():
            if dataset_state(output) == "completed" and not dataset_requires_regeneration(output):
                emit_event("result", status="skipped", output=str(output),
                           message="A completed dataset with these settings already exists.")
                return 0
            archived = archive_incomplete_output(output)
            emit_event("log", message=f"Preserved earlier attempt: {archived}")

        owns_output = True
        emit_event("output", path=str(output))
        last_update, last_percent, last_label = 0.0, -1, ""

        def report(fraction: float, label: str) -> None:
            nonlocal last_update, last_percent, last_label
            percent = int(fraction * 100)
            now = time.monotonic()
            if label == last_label and percent != 100:
                if percent == last_percent or now - last_update < 0.1:
                    return
            emit_event("progress", fraction=fraction, label=label)
            last_update, last_percent, last_label = now, percent, label

        report(0.0, "Starting")
        isolate_packets(
            run_directory, config, output_directory=output,
            progress=progress_range(report, 0.0, 0.65),
        )
        extract_features(output, config, progress=progress_range(report, 0.65, 0.99))
        write_attempt_status(output, "completed")
        report(1.0, "Complete")
        emit_event("result", status="completed", output=str(output), message="Dataset built.")
        return 0
    except Exception as error:
        message = str(error)
        if (
            owns_output and output is not None
            and dataset_state(output) != "completed"
        ):
            try:
                write_attempt_status(output, "failed", message)
            except OSError as status_error:
                message += f"; could not record attempt status: {status_error}"
        traceback.print_exc()
        emit_event("result", status="failed", output=str(output) if owns_output else None,
                   message=message)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one run for the batch GUI.")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--processing-config", type=Path, required=True)
    options = parser.parse_args()
    return build_run(options.run_directory, options.processing_config)


if __name__ == "__main__":
    raise SystemExit(main())
