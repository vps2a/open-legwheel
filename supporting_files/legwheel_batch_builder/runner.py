"""Sequential, cancellable batch orchestration on Qt's event loop."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from legwheel_dataset.config import ProcessingConfig
from legwheel_dataset.config_creator import render_processing_config, write_processing_config

from .model import (
    EVENT_PREFIX, BuildResult, RunInfo, dataset_state, write_attempt_status,
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


class BatchRunner(QObject):
    """Own one worker process at a time; never perform analysis in the GUI."""

    # Overall fraction, current fraction, run name, processing phase and ETA.
    progress = Signal(float, float, str, str, str)
    result_ready = Signal(object)
    run_started = Signal(object)
    log = Signal(str)
    finished = Signal(object, bool)

    def __init__(self, ros_python: str = "/usr/bin/python3", parent=None) -> None:
        super().__init__(parent)
        self.ros_python = ros_python
        self.running = False
        self.process: QProcess | None = None
        self.results: list[BuildResult] = []
        self._runs: list[RunInfo] = []
        self._prepared: list[RunInfo] = []
        self._prepare_index = 0
        self._next_index = 0
        self._current: RunInfo | None = None
        self._output: Path | None = None
        self._fraction = 0.0
        self._label = ""
        self._phase = ""
        self._cancelled = False
        self._cancel_progress = 0.0
        self._stdout = b""
        self._stderr = ""
        self._worker_result: dict | None = None
        self._work_done = 0.0
        self._work_seconds = 0.0
        self._run_started = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._emit_progress)

    def start(self, runs: list[RunInfo], config: ProcessingConfig) -> None:
        """Apply the approved settings to selected runs before building them."""
        if self.running:
            raise RuntimeError("A batch is already running.")
        if not runs:
            raise ValueError("Select at least one run.")
        config.validate()
        self._runs = list(runs)
        self.results, self._prepared = [], []
        self._prepare_index = self._next_index = 0
        self._current = None
        self._fraction = self._work_done = self._work_seconds = 0.0
        self._cancelled = False
        self._cancel_progress = 0.0
        self._label, self._phase = "Preparing batch", "Saving shared processing settings"
        self._config_text = render_processing_config(
            config, datetime.now(timezone.utc).isoformat(),
        )
        self.running = True
        self._timer.start()
        self._emit_progress()
        QTimer.singleShot(0, self._prepare_next)

    def _record(self, result: BuildResult) -> None:
        self.results.append(result)
        self.result_ready.emit(result)

    def _prepare_next(self) -> None:
        if not self.running:
            return
        if self._cancelled:
            self._finish()
            return
        if self._prepare_index == len(self._runs):
            self._start_next()
            return
        run = self._runs[self._prepare_index]
        try:
            write_processing_config(run.directory / "postprocess.yaml", self._config_text)
            self._prepared.append(run)
        except OSError as error:
            self._record(BuildResult(run, "failed", f"Cannot save postprocess.yaml: {error}"))
        self._prepare_index += 1
        # Yield between files so even a large batch stays responsive to Cancel.
        QTimer.singleShot(0, self._prepare_next)

    def _start_next(self) -> None:
        if not self.running:
            return
        if self._cancelled or self._next_index == len(self._prepared):
            self._finish()
            return
        self._current = self._prepared[self._next_index]
        self._next_index += 1
        self.run_started.emit(self._current)
        self._label, self._phase = self._current.label, "Starting worker"
        self._fraction, self._output, self._worker_result = 0.0, None, None
        self._stdout, self._stderr = b"", ""
        self._run_started = time.monotonic()
        self._emit_progress()

        process = QProcess(self)
        self.process = process
        environment = QProcessEnvironment.systemEnvironment()
        # The GUI can run under uv, while the worker must use ROS's system
        # Python. Preserve the sourced workspace and expose this helper package.
        package_root = str(Path(__file__).resolve().parent.parent)
        inherited = environment.value("PYTHONPATH")
        environment.insert("PYTHONPATH", package_root + (os.pathsep + inherited if inherited else ""))
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProgram(self.ros_python)
        process.setArguments(self._worker_arguments(self._current))
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._process_finished)
        process.start()

    def _worker_arguments(self, run: RunInfo) -> list[str]:
        return [
            "-u", "-m", "legwheel_batch_builder.worker",
            "--run-directory", str(run.directory),
            "--processing-config", str(run.directory / "postprocess.yaml"),
        ]

    def _read_stdout(self) -> None:
        self._stdout += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._stdout:
            line, self._stdout = self._stdout.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace")
            if not text.startswith(EVENT_PREFIX):
                if text.strip():
                    self.log.emit(text)
                continue
            try:
                event = json.loads(text[len(EVENT_PREFIX):])
                kind = event["type"]
                if kind == "progress":
                    self._fraction = max(self._fraction, min(1.0, float(event["fraction"])))
                    self._phase = str(event["label"])
                    self._emit_progress()
                elif kind == "output":
                    self._output = Path(event["path"])
                elif kind == "result":
                    self._worker_result = event
                elif kind == "log":
                    self.log.emit(str(event["message"]))
            except (ValueError, KeyError, TypeError) as error:
                self.log.emit(f"Invalid worker progress message: {error}")

    def _read_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr = (self._stderr + text)[-20000:]
        if text.strip():
            self.log.emit(text.rstrip())

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self._current is not None:
            self._complete_current(BuildResult(
                self._current, "failed", f"Cannot start ROS Python: {self.process.errorString()}",
            ))

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        if self._current is None:
            return
        self._read_stdout()
        self._read_stderr()
        event = self._worker_result or {}
        output = Path(event["output"]) if event.get("output") else self._output
        completed_result = event.get("status") in {"completed", "skipped"}
        if completed_result and (exit_code == 0 or self._cancelled):
            result = BuildResult(self._current, event["status"], event["message"], output)
        elif self._cancelled and output is not None and dataset_state(output) == "completed":
            # Cancellation can race the final result message. Keep a dataset
            # that has already completed both stages rather than marking it bad.
            result = BuildResult(
                self._current, "completed", "Dataset finished before cancellation.", output,
            )
        elif self._cancelled:
            message = "Interrupted by the user; partial output preserved."
            if output is not None:
                try:
                    write_attempt_status(output, "cancelled", message)
                except OSError as error:
                    message += f" Could not record cancellation: {error}"
            result = BuildResult(self._current, "cancelled", message, output)
        else:
            message = (
                event.get("message") or self._stderr.strip()
                or f"Worker exited with code {exit_code}."
            )
            if output is not None and dataset_state(output) != "completed":
                try:
                    write_attempt_status(output, "failed", message)
                except OSError as error:
                    message += f" Could not record failure: {error}"
            result = BuildResult(self._current, "failed", message, output)
        self._complete_current(result)

    def _complete_current(self, result: BuildResult) -> None:
        if result.status in {"completed", "skipped"}:
            self._fraction = 1.0
        if self._fraction > 0 and result.status in {"completed", "failed"}:
            self._work_done += self._fraction
            self._work_seconds += time.monotonic() - self._run_started
        self._record(result)
        self._current = None
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        QTimer.singleShot(0, self._start_next)

    def _emit_progress(self) -> None:
        total = len(self._runs)
        active_fraction = self._fraction if self._current is not None else 0.0
        overall = (len(self.results) + active_fraction) / total if total else 0.0
        eta = "Estimating…"
        work = self._work_done + (self._fraction if self._current else 0.0)
        elapsed = self._work_seconds
        if self._current is not None:
            elapsed += time.monotonic() - self._run_started
        if work >= 0.01 and elapsed >= 1.0:
            eta = format_duration(max(0.0, total - len(self.results) - active_fraction) * elapsed / work)
        if self._cancelled:
            overall, eta = self._cancel_progress, "Cancelled"
        elif not self.running:
            eta = "0m 00s"
        self.progress.emit(overall, self._fraction, self._label, self._phase, eta)

    def cancel(self) -> None:
        """Kill the worker directly; preserve its files after the OS confirms exit."""
        if not self.running or self._cancelled:
            return
        self._cancelled = True
        active_fraction = self._fraction if self._current is not None else 0.0
        self._cancel_progress = (len(self.results) + active_fraction) / len(self._runs)
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self._phase = "Cancelling current run…"
            self.process.kill()
            self._emit_progress()
        else:
            self._finish()

    def _finish(self) -> None:
        if not self.running:
            return
        if self._cancelled:
            recorded = {result.run.directory for result in self.results}
            for run in self._runs:
                if run.directory not in recorded:
                    self._record(BuildResult(
                        run, "not_started", "Batch cancelled before this run started.",
                    ))
        self._timer.stop()
        self.running = False
        self._phase = "Batch cancelled" if self._cancelled else "Batch finished"
        self._emit_progress()
        self.finished.emit(list(self.results), self._cancelled)
