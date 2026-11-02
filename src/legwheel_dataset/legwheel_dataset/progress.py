"""Optional work-based progress reporting for the offline dataset pipeline."""

from collections.abc import Callable, Iterator, Sequence
import shutil
import sys
import time
from typing import TextIO, TypeVar


ProgressCallback = Callable[[float, str], None]
T = TypeVar("T")


def progress_range(
    callback: ProgressCallback | None, start: float, end: float,
) -> ProgressCallback | None:
    """Map a stage's 0..1 progress into its share of the complete build."""
    if callback is None:
        return None

    def report(fraction: float, label: str) -> None:
        callback(start + (end - start) * fraction, label)

    return report


def progress_items(
    items: Sequence[T], callback: ProgressCallback | None,
    start: float, end: float, label: str,
) -> Iterator[T]:
    """Count completed items, including candidates rejected with ``continue``."""
    if callback:
        callback(start, label)
    for index, item in enumerate(items, start=1):
        yield item
        if callback:
            callback(start + (end - start) * index / len(items), label)
    if callback:
        callback(end, label)


class TerminalProgress:
    """One monotonic terminal bar; 100% is reserved for a successful build.

    Stage weights describe completed work, not an estimate of remaining time.
    Output goes to stderr so the command's normal result stays on stdout.
    """

    def __init__(self, run_name: str, stream: TextIO | None = None) -> None:
        self.run_name = run_name
        self.stream = sys.stderr if stream is None else stream
        self._fraction = 0.0
        self._last_update = 0.0
        self._last_percent = -1
        self._last_label = ""
        self._line_width = 0

    def __enter__(self) -> "TerminalProgress":
        self.update(0.0, "Starting")
        return self

    def update(self, fraction: float, label: str) -> None:
        """Refresh at most ten times a second, except at phase boundaries."""
        self._fraction = max(self._fraction, min(1.0, fraction))
        percent = int(self._fraction * 100)
        now = time.monotonic()
        if label == self._last_label and percent != 100:
            if percent == self._last_percent or now - self._last_update < 0.1:
                return
        filled = percent * 28 // 100
        bar = "#" * filled + "-" * (28 - filled)
        line = f"{self.run_name}: [{bar}] {percent:3d}% {label}"
        interactive = self.stream.isatty()
        previous_width = self._line_width
        if interactive:
            # Leave the final column free so long phase names cannot wrap the
            # updating bar onto a second terminal line.
            width = max(1, shutil.get_terminal_size().columns - 1)
            line = line[:width]
            previous_width = min(previous_width, width)
        self.stream.write(
            "\r" + line.ljust(previous_width) if interactive else line + "\n"
        )
        self.stream.flush()
        self._line_width = len(line)
        self._last_update = now
        self._last_percent = percent
        self._last_label = label

    def __exit__(self, error_type, error, traceback) -> None:
        if error_type is None:
            self.update(1.0, "Complete")
        if self.stream.isatty():
            self.stream.write("\n")
            self.stream.flush()
