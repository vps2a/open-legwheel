"""Application entry point for the standalone LegWheel dataset browser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from .window import DatasetBrowserWindow


def _arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect derived LegWheel packets and manually classify terrain."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional configuration-hash directory to open instead of the last dataset.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Start the desktop browser and return its Qt exit status."""
    options = _arguments(arguments)
    QCoreApplication.setOrganizationName("ESA SSP LegWheel")
    QCoreApplication.setApplicationName("LegWheel Dataset Browser")
    application = QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    window = DatasetBrowserWindow(options.dataset)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
