"""Application entry point for the standalone dataset exporter."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select terrain packets and export a Parquet dataset without ROS."
    )
    parser.add_argument(
        "--directory", type=Path, help="Initial recordings directory or derived dataset folder."
    )
    options = parser.parse_args(arguments)
    application = QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    from .window import ExporterWindow

    window = ExporterWindow(options.directory)
    window.show()
    return application.exec()
