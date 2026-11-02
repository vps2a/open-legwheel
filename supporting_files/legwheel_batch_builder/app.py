"""Application entry point for the batch dataset builder."""

import argparse
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QMessageBox


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build datasets for selected experiments and runs.")
    parser.add_argument("--directory", type=Path, help="Initial parent directory of experiments.")
    parser.add_argument(
        "--ros-python", default="/usr/bin/python3",
        help="Python interpreter with ROS 2 support.",
    )
    options = parser.parse_args(arguments)
    application = QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    try:
        from .window import BatchBuilderWindow
    except ImportError as error:
        QMessageBox.critical(
            None, "Batch builder environment unavailable",
            "Build and source the ROS workspace before opening this tool, then run uv sync "
            f"in supporting_files.\n\n{error}",
        )
        return 1
    window = BatchBuilderWindow(options.directory, options.ros_python)
    window.show()
    return application.exec()
