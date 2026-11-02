#!/usr/bin/env python3
"""Launch the offline exported-packet telemetry visualiser without ROS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Export folder or packet Parquet file to open.")
    options = parser.parse_args(arguments)
    # Support both the requested standalone file and an installed entry point.
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from PySide6.QtWidgets import QApplication
    from ML_tools.window import VisualiserWindow

    application = QApplication(sys.argv[:1])
    application.setOrganizationName("ESA SSP LegWheel")
    application.setApplicationName("Telemetry Classification Visualiser")
    application.setStyle("Fusion")
    window = VisualiserWindow(options.dataset)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
