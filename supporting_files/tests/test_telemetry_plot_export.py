"""Verify image pixels/DPI through toolbar and experiment export paths."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib as mpl
from PIL import Image
from PySide6.QtWidgets import QApplication

from ML_tools.analysis import load_dataset
from ML_tools.experiments import save_experiment
from ML_tools.plot_export import validate_export_dpi
from ML_tools.training import train_experiment
from ML_tools.window import VisualiserWindow
from tests.test_telemetry_training import make_training_export, setup_config


class PlotExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.window = VisualiserWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.temporary.cleanup()

    def test_shared_setting_increases_every_toolbar_export_without_resizing_screen(
        self,
    ):
        self.assertEqual(self.window.export_dpi_spin.value(), 300)
        panels = [*self.window.panels, self.window.training_page.results]
        with mpl.rc_context({"savefig.dpi": "figure", "savefig.directory": ""}):
            for dpi in (300, 600):
                self.window.export_dpi_spin.setValue(dpi)
                for index, panel in enumerate(panels):
                    panel.figure.set_size_inches(2, 1.5)
                    screen_dpi = panel.figure.dpi
                    output = self.root / f"panel_{index}_{dpi}.png"
                    with patch(
                        "matplotlib.backends.backend_qt.QtWidgets.QFileDialog.getSaveFileName",
                        return_value=(str(output), "PNG (*.png)"),
                    ):
                        # Trigger the actual toolbar action, not a different save path.
                        action = next(
                            action
                            for action in panel.toolbar.actions()
                            if action.text() == "Save"
                        )
                        action.trigger()
                    with Image.open(output) as exported:
                        self.assertEqual(exported.size, (2 * dpi, round(1.5 * dpi)))
                        self.assertAlmostEqual(exported.info["dpi"][0], dpi, delta=0.1)
                    self.assertEqual(panel.figure.dpi, screen_dpi)
                    self.assertEqual(mpl.rcParams["savefig.dpi"], "figure")

    def test_experiment_save_uses_shared_dpi_without_invalidating_models(self):
        dataset = load_dataset(make_training_export(self.root / "export"))
        experiment = train_experiment(dataset, setup_config(dataset))
        self.window.set_dataset(dataset)
        page = self.window.training_page
        page._completed(experiment)
        page.refresh_summary()
        self.window.export_dpi_spin.setValue(150)
        self.assertFalse(page.stale)
        self.assertTrue(page.save_button.isEnabled())
        self.assertIs(page.experiment, experiment)
        with (
            patch(
                "ML_tools.training_page.QFileDialog.getExistingDirectory",
                return_value=str(self.root),
            ),
            patch(
                "ML_tools.training_page.save_experiment", wraps=save_experiment
            ) as save,
        ):
            page.choose_save()
        self.assertEqual(save.call_args.kwargs["export_dpi"], 150)
        folder = next(self.root.glob("telemetry_experiment_*"))
        manifest = json.loads((folder / "experiment.json").read_text())
        self.assertEqual(manifest["plot_export_dpi"], 150)
        for partition in ("training", "validation"):
            with Image.open(folder / "figures" / f"{partition}.png") as exported:
                # Comparison figures have an explicit 11 × 7 inch layout.
                self.assertEqual(exported.size, (1650, 1050))
                self.assertAlmostEqual(exported.info["dpi"][0], 150, delta=0.1)

    def test_invalid_export_resolution_is_rejected_before_creating_files(self):
        for dpi in (0, -300, 1201, 300.5, True):
            with (
                self.subTest(dpi=dpi),
                self.assertRaisesRegex(ValueError, "Export DPI"),
            ):
                validate_export_dpi(dpi)


if __name__ == "__main__":
    unittest.main()
