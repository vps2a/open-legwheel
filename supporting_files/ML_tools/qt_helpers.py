"""Small desktop controls shared by exploration and training."""

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QDoubleSpinBox

from .plot_export import DEFAULT_EXPORT_DPI, validate_export_dpi


class ExportNavigationToolbar(NavigationToolbar2QT):
    """Keep the native save dialog while applying the chosen export resolution."""

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.export_dpi = DEFAULT_EXPORT_DPI

    def save_figure(self, *args):
        validate_export_dpi(self.export_dpi)
        # Override only export resolution. The canvas keeps its screen DPI, and
        # Matplotlib can still remember the last directory chosen in its dialog.
        previous = mpl.rcParams["savefig.dpi"]
        try:
            mpl.rcParams["savefig.dpi"] = self.export_dpi
            return super().save_figure(*args)
        finally:
            mpl.rcParams["savefig.dpi"] = previous


class DecimalSpinBox(QDoubleSpinBox):
    """Accept either decimal separator regardless of the desktop locale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setKeyboardTracking(False)

    def validate(self, text, position):
        return super().validate(text.replace(",", "."), position)

    def valueFromText(self, text):
        return super().valueFromText(text.replace(",", "."))
