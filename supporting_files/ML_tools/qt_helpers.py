"""Small desktop controls shared by exploration and training."""

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QDoubleSpinBox


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
