"""Shared raster-export resolution, independent of a figure's screen DPI."""

DEFAULT_EXPORT_DPI = 300
MIN_EXPORT_DPI = 72
MAX_EXPORT_DPI = 1200


def validate_export_dpi(dpi: int) -> None:
    if (
        isinstance(dpi, bool)
        or not isinstance(dpi, int)
        or not MIN_EXPORT_DPI <= dpi <= MAX_EXPORT_DPI
    ):
        raise ValueError(
            f"Export DPI must be a whole number between {MIN_EXPORT_DPI} and {MAX_EXPORT_DPI}."
        )
