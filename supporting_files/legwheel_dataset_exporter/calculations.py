"""Numerical calculations with no filesystem, GUI or ROS dependencies.

Intervals are closed: sharing even one boundary counts as overlap. Spatial
intervals describe wheel travel during an FFT window, not the camera footprint.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

TAU = 2.0 * math.pi
ANGLE_TOLERANCE_RAD = 1e-12


@dataclass(frozen=True)
class WindowCoverage:
    start_time_ns: int
    end_time_ns: int
    start_angle_rad: float
    end_angle_rad: float
    lower_angle_rad: float
    upper_angle_rad: float


def window_coverage(
    contact_time_ns: int,
    relative_times_ns: NDArray,
    tick_times_ns: NDArray,
    tick_angles_rad: NDArray,
    maximum_gap_ns: int,
) -> WindowCoverage:
    """Interpolate boundaries and conservatively include surrounding ticks.

    The bounds assume monotonic movement between ticks. We reject observable
    reversals and event intervals beyond the interpolation limit. Long intervals
    between count changes do not themselves establish missing sensor messages.
    Absolute nanosecond timestamps are subtracted as integers before division
    so floating-point rounding cannot erase short intervals in epoch times.
    """
    relative = np.asarray(relative_times_ns)
    times = np.asarray(tick_times_ns)
    angles = np.asarray(tick_angles_rad, dtype=float)
    for name, array in (("FFT timestamps", relative), ("Encoder timestamps", times)):
        if array.ndim != 1 or array.size < 2 or array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain at least two integer timestamps.")
        if np.any(np.diff(array) <= 0):
            raise ValueError(f"{name} must be strictly increasing.")
    if angles.shape != times.shape or not np.all(np.isfinite(angles)):
        raise ValueError("Encoder angles must be finite and match the tick timestamps.")
    if maximum_gap_ns <= 0:
        raise ValueError("Maximum encoder movement-event interval must be positive.")

    start = int(contact_time_ns) + int(relative[0])
    end = int(contact_time_ns) + int(relative[-1])
    if start < int(times[0]) or end > int(times[-1]):
        raise ValueError("FFT window extends outside the recorded encoder trajectory.")

    left = max(0, int(np.searchsorted(times, start, side="right")) - 1)
    right = min(times.size - 1, int(np.searchsorted(times, end, side="left")))
    intervals_ns = np.diff(times[left : right + 1])
    if np.any(intervals_ns > maximum_gap_ns):
        raise ValueError(
            f"FFT window spans encoder movement events {int(intervals_ns.max()) / 1e9:.9f} s apart; "
            f"exporter interpolation limit is {maximum_gap_ns / 1e9:g} s."
        )
    movement = np.diff(angles[left : right + 1])
    if np.any(movement > ANGLE_TOLERANCE_RAD) and np.any(movement < -ANGLE_TOLERANCE_RAD):
        raise ValueError("FFT window crosses a detected reversal in the encoder trajectory.")

    def boundary(timestamp: int) -> tuple[float, float, float]:
        index = int(np.searchsorted(times, timestamp, side="left"))
        if int(times[index]) == timestamp:
            value = float(angles[index])
            return value, value, value
        before, after = float(angles[index - 1]), float(angles[index])
        fraction = (timestamp - int(times[index - 1])) / (int(times[index]) - int(times[index - 1]))
        estimate = before + fraction * (after - before)
        return estimate, min(before, after), max(before, after)

    start_angle, start_low, start_high = boundary(start)
    end_angle, end_low, end_high = boundary(end)
    return WindowCoverage(
        start,
        end,
        start_angle,
        end_angle,
        min(start_low, end_low),
        max(start_high, end_high),
    )


def frequency_bands(width_hz: float, maximum_hz: float) -> NDArray:
    """Return uniform [lower, upper] edges with a clipped final band."""
    if not math.isfinite(width_hz) or width_hz <= 0:
        raise ValueError("Band width must be a finite positive number.")
    if not math.isfinite(maximum_hz) or maximum_hz <= 0:
        raise ValueError("Maximum frequency must be a finite positive number.")
    count = math.ceil(maximum_hz / width_hz)
    if count > 10_000:
        raise ValueError("Band width would create more than 10,000 bands; choose a larger width.")
    edges = np.minimum(np.arange(count + 1, dtype=float) * width_hz, maximum_hz)
    return np.column_stack((edges[:-1], edges[1:]))


def band_power(frequencies_hz: NDArray, psd: NDArray, bands: NDArray, nyquist_hz: float) -> NDArray:
    """Integrate saved PSD bins using sum(PSD * frequency spacing), excluding DC.

    Bins belong to [lower, upper), except that the final edge includes Nyquist.
    This discrete periodogram convention preserves total non-DC power without
    inventing finer frequency resolution. Unsupported or invalid bands are NaN.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    power = np.asarray(psd, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2 or not np.all(np.isfinite(frequencies)):
        raise ValueError("Spectrum needs at least two finite frequency bins.")
    spacing = np.diff(frequencies)
    if (
        spacing[0] <= 0
        or not np.allclose(spacing, spacing[0])
        or not np.isclose(frequencies[0], 0.0)
        or frequencies[-1] > nyquist_hz + 1e-9
    ):
        raise ValueError("Expected a uniform one-sided FFT grid beginning at zero.")
    if power.ndim != 2 or power.shape[0] != frequencies.size:
        raise ValueError("PSD must have shape (frequency bins, channels).")
    result = np.full((len(bands), power.shape[1]), np.nan)
    for index, (lower, upper) in enumerate(bands):
        if upper > nyquist_hz + 1e-9:
            continue
        mask = (frequencies > 0) & (frequencies >= lower) & (frequencies < upper)
        if np.isclose(upper, nyquist_hz):
            mask |= (frequencies > 0) & np.isclose(frequencies, upper, rtol=0, atol=1e-9)
        if not np.any(mask):
            continue
        values = power[mask]
        valid = np.all(np.isfinite(values) & (values >= 0), axis=0)
        result[index, valid] = values[:, valid].sum(axis=0) * spacing[0]
    return result


def spatial_overlap(first: WindowCoverage, second: WindowCoverage) -> bool:
    """Compare closed arcs modulo one revolution, including repeated laps."""
    first_width = first.upper_angle_rad - first.lower_angle_rad
    second_width = second.upper_angle_rad - second.lower_angle_rad
    if max(first_width, second_width) >= TAU - ANGLE_TOLERANCE_RAD:
        return True
    offset = (second.lower_angle_rad - first.lower_angle_rad) % TAU
    return (
        offset <= first_width + ANGLE_TOLERANCE_RAD
        or offset + second_width >= TAU - ANGLE_TOLERANCE_RAD
    )


def maximum_interval_count(intervals: list[tuple[int | float, int | float]]) -> int:
    """Find the exact maximum for closed line intervals by earliest finish."""
    count = 0
    previous_end = -math.inf
    for start, end in sorted(intervals, key=lambda interval: interval[1]):
        if start > previous_end:
            count += 1
            previous_end = end
    return count


def maximum_spatial_count(windows: list[WindowCoverage], check_cancel: Callable[[], None]) -> int:
    """Find the exact maximum disjoint subset of circular arcs.

    Any nonempty solution contains an anchor arc. Cutting the circle after that
    arc leaves a line-interval problem. Trying each anchor is O(n² log n), avoids
    a general graph solver, and also handles arcs crossing the zero-angle seam.
    """
    best = 0
    for anchor in windows:
        check_cancel()
        width = anchor.upper_angle_rad - anchor.lower_angle_rad
        candidates = []
        available = TAU - width
        for other in windows:
            if other is anchor or spatial_overlap(anchor, other):
                continue
            start = (other.lower_angle_rad - anchor.upper_angle_rad) % TAU
            end = start + other.upper_angle_rad - other.lower_angle_rad
            if start > ANGLE_TOLERANCE_RAD and end < available - ANGLE_TOLERANCE_RAD:
                # Expand just the end to apply the same tolerance as the overlap
                # test while keeping the greedy interval algorithm unchanged.
                candidates.append((start, end + ANGLE_TOLERANCE_RAD))
        best = max(best, 1 + maximum_interval_count(candidates))
    return best


def analyze_overlaps(
    windows: list[WindowCoverage],
    check_cancel: Callable[[], None] = lambda: None,
) -> tuple[list[list[int]], list[list[int]], int, int]:
    """Return symmetric neighbor indices and two separately optimized counts."""
    temporal = [[] for _ in windows]
    spatial = [[] for _ in windows]
    for index, first in enumerate(windows):
        check_cancel()
        for other_index in range(index + 1, len(windows)):
            second = windows[other_index]
            if max(first.start_time_ns, second.start_time_ns) <= min(
                first.end_time_ns, second.end_time_ns
            ):
                temporal[index].append(other_index)
                temporal[other_index].append(index)
            if spatial_overlap(first, second):
                spatial[index].append(other_index)
                spatial[other_index].append(index)
    temporal_count = maximum_interval_count([(w.start_time_ns, w.end_time_ns) for w in windows])
    return temporal, spatial, temporal_count, maximum_spatial_count(windows, check_cancel)
