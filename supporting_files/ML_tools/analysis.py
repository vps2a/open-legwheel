"""Read exported packets and calculate features independently of the GUI.

Each row remains one packet, retaining its run and terrain grouping for future
model evaluation. No images are loaded and no source files are modified.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CLASS_NAMES = ("Firm", "Loose", "Firm → Loose", "Loose → Firm")
CLASS_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MOTOR_CHANNELS = tuple(
    f"{joint}_joint.{field}"
    for joint in ("hip", "knee", "wheel")
    for field in ("velocity", "effort")
)
IMU_CHANNELS = tuple(
    f"imu.{quantity}.{axis}"
    for axis in "xyz"
    for quantity in ("angular_velocity", "linear_acceleration")
)
BAND_CHANNELS = MOTOR_CHANNELS + IMU_CHANNELS
WINDOWS = ("spectral", "local")
TerrainKey = tuple[str, str]

REQUIRED_COLUMNS = (
    "export_id",
    "source_id",
    "class_id",
    "terrain_setup_id",
    "terrain_and_light_ID",
    "channel_names",
    "local_values",
    "local_valid",
    "spectral_values",
    "spectral_valid",
    "band_edges_hz",
    "band_power",
)
# Older exports remain usable for exploration. Missing training measurements
# stay explicitly unavailable rather than being replaced with invented values.
TRAINING_COLUMNS = (
    "experiment_id",
    "run_number",
    "along_slope_rad",
    "cross_slope_rad",
    "fft_start_time_ns",
    "fft_end_time_ns",
    "metadata_json",
)


class DatasetFormatError(ValueError):
    """The selected files are not a compatible exported packet dataset."""


class LoadCancelled(Exception):
    """The user cancelled a background dataset load."""


@dataclass(frozen=True)
class BandSummary:
    """Equal-packet mean, sample standard deviation and valid count per band."""

    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class TelemetryDataset:
    root: Path
    packet_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    terrain_keys: tuple[TerrainKey, ...]
    class_ids: np.ndarray  # -1 = unlabelled; -2 = unsupported label.
    standard_deviations: dict[str, np.ndarray]  # [packet, motor channel]
    band_edges_hz: np.ndarray  # [band, lower/upper edge]
    band_powers: np.ndarray  # [packet, band, motor + IMU channel]
    issues: tuple[str, ...]
    export_version: str | None = None
    run_names: dict[str, str] = field(default_factory=dict)
    wheel_speeds: dict[str, np.ndarray] = field(default_factory=dict)
    slopes: np.ndarray | None = None
    fft_bounds_ns: np.ndarray | None = None
    local_bounds_ns: np.ndarray | None = None
    fingerprint: str = ""

    def selection(
        self,
        classes: set[int],
        terrains: set[TerrainKey],
    ) -> np.ndarray:
        """Filter packets; class zero is a valid label, never a missing value."""
        return np.isin(self.class_ids, list(classes)) & self.terrain_selection(terrains)

    def terrain_selection(self, terrains: set[TerrainKey]) -> np.ndarray:
        return np.fromiter(
            (key in terrains for key in self.terrain_keys), bool, len(self.packet_ids)
        )


def mean_wheel_speed(values, valid, names: list[str]) -> float:
    """Mean absolute angular speed, using the saved validity mask."""
    try:
        data, mask = np.asarray(values, float), np.asarray(valid, bool)
        column = names.index("wheel_joint.velocity")
        if data.ndim != 2 or data.shape != mask.shape or data.shape[1] != len(names):
            return float("nan")
        usable = mask[:, column] & np.isfinite(data[:, column])
        return (
            float(np.mean(np.abs(data[usable, column])))
            if np.any(usable)
            else float("nan")
        )
    except (TypeError, ValueError):
        return float("nan")


def _time_pair(start, end) -> tuple[int, int]:
    # Integer nanoseconds must never pass through float64 (which loses precision).
    if (
        all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (start, end)
        )
        and 0 <= start <= end <= np.iinfo(np.int64).max
    ):
        return start, end
    return -1, -1


def window_standard_deviations(values, valid, names: list[str]) -> np.ndarray:
    """Population STD (ddof=0) of finite, valid samples; require at least two.

    This is variation within one observed packet window, distinct from the
    sample STD (ddof=1) used for variation between packet band powers.
    """
    result = np.full(len(MOTOR_CHANNELS), np.nan)
    try:
        data = np.asarray(values, dtype=float)
        mask = np.asarray(valid, dtype=bool)
    except (TypeError, ValueError):
        return result
    if data.ndim != 2 or mask.shape != data.shape or data.shape[1] != len(names):
        return result
    for output_column, name in enumerate(MOTOR_CHANNELS):
        if name not in names:
            continue
        column = names.index(name)
        samples = data[:, column]
        usable = mask[:, column] & np.isfinite(samples)
        if np.count_nonzero(usable) >= 2:
            result[output_column] = np.std(samples[usable], ddof=0)
    return result


def summarize_band_power(values: np.ndarray) -> BandSummary:
    """Summarize [packet, band] values without giving each run equal weight."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Band values must have shape [packet, band].")
    valid = np.isfinite(values) & (values >= 0)
    count = valid.sum(axis=0)
    mean = np.full(values.shape[1], np.nan)
    np.divide(np.where(valid, values, 0).sum(axis=0), count, out=mean, where=count > 0)
    residuals = np.where(valid, values - mean, 0)
    variance = np.full_like(mean, np.nan)
    np.divide((residuals**2).sum(axis=0), count - 1, out=variance, where=count > 1)
    return BandSummary(mean, np.sqrt(variance), count)


def histogram_edges(values: np.ndarray, bins: int, logarithmic: bool) -> np.ndarray:
    """One shared bin grid for every visible class in a channel."""
    values = np.asarray(values, dtype=float)
    values = values[
        np.isfinite(values) & ((values > 0) if logarithmic else (values >= 0))
    ]
    if not values.size:
        return np.array([])
    lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        if logarithmic:
            lower, upper = lower / 1.2, upper * 1.2
        else:
            margin = max(abs(lower) * 0.05, 0.01)
            lower, upper = max(0.0, lower - margin), upper + margin
    return (np.geomspace if logarithmic else np.linspace)(lower, upper, bins + 1)


def _packet_files(path: Path) -> tuple[Path, list[Path]]:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".parquet":
        if path.parent.name == "packets":
            path = path.parent.parent
        elif path.name == "runs.parquet":
            path = path.parent
        else:
            raise DatasetFormatError(
                "Choose the export folder or a Parquet file inside its packets folder."
            )
    if path.name == "packets" and path.is_dir():
        path = path.parent
    files = sorted((path / "packets").glob("*.parquet"))
    if not files:
        raise DatasetFormatError(
            "No packets/*.parquet files found. Choose the exported dataset folder."
        )
    return path, files


def _band_edges(value) -> np.ndarray:
    try:
        edges = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise DatasetFormatError(
            "Frequency band edges must be numerical pairs."
        ) from error
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or not len(edges)
        or not np.all(np.isfinite(edges))
        or np.any(edges[:, 0] < 0)
        or np.any(edges[:, 1] <= edges[:, 0])
        or np.any(edges[1:, 0] < edges[:-1, 1])
    ):
        raise DatasetFormatError(
            "Expected ordered, non-overlapping frequency bands with positive widths."
        )
    return edges


def load_dataset(
    path: Path,
    progress: Callable[[int, int], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> TelemetryDataset:
    """Read all shards, selecting numerical columns instead of embedded media."""
    root, files = _packet_files(Path(path))
    packet_files = [pq.ParquetFile(file) for file in files]
    for file, parquet in zip(files, packet_files):
        missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise DatasetFormatError(
                f"{file.name} is missing columns: {', '.join(sorted(missing))}"
            )
    total = sum(parquet.metadata.num_rows for parquet in packet_files)
    ids, sources, terrains, labels, powers = [], [], [], [], []
    run_names, slopes, fft_bounds, local_bounds = {}, [], [], []
    speeds = {window: [] for window in WINDOWS}
    deviations = {window: [] for window in WINDOWS}
    seen = set()
    issues = Counter()
    band_edges = None
    for parquet in packet_files:
        columns = list(REQUIRED_COLUMNS) + [
            name for name in TRAINING_COLUMNS if name in parquet.schema_arrow.names
        ]
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            if check_cancel:
                check_cancel()
            for row in batch.to_pylist():
                if check_cancel:
                    check_cancel()
                packet_id = row["export_id"]
                if not isinstance(packet_id, str) or not packet_id or packet_id in seen:
                    raise DatasetFormatError(
                        f"Missing or duplicate export_id: {packet_id!r}"
                    )
                seen.add(packet_id)
                names = row["channel_names"]
                if (
                    not isinstance(names, list)
                    or not all(isinstance(name, str) for name in names)
                    or len(set(names)) != len(names)
                ):
                    raise DatasetFormatError(f"Invalid channel_names in {packet_id}.")
                edges = _band_edges(row["band_edges_hz"])
                if band_edges is None:
                    band_edges = edges
                elif edges.shape != band_edges.shape or not np.array_equal(
                    edges, band_edges
                ):
                    raise DatasetFormatError(
                        "Packets have different frequency bands; export them on a common band layout."
                    )
                packet_power = np.full((len(edges), len(BAND_CHANNELS)), np.nan)
                try:
                    saved = np.asarray(row["band_power"], dtype=float)
                except (TypeError, ValueError):
                    saved = np.array([])
                if saved.shape == (len(edges), len(names)):
                    for column, name in enumerate(BAND_CHANNELS):
                        if name in names:
                            packet_power[:, column] = saved[:, names.index(name)]
                    packet_power[~np.isfinite(packet_power) | (packet_power < 0)] = (
                        np.nan
                    )
                else:
                    issues["Packets with malformed band-power arrays"] += 1
                missing_channels = set(BAND_CHANNELS) - set(names)
                if missing_channels:
                    issues["Packets with missing plotted channels"] += 1
                for window in WINDOWS:
                    std = window_standard_deviations(
                        row[f"{window}_values"], row[f"{window}_valid"], names
                    )
                    deviations[window].append(std)
                    speeds[window].append(
                        mean_wheel_speed(
                            row[f"{window}_values"], row[f"{window}_valid"], names
                        )
                    )
                    if not np.all(np.isfinite(std)):
                        issues[
                            f"Packets with unavailable {window}-window STD channels"
                        ] += 1
                ids.append(packet_id)
                source = row["source_id"]
                if not isinstance(source, str) or not source:
                    raise DatasetFormatError(f"Missing source_id in {packet_id}.")
                sources.append(source)
                experiment, run = row.get("experiment_id"), row.get("run_number")
                run_names[source] = (
                    f"{experiment} / run {run}"
                    if experiment is not None and run is not None
                    else source
                )
                slopes.append([row.get("along_slope_rad"), row.get("cross_slope_rad")])
                fft_bounds.append(
                    _time_pair(row.get("fft_start_time_ns"), row.get("fft_end_time_ns"))
                )
                try:
                    metadata = json.loads(row.get("metadata_json") or "{}")
                    local_bounds.append(
                        _time_pair(
                            metadata.get("local_start_time_ns"),
                            metadata.get("local_end_time_ns"),
                        )
                    )
                except (ValueError, TypeError, AttributeError):
                    local_bounds.append((-1, -1))
                terrains.append(
                    (
                        row["terrain_setup_id"] or "(Unknown setup)",
                        row["terrain_and_light_ID"] or "(Unknown terrain/light)",
                    )
                )
                label = row["class_id"]
                labels.append(
                    -1 if label is None else label if label in range(4) else -2
                )
                powers.append(packet_power)
            if progress:
                progress(len(ids), total)
    if not ids:
        raise DatasetFormatError("The packet files are empty.")
    # Version metadata is informational; figures always use the saved powers.
    version = None
    manifest = root / "export_manifest.json"
    if manifest.is_file():
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            version = (
                str(document["export_version"])
                if "export_version" in document
                else None
            )
        except (OSError, ValueError, TypeError):
            issues["Export manifest could not be read"] += 1
    dataset = TelemetryDataset(
        root,
        tuple(ids),
        tuple(sources),
        tuple(terrains),
        np.asarray(labels, dtype=int),
        {key: np.asarray(value) for key, value in deviations.items()},
        band_edges,
        np.asarray(powers),
        tuple(f"{name}: {count}" for name, count in sorted(issues.items())),
        version,
        run_names,
        {key: np.asarray(value) for key, value in speeds.items()},
        np.asarray(slopes, dtype=float),
        np.asarray(fft_bounds, dtype=np.int64),
        np.asarray(local_bounds, dtype=np.int64),
    )
    # Identity follows the numerical inputs and labels, not the folder location.
    # A moved export can therefore reopen an experiment; changed labels cannot.
    digest = hashlib.sha256()
    for strings in (dataset.packet_ids, dataset.source_ids, dataset.terrain_keys):
        digest.update(json.dumps(strings, ensure_ascii=True).encode())
    arrays = [
        dataset.class_ids,
        dataset.band_edges_hz,
        dataset.band_powers,
        dataset.slopes,
        dataset.fft_bounds_ns,
        dataset.local_bounds_ns,
        *dataset.standard_deviations.values(),
        *dataset.wheel_speeds.values(),
    ]
    for array in arrays:
        if np.issubdtype(array.dtype, np.integer):
            canonical = array.astype("<i8")
        else:
            canonical = np.where(np.isfinite(array), array, np.nan).astype("<f8")
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return replace(dataset, fingerprint=digest.hexdigest())
