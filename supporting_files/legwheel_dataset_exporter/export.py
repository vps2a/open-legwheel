"""Stream a validated selection into portable, typed Parquet tables.

The export is built in a temporary sibling directory. Only a complete export
gets its final name, so cancellation and failures cannot leave a partial dataset
that looks ready for use. No source files are ever written.
"""

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import ClassLabel, Features, Image, List, Value
from PIL import Image as PillowImage

from .calculations import analyze_overlaps, band_power, frequency_bands
from .sources import CLASS_NAMES, Packet, SourceDataset, load_source, read_npz

EXPORT_VERSION = "1.1"
ROWS_PER_GROUP = 16
SHARD_TARGET_BYTES = 128 * 1024 * 1024


class ExportCancelled(Exception):
    """A cooperative cancellation, with temporary output removed."""


@dataclass(frozen=True)
class ExportOptions:
    band_width_hz: float = 10.0
    include_full_frames: bool = False
    include_point_maps: bool = False


# Frozen options are safe to share; callers can supply a different immutable set.
DEFAULT_OPTIONS = ExportOptions()


@dataclass(frozen=True)
class ExportResult:
    directory: Path
    packet_count: int
    runs: list[dict]


def clean_json(value: Any) -> Any:
    """Use portable nulls for nonfinite numbers, preserving integer timestamps."""
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_text(value: Any) -> str:
    return json.dumps(clean_json(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def portable_provenance(source: SourceDataset) -> dict:
    """Exclude machine-local paths and generation times from published identity."""
    excluded = {
        "run_directory",
        "bag_directory",
        "experiment_config_source",
        "created_utc",
        "stage2_completed_utc",
    }
    return {key: value for key, value in source.provenance.items() if key not in excluded}


def channel_unit(name: str) -> str:
    if name.startswith("imu.angular_velocity.") or name.endswith(".velocity"):
        return "rad/s"
    if name.startswith("imu.linear_acceleration."):
        return "m/s^2"
    if name.endswith(".position"):
        return "rad"
    if name.endswith(".effort"):
        return "N*m"
    return "unknown"


def packet_features(options: ExportOptions) -> Features:
    """Define a stable schema even when labels or optional values are all null."""
    text, number, integer = Value("string"), Value("float64"), Value("int64")
    vector = List(number)
    matrix = List(vector)
    features = Features(
        {
            **{
                name: text
                for name in (
                    "export_id",
                    "packet_id",
                    "source_id",
                    "experiment_id",
                    "run_number",
                    "processing_id",
                    "terrain_and_light_ID",
                    "laboratory_id",
                    "terrain_setup_id",
                    "lighting_id",
                    "classification_method",
                    "model_id",
                    "split",
                    "metadata_json",
                    "features_json",
                )
            },
            "class_id": ClassLabel(names=list(CLASS_NAMES)),
            "packet_number": integer,
            "image_time_ns": integer,
            "contact_time_ns": integer,
            "lap_index": integer,
            "sample_rate_hz": number,
            "depth_scale_m": number,
            "channel_names": List(text),
            "channel_units": List(text),
            **{
                name: matrix
                for name in (
                    "local_values",
                    "spectral_values",
                    "spectral_power",
                    "dominant_frequency_hz",
                    "dominant_power",
                    "band_edges_hz",
                    "band_power",
                )
            },
            "local_valid": List(List(Value("bool"))),
            "spectral_valid": List(List(Value("bool"))),
            "local_relative_times_ns": List(integer),
            "spectral_relative_times_ns": List(integer),
            "frequencies_hz": vector,
            **{
                name: number
                for name in (
                    "imu_uncertainty_deg",
                    "along_slope_rad",
                    "along_slope_deg",
                    "cross_slope_rad",
                    "cross_slope_deg",
                    "plane_offset",
                    "plane_rmse_m",
                    "plane_inlier_ratio",
                    "valid_depth_fraction",
                )
            },
            "plane_normal": vector,
            "plane_point_count": integer,
            "plane_inlier_count": integer,
            "rgb_crop_origin_uv": List(integer),
            "depth_crop_origin_uv": List(integer),
            "rgb_patch": Image(),
            "depth_patch": Image(),
            "rgb_patch_mask": Image(),
            "depth_patch_mask": Image(),
            # Stage 1 geometry stays numerical and accessible, rather than becoming
            # an opaque NPZ blob. Full-image masks are already represented by the
            # patch masks together with crop origins and saved image calibration.
            "geometry": {
                **{
                    name: matrix
                    for name in (
                        "rgb_patch_polygon_pixels",
                        "depth_patch_polygon_pixels",
                        "patch_inner_points_base_m",
                        "patch_outer_points_base_m",
                        "rgb_inner_track_pixels",
                        "rgb_outer_track_pixels",
                        "depth_inner_track_pixels",
                        "depth_outer_track_pixels",
                    )
                },
                **{
                    name: List(Value("bool"))
                    for name in (
                        "rgb_inner_track_valid",
                        "rgb_outer_track_valid",
                        "depth_inner_track_valid",
                        "depth_outer_track_valid",
                    )
                },
            },
            "fft_start_time_ns": integer,
            "fft_end_time_ns": integer,
            "fft_start_angle_rad": number,
            "fft_end_angle_rad": number,
            "fft_lower_angle_bound_rad": number,
            "fft_upper_angle_bound_rad": number,
            "temporal_overlap_ids": List(text),
            "spatial_overlap_ids": List(text),
            "temporal_overlap_packet_numbers": List(integer),
            "spatial_overlap_packet_numbers": List(integer),
        }
    )
    if options.include_full_frames:
        features.update(rgb_frame=Image(), depth_frame=Image())
    if options.include_point_maps:
        features["point_map"] = {
            "points_base_m": matrix,
            "pixel_uv": matrix,
            "depth_m": vector,
            "ransac_inlier_mask": List(Value("bool")),
        }
    return features


def png_image(path: Path, *, depth: bool = False, mask: bool = False) -> dict:
    """Embed lossless PNG bytes, never a link to the source computer."""
    array = np.load(path, allow_pickle=False)
    if mask:
        if array.ndim != 2 or array.dtype.kind != "b":
            raise ValueError(f"Expected a 2-D boolean mask: {path.name}")
        array = array.astype(np.uint8) * 255
    elif depth:
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(f"Expected unsigned 16-bit depth: {path.name}")
    elif array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit RGB image: {path.name}")
    output = BytesIO()
    PillowImage.fromarray(array).save(output, format="PNG")
    return {"bytes": output.getvalue(), "path": None}


def validate_telemetry(packet: Packet, arrays: dict) -> list[str]:
    names = packet.metadata["telemetry_channel_names"]
    if (
        len(names) != 15
        or len(set(names)) != 15
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError("Expected 15 distinct telemetry channel names.")
    for prefix in ("local", "spectral"):
        values, valid = arrays[f"{prefix}_values"], arrays[f"{prefix}_valid"]
        times = arrays[f"{prefix}_relative_times_ns"]
        if (
            values.shape != (len(times), len(names))
            or valid.shape != values.shape
            or valid.dtype.kind != "b"
        ):
            raise ValueError(f"Inconsistent {prefix} telemetry shapes or validity mask.")
        if (
            times.ndim != 1
            or times.dtype.kind not in "iu"
            or len(times) < 2
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"Invalid {prefix} telemetry timestamps.")
        if not np.all(np.isfinite(values[valid])):
            raise ValueError(f"{prefix} telemetry marks nonfinite measurements as valid.")
    return names


def packet_row(packet: Packet, options: ExportOptions, bands: np.ndarray) -> dict:
    """Load one packet at a time; preserve original precision and validity masks."""
    source, metadata, features = packet.source, packet.metadata, packet.features
    telemetry = read_npz(packet.directory / "telemetry.npz")
    names = validate_telemetry(packet, telemetry)
    spectra = read_npz(packet.directory / "stage2/features.npz")
    if spectra["spectral_power"].shape[1:] != (len(names),):
        raise ValueError("Spectrum channel count does not match telemetry.")
    expected_grid = np.fft.rfftfreq(len(telemetry["spectral_values"]), d=1 / source.sample_rate_hz)
    if spectra["frequencies_hz"].shape != expected_grid.shape or not np.allclose(
        spectra["frequencies_hz"], expected_grid
    ):
        raise ValueError("Saved FFT grid does not match the sample rate and window length.")
    coverage = packet.coverage
    if coverage is None:
        raise ValueError("Missing FFT trajectory coverage.")
    full_id, laboratory, terrain, lighting = source.terrain
    row = {
        "export_id": packet.export_id,
        "packet_id": packet.packet_id,
        "source_id": source.source_id,
        "experiment_id": source.experiment_id,
        "run_number": source.run_number,
        "processing_id": source.processing_id,
        "packet_number": packet.number,
        "terrain_and_light_ID": full_id,
        "laboratory_id": laboratory,
        "terrain_setup_id": terrain,
        "lighting_id": lighting,
        "split": None,
        **{name: features.get(name) for name in ("class_id", "classification_method", "model_id")},
        "image_time_ns": int(metadata["image_time_ns"]),
        "contact_time_ns": int(metadata["contact_time_ns"]),
        "lap_index": metadata.get("lap_index"),
        "sample_rate_hz": source.sample_rate_hz,
        "depth_scale_m": float(source.provenance["processing_config"].get("depth_scale_m", 0.001)),
        "channel_names": names,
        "channel_units": [channel_unit(name) for name in names],
        **{
            name: telemetry[name]
            for name in (
                "local_values",
                "local_valid",
                "local_relative_times_ns",
                "spectral_values",
                "spectral_valid",
                "spectral_relative_times_ns",
            )
        },
        **{
            name: spectra[name]
            for name in (
                "frequencies_hz",
                "spectral_power",
                "dominant_frequency_hz",
                "dominant_power",
            )
        },
        "band_edges_hz": bands,
        "band_power": band_power(
            spectra["frequencies_hz"], spectra["spectral_power"], bands, source.sample_rate_hz / 2
        ),
        **{
            name: features.get(name)
            for name in (
                "along_slope_rad",
                "along_slope_deg",
                "cross_slope_rad",
                "cross_slope_deg",
                "plane_normal",
                "plane_offset",
                "plane_rmse_m",
                "plane_point_count",
                "plane_inlier_count",
                "plane_inlier_ratio",
                "valid_depth_fraction",
                "rgb_crop_origin_uv",
                "depth_crop_origin_uv",
            )
        },
        "imu_uncertainty_deg": features.get("IMU uncertainty", metadata.get("imu_uncertainty_deg")),
        "metadata_json": json_text(metadata),
        "features_json": json_text(features),
        "geometry": read_npz(packet.directory / "geometry.npz"),
        "rgb_patch": png_image(packet.directory / "stage2/rgb_patch.npy"),
        "depth_patch": png_image(packet.directory / "stage2/depth_patch.npy", depth=True),
        "rgb_patch_mask": png_image(packet.directory / "stage2/rgb_patch_mask.npy", mask=True),
        "depth_patch_mask": png_image(packet.directory / "stage2/depth_patch_mask.npy", mask=True),
        "fft_start_time_ns": coverage.start_time_ns,
        "fft_end_time_ns": coverage.end_time_ns,
        "fft_start_angle_rad": coverage.start_angle_rad,
        "fft_end_angle_rad": coverage.end_angle_rad,
        "fft_lower_angle_bound_rad": coverage.lower_angle_rad,
        "fft_upper_angle_bound_rad": coverage.upper_angle_rad,
    }
    # Check crop and mask dimensions together; a mismatched mask must not travel
    # unnoticed into a published training example.
    for prefix in ("rgb", "depth"):
        image = PillowImage.open(BytesIO(row[f"{prefix}_patch"]["bytes"]))
        mask = PillowImage.open(BytesIO(row[f"{prefix}_patch_mask"]["bytes"]))
        if image.size != mask.size:
            raise ValueError(f"{prefix} patch and mask dimensions disagree.")
    if options.include_full_frames:
        row["rgb_frame"] = png_image(packet.directory / "rgb.npy")
        row["depth_frame"] = png_image(packet.directory / "depth.npy", depth=True)
    if options.include_point_maps:
        row["point_map"] = read_npz(packet.directory / "stage2/point_map.npz")
        points = row["point_map"]
        size = len(points["depth_m"])
        for key, shape in (
            ("points_base_m", (size, 3)),
            ("pixel_uv", (size, 2)),
            ("ransac_inlier_mask", (size,)),
        ):
            if points[key].shape != shape:
                raise ValueError(f"Point-map {key} has inconsistent dimensions.")
    return row


def refresh_selection(packets: list[Packet], check_cancel: Callable[[], None]) -> list[Packet]:
    """Re-read sources so labels edited since GUI discovery cannot be stale."""
    if not packets:
        raise ValueError("Select at least one packet.")
    sources, result, runs, seen = {}, [], {}, set()
    for selected in packets:
        check_cancel()
        root = selected.source.root
        if root not in sources:
            sources[root] = load_source(root, check_cancel)
        source = sources[root]
        if source.run_key in runs and runs[source.run_key] != root:
            raise ValueError("Select only one source processing version per experiment/run.")
        runs[source.run_key] = root
        packet_index = {packet.packet_id: packet for packet in source.packets}
        if selected.packet_id not in packet_index:
            raise ValueError(f"Selected packet no longer exists: {selected.packet_id}")
        packet = packet_index[selected.packet_id]
        if packet.export_id in seen:
            raise ValueError(f"Packet selected more than once: {packet.packet_id}")
        if packet.problem:
            raise ValueError(
                f"{source.experiment_id}/run_{source.run_number}, packet {packet.number}: {packet.problem}"
            )
        seen.add(packet.export_id)
        result.append(packet)
    return sorted(result, key=lambda packet: (packet.source.source_id, packet.number))


def export_dataset(
    packets: list[Packet],
    destination: Path,
    name: str,
    options: ExportOptions = DEFAULT_OPTIONS,
    progress: Callable[[int, str], None] = lambda percent, message: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> ExportResult:
    """Perform coverage/selection validation, a second overlap pass, then export."""
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,99}", name) or name.endswith("."):
        raise ValueError(
            "Use a dataset name beginning with a letter or digit, with no path separators (100 characters maximum)."
        )
    destination = destination.expanduser().resolve()
    if not destination.is_dir():
        raise ValueError("Choose an existing destination directory.")
    progress(0, "Validating selected packets and trajectory coverage")
    packets = refresh_selection(packets, check_cancel)
    bands = frequency_bands(
        options.band_width_hz, max(packet.source.sample_rate_hz / 2 for packet in packets)
    )
    schema = packet_features(options).arrow_schema
    groups: dict[str, list[Packet]] = {}
    for packet in packets:
        groups.setdefault(packet.source.source_id, []).append(packet)

    summaries, neighbors = [], {}
    for index, group in enumerate(groups.values()):
        source = group[0].source
        progress(
            5 + int(15 * index / len(groups)),
            f"Analyzing overlaps: {source.experiment_id}/run_{source.run_number}",
        )
        temporal, spatial, time_count, spatial_count = analyze_overlaps(
            [packet.coverage for packet in group], check_cancel
        )
        for number, packet in enumerate(group):
            neighbors[packet.export_id] = {
                "temporal_overlap_ids": [group[i].export_id for i in temporal[number]],
                "spatial_overlap_ids": [group[i].export_id for i in spatial[number]],
                "temporal_overlap_packet_numbers": [group[i].number for i in temporal[number]],
                "spatial_overlap_packet_numbers": [group[i].number for i in spatial[number]],
            }
        full_id, laboratory, terrain, lighting = source.terrain
        summaries.append(
            {
                "source_id": source.source_id,
                "experiment_id": source.experiment_id,
                "run_number": source.run_number,
                "processing_id": source.processing_id,
                "terrain_and_light_ID": full_id,
                "laboratory_id": laboratory,
                "terrain_setup_id": terrain,
                "lighting_id": lighting,
                "selected_packet_count": len(group),
                "source_packet_count": len(source.packets),
                "maximum_temporal_nonoverlapping_count": time_count,
                "maximum_spatial_nonoverlapping_count": spatial_count,
                "maximum_encoder_gap_ns": source.maximum_encoder_gap_ns,
                "provenance_json": json_text(portable_provenance(source)),
            }
        )

    settings = {"export_version": EXPORT_VERSION, **asdict(options)}
    digest = hashlib.sha256(json_text({"settings": settings, "runs": summaries}).encode())
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination))
    writer = None
    try:
        (temporary / "packets").mkdir()
        shard, shard_bytes, pending = 0, 0, []

        def flush() -> None:
            nonlocal writer, shard, shard_bytes
            if not pending:
                return
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary / "packets" / f"part-{shard:05d}.parquet", schema, compression="zstd"
                )
            table = pa.Table.from_pylist(pending, schema=schema)
            writer.write_table(table)
            shard_bytes += table.nbytes
            pending.clear()
            if shard_bytes >= SHARD_TARGET_BYTES:
                writer.close()
                writer = None
                shard += 1
                shard_bytes = 0

        for index, packet in enumerate(packets):
            check_cancel()
            progress(
                20 + int(70 * index / len(packets)), f"Exporting packet {index + 1}/{len(packets)}"
            )
            try:
                row = packet_row(packet, options, bands)
                row.update(neighbors[packet.export_id])
                # Only schema fields are serialized: geometry may contain extra
                # Stage 1 mask arrays that are represented by the patch images.
                table = pa.Table.from_pylist([clean_json(row)], schema=schema)
            except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
                raise ValueError(
                    f"{packet.source.experiment_id}/run_{packet.source.run_number}, packet {packet.number}: {error}"
                ) from error
            # Hash normalized exported records, not source filenames, ZIP header
            # timestamps or destination paths. Selection order is canonical.
            digest.update(table.to_batches()[0].serialize().to_pybytes())
            pending.append(table.to_pylist()[0])
            if len(pending) >= ROWS_PER_GROUP:
                flush()
        flush()
        if writer is not None:
            writer.close()
            writer = None
        check_cancel()
        progress(92, "Writing run summaries and documentation")
        run_schema = pa.schema(
            [
                pa.field(key, pa.int64() if isinstance(value, int) else pa.string())
                for key, value in summaries[0].items()
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(summaries, schema=run_schema),
            temporary / "runs.parquet",
            compression="zstd",
        )
        identity = digest.hexdigest()
        output = destination / f"{name}_{identity[:12]}"
        if output.exists():
            raise ValueError(
                f"This dataset already exists; choose another name or destination: {output}"
            )
        checksums = {}
        for path in sorted(temporary.rglob("*.parquet")):
            check_cancel()
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    check_cancel()
                    checksum.update(block)
            checksums[str(path.relative_to(temporary))] = checksum.hexdigest()
        manifest = {
            **settings,
            "name": name,
            "sha256": identity,
            "created_utc": datetime.now(UTC).isoformat(),
            "packet_count": len(packets),
            "class_counts": dict(
                Counter(
                    "unlabelled" if packet.label is None else str(packet.label)
                    for packet in packets
                )
            ),
            "class_names": list(CLASS_NAMES),
            "band_edges_hz": bands.tolist(),
            "selected_export_ids": [packet.export_id for packet in packets],
            "runs": summaries,
            "parquet_sha256": checksums,
        }
        (temporary / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        from .readme import export_readme

        (temporary / "README.md").write_text(
            export_readme(name, options, len(packets)), encoding="utf-8"
        )
        check_cancel()
        temporary.rename(output)
        progress(100, f"Exported {len(packets)} packets")
        return ExportResult(output, len(packets), summaries)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            # Cleanup still matters if a disk error also prevents closing the
            # Parquet footer. This directory was created by this export alone.
            if temporary.exists():
                shutil.rmtree(temporary)
