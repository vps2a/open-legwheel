"""Read-only discovery and validation of derived packet folders.

Do not use the browser's dataset loader here: opening an old dataset through
that loader can write classification upgrades. Export must never edit sources.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from legwheel_dataset_browser.classification import (
    CLASSIFICATION_FIELDS,
    TERRAIN_CLASSES,
)

from .calculations import WindowCoverage, window_coverage

CLASS_NAMES = TERRAIN_CLASSES
REQUIRED_PACKET_FILES = (
    "metadata.json",
    "telemetry.npz",
    "geometry.npz",
    "stage2/features.json",
    "stage2/features.npz",
    "stage2/rgb_patch.npy",
    "stage2/depth_patch.npy",
    "stage2/rgb_patch_mask.npy",
    "stage2/depth_patch_mask.npy",
)
# The saved trajectory contains count changes, not every encoder message.
# This exporter-only limit bounds interpolation between movement events; it
# must not inherit the pitch estimator's message-dropout threshold.
MAXIMUM_ENCODER_EVENT_INTERVAL_NS = 2_000_000_000


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path.name}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}, line {number}: expected an object.")
            records.append(value)
    return records


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def terrain_parts(provenance: dict) -> tuple[str | None, str | None, str | None, str | None]:
    """Split KK_01_L1 into full ID, laboratory, physical setup and lighting."""
    embedded = (
        provenance.get("experiment_config", {}).get("environment", {}).get("terrain_and_light_ID")
    )
    identifier = provenance.get("terrain_and_light_ID") or embedded
    if identifier is not None and embedded is not None and identifier != embedded:
        raise ValueError("Terrain/light identifiers disagree between provenance copies.")
    if identifier is None:
        return None, None, None, None
    match = re.fullmatch(r"([A-Za-z0-9]{2})_([A-Za-z0-9_-]+)_(L[0-9]+)", str(identifier))
    if match is None:
        raise ValueError(f"Terrain/light ID {identifier!r} must follow KK_01_L1 format.")
    laboratory, arrangement, lighting = match.groups()
    return identifier, laboratory, f"{laboratory}_{arrangement}", lighting


@dataclass
class SourceDataset:
    root: Path
    provenance: dict[str, Any]
    experiment_id: str
    run_number: str
    processing_id: str
    terrain: tuple[str | None, str | None, str | None, str | None]
    maximum_encoder_gap_ns: int
    packets: list[Packet] = field(default_factory=list)

    @property
    def run_key(self) -> tuple[str, str]:
        return self.experiment_id, self.run_number

    @property
    def source_id(self) -> str:
        identity = json.dumps([*self.run_key, self.processing_id], separators=(",", ":"))
        return hashlib.sha256(identity.encode()).hexdigest()[:24]

    @property
    def sample_rate_hz(self) -> float:
        value = float(self.provenance["processing_config"]["telemetry_resample_hz"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Telemetry sample rate must be finite and positive.")
        return value


@dataclass
class Packet:
    source: SourceDataset
    number: int
    directory: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    features: dict[str, Any]
    coverage: WindowCoverage | None = None
    problem: str = ""

    @property
    def packet_id(self) -> str:
        return str(self.manifest["packet_id"])

    @property
    def export_id(self) -> str:
        return f"{self.source.source_id}:{self.packet_id}"

    @property
    def label(self) -> int | None:
        return self.features.get("class_id", self.metadata.get("class_id"))


def load_source(root: Path, check_cancel: Callable[[], None] = lambda: None) -> SourceDataset:
    """Load lightweight packet headers and coverage; keep blocked packets visible."""
    root = root.expanduser().resolve()
    provenance = read_json(root / "provenance.json")
    records = read_jsonl(root / "manifest.jsonl")
    if not records:
        raise ValueError("Manifest contains no packets.")
    records.sort(key=lambda item: (int(item["image_time_ns"]), str(item["packet_id"])))
    if len({item["packet_id"] for item in records}) != len(records):
        raise ValueError("Manifest contains duplicate packet IDs.")

    def directory(record: dict) -> Path:
        path = (root / record["packet_path"]).resolve()
        if not path.is_relative_to(root / "packets"):
            raise ValueError("Packet path must remain inside the source packets directory.")
        return path

    first = read_json(directory(records[0]) / "metadata.json")
    source = SourceDataset(
        root,
        provenance,
        str(first["experiment_id"]),
        str(first["run_number"]),
        str(provenance.get("dataset_identifier") or root.name),
        terrain_parts(provenance),
        MAXIMUM_ENCODER_EVENT_INTERVAL_NS,
    )
    _ = source.sample_rate_hz  # Validate configuration before presenting packets.
    features = read_jsonl(root / "features.jsonl")
    feature_index = {str(record["packet_id"]): record for record in features}
    if len(feature_index) != len(features):
        raise ValueError("features.jsonl contains duplicate packet IDs.")
    trajectory_problem = ""
    try:
        trajectory = read_npz(root / "packets" / "run_motion_analysis.npz")
        tick_times = trajectory["tick_times_ns"]
        angles = trajectory.get(
            "theta_y_motion_at_ticks_rad", trajectory.get("theta_y_at_ticks_rad")
        )
        if angles is None:
            raise ValueError("Missing corrected motion angles in encoder trajectory.")
    except (OSError, ValueError, KeyError) as error:
        tick_times, angles = np.array([]), np.array([])
        trajectory_problem = f"Cannot map FFT positions: {error}"

    for number, record in enumerate(records, 1):
        check_cancel()
        packet = Packet(source, number, directory(record), record, {}, {})
        source.packets.append(packet)
        try:
            packet.metadata = read_json(packet.directory / "metadata.json")
            packet_features = read_json(packet.directory / "stage2/features.json")
            aggregate = feature_index.get(packet.packet_id)
            if aggregate is None:
                raise ValueError("Packet is absent from features.jsonl.")
            packet.features = {**packet_features, **aggregate}
            for document in (packet.metadata, packet_features):
                if str(document.get("packet_id")) != packet.packet_id:
                    raise ValueError("Packet IDs disagree between saved files.")
            if (
                str(packet.metadata["experiment_id"]),
                str(packet.metadata["run_number"]),
            ) != source.run_key:
                raise ValueError("Packet belongs to a different experiment or run.")
            for name in CLASSIFICATION_FIELDS:
                values = [
                    doc[name]
                    for doc in (packet.metadata, packet_features, aggregate)
                    if name in doc
                ]
                if values and any(value != values[0] for value in values):
                    raise ValueError(
                        f"Conflicting {name}; reconcile labels in the source before exporting."
                    )
                packet.features[name] = values[0] if values else None
            if packet.label is not None and (
                type(packet.label) is not int or packet.label not in range(4)
            ):
                raise ValueError("Class ID must be null or an integer from 0 to 3.")
            if provenance.get("requires_regeneration"):
                raise ValueError("Source dataset is marked as requiring regeneration.")
            if not packet.features.get("valid"):
                raise ValueError("Packet has no successful Stage 2 feature extraction.")
            missing = [
                name for name in REQUIRED_PACKET_FILES if not (packet.directory / name).is_file()
            ]
            if missing:
                raise ValueError("Missing required files: " + ", ".join(missing))
            if trajectory_problem:
                raise ValueError(trajectory_problem)
            with np.load(packet.directory / "telemetry.npz", allow_pickle=False) as telemetry:
                packet.coverage = window_coverage(
                    int(packet.metadata["contact_time_ns"]),
                    telemetry["spectral_relative_times_ns"],
                    tick_times,
                    angles,
                    source.maximum_encoder_gap_ns,
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            packet.problem = str(error)
    return source


def discover_sources(
    root: Path,
    check_cancel: Callable[[], None] = lambda: None,
) -> tuple[list[SourceDataset], list[str]]:
    """Discover derived folders recursively while skipping raw bags and environments."""
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    sources, warnings = [], []

    def report_walk_error(error: OSError) -> None:
        warnings.append(str(error))

    for folder, children, files in os.walk(root, onerror=report_walk_error):
        check_cancel()
        children[:] = sorted(
            name
            for name in children
            if name
            not in {
                "rosbag",
                ".git",
                ".venv",
                "node_modules",
                "__pycache__",
            }
            and not name.startswith(".export-")
        )
        if "manifest.jsonl" not in files or "provenance.json" not in files:
            continue
        children.clear()
        try:
            sources.append(load_source(Path(folder), check_cancel))
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f"{folder}: {error}")
    sources.sort(key=lambda source: (source.experiment_id, source.run_number, source.processing_id))
    return sources, warnings
