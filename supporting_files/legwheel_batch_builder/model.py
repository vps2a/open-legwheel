"""Filesystem operations shared by the batch GUI and its ROS worker.

This module has no Qt or ROS imports. Processing settings and YAML writing come
from legwheel_dataset so individual and batch builds use the same definitions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import yaml

from legwheel_dataset.config_creator import write_processing_config
from legwheel_dataset.terrain import EXPERIMENT_MAPPINGS


EVENT_PREFIX = "LEGWHEEL_BATCH_EVENT "
ATTEMPT_STATUS_FILE = "batch_build_status.json"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a mapping and present malformed YAML as a useful validation error."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping.")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def dataset_state(directory: Path) -> str:
    """Classify one output directory, including attempts from older builders."""
    attempt = _read_json(directory / ATTEMPT_STATUS_FILE)
    analysis = _read_json(directory / "packets" / "analysis_report.json")
    provenance = _read_json(directory / "provenance.json")
    if attempt.get("status") in {"failed", "cancelled"}:
        return "failed"
    if str(analysis.get("status", "")).startswith("failed"):
        return "failed"
    if (
        provenance.get("stage") == "stage2_complete"
        and (directory / "manifest.jsonl").is_file()
        and (directory / "features.jsonl").is_file()
        and (directory / "packets").is_dir()
    ):
        return "completed"
    return "partial"


def dataset_requires_regeneration(directory: Path) -> bool:
    """Keep completed-but-outdated datasets out of the build-skip path."""
    return bool(_read_json(directory / "provenance.json").get("requires_regeneration"))


@dataclass
class DatasetCounts:
    completed: int = 0
    partial: int = 0
    failed: int = 0

    def label(self) -> str:
        """Only show the extra counts when those kinds of output exist."""
        parts = [f"Completed Datasets: {self.completed}"]
        if self.partial:
            parts.append(f"Partial Datasets: {self.partial}")
        if self.failed:
            parts.append(f"Failed Attempts: {self.failed}")
        return "  ·  ".join(parts)


def count_datasets(run_directory: Path) -> DatasetCounts:
    counts = DatasetCounts()
    derived = run_directory / "derived"
    if not derived.is_dir():
        return counts
    for directory in derived.iterdir():
        if directory.is_dir() and any(
            (directory / name).exists()
            for name in (ATTEMPT_STATUS_FILE, "provenance.json", "manifest.jsonl", "packets")
        ):
            state = dataset_state(directory)
            if state == "completed":
                counts.completed += 1
            elif state == "failed":
                counts.failed += 1
            else:
                counts.partial += 1
    return counts


@dataclass
class RunInfo:
    directory: Path
    experiment_name: str
    number: int
    valid: bool | None = None
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    counts: DatasetCounts = field(default_factory=DatasetCounts)

    @property
    def label(self) -> str:
        return f"{self.experiment_name}/{self.directory.name}"


@dataclass
class ExperimentInfo:
    directory: Path
    experiment_id: str
    runs: list[RunInfo]


@dataclass
class BuildResult:
    run: RunInfo
    status: str
    message: str
    output: Path | None = None


def inspect_run(directory: Path, experiment_name: str) -> RunInfo:
    """Keep unusable and unreviewed runs visible, with explicit explanations."""
    run = RunInfo(directory, experiment_name, int(directory.name.removeprefix("run_")))
    try:
        config = load_yaml(directory / "run_config.yaml")
        value = config.get("run_valid")
        run.valid = value if isinstance(value, bool) else None
        if run.valid is False:
            run.warnings.append("Marked invalid")
        elif run.valid is None:
            run.warnings.append("Validity not reviewed")
    except (OSError, ValueError) as error:
        run.problems.append(f"Cannot read run_config.yaml: {error}")

    bag = directory / "rosbag"
    if not bag.is_dir() or not (bag / "metadata.yaml").is_file() or not any(bag.glob("*.mcap")):
        run.problems.append("Missing rosbag directory, metadata.yaml or MCAP recording")
    for pattern, description in (
        ("experiment_config_run_*_snapshot.yaml", "experiment configuration snapshot"),
        ("camera_driver_config_run_*_snapshot.yaml", "camera configuration snapshot"),
    ):
        snapshots = list(directory.glob(pattern))
        if len(snapshots) != 1:
            run.problems.append(f"Expected one {description}; found {len(snapshots)}")
        else:
            try:
                load_yaml(snapshots[0])
            except (OSError, ValueError) as error:
                run.problems.append(str(error))
    try:
        run.counts = count_datasets(directory)
        outdated = sum(
            dataset_requires_regeneration(path.parent)
            for path in (directory / "derived").glob("*/provenance.json")
        )
        if outdated:
            subject = "1 dataset requires" if outdated == 1 else f"{outdated} datasets require"
            run.warnings.append(f"{subject} regeneration (terrain/light ID updated)")
    except OSError as error:
        run.problems.append(f"Cannot inspect existing datasets: {error}")
    return run


def discover_experiments(root: Path) -> tuple[list[ExperimentInfo], list[str]]:
    """Inspect immediate child folders, recognizing experiment YAML by contents."""
    experiments, warnings = [], []
    for directory in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not directory.is_dir():
            continue
        identifiers = set()
        for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
            try:
                config = load_yaml(path)
            except (OSError, ValueError) as error:
                warnings.append(f"{directory.name}/{path.name}: {error}")
                continue
            identifier = config.get("experiment_id")
            if isinstance(identifier, str) and identifier.strip() and all(
                isinstance(config.get(name), dict) for name in EXPERIMENT_MAPPINGS
            ):
                identifiers.add(identifier)
        if not identifiers:
            continue
        if len(identifiers) > 1:
            warnings.append(f"{directory.name}: multiple experiment IDs; run snapshots will be used.")
        runs = [
            inspect_run(path, directory.name)
            for path in directory.iterdir()
            if path.is_dir() and re.fullmatch(r"run_\d+", path.name)
        ]
        runs.sort(key=lambda run: run.number)
        experiments.append(ExperimentInfo(directory, ", ".join(sorted(identifiers)), runs))
    return experiments, warnings


def mark_run_valid(run: RunInfo) -> RunInfo:
    """Save immediately, preserving every other run-configuration value."""
    path = run.directory / "run_config.yaml"
    config = load_yaml(path)
    config["run_valid"] = True
    write_processing_config(path, yaml.safe_dump(config, sort_keys=False))
    return inspect_run(run.directory, run.experiment_name)


def write_attempt_status(directory: Path, status: str, message: str = "") -> None:
    """Record failures/cancellations without changing the pipeline's own reports."""
    # Some failures occur before packet isolation creates its output directory.
    # Retain those known attempts too, so they appear in the next run listing.
    directory.mkdir(parents=True, exist_ok=True)
    contents = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
    write_processing_config(directory / ATTEMPT_STATUS_FILE, json.dumps(contents, indent=2) + "\n")


def archive_incomplete_output(directory: Path) -> Path:
    """Preserve incomplete or outdated output before rebuilding at its normal path."""
    if not directory.is_dir():
        raise ValueError(f"Dataset output path is not a directory: {directory}")
    if dataset_state(directory) == "completed" and not dataset_requires_regeneration(directory):
        raise ValueError("A completed dataset must never be moved for a retry.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = directory.with_name(f"{directory.name}_attempt_{timestamp}_{uuid4().hex[:8]}")
    directory.rename(archived)
    return archived
