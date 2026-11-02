"""Inspect experiment-wide terrain IDs and prepare metadata-only corrections.

Inspection tolerates missing/invalid IDs so the batch GUI can repair old files.
Dataset generation uses the strict checks below and never invents an ID.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml

from legwheel_experiments.terrain import (
    TERRAIN_ID_FIELD, experiment_terrain_id, validate_terrain_id,
)


EXPERIMENT_MAPPINGS = ("rig_config", "leg_config", "wheel_config", "electronics_hardware")
REGENERATION_REASON = "terrain_and_light_ID was added or changed; regenerate this dataset."


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
        value = json.loads(contents) if path.suffix == ".json" else yaml.safe_load(contents)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read metadata in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def _environment(document: dict[str, Any]) -> dict[str, Any]:
    environment = document.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment must be a mapping before an ID can be added.")
    return environment


@dataclass
class TerrainIdDocument:
    """An experiment YAML/snapshot or a dataset's provenance JSON."""

    path: Path
    contents: dict[str, Any]
    provenance: bool = False

    def values(self) -> list[Any]:
        """Include both provenance copies so a hidden disagreement is detected."""
        if not self.provenance:
            return [_environment(self.contents).get(TERRAIN_ID_FIELD)]
        experiment = self.contents.get("experiment_config", {})
        if not isinstance(experiment, dict):
            raise ValueError(f"experiment_config must be a mapping in {self.path}.")
        return [
            self.contents.get(TERRAIN_ID_FIELD),
            _environment(experiment).get(TERRAIN_ID_FIELD),
        ]

    def needs_update(self, terrain_id: str) -> bool:
        """Return whether either stored copy needs the selected value."""
        return any(value != terrain_id for value in self.values())

    def updated_text(self, terrain_id: str) -> str:
        """Change only the ID and, for old datasets, the regeneration notice."""
        updated = deepcopy(self.contents)
        if self.provenance:
            updated[TERRAIN_ID_FIELD] = terrain_id
            experiment = updated.setdefault("experiment_config", {})
            experiment.setdefault("environment", {})[TERRAIN_ID_FIELD] = terrain_id
            # Keep the old dataset identifier, files and classifications. Its
            # corrected metadata must not make an old analysis reusable.
            updated["requires_regeneration"] = True
            updated["regeneration_reason"] = REGENERATION_REASON
            return json.dumps(updated, indent=2) + "\n"
        updated.setdefault("environment", {})[TERRAIN_ID_FIELD] = terrain_id
        return yaml.safe_dump(updated, sort_keys=False)


@dataclass
class TerrainIdReview:
    """Every relevant metadata file from one experiment, including unselected runs."""

    directory: Path
    documents: list[TerrainIdDocument]

    @property
    def known_ids(self) -> list[str]:
        """Return the distinct valid IDs without treating case variants as equal."""
        values = set()
        for document in self.documents:
            for value in document.values():
                try:
                    values.add(validate_terrain_id(value))
                except ValueError:
                    continue
        return sorted(values)

    @property
    def needs_resolution(self) -> bool:
        """Prompt for missing/invalid values and for case-sensitive conflicts."""
        known_ids = self.known_ids
        if len(known_ids) != 1:
            return True
        return any(document.needs_update(known_ids[0]) for document in self.documents)

    def updates(self, terrain_id: str) -> dict[Path, str]:
        """Prepare all changed files before the GUI attempts any writes."""
        validate_terrain_id(terrain_id)
        return {
            document.path: document.updated_text(terrain_id)
            for document in self.documents if document.needs_update(terrain_id)
        }

    def affected_datasets(self, terrain_id: str) -> list[Path]:
        """List the precise datasets to warn about before their metadata changes."""
        return [
            document.path.parent for document in self.documents
            if document.provenance and document.needs_update(terrain_id)
        ]


def review_terrain_ids(directory: Path) -> TerrainIdReview:
    """Read root experiment configs, every run snapshot and every dataset provenance."""
    documents = []
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        try:
            contents = _load_mapping(path)
        except ValueError:
            # Unrelated YAML is not an experiment config. Discovery reports
            # malformed root files separately; known snapshots below are strict.
            continue
        if isinstance(contents.get("experiment_id"), str) and all(
            isinstance(contents.get(name), dict) for name in EXPERIMENT_MAPPINGS
        ):
            documents.append(TerrainIdDocument(path, contents))
    for run in sorted(directory.iterdir()):
        if not run.is_dir() or re.fullmatch(r"run_\d+", run.name) is None:
            continue
        for path in sorted(run.glob("experiment_config_run_*_snapshot.yaml")):
            documents.append(TerrainIdDocument(path, _load_mapping(path)))
        for path in sorted((run / "derived").glob("*/provenance.json")):
            documents.append(TerrainIdDocument(path, _load_mapping(path), provenance=True))
    # Validate mapping shapes now, before a dialog can offer to change anything.
    for document in documents:
        try:
            document.values()
        except ValueError as error:
            raise ValueError(f"{document.path}: {error}") from error
    return TerrainIdReview(directory, documents)


def require_run_terrain_id(run_directory: Path, experiment: dict[str, Any]) -> str:
    """Reject missing or conflicting IDs before individual or batch analysis starts."""
    try:
        terrain_id = experiment_terrain_id(experiment)
        directory = (
            run_directory.parent if re.fullmatch(r"run_\d+", run_directory.name)
            else run_directory
        )
        review = review_terrain_ids(directory)
        if any(document.needs_update(terrain_id) for document in review.documents):
            raise ValueError("Missing or conflicting terrain_and_light_ID values in this experiment.")
    except ValueError as error:
        raise ValueError(
            f"{error} Open the experiment in the batch dataset builder to add or resolve its IDs."
        ) from error
    return terrain_id


def require_provenance_terrain_id(provenance: dict[str, Any]) -> str:
    """Prevent Stage 2 from blessing metadata that explicitly requires a fresh build."""
    if provenance.get("requires_regeneration"):
        raise ValueError(
            "This dataset requires regeneration. Run build_dataset or use the batch builder."
        )
    terrain_id = validate_terrain_id(provenance.get(TERRAIN_ID_FIELD))
    if experiment_terrain_id(provenance.get("experiment_config", {})) != terrain_id:
        raise ValueError("Conflicting terrain_and_light_ID values in provenance.json.")
    return terrain_id
