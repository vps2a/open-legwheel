"""Validation for the experiment-wide terrain and lighting setup identifier."""

import re
from typing import Any


TERRAIN_ID_FIELD = "terrain_and_light_ID"


def validate_terrain_id(value: Any) -> str:
    """Require a case-sensitive code, preserving numeric strings and leading zeros."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
        or re.search(r"[A-Za-z0-9]", value) is None
    ):
        raise ValueError(
            "environment.terrain_and_light_ID must be a string containing only "
            "ASCII letters, digits, underscores and hyphens, with at least one "
            "letter or digit. Values are case-sensitive; quote numeric IDs in YAML."
        )
    return value


def experiment_terrain_id(experiment: dict[str, Any]) -> str:
    """Read the required ID from an experiment mapping without changing its case."""
    if not isinstance(experiment, dict):
        raise ValueError("Experiment configuration must be a mapping.")
    environment = experiment.get("environment")
    return validate_terrain_id(
        environment.get(TERRAIN_ID_FIELD) if isinstance(environment, dict) else None
    )
