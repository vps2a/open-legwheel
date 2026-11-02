"""Explicit feature definitions, transition filtering and run-based partitions.

Nothing in this module fits a statistical transform. Feature extraction is local
to each packet; learned preprocessing belongs inside the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import BAND_CHANNELS, MOTOR_CHANNELS, TelemetryDataset

FEATURE_SCHEMA_VERSION = 1
PARTITIONS = ("training", "validation", "testing")
ROLES = ("excluded",) + PARTITIONS


class ConfigurationError(ValueError):
    """A feature selection or split cannot be used for this experiment."""


@dataclass(frozen=True)
class FeatureSpec:
    kind: str  # std, power, speed or slope
    channel: str
    band: int = -1

    @property
    def name(self) -> str:
        suffix = f":band_{self.band}" if self.kind == "power" else ""
        return f"{self.kind}:{self.channel}{suffix}"

    @property
    def input_unit(self) -> str:
        if self.kind == "slope":
            return "rad"
        unit = (
            "N*m"
            if self.channel.endswith("effort")
            else "m/s^2"
            if "linear_acceleration" in self.channel
            else "rad/s"
        )
        return f"({unit})^2" if self.kind == "power" else unit


@dataclass(frozen=True)
class FeatureSettings:
    features: tuple[FeatureSpec, ...]
    window: str = "spectral"
    log_powers: bool = True
    exclude_transitions: bool = True
    transition_margin_s: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    features: FeatureSettings
    run_roles: dict[str, str]
    split_mode: str = "setup"
    linear_c: float = 1.0
    rbf_c: float = 1.0
    rbf_gamma: str | float = "scale"


@dataclass(frozen=True)
class PreparedData:
    matrix: np.ndarray  # Raw selected features, before training-only transforms.
    feature_names: tuple[str, ...]
    power_columns: tuple[int, ...]
    masks: dict[str, np.ndarray]
    transition_excluded: np.ndarray
    missing_features: np.ndarray
    assigned: dict[str, np.ndarray]


def available_features(dataset: TelemetryDataset) -> tuple[FeatureSpec, ...]:
    return (
        *(FeatureSpec("std", name) for name in MOTOR_CHANNELS),
        *(
            FeatureSpec("power", name, band)
            for name in BAND_CHANNELS
            for band in range(len(dataset.band_edges_hz))
        ),
        FeatureSpec("speed", "wheel_joint.velocity"),
        FeatureSpec("slope", "along"),
        FeatureSpec("slope", "cross"),
    )


def feature_matrix(dataset: TelemetryDataset, settings: FeatureSettings) -> np.ndarray:
    """Construct identical, ordered input columns for all three models."""
    if settings.window not in ("local", "spectral"):
        raise ConfigurationError("Choose a local or spectral telemetry window.")
    if not settings.features or len(set(settings.features)) != len(settings.features):
        raise ConfigurationError("Select at least one feature, without duplicates.")
    known = set(available_features(dataset))
    columns = []
    for feature in settings.features:
        if feature not in known:
            raise ConfigurationError(f"Unavailable feature: {feature.name}")
        if feature.kind == "std":
            values = dataset.standard_deviations[settings.window][
                :, MOTOR_CHANNELS.index(feature.channel)
            ]
        elif feature.kind == "power":
            values = dataset.band_powers[
                :, feature.band, BAND_CHANNELS.index(feature.channel)
            ]
        elif feature.kind == "speed":
            values = dataset.wheel_speeds[settings.window]
        else:
            values = dataset.slopes[:, 0 if feature.channel == "along" else 1]
        columns.append(values)
    return np.column_stack(columns)


def transition_exclusions(
    dataset: TelemetryDataset,
    sources: set[str],
    margin_s: float,
) -> np.ndarray:
    """Intersect closed FFT windows with annotated local transition intervals.

    Compare only within a run: separate recordings do not share a clock/angle
    origin. Unchecked transition classes still supply annotation context.
    """
    if not np.isfinite(margin_s) or not 0 <= margin_s <= 60:
        raise ConfigurationError("Transition padding must be between 0 and 60 seconds.")
    result = np.zeros(len(dataset.packet_ids), dtype=bool)
    padding_ns = round(margin_s * 1_000_000_000)
    source_ids = np.asarray(dataset.source_ids)
    for source in sources:
        same_run = source_ids == source
        transitions = same_run & np.isin(dataset.class_ids, [2, 3])
        candidates = same_run & np.isin(dataset.class_ids, [0, 1])
        if not np.any(transitions) or not np.any(candidates):
            continue
        if (
            dataset.fft_bounds_ns is None
            or dataset.local_bounds_ns is None
            or np.any(dataset.fft_bounds_ns[candidates] < 0)
            or np.any(dataset.local_bounds_ns[transitions] < 0)
        ):
            raise ConfigurationError(
                f"Transition timing is unavailable for {dataset.run_names[source]}. "
                "Use a complete export or explicitly disable transition exclusion."
            )
        windows = dataset.fft_bounds_ns[candidates]
        intervals = dataset.local_bounds_ns[transitions]
        overlaps = np.zeros(len(windows), dtype=bool)
        # Subtract Python integers rather than converting nanoseconds to floats.
        for start, end in intervals:
            overlaps |= (windows[:, 0] <= int(end) + padding_ns) & (
                windows[:, 1] >= int(start) - padding_ns
            )
        result[np.flatnonzero(candidates)] = overlaps
    return result


def prepare_data(dataset: TelemetryDataset, config: ExperimentConfig) -> PreparedData:
    """Validate whole-run assignments and report exclusions before fitting."""
    sources = set(dataset.source_ids)
    if set(config.run_roles) - sources or any(
        role not in ROLES for role in config.run_roles.values()
    ):
        raise ConfigurationError("The split contains unknown runs or partition names.")
    if config.split_mode not in ("run", "setup"):
        raise ConfigurationError("Choose new-run or new-setup evaluation.")
    if (
        not np.isfinite(config.linear_c)
        or config.linear_c <= 0
        or not np.isfinite(config.rbf_c)
        or config.rbf_c <= 0
    ):
        raise ConfigurationError("Both SVM C values must be positive and finite.")
    if config.rbf_gamma != "scale":
        try:
            gamma = float(config.rbf_gamma)
        except (ValueError, TypeError) as error:
            raise ConfigurationError(
                "RBF gamma must be Automatic or a positive number."
            ) from error
        if not np.isfinite(gamma) or gamma <= 0:
            raise ConfigurationError("RBF gamma must be positive and finite.")
    roles = np.asarray(
        [config.run_roles.get(source, "excluded") for source in dataset.source_ids]
    )
    if config.split_mode == "setup":
        for setup in {key[0] for key in dataset.terrain_keys}:
            members = np.asarray([key[0] == setup for key in dataset.terrain_keys])
            assigned_roles = set(roles[members]) - {"excluded"}
            if len(assigned_roles) > 1:
                raise ConfigurationError(
                    f"Physical setup {setup} appears in multiple partitions."
                )
            if assigned_roles and setup.startswith("(Unknown"):
                raise ConfigurationError(
                    "New-setup evaluation requires known physical terrain setup IDs."
                )
    matrix = feature_matrix(dataset, config.features)
    binary = np.isin(dataset.class_ids, [0, 1])
    missing = ~np.all(np.isfinite(matrix), axis=1)
    selected_sources = {
        source for source, role in config.run_roles.items() if role != "excluded"
    }
    transition = (
        transition_exclusions(
            dataset, selected_sources, config.features.transition_margin_s
        )
        if config.features.exclude_transitions
        else np.zeros(len(binary), dtype=bool)
    )
    assigned = {role: (roles == role) & binary for role in PARTITIONS}
    masks = {role: mask & ~missing & ~transition for role, mask in assigned.items()}
    return PreparedData(
        matrix,
        tuple(feature.name for feature in config.features.features),
        tuple(
            i
            for i, feature in enumerate(config.features.features)
            if feature.kind == "power"
        ),
        masks,
        transition,
        missing,
        assigned,
    )


def require_trainable(dataset: TelemetryDataset, prepared: PreparedData) -> None:
    # Both classes are necessary for fitting and for a meaningful ROC/TPR/FPR
    # comparison. Empty optional testing partitions are deliberately allowed.
    for role in ("training", "validation"):
        labels = dataset.class_ids[prepared.masks[role]]
        if set(labels) != {0, 1}:
            raise ConfigurationError(
                f"{role.title()} needs at least one eligible firm and one eligible loose packet."
            )
