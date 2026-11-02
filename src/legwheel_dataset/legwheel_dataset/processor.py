"""Stage 1 packet isolation and Stage 2 feature extraction."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from legwheel_encoder.config import (
    ENCODER_POSITION_VERSION,
    TICKS_TOPIC,
    EncoderCalibration,
)

from legwheel_dataset.bag_reader import (
    DEPTH_IMAGE_TOPIC,
    RGB_IMAGE_TOPIC,
    BagIndex,
    ImageMetadata,
    extract_selected_images,
    index_bag,
)
from legwheel_dataset.config import LEGACY_CONFIG_FIELDS, ProcessingConfig
from legwheel_dataset.features import (
    calculate_spectra,
    fit_plane_ransac,
    slope_from_plane,
)
from legwheel_dataset.images import (
    decode_image,
    depth_point_map_in_base,
    masked_crop,
    polygon_mask,
)
from legwheel_kinematics.imu_pitch import (
    PITCH_ESTIMATOR_VERSION,
    PitchTrajectory,
    estimate_pitch,
)
from legwheel_dataset.patch_geometry import (
    PatchProjection,
    project_full_track_boundaries,
    project_track_patch,
)
from legwheel_dataset.progress import ProgressCallback, progress_items, progress_range
from legwheel_dataset.run_analysis import (
    CorrectedThetaTrajectory,
    build_corrected_theta_trajectory,
    effective_rate_statistics,
    find_steady_torque_period,
    time_weighted_mean_absolute_speed,
)
from legwheel_dataset.synchronization import (
    ImagePair,
    interpolate_scalar,
    pair_image_timestamps,
    resample_channels,
)
from legwheel_dataset.terrain import require_provenance_terrain_id, require_run_terrain_id
from legwheel_kinematics.config import KINEMATICS_MODEL_VERSION, KinematicsConfig
from legwheel_kinematics.projection import CameraCalibration


# Only measurements acquired near 100 Hz belong on the common ML telemetry
# grid. Central-encoder ticks remain a motion reference for theta_y correction,
# but are deliberately excluded from packet arrays and spectral features.
RESAMPLED_TELEMETRY_STREAM_NAMES = ("leg", "wheel", "imu")
RESAMPLED_TELEMETRY_CHANNEL_COUNT = 15


@dataclass(frozen=True)
class ValidationCameraView:
    """Everything needed to plot one midpoint camera projection."""

    label: str
    image_topic: str
    metadata: ImageMetadata
    calibration: CameraCalibration
    theta_y_motion_rad: float
    theta_y_kinematic_rad: float
    knee_joint_rad: float | None
    theta_p_rad: float
    patch: PatchProjection
    inner_track_pixels: np.ndarray
    inner_track_valid: np.ndarray
    outer_track_pixels: np.ndarray
    outer_track_valid: np.ndarray
    visibility_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ValidationPreview:
    """Midpoint RGB/depth pair selected for interactive validation."""

    bag_directory: Path
    pair: ImagePair
    steady_midpoint_ns: int
    steady_start_ns: int
    steady_end_ns: int
    patch_centre_motion_angle_rad: float
    patch_centre_kinematic_angle_rad: float
    travel_direction_motion: float
    travel_direction_kinematic: float
    rgb: ValidationCameraView
    depth: ValidationCameraView


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return values


def _find_unique(run_directory: Path, pattern: str, description: str) -> Path:
    candidates = sorted(run_directory.glob(pattern))
    if len(candidates) != 1:
        raise ValueError(
            f"Run directory must contain exactly one {description}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def find_experiment_snapshot(run_directory: Path) -> Path:
    """Find the unique experiment snapshot saved by RunRecorder."""
    return _find_unique(
        run_directory,
        "experiment_config_run_*_snapshot.yaml",
        "experiment configuration snapshot",
    )


def _load_run_config(run_directory: Path) -> dict[str, Any]:
    return _load_yaml(run_directory / "run_config.yaml")


def _validate_registered_depth(run_directory: Path) -> dict[str, Any]:
    """Require the simple, calibrated color-aligned depth mode agreed for Stage 2."""
    camera_path = _find_unique(
        run_directory,
        "camera_driver_config_run_*_snapshot.yaml",
        "camera-driver configuration snapshot",
    )
    camera_config = _load_yaml(camera_path)
    if camera_config.get("depth_registration") is not True:
        raise ValueError("camera_depth_registration_must_be_true")
    if str(camera_config.get("align_target_stream", "")).upper() != "COLOR":
        raise ValueError("camera_align_target_stream_must_be_COLOR")
    return camera_config


def _dataset_identifier(
    experiment_config: dict[str, Any],
    run_config: dict[str, Any],
    processing_config: ProcessingConfig,
) -> str:
    payload = json.dumps(
        {
            "kinematics_model_version": KINEMATICS_MODEL_VERSION,
            "pitch_estimator_version": PITCH_ESTIMATOR_VERSION,
            "encoder_position_version": ENCODER_POSITION_VERSION,
            "experiment": experiment_config,
            "run": run_config,
            "processing": processing_config.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def dataset_output_directory(
    run_directory: Path, processing_config: ProcessingConfig,
) -> Path:
    """Resolve the normal output path without opening the bag or writing files.

    Batch processing uses this same identity to skip completed datasets and
    preserve incomplete attempts before retrying them.
    """
    run_directory = run_directory.expanduser().resolve()
    processing_config.validate()
    experiment_config = _load_yaml(find_experiment_snapshot(run_directory))
    require_run_terrain_id(run_directory, experiment_config)
    identifier = _dataset_identifier(
        experiment_config, _load_run_config(run_directory), processing_config,
    )
    return run_directory / "derived" / identifier


def _finite_column(
    times: np.ndarray,
    values: np.ndarray,
    column: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values[:, column])
    return times[finite], values[finite, column]


def _estimate_run_pitch(
    bag_index: BagIndex,
    experiment_config: dict[str, Any],
    geometry: KinematicsConfig,
    processing_config: ProcessingConfig,
    *,
    progress: ProgressCallback | None = None,
) -> PitchTrajectory:
    """Estimate beam pitch once, using the complete recorded IMU timeline.

    The stationary samples before/after wheel motion are useful for gyro bias
    estimation, so this must run before any steady-torque or packet filtering.
    Encoder ticks supply circular acceleration compensation; knee geometry is
    deliberately absent from the pitch estimator's inputs.
    """
    if any(
        frame_id and not frame_id.endswith("optical_frame")
        for frame_id in bag_index.imu_frame_ids
    ):
        raise ValueError(
            "Camera IMU pitch requires Gemini optical-axis measurements; "
            f"recorded frames: {sorted(bag_index.imu_frame_ids)}"
        )
    imu_times, imu_values = bag_index.streams["imu"].arrays()
    encoder_times, encoder_motion_angles, _ = _recorded_encoder_motion(
        bag_index, experiment_config
    )
    # Tick direction belongs to the hardware/motion convention. Compensation
    # needs the signed yaw used by the transform chain, including its -1 map.
    encoder_angles = encoder_motion_angles * geometry.theta_y_kinematic_sign
    pitch = estimate_pitch(
        imu_times,
        imu_values[:, :3],
        imu_values[:, 3:6],
        encoder_times,
        encoder_angles,
        geometry,
        maximum_gap_s=processing_config.maximum_interpolation_gap_ms / 1000.0,
        maximum_uncertainty_deg=processing_config.maximum_pitch_uncertainty_deg,
        progress=progress,
    )
    pitch.report["recorded_imu_frame_ids"] = sorted(bag_index.imu_frame_ids)
    return pitch


def _optional_knee_angle(
    times_ns: np.ndarray,
    values_rad: np.ndarray,
    timestamp_ns: int,
) -> float | None:
    """Keep knee angle as diagnostic metadata without using it as a pose input."""
    if (
        times_ns.size < 2
        or timestamp_ns < times_ns[0]
        or timestamp_ns > times_ns[-1]
    ):
        return None
    return interpolate_scalar(times_ns, values_rad, timestamp_ns)


def _packet_imu_uncertainty_deg(metadata: dict[str, Any]) -> float | None:
    """Return the packet's worst exposure-time two-sigma pitch bound in degrees.

    Older IMU packets already saved separate one-sigma RGB/depth values. Those
    are enough to populate the new feature without treating unknown uncertainty
    as zero. This value is descriptive and never controls packet acceptance.
    """
    try:
        uncertainty = metadata.get("imu_uncertainty_deg")
        if uncertainty is None:
            exposure_uncertainties = (
                float(metadata["theta_p_rgb_uncertainty_rad"]),
                float(metadata["theta_p_depth_uncertainty_rad"]),
            )
            if any(not math.isfinite(value) or value < 0.0
                   for value in exposure_uncertainties):
                return None
            uncertainty = math.degrees(2.0 * max(exposure_uncertainties))
        uncertainty = float(uncertainty)
    except (KeyError, TypeError, ValueError):
        return None
    return uncertainty if math.isfinite(uncertainty) and uncertainty >= 0.0 else None


def _resample_all_streams(
    bag_index: BagIndex,
    centre_time_ns: int,
    start_time_ns: int,
    end_time_ns: int,
    config: ProcessingConfig,
    valid_start_time_ns: int,
    valid_end_time_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Resample the 15 ML channels without adding encoder or ramp data."""
    sampled_streams = []
    sampled_masks = []
    channel_names = []
    relative_times_ns = None
    for stream_name in RESAMPLED_TELEMETRY_STREAM_NAMES:
        stream = bag_index.streams[stream_name]
        source_times_ns, source_values = stream.arrays()
        inside = (
            (source_times_ns >= valid_start_time_ns)
            & (source_times_ns <= valid_end_time_ns)
        )
        source_times_ns = source_times_ns[inside]
        source_values = source_values[inside]
        if source_times_ns.size < 2:
            raise ValueError(f"telemetry_stream_{stream_name}_has_too_few_samples")
        values, valid, stream_relative_times_ns = resample_channels(
            source_times_ns,
            source_values,
            centre_time_ns,
            start_time_ns,
            end_time_ns,
            config.telemetry_resample_hz,
            config.maximum_interpolation_gap_ms,
        )
        if relative_times_ns is None:
            relative_times_ns = stream_relative_times_ns
        elif not np.array_equal(relative_times_ns, stream_relative_times_ns):
            raise RuntimeError("Telemetry streams produced different sample grids.")
        sampled_streams.append(values)
        sampled_masks.append(valid)
        channel_names.extend(stream.channel_names)
    if len(channel_names) != RESAMPLED_TELEMETRY_CHANNEL_COUNT:
        raise RuntimeError(
            f"Expected {RESAMPLED_TELEMETRY_CHANNEL_COUNT} telemetry channels, "
            f"received {len(channel_names)}."
        )
    return (
        np.column_stack(sampled_streams),
        np.column_stack(sampled_masks),
        relative_times_ns,
        channel_names,
    )


def _window_has_complete_timestamp_coverage(
    bag_index: BagIndex,
    start_time_ns: int,
    end_time_ns: int,
    maximum_gap_ns: int,
    valid_start_time_ns: int,
    valid_end_time_ns: int,
) -> bool:
    """Check local-window timestamp coverage before doing any interpolation."""
    for stream_name in RESAMPLED_TELEMETRY_STREAM_NAMES:
        times_ns, _ = bag_index.streams[stream_name].arrays()
        times_ns = times_ns[
            (times_ns >= valid_start_time_ns) & (times_ns <= valid_end_time_ns)
        ]
        if (
            times_ns.size < 2
            or start_time_ns < times_ns[0]
            or end_time_ns > times_ns[-1]
        ):
            return False
        left = max(0, int(np.searchsorted(times_ns, start_time_ns)) - 1)
        right = min(times_ns.size, int(np.searchsorted(times_ns, end_time_ns)) + 1)
        if np.any(np.diff(times_ns[left:right]) > maximum_gap_ns):
            return False
    return True


def _normalise_rgb(image: np.ndarray, encoding: str) -> tuple[np.ndarray, str]:
    if encoding == "bgr8":
        return image[:, :, ::-1], "rgb8"
    if encoding == "bgra8":
        return image[:, :, [2, 1, 0, 3]], "rgba8"
    return image, encoding


def _image_writer(path: Path, kind: str):
    def write(message) -> None:
        image = decode_image(message)
        if kind == "rgb":
            image, _ = _normalise_rgb(image, str(message.encoding))
        np.save(path, image, allow_pickle=False)

    return write


def _camera_inputs(
    image_metadata,
    calibrations,
    timestamp_ns: int,
) -> CameraCalibration:
    calibration = calibrations.at(timestamp_ns)
    camera_info_frame = calibrations.frame_id_at(timestamp_ns)
    if image_metadata.frame_id and camera_info_frame:
        if image_metadata.frame_id != camera_info_frame:
            raise ValueError("image_and_camera_info_frame_ids_differ")
    if (
        calibration.width != image_metadata.width
        or calibration.height != image_metadata.height
    ):
        raise ValueError("image_and_camera_info_dimensions_differ")
    return calibration


def _safe_identifier(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(value))
    return cleaned or "unknown"


def _run_number(run_directory: Path) -> str:
    match = re.search(r"run_(\d+)$", run_directory.name)
    return match.group(1) if match else _safe_identifier(run_directory.name)


def _theta_filename_value(theta_y_rad: float) -> str:
    rendered = f"{abs(theta_y_rad):.3f}".replace(".", "p")
    return ("m" if theta_y_rad < 0.0 else "") + rendered


def _packet_identifier(
    experiment_id: str,
    run_number: str,
    theta_y_rad: float,
    rgb_time_ns: int,
    depth_time_ns: int,
    dataset_identifier: str,
) -> str:
    hash_payload = (
        f"{experiment_id}|{run_number}|{rgb_time_ns}|{depth_time_ns}|"
        f"{dataset_identifier}"
    ).encode("utf-8")
    packet_hash = hashlib.sha256(hash_payload).hexdigest()[:8]
    return (
        f"e{_safe_identifier(experiment_id)}_r{run_number}_"
        f"theta_{_theta_filename_value(theta_y_rad)}_{packet_hash}"
    )


def _recorded_encoder_motion(bag_index, experiment_config):
    """Use the first finite count in the whole recording as the shared zero."""
    calibration = EncoderCalibration.from_experiment_dict(experiment_config)
    times, values = bag_index.streams["ticks"].arrays()
    times, ticks = _finite_column(times, values, 0)
    if times.size < 2:
        raise ValueError("too_few_rotation_encoder_samples")
    origin = {
        "topic": TICKS_TOPIC,
        "timestamp_source": "bag_receipt_time",
        "origin": "first_recorded_tick",
        "origin_time_ns": int(times[0]),
        "origin_ticks": float(ticks[0]),
        "ticks_per_legwheel_revolution": calibration.ticks_per_legwheel_revolution,
    }
    return times, calibration.motion_angle(ticks, ticks[0]), origin


def _describe_wheel_control(
    bag_index: BagIndex, wheel_settings: dict[str, Any],
    start_time_ns: int, end_time_ns: int,
) -> dict[str, Any]:
    """Describe the selected mode without mistaking requested torque for output."""
    wheel_control_report = {
        "control_mode": wheel_settings.get("control_mode", "raw_torque"),
        "max_wheel_speed_rad_s": wheel_settings.get("max_wheel_speed_rad_s"),
        "velocity_governor_kp": wheel_settings.get("velocity_governor_kp"),
        "velocity_governor_ki": wheel_settings.get("velocity_governor_ki"),
        "interval_source": "requested_torque_plateau",
        "interval_meaning": "requested drive phase; motor torque may vary under the governor",
        "active_braking_enabled": False,
        "governed_command_available": "governed_torque_command" in bag_index.streams,
    }
    governed_stream = bag_index.streams.get("governed_torque_command")
    if governed_stream is not None:
        command_times, command_values = governed_stream.arrays()
        during_run = (
            (command_times >= start_time_ns)
            & (command_times <= end_time_ns)
        )
        commands_nm = command_values[during_run, 0]
        commands_nm = commands_nm[np.isfinite(commands_nm)]
        if commands_nm.size:
            wheel_control_report["governed_command"] = {
                "sample_count": int(commands_nm.size),
                "minimum_nm": float(np.min(commands_nm)),
                "maximum_nm": float(np.max(commands_nm)),
                "mean_absolute_nm": float(np.mean(np.abs(commands_nm))),
            }
    return wheel_control_report


def _run_motion_analysis(
    bag_index: BagIndex,
    experiment_config: dict[str, Any],
    run_config: dict[str, Any],
    kinematics_config: KinematicsConfig,
    processing_config: ProcessingConfig,
) -> tuple[CorrectedThetaTrajectory, dict[str, Any]]:
    """Calculate the accepted run, corrected theta_y and acquisition rates."""
    try:
        target_torque_nm = float(
            run_config["wheel_parameters"]["commanded_wheel_torque_nm"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run_config_missing_commanded_wheel_torque_nm") from error

    torque_times, torque_values = bag_index.streams["torque_command"].arrays()
    torque_period = find_steady_torque_period(
        torque_times,
        torque_values[:, 0],
        target_torque_nm,
        processing_config.minimum_target_torque_fraction,
    )

    encoder_times, encoder_angles, encoder_origin = _recorded_encoder_motion(
        bag_index, experiment_config
    )
    wheel_times, wheel_values = bag_index.streams["wheel"].arrays()
    wheel_velocity_column = bag_index.streams["wheel"].channel_names.index(
        "wheel_joint.velocity"
    )
    wheel_times, wheel_velocity = _finite_column(
        wheel_times,
        wheel_values,
        wheel_velocity_column,
    )

    track_radius_m = kinematics_config.track_center_radius_m
    trajectory = build_corrected_theta_trajectory(
        encoder_times,
        encoder_angles,
        wheel_times,
        wheel_velocity,
        torque_period,
        kinematics_config.wheel_radius_mm / 1000.0,
        processing_config.grouser_effective_height_m,
        track_radius_m,
        processing_config.maximum_interpolation_gap_ms,
    )
    theta_y_end_motion_rad = float(trajectory.theta_y_at_ticks_rad[-1])
    theta_y_start_motion_rad = float(trajectory.theta_y_at_ticks_rad[0])
    angular_displacement_rad = theta_y_end_motion_rad - theta_y_start_motion_rad
    theta_y_end_kinematic_rad = kinematics_config.theta_y_to_kinematic(
        theta_y_end_motion_rad
    )
    travel_direction_kinematic = kinematics_config.direction_to_kinematic(
        trajectory.travel_direction
    )
    forward_recording = kinematics_config.is_forward_motion(
        trajectory.travel_direction
    )

    omega_avg_rad_s = time_weighted_mean_absolute_speed(
        trajectory.wheel_times_ns,
        trajectory.wheel_velocity_rad_s,
        trajectory.start_time_ns,
        trajectory.end_time_ns,
        trajectory.maximum_gap_ns,
    )
    duration_s = (trajectory.end_time_ns - trajectory.start_time_ns) * 1e-9
    average_rig_angular_speed_rad_s = abs(angular_displacement_rad / duration_s)
    if average_rig_angular_speed_rad_s <= 0.0:
        raise ValueError("average_corrected_rig_speed_is_zero")

    source_rates = {}
    rate_limit_violations = []
    for stream_name in RESAMPLED_TELEMETRY_STREAM_NAMES:
        stream_times, _ = bag_index.streams[stream_name].arrays()
        statistics = effective_rate_statistics(
            stream_times,
            trajectory.start_time_ns,
            trajectory.end_time_ns,
        )
        source_rates[stream_name] = statistics
        median_rate = statistics["median_rate_hz"]
        if median_rate is None or processing_config.telemetry_resample_hz > (
            float(median_rate)
            * (1.0 + processing_config.source_rate_tolerance_fraction)
        ):
            rate_limit_violations.append(stream_name)

    steady_wheel = trajectory.wheel_velocity_rad_s[
        (trajectory.wheel_times_ns >= trajectory.start_time_ns)
        & (trajectory.wheel_times_ns <= trajectory.end_time_ns)
    ]
    wheel_control_report = _describe_wheel_control(
        bag_index, run_config["wheel_parameters"],
        trajectory.start_time_ns, trajectory.end_time_ns,
    )
    report = {
        "status": "analysed",
        "encoder_position_version": ENCODER_POSITION_VERSION,
        "encoder_position": encoder_origin,
        "wheel_control": wheel_control_report,
        "torque_period": {
            "threshold_start_time_ns": torque_period.start_time_ns,
            "threshold_end_time_ns": torque_period.end_time_ns,
            "configured_target_nm": torque_period.configured_target_nm,
            "minimum_target_fraction": (
                processing_config.minimum_target_torque_fraction
            ),
            "acceptance_threshold_nm": torque_period.acceptance_threshold_nm,
            "held_command_at_start_nm": torque_period.held_command_at_start_nm,
        },
        "corrected_motion": {
            "t0_ns": trajectory.start_time_ns,
            "end_time_ns": trajectory.end_time_ns,
            "encoder_angle_at_t0_rad": float(
                trajectory.raw_encoder_angles_rad[0]
            ),
            # The unqualified legacy fields remain encoder-motion quantities.
            # Explicit fields prevent motion timing from being confused with
            # the opposite-handed notebook/base geometry.
            "theta_y_end_rad": theta_y_end_motion_rad,
            "theta_y_start_motion_rad": theta_y_start_motion_rad,
            "angular_displacement_motion_rad": angular_displacement_rad,
            "theta_y_end_motion_rad": theta_y_end_motion_rad,
            "theta_y_end_kinematic_rad": theta_y_end_kinematic_rad,
            "travel_direction": trajectory.travel_direction,
            "travel_direction_motion": trajectory.travel_direction,
            "travel_direction_kinematic": travel_direction_kinematic,
            "theta_y_kinematic_sign": (
                kinematics_config.theta_y_kinematic_sign
            ),
            "forward_recording": forward_recording,
            "average_angular_speed_rad_s": average_rig_angular_speed_rad_s,
            "complete_encoder_interval_count": int(
                trajectory.correction_factors.size
            ),
            "rejected_reverse_transition_count": (
                trajectory.rejected_reverse_transition_count
            ),
            "track_radius_m": track_radius_m,
            "wheel_to_track_radius_ratio": trajectory.radius_ratio,
            "correction_factor_min": float(np.min(trajectory.correction_factors)),
            "correction_factor_max": float(np.max(trajectory.correction_factors)),
            "correction_factor_mean": float(np.mean(trajectory.correction_factors)),
        },
        "wheel_speed": {
            "time_weighted_mean_absolute_rad_s": omega_avg_rad_s,
            "minimum_absolute_rad_s": float(np.min(np.abs(steady_wheel))),
            "maximum_absolute_rad_s": float(np.max(np.abs(steady_wheel))),
            "median_absolute_rad_s": float(np.median(np.abs(steady_wheel))),
            "standard_deviation_absolute_rad_s": float(
                np.std(np.abs(steady_wheel))
            ),
            "sample_count": int(steady_wheel.size),
        },
        "local_window": {
            "derivation": "corrected_theta_y_patch_boundary_crossings",
            "alpha_sp_rad": processing_config.alpha_sp_rad,
        },
        "measured_effective_rates": source_rates,
        "motion_reference_effective_rate": {
            "stream": "rotation_encoder_ticks",
            **effective_rate_statistics(
                encoder_times,
                trajectory.start_time_ns,
                trajectory.end_time_ns,
            ),
        },
        "resampled_telemetry_streams": list(
            RESAMPLED_TELEMETRY_STREAM_NAMES
        ),
        "resampled_telemetry_channel_count": (
            RESAMPLED_TELEMETRY_CHANNEL_COUNT
        ),
        "excluded_from_packet_telemetry": [
            "rotation_joint.position",
            "rotation_joint.velocity",
            "rotation_encoder.ticks",
        ],
        "requested_resample_rate_hz": processing_config.telemetry_resample_hz,
        "source_rate_tolerance_fraction": (
            processing_config.source_rate_tolerance_fraction
        ),
        "rate_limit_violations": rate_limit_violations,
        "experiment_encoder_setup": experiment_config.get(
            "electronics_hardware", {}
        ).get("encoder_setup", {}),
    }
    return trajectory, report


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _patch_visibility_reasons(
    projection: PatchProjection,
    calibration: CameraCalibration,
) -> tuple[str, ...]:
    """Explain why a projected patch is not completely inside an image."""
    if projection.fully_visible:
        return ("complete_patch_visible",)

    pixels = np.vstack((projection.inner_pixels, projection.outer_pixels))
    in_front = np.concatenate((
        projection.inner_in_front,
        projection.outer_in_front,
    ))
    reasons = []
    behind_count = int(np.count_nonzero(~in_front))
    if behind_count:
        reasons.append(f"{behind_count}_patch_points_behind_camera")

    finite = np.all(np.isfinite(pixels), axis=1)
    nonfinite_count = int(np.count_nonzero(~finite))
    if nonfinite_count:
        reasons.append(f"{nonfinite_count}_patch_points_nonfinite")
    if np.any(finite):
        finite_pixels = pixels[finite]
        for name, count in (
            ("left_of_image", np.count_nonzero(finite_pixels[:, 0] < 0.0)),
            (
                "right_of_image",
                np.count_nonzero(finite_pixels[:, 0] >= calibration.width),
            ),
            ("above_image", np.count_nonzero(finite_pixels[:, 1] < 0.0)),
            (
                "below_image",
                np.count_nonzero(finite_pixels[:, 1] >= calibration.height),
            ),
        ):
            if count:
                reasons.append(f"{int(count)}_patch_points_{name}")
    return tuple(reasons or ["patch_projection_invalid"])


def _camera_validation_view(
    label: str,
    image_topic: str,
    metadata: ImageMetadata,
    calibration: CameraCalibration,
    kinematics_config: KinematicsConfig,
    theta_y_motion_rad: float,
    knee_joint_rad: float | None,
    theta_p_rad: float,
    patch_centre_motion_angle_rad: float,
    processing_config: ProcessingConfig,
) -> ValidationCameraView:
    """Project one image-time patch after converting motion into geometry."""
    theta_y_kinematic_rad = kinematics_config.theta_y_to_kinematic(
        theta_y_motion_rad
    )
    patch_centre_kinematic_angle_rad = (
        kinematics_config.theta_y_to_kinematic(
            patch_centre_motion_angle_rad
        )
    )
    patch = project_track_patch(
        kinematics_config,
        theta_y_kinematic_rad,
        theta_p_rad,
        patch_centre_kinematic_angle_rad,
        processing_config.alpha_sp_rad,
        calibration,
        processing_config.track_point_count,
    )
    (
        inner_track_pixels,
        inner_track_valid,
        outer_track_pixels,
        outer_track_valid,
    ) = project_full_track_boundaries(
        kinematics_config,
        theta_y_kinematic_rad,
        theta_p_rad,
        calibration,
        processing_config.track_point_count,
    )
    return ValidationCameraView(
        label=label,
        image_topic=image_topic,
        metadata=metadata,
        calibration=calibration,
        theta_y_motion_rad=theta_y_motion_rad,
        theta_y_kinematic_rad=theta_y_kinematic_rad,
        knee_joint_rad=knee_joint_rad,
        theta_p_rad=theta_p_rad,
        patch=patch,
        inner_track_pixels=inner_track_pixels,
        inner_track_valid=inner_track_valid,
        outer_track_pixels=outer_track_pixels,
        outer_track_valid=outer_track_valid,
        visibility_reasons=_patch_visibility_reasons(patch, calibration),
    )


def _build_validation_preview(
    run_directory: Path,
    pairs: list[ImagePair],
    bag_index: BagIndex,
    trajectory: CorrectedThetaTrajectory,
    pitch_trajectory: PitchTrajectory,
    knee_times: np.ndarray,
    knee_values: np.ndarray,
    kinematics_config: KinematicsConfig,
    processing_config: ProcessingConfig,
) -> tuple[ValidationPreview | None, dict[str, Any]]:
    """Select and project the pair nearest the steady-motion midpoint."""
    steady_midpoint_ns = (
        trajectory.start_time_ns + trajectory.end_time_ns
    ) // 2
    pair = _select_midpoint_pair(
        pairs,
        trajectory.start_time_ns,
        trajectory.end_time_ns,
    )
    if pair is None:
        return None, {
            "available": False,
            "reason": "no_synchronized_pair_inside_steady_motion_interval",
            "steady_midpoint_ns": steady_midpoint_ns,
        }

    try:
        theta_y_image_motion_rad = trajectory.theta_y_at(pair.image_time_ns)
        patch_centre_motion_angle_rad = (
            theta_y_image_motion_rad
            + trajectory.travel_direction * processing_config.alpha_off_rad
        )
        rgb_metadata = bag_index.rgb_images[pair.rgb_index]
        depth_metadata = bag_index.depth_images[pair.depth_index]
        rgb_calibration = _camera_inputs(
            rgb_metadata,
            bag_index.rgb_calibrations,
            pair.rgb_time_ns,
        )
        depth_calibration = _camera_inputs(
            depth_metadata,
            bag_index.depth_calibrations,
            pair.depth_time_ns,
        )
        rgb_view = _camera_validation_view(
            "RGB",
            RGB_IMAGE_TOPIC,
            rgb_metadata,
            rgb_calibration,
            kinematics_config,
            trajectory.theta_y_at(pair.rgb_time_ns),
            _optional_knee_angle(knee_times, knee_values, pair.rgb_time_ns),
            pitch_trajectory.sample(pair.rgb_time_ns),
            patch_centre_motion_angle_rad,
            processing_config,
        )
        depth_view = _camera_validation_view(
            "Depth",
            DEPTH_IMAGE_TOPIC,
            depth_metadata,
            depth_calibration,
            kinematics_config,
            trajectory.theta_y_at(pair.depth_time_ns),
            _optional_knee_angle(knee_times, knee_values, pair.depth_time_ns),
            pitch_trajectory.sample(pair.depth_time_ns),
            patch_centre_motion_angle_rad,
            processing_config,
        )
    except (IndexError, ValueError) as error:
        return None, {
            "available": False,
            "reason": f"midpoint_pair_projection_failed: {error}",
            "steady_midpoint_ns": steady_midpoint_ns,
            "selected_image_time_ns": pair.image_time_ns,
        }

    preview = ValidationPreview(
        bag_directory=run_directory / "rosbag",
        pair=pair,
        steady_midpoint_ns=steady_midpoint_ns,
        steady_start_ns=trajectory.start_time_ns,
        steady_end_ns=trajectory.end_time_ns,
        patch_centre_motion_angle_rad=patch_centre_motion_angle_rad,
        patch_centre_kinematic_angle_rad=(
            kinematics_config.theta_y_to_kinematic(
                patch_centre_motion_angle_rad
            )
        ),
        travel_direction_motion=trajectory.travel_direction,
        travel_direction_kinematic=(
            kinematics_config.direction_to_kinematic(
                trajectory.travel_direction
            )
        ),
        rgb=rgb_view,
        depth=depth_view,
    )
    summary = {
        "available": True,
        "selection": "synchronized_pair_closest_to_steady_motion_midpoint",
        "steady_midpoint_ns": steady_midpoint_ns,
        "selected_image_time_ns": pair.image_time_ns,
        "midpoint_offset_ms": (
            (pair.image_time_ns - steady_midpoint_ns) / 1_000_000.0
        ),
        "rgb_time_ns": pair.rgb_time_ns,
        "depth_time_ns": pair.depth_time_ns,
        "sync_error_ms": pair.sync_error_ns / 1_000_000.0,
        "theta_y_kinematic_sign": kinematics_config.theta_y_kinematic_sign,
        "travel_direction_motion": trajectory.travel_direction,
        "travel_direction_kinematic": (
            kinematics_config.direction_to_kinematic(
                trajectory.travel_direction
            )
        ),
        "patch_centre_motion_angle_rad": patch_centre_motion_angle_rad,
        "patch_centre_kinematic_angle_rad": (
            kinematics_config.theta_y_to_kinematic(
                patch_centre_motion_angle_rad
            )
        ),
        "rgb": {
            "resolution_pixels": [
                rgb_metadata.width,
                rgb_metadata.height,
            ],
            "theta_y_motion_rad": rgb_view.theta_y_motion_rad,
            "theta_y_kinematic_rad": rgb_view.theta_y_kinematic_rad,
            "knee_joint_rad": rgb_view.knee_joint_rad,
            "theta_p_rad": rgb_view.theta_p_rad,
            "complete_patch_visible": rgb_view.patch.fully_visible,
            "visibility_reasons": list(rgb_view.visibility_reasons),
        },
        "depth": {
            "resolution_pixels": [
                depth_metadata.width,
                depth_metadata.height,
            ],
            "theta_y_motion_rad": depth_view.theta_y_motion_rad,
            "theta_y_kinematic_rad": depth_view.theta_y_kinematic_rad,
            "knee_joint_rad": depth_view.knee_joint_rad,
            "theta_p_rad": depth_view.theta_p_rad,
            "complete_patch_visible": depth_view.patch.fully_visible,
            "visibility_reasons": list(depth_view.visibility_reasons),
        },
    }
    return preview, summary


def _select_midpoint_pair(
    pairs: list[ImagePair],
    steady_start_ns: int,
    steady_end_ns: int,
) -> ImagePair | None:
    """Return the eligible pair closest to the steady interval midpoint."""
    steady_midpoint_ns = (steady_start_ns + steady_end_ns) // 2
    eligible_pairs = [
        pair for pair in pairs
        if steady_start_ns <= pair.image_time_ns <= steady_end_ns
    ]
    if not eligible_pairs:
        return None
    return min(
        eligible_pairs,
        key=lambda candidate: (
            abs(candidate.image_time_ns - steady_midpoint_ns),
            candidate.image_time_ns,
        ),
    )


def _analyse_processing_configuration(
    run_directory: Path,
    processing_config: ProcessingConfig,
    experiment_config_path: Path | None = None,
) -> tuple[dict[str, Any], ValidationPreview | None]:
    """Analyse a run and retain one midpoint pair for optional plotting."""
    run_directory = run_directory.resolve()
    processing_config.validate()
    if experiment_config_path is None:
        experiment_config_path = find_experiment_snapshot(run_directory)
    experiment_config = _load_yaml(experiment_config_path.resolve())
    run_config = _load_run_config(run_directory)
    _validate_registered_depth(run_directory)
    kinematics_config = KinematicsConfig.from_experiment_dict(experiment_config)
    bag_index = index_bag(run_directory / "rosbag")
    trajectory, report = _run_motion_analysis(
        bag_index,
        experiment_config,
        run_config,
        kinematics_config,
        processing_config,
    )
    pitch_trajectory = _estimate_run_pitch(
        bag_index, experiment_config, kinematics_config, processing_config
    )
    report["pitch_estimation"] = pitch_trajectory.report

    spectral_sample_count = max(
        2,
        int(round(
            processing_config.spectral_window_ms
            * processing_config.telemetry_resample_hz
            / 1000.0
        )) + 1,
    )
    report["spectral_analysis"] = {
        "spectral_window_ms": processing_config.spectral_window_ms,
        "sample_count": spectral_sample_count,
        "nyquist_frequency_hz": processing_config.telemetry_resample_hz / 2.0,
        "fft_bin_spacing_hz": (
            processing_config.telemetry_resample_hz / spectral_sample_count
        ),
    }

    rgb_times = np.asarray(
        [image.timestamp_ns for image in bag_index.rgb_images], dtype=np.int64
    )
    depth_times = np.asarray(
        [image.timestamp_ns for image in bag_index.depth_images], dtype=np.int64
    )
    pairs = pair_image_timestamps(
        rgb_times,
        depth_times,
        int(round(processing_config.rgb_depth_max_delta_ms * 1_000_000.0)),
    )
    leg_times, leg_values = bag_index.streams["leg"].arrays()
    knee_column = bag_index.streams["leg"].channel_names.index(
        "knee_joint.position"
    )
    knee_times, knee_values = _finite_column(
        leg_times, leg_values, knee_column
    )
    forward_recording = bool(
        report["corrected_motion"]["forward_recording"]
    )
    if forward_recording:
        preview, preview_summary = _build_validation_preview(
            run_directory,
            pairs,
            bag_index,
            trajectory,
            pitch_trajectory,
            knee_times,
            knee_values,
            kinematics_config,
            processing_config,
        )
    else:
        preview = None
        preview_summary = {
            "available": False,
            "reason": "reverse_recording_rejected",
            "steady_midpoint_ns": (
                trajectory.start_time_ns + trajectory.end_time_ns
            ) // 2,
        }
    report["visualization_preview"] = preview_summary
    depth_pixel_counts = []
    validation_pairs = pairs if forward_recording else []
    for pair in validation_pairs:
        if not trajectory.start_time_ns <= pair.image_time_ns <= trajectory.end_time_ns:
            continue
        try:
            theta_y_image_motion_rad = trajectory.theta_y_at(
                pair.image_time_ns
            )
            theta_y_depth_motion_rad = trajectory.theta_y_at(
                pair.depth_time_ns
            )
            patch_centre_motion_angle_rad = (
                theta_y_image_motion_rad
                + trajectory.travel_direction * processing_config.alpha_off_rad
            )
            depth_metadata = bag_index.depth_images[pair.depth_index]
            calibration = _camera_inputs(
                depth_metadata,
                bag_index.depth_calibrations,
                pair.depth_time_ns,
            )
            projection = project_track_patch(
                kinematics_config,
                kinematics_config.theta_y_to_kinematic(
                    theta_y_depth_motion_rad
                ),
                pitch_trajectory.sample(pair.depth_time_ns),
                kinematics_config.theta_y_to_kinematic(
                    patch_centre_motion_angle_rad
                ),
                processing_config.alpha_sp_rad,
                calibration,
                processing_config.track_point_count,
            )
            if not projection.fully_visible:
                continue
            mask = polygon_mask(
                depth_metadata.height,
                depth_metadata.width,
                projection.polygon_pixels,
            )
            depth_pixel_counts.append(int(np.count_nonzero(mask)))
            if len(depth_pixel_counts) >= 20:
                break
        except ValueError:
            continue
    report["estimated_depth_patch_pixels"] = {
        "sampled_fully_visible_packet_count": len(depth_pixel_counts),
        "minimum": min(depth_pixel_counts) if depth_pixel_counts else None,
        "median": (
            float(np.median(depth_pixel_counts)) if depth_pixel_counts else None
        ),
        "maximum": max(depth_pixel_counts) if depth_pixel_counts else None,
    }
    report["configuration_valid"] = bool(
        forward_recording
        and not report["rate_limit_violations"]
        and depth_pixel_counts
    )
    return report, preview


def validate_processing_configuration(
    run_directory: Path,
    processing_config: ProcessingConfig,
    experiment_config_path: Path | None = None,
) -> dict[str, Any]:
    """Analyse one run without retaining optional visualization data."""
    report, _ = _analyse_processing_configuration(
        run_directory,
        processing_config,
        experiment_config_path,
    )
    return report


def validate_processing_configuration_with_preview(
    run_directory: Path,
    processing_config: ProcessingConfig,
    experiment_config_path: Path | None = None,
) -> tuple[dict[str, Any], ValidationPreview | None]:
    """Analyse one run and return its midpoint visualization inputs."""
    return _analyse_processing_configuration(
        run_directory,
        processing_config,
        experiment_config_path,
    )


def isolate_packets(
    run_directory: Path,
    processing_config: ProcessingConfig,
    experiment_config_path: Path | None = None,
    output_directory: Path | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Convert synchronized MCAP data into self-contained Stage 1 packets."""
    run_directory = run_directory.resolve()
    processing_config.validate()
    if experiment_config_path is None:
        experiment_config_path = find_experiment_snapshot(run_directory)
    experiment_config_path = experiment_config_path.resolve()
    experiment_config = _load_yaml(experiment_config_path)
    terrain_id = require_run_terrain_id(run_directory, experiment_config)
    run_config = _load_run_config(run_directory)
    camera_driver_config = _validate_registered_depth(run_directory)
    kinematics_config = KinematicsConfig.from_experiment_dict(experiment_config)
    bag_directory = run_directory / "rosbag"
    if not bag_directory.is_dir():
        raise FileNotFoundError(f"Rosbag directory does not exist: {bag_directory}")

    dataset_identifier = _dataset_identifier(
        experiment_config,
        run_config,
        processing_config,
    )
    if output_directory is None:
        output_directory = run_directory / "derived" / dataset_identifier
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(
            "Derived output already exists; raw results are never overwritten: "
            f"{output_directory}"
        )
    packets_directory = output_directory / "packets"
    packets_directory.mkdir(parents=True)

    bag_index = index_bag(
        bag_directory, progress=progress_range(progress, 0.0, 0.20),
    )
    if progress:
        progress(0.20, "Analysing run motion")
    trajectory, analysis_report = _run_motion_analysis(
        bag_index,
        experiment_config,
        run_config,
        kinematics_config,
        processing_config,
    )
    analysis_report.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_identifier": dataset_identifier,
            "rejected_candidates": [],
            "rejection_counts": {},
        }
    )
    analysis_report_path = packets_directory / "analysis_report.json"
    _write_json(analysis_report_path, analysis_report)

    if not analysis_report["corrected_motion"]["forward_recording"]:
        analysis_report["status"] = "failed_reverse_recording"
        _write_json(analysis_report_path, analysis_report)
        raise ValueError(
            "Reverse recording rejected: ML packets require motion in the "
            "direction faced by the camera. See "
            f"{analysis_report_path}"
        )
    if analysis_report["rate_limit_violations"]:
        analysis_report["status"] = "failed_source_rate_validation"
        _write_json(analysis_report_path, analysis_report)
        names = ", ".join(analysis_report["rate_limit_violations"])
        raise ValueError(
            "telemetry_resample_hz exceeds the measured effective source "
            f"rate (including tolerance) for: {names}. See {analysis_report_path}"
        )
    try:
        pitch_trajectory = _estimate_run_pitch(
            bag_index, experiment_config, kinematics_config, processing_config,
            progress=progress_range(progress, 0.25, 0.60),
        )
    except ValueError as error:
        analysis_report["status"] = "failed_pitch_estimation"
        analysis_report["pitch_estimation_error"] = str(error)
        _write_json(analysis_report_path, analysis_report)
        raise
    analysis_report["pitch_estimation"] = pitch_trajectory.report
    _write_json(analysis_report_path, analysis_report)

    # Keep the full estimate for inspection, including rejected impact samples.
    # Packet metadata below stores only the image-time estimates actually used.
    np.savez_compressed(
        packets_directory / "run_pitch_estimation.npz",
        times_ns=pitch_trajectory.times_ns,
        theta_p_rad=pitch_trajectory.theta_p_rad,
        uncertainty_rad=pitch_trajectory.uncertainty_rad,
        available=pitch_trajectory.available,
        valid=pitch_trajectory.valid,
        acceleration_rejected=pitch_trajectory.acceleration_rejected,
    )

    np.savez_compressed(
        packets_directory / "run_motion_analysis.npz",
        tick_times_ns=trajectory.tick_times_ns,
        raw_encoder_angles_rad=trajectory.raw_encoder_angles_rad,
        # Retained as a legacy alias; this array is in motion coordinates.
        theta_y_at_ticks_rad=trajectory.theta_y_at_ticks_rad,
        theta_y_motion_at_ticks_rad=trajectory.theta_y_at_ticks_rad,
        theta_y_kinematic_at_ticks_rad=(
            kinematics_config.theta_y_kinematic_sign
            * trajectory.theta_y_at_ticks_rad
        ),
        correction_factors=trajectory.correction_factors,
        predicted_interval_angles_rad=trajectory.predicted_interval_angles_rad,
        encoder_interval_angles_rad=trajectory.encoder_interval_angles_rad,
    )

    rgb_times_ns = np.asarray(
        [image.timestamp_ns for image in bag_index.rgb_images], dtype=np.int64
    )
    depth_times_ns = np.asarray(
        [image.timestamp_ns for image in bag_index.depth_images], dtype=np.int64
    )
    synchronized_pairs = pair_image_timestamps(
        rgb_times_ns,
        depth_times_ns,
        int(round(processing_config.rgb_depth_max_delta_ms * 1_000_000.0)),
    )

    leg_times_ns, leg_values = bag_index.streams["leg"].arrays()
    knee_column = bag_index.streams["leg"].channel_names.index(
        "knee_joint.position"
    )
    knee_times_ns, knee_values = _finite_column(
        leg_times_ns,
        leg_values,
        knee_column,
    )

    manifest_records = []
    image_requests = {}
    last_saved_patch_centre_motion_rad = None
    half_patch_spread_rad = processing_config.alpha_sp_rad / 2.0
    half_spectral_window_ns = int(round(
        processing_config.spectral_window_ms * 500_000.0
    ))
    maximum_gap_ns = int(round(
        processing_config.maximum_interpolation_gap_ms * 1_000_000.0
    ))
    experiment_id = str(experiment_config.get("experiment_id", "unknown"))
    run_number = _run_number(run_directory)

    def reject(pair, reason: str) -> None:
        counts = analysis_report["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1
        analysis_report["rejected_candidates"].append(
            {
                "rgb_time_ns": pair.rgb_time_ns,
                "depth_time_ns": pair.depth_time_ns,
                "image_time_ns": pair.image_time_ns,
                "reason": reason,
            }
        )

    for pair in progress_items(
        synchronized_pairs, progress, 0.62, 0.82, "Isolating packets",
    ):
        if pair.image_time_ns < trajectory.start_time_ns:
            reject(pair, "image_before_t0")
            continue
        if pair.image_time_ns > trajectory.end_time_ns:
            reject(pair, "image_after_corrected_run_end")
            continue
        try:
            theta_y_image_motion_rad = trajectory.theta_y_at(
                pair.image_time_ns
            )
            patch_centre_motion_angle_rad = (
                theta_y_image_motion_rad
                + trajectory.travel_direction * processing_config.alpha_off_rad
            )
            if (
                last_saved_patch_centre_motion_rad is not None
                and trajectory.travel_direction * (
                    patch_centre_motion_angle_rad
                    - last_saved_patch_centre_motion_rad
                ) < processing_config.alpha_sp_rad
            ):
                reject(pair, "patch_overlaps_previous_packet")
                continue

            theta_y_rgb_motion_rad = trajectory.theta_y_at(pair.rgb_time_ns)
            theta_y_depth_motion_rad = trajectory.theta_y_at(
                pair.depth_time_ns
            )
            # Contact timing stays in the encoder-motion convention. Spatial
            # projection uses the opposite-handed notebook/base convention.
            theta_y_image_kinematic_rad = (
                kinematics_config.theta_y_to_kinematic(
                    theta_y_image_motion_rad
                )
            )
            theta_y_rgb_kinematic_rad = (
                kinematics_config.theta_y_to_kinematic(
                    theta_y_rgb_motion_rad
                )
            )
            theta_y_depth_kinematic_rad = (
                kinematics_config.theta_y_to_kinematic(
                    theta_y_depth_motion_rad
                )
            )
            patch_centre_kinematic_angle_rad = (
                kinematics_config.theta_y_to_kinematic(
                    patch_centre_motion_angle_rad
                )
            )
            travel_direction_kinematic = (
                kinematics_config.direction_to_kinematic(
                    trajectory.travel_direction
                )
            )
            contact_time_ns, angular_residual_rad = (
                trajectory.first_future_crossing_time(
                    pair.image_time_ns,
                    processing_config.alpha_off_rad,
                )
            )
            local_start_ns, _ = trajectory.crossing_time(
                patch_centre_motion_angle_rad
                - trajectory.travel_direction * half_patch_spread_rad
            )
            local_end_ns, _ = trajectory.crossing_time(
                patch_centre_motion_angle_rad
                + trajectory.travel_direction * half_patch_spread_rad
            )
            local_window_ms = (local_end_ns - local_start_ns) * 1e-6

            rgb_metadata = bag_index.rgb_images[pair.rgb_index]
            depth_metadata = bag_index.depth_images[pair.depth_index]
            rgb_calibration = _camera_inputs(
                rgb_metadata,
                bag_index.rgb_calibrations,
                pair.rgb_time_ns,
            )
            depth_calibration = _camera_inputs(
                depth_metadata,
                bag_index.depth_calibrations,
                pair.depth_time_ns,
            )
            knee_joint_rgb = _optional_knee_angle(
                knee_times_ns, knee_values, pair.rgb_time_ns
            )
            knee_joint_depth = _optional_knee_angle(
                knee_times_ns, knee_values, pair.depth_time_ns
            )
            # The two exposures need their own acquisition-time pose. Sampling
            # still checks actual coverage, but retains gyro-supported estimates
            # during impacts or low confidence. Save their uncertainty instead
            # of rejecting these packets. No pose is invented across a gap.
            theta_p_rgb = pitch_trajectory.sample(pair.rgb_time_ns)
            theta_p_depth = pitch_trajectory.sample(pair.depth_time_ns)
            theta_p_rgb_uncertainty_rad = pitch_trajectory.uncertainty_at(
                pair.rgb_time_ns
            )
            theta_p_depth_uncertainty_rad = pitch_trajectory.uncertainty_at(
                pair.depth_time_ns
            )

            rgb_projection = project_track_patch(
                kinematics_config,
                theta_y_rgb_kinematic_rad,
                theta_p_rgb,
                patch_centre_kinematic_angle_rad,
                processing_config.alpha_sp_rad,
                rgb_calibration,
                processing_config.track_point_count,
            )
            depth_projection = project_track_patch(
                kinematics_config,
                theta_y_depth_kinematic_rad,
                theta_p_depth,
                patch_centre_kinematic_angle_rad,
                processing_config.alpha_sp_rad,
                depth_calibration,
                processing_config.track_point_count,
            )
            if not rgb_projection.fully_visible:
                raise ValueError("rgb_patch_not_fully_visible")
            if not depth_projection.fully_visible:
                raise ValueError("depth_patch_not_fully_visible")

            rgb_mask = polygon_mask(
                rgb_metadata.height,
                rgb_metadata.width,
                rgb_projection.polygon_pixels,
            )
            depth_mask = polygon_mask(
                depth_metadata.height,
                depth_metadata.width,
                depth_projection.polygon_pixels,
            )
            if not np.any(rgb_mask):
                raise ValueError("rgb_patch_mask_is_empty")
            if not np.any(depth_mask):
                raise ValueError("depth_patch_mask_is_empty")

            if not _window_has_complete_timestamp_coverage(
                bag_index,
                local_start_ns,
                local_end_ns,
                maximum_gap_ns,
                trajectory.start_time_ns,
                trajectory.end_time_ns,
            ):
                raise ValueError("local_window_has_incomplete_telemetry_coverage")

            local_values, local_valid, local_times, channel_names = (
                _resample_all_streams(
                    bag_index,
                    contact_time_ns,
                    local_start_ns,
                    local_end_ns,
                    processing_config,
                    trajectory.start_time_ns,
                    trajectory.end_time_ns,
                )
            )
            spectral_values, spectral_valid, spectral_times, _ = (
                _resample_all_streams(
                    bag_index,
                    contact_time_ns,
                    contact_time_ns - half_spectral_window_ns,
                    contact_time_ns + half_spectral_window_ns,
                    processing_config,
                    trajectory.start_time_ns,
                    trajectory.end_time_ns,
                )
            )
            rgb_full = project_full_track_boundaries(
                kinematics_config,
                theta_y_rgb_kinematic_rad,
                theta_p_rgb,
                rgb_calibration,
                processing_config.track_point_count,
            )
            depth_full = project_full_track_boundaries(
                kinematics_config,
                theta_y_depth_kinematic_rad,
                theta_p_depth,
                depth_calibration,
                processing_config.track_point_count,
            )
        except ValueError as error:
            reject(pair, str(error))
            continue

        packet_id = _packet_identifier(
            experiment_id,
            run_number,
            theta_y_image_motion_rad,
            pair.rgb_time_ns,
            pair.depth_time_ns,
            dataset_identifier,
        )
        packet_directory = packets_directory / packet_id
        packet_directory.mkdir()
        stored_rgb_encoding = {
            "bgr8": "rgb8",
            "bgra8": "rgba8",
        }.get(rgb_metadata.encoding, rgb_metadata.encoding)
        lap_index = int(math.floor(
            abs(theta_y_image_motion_rad) / (2.0 * math.pi)
        ))
        metadata = {
            "packet_id": packet_id,
            # Classification describes the highlighted physical terrain patch.
            "class_id": None,
            "classification_method": None,
            "model_id": None,
            "experiment_id": experiment_id,
            "run_number": run_number,
            "lap_index": lap_index,
            "rgb_time_ns": pair.rgb_time_ns,
            "depth_time_ns": pair.depth_time_ns,
            "image_time_ns": pair.image_time_ns,
            "sync_error_ns": pair.sync_error_ns,
            "contact_time_ns": contact_time_ns,
            "contact_delay_ns": contact_time_ns - pair.image_time_ns,
            "contact_angular_residual_rad": angular_residual_rad,
            # Unqualified spatial fields use the kinematic/base convention so
            # existing Stage 2 and Foxglove consumers remain geometrically
            # correct. Explicit motion fields preserve the contact-time basis.
            "travel_direction": travel_direction_kinematic,
            "travel_direction_motion": trajectory.travel_direction,
            "travel_direction_kinematic": travel_direction_kinematic,
            "theta_y_kinematic_sign": (
                kinematics_config.theta_y_kinematic_sign
            ),
            "theta_y_image_rad": theta_y_image_kinematic_rad,
            "theta_y_rgb_rad": theta_y_rgb_kinematic_rad,
            "theta_y_depth_rad": theta_y_depth_kinematic_rad,
            "theta_y_image_kinematic_rad": theta_y_image_kinematic_rad,
            "theta_y_rgb_kinematic_rad": theta_y_rgb_kinematic_rad,
            "theta_y_depth_kinematic_rad": theta_y_depth_kinematic_rad,
            "theta_y_image_motion_rad": theta_y_image_motion_rad,
            "theta_y_rgb_motion_rad": theta_y_rgb_motion_rad,
            "theta_y_depth_motion_rad": theta_y_depth_motion_rad,
            "patch_centre_angle_rad": (
                patch_centre_kinematic_angle_rad
            ),
            "patch_centre_motion_angle_rad": (
                patch_centre_motion_angle_rad
            ),
            "patch_centre_kinematic_angle_rad": (
                patch_centre_kinematic_angle_rad
            ),
            "patch_spread_rad": processing_config.alpha_sp_rad,
            "alpha_off_rad": processing_config.alpha_off_rad,
            "local_window_ms": local_window_ms,
            "local_start_time_ns": local_start_ns,
            "local_end_time_ns": local_end_ns,
            "knee_joint_rgb_rad": knee_joint_rgb,
            "knee_joint_depth_rad": knee_joint_depth,
            "theta_p_rgb_rad": theta_p_rgb,
            "theta_p_depth_rad": theta_p_depth,
            "theta_p_source": "camera_imu",
            "pitch_estimator_version": PITCH_ESTIMATOR_VERSION,
            "theta_p_rgb_uncertainty_rad": theta_p_rgb_uncertainty_rad,
            "theta_p_depth_uncertainty_rad": theta_p_depth_uncertainty_rad,
            "imu_uncertainty_deg": math.degrees(2.0 * max(
                theta_p_rgb_uncertainty_rad, theta_p_depth_uncertainty_rad,
            )),
            "rgb_calibration": rgb_calibration.to_dict(),
            "depth_calibration": depth_calibration.to_dict(),
            "rgb_frame_id": rgb_metadata.frame_id,
            "depth_frame_id": depth_metadata.frame_id,
            "rgb_source_encoding": rgb_metadata.encoding,
            "rgb_stored_encoding": stored_rgb_encoding,
            "depth_encoding": depth_metadata.encoding,
            "T_rgb_camera_in_base": rgb_projection.T_camera_in_base.tolist(),
            "T_depth_camera_in_base": depth_projection.T_camera_in_base.tolist(),
            "telemetry_channel_names": channel_names,
        }
        _write_json(packet_directory / "metadata.json", metadata)
        np.savez_compressed(
            packet_directory / "geometry.npz",
            rgb_mask=np.packbits(rgb_mask, axis=None),
            rgb_mask_shape=np.asarray(rgb_mask.shape, dtype=np.int64),
            depth_mask=np.packbits(depth_mask, axis=None),
            depth_mask_shape=np.asarray(depth_mask.shape, dtype=np.int64),
            rgb_patch_polygon_pixels=rgb_projection.polygon_pixels,
            depth_patch_polygon_pixels=depth_projection.polygon_pixels,
            patch_inner_points_base_m=rgb_projection.inner_points_base_m,
            patch_outer_points_base_m=rgb_projection.outer_points_base_m,
            rgb_inner_track_pixels=rgb_full[0],
            rgb_inner_track_valid=rgb_full[1],
            rgb_outer_track_pixels=rgb_full[2],
            rgb_outer_track_valid=rgb_full[3],
            depth_inner_track_pixels=depth_full[0],
            depth_inner_track_valid=depth_full[1],
            depth_outer_track_pixels=depth_full[2],
            depth_outer_track_valid=depth_full[3],
        )
        np.savez_compressed(
            packet_directory / "telemetry.npz",
            local_values=local_values,
            local_valid=local_valid,
            local_relative_times_ns=local_times,
            spectral_values=spectral_values,
            spectral_valid=spectral_valid,
            spectral_relative_times_ns=spectral_times,
        )
        image_requests[(RGB_IMAGE_TOPIC, pair.rgb_time_ns)] = _image_writer(
            packet_directory / "rgb.npy", "rgb"
        )
        image_requests[(DEPTH_IMAGE_TOPIC, pair.depth_time_ns)] = _image_writer(
            packet_directory / "depth.npy", "depth"
        )
        manifest_records.append(
            {
                "packet_id": packet_id,
                "image_time_ns": pair.image_time_ns,
                "contact_time_ns": contact_time_ns,
                "theta_y_image_rad": theta_y_image_kinematic_rad,
                "theta_y_image_motion_rad": theta_y_image_motion_rad,
                "patch_centre_angle_rad": (
                    patch_centre_kinematic_angle_rad
                ),
                "patch_centre_motion_angle_rad": (
                    patch_centre_motion_angle_rad
                ),
                "lap_index": lap_index,
                "sync_error_ns": pair.sync_error_ns,
                "packet_path": str(packet_directory.relative_to(output_directory)),
            }
        )
        last_saved_patch_centre_motion_rad = patch_centre_motion_angle_rad

    found_images = extract_selected_images(
        bag_directory, image_requests,
        progress=progress_range(progress, 0.82, 0.98),
    )
    missing_images = set(image_requests) - found_images
    if missing_images:
        analysis_report["status"] = "failed_image_recovery"
        analysis_report["missing_selected_image_count"] = len(missing_images)
        _write_json(analysis_report_path, analysis_report)
        raise RuntimeError(
            f"Second bag pass could not recover {len(missing_images)} selected images."
        )

    manifest_path = output_directory / "manifest.jsonl"
    if progress:
        progress(0.98, "Writing packet reports")
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage1_complete",
        "terrain_and_light_ID": terrain_id,
        "requires_regeneration": False,
        "dataset_identifier": dataset_identifier,
        "run_directory": str(run_directory),
        "kinematics_model_version": KINEMATICS_MODEL_VERSION,
        "pitch_estimator_version": PITCH_ESTIMATOR_VERSION,
        "encoder_position_version": ENCODER_POSITION_VERSION,
        "encoder_position": analysis_report["encoder_position"],
        "pitch_estimation": pitch_trajectory.report,
        "bag_directory": str(bag_directory),
        "experiment_config_source": str(experiment_config_path),
        "experiment_config": experiment_config,
        "run_config": run_config,
        "camera_driver_config": camera_driver_config,
        "processing_config": processing_config.to_dict(),
        "packet_count": len(manifest_records),
    }
    _write_json(output_directory / "provenance.json", provenance)
    quality_report = {
        "rgb_image_count": len(bag_index.rgb_images),
        "depth_image_count": len(bag_index.depth_images),
        "synchronized_pair_count": len(synchronized_pairs),
        "packet_count": len(manifest_records),
        "rejected": analysis_report["rejection_counts"],
    }
    _write_json(output_directory / "quality_report.json", quality_report)
    analysis_report["status"] = "stage1_complete"
    analysis_report["accepted_packet_count"] = len(manifest_records)
    _write_json(analysis_report_path, analysis_report)
    if progress:
        progress(1.0, "Packet isolation complete")
    return output_directory


def _read_manifest(dataset_directory: Path) -> list[dict[str, Any]]:
    records = []
    with (dataset_directory / "manifest.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _unpack_mask(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
    dimensions = tuple(int(value) for value in shape)
    pixel_count = int(np.prod(dimensions))
    return np.unpackbits(packed, count=pixel_count).reshape(dimensions).astype(bool)


def _mask_crop_origin_uv(mask: np.ndarray) -> list[int]:
    """Return the full-image pixel coordinate of a mask crop's top-left corner."""
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("cannot_find_crop_origin_for_empty_mask")
    return [int(columns.min()), int(rows.min())]


def extract_features(
    dataset_directory: Path,
    processing_config: ProcessingConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Extract patches, point maps, plane slopes, and all-channel spectra."""
    dataset_directory = dataset_directory.resolve()
    provenance_path = dataset_directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require_provenance_terrain_id(provenance)
    if processing_config is None:
        processing_config = ProcessingConfig(**{
            name: value for name, value in provenance["processing_config"].items()
            if name not in LEGACY_CONFIG_FIELDS
        })
        processing_config.validate()

    feature_records = []
    for record in progress_items(
        _read_manifest(dataset_directory), progress, 0.0, 0.98,
        "Extracting packet features",
    ):
        packet_directory = dataset_directory / record["packet_path"]
        stage2_directory = packet_directory / "stage2"
        stage2_directory.mkdir(exist_ok=False)
        metadata = json.loads(
            (packet_directory / "metadata.json").read_text(encoding="utf-8")
        )
        # Older Stage 1 datasets have no classification fields. Upgrade them
        # here too, and preserve labels assigned before feature extraction.
        classification = {
            name: metadata.get(name)
            for name in ("class_id", "classification_method", "model_id")
        }
        if any(name not in metadata for name in classification):
            metadata.update(classification)
            _write_json(packet_directory / "metadata.json", metadata)
        imu_uncertainty_deg = _packet_imu_uncertainty_deg(metadata)
        # Include this feature even when an unrelated plane/spectrum extraction
        # fails. A high value does not invalidate a packet or its other features.
        feature_record = {
            "packet_id": metadata["packet_id"],
            **classification,
            "valid": False,
            "IMU uncertainty": imu_uncertainty_deg,
        }
        try:
            rgb = np.load(packet_directory / "rgb.npy", allow_pickle=False)
            depth = np.load(packet_directory / "depth.npy", allow_pickle=False)
            geometry = np.load(
                packet_directory / "geometry.npz", allow_pickle=False
            )
            rgb_mask = _unpack_mask(
                geometry["rgb_mask"], geometry["rgb_mask_shape"]
            )
            depth_mask = _unpack_mask(
                geometry["depth_mask"], geometry["depth_mask_shape"]
            )
            rgb_patch, rgb_patch_mask = masked_crop(rgb, rgb_mask)
            depth_patch, depth_patch_mask = masked_crop(depth, depth_mask)
            rgb_crop_origin_uv = _mask_crop_origin_uv(rgb_mask)
            depth_crop_origin_uv = _mask_crop_origin_uv(depth_mask)
            np.save(stage2_directory / "rgb_patch.npy", rgb_patch)
            np.save(stage2_directory / "depth_patch.npy", depth_patch)
            np.save(stage2_directory / "rgb_patch_mask.npy", rgb_patch_mask)
            np.save(stage2_directory / "depth_patch_mask.npy", depth_patch_mask)

            depth_calibration = CameraCalibration.from_dict(
                metadata["depth_calibration"]
            )
            point_map = depth_point_map_in_base(
                depth,
                depth_mask,
                depth_calibration,
                processing_config.depth_scale_m,
                np.asarray(metadata["T_depth_camera_in_base"], dtype=float),
            )
            plane = fit_plane_ransac(
                point_map.points_base_m,
                processing_config.plane_ransac_iterations,
                processing_config.plane_inlier_threshold_m,
                processing_config.minimum_plane_points,
            )
            along_slope, cross_slope = slope_from_plane(
                plane.normal,
                float(metadata["patch_centre_kinematic_angle_rad"]),
                float(metadata["travel_direction_kinematic"]),
            )
            np.savez_compressed(
                stage2_directory / "point_map.npz",
                points_base_m=point_map.points_base_m,
                pixel_uv=point_map.pixel_uv,
                depth_m=point_map.depth_m,
                ransac_inlier_mask=plane.inlier_mask,
            )

            telemetry = np.load(
                packet_directory / "telemetry.npz", allow_pickle=False
            )
            (
                frequencies_hz,
                power,
                dominant_frequency_hz,
                dominant_power,
            ) = calculate_spectra(
                telemetry["spectral_values"],
                telemetry["spectral_valid"],
                processing_config.telemetry_resample_hz,
                processing_config.minimum_spectral_valid_fraction,
                processing_config.maximum_interpolation_gap_ms,
            )
            np.savez_compressed(
                stage2_directory / "features.npz",
                plane_normal=plane.normal,
                plane_offset=plane.offset,
                frequencies_hz=frequencies_hz,
                spectral_power=power,
                dominant_frequency_hz=dominant_frequency_hz,
                dominant_power=dominant_power,
                imu_uncertainty_deg=(
                    imu_uncertainty_deg if imu_uncertainty_deg is not None else np.nan
                ),
            )
            feature_record.update(
                {
                    "valid": True,
                    "along_slope_rad": along_slope,
                    "along_slope_deg": float(np.degrees(along_slope)),
                    "cross_slope_rad": cross_slope,
                    "cross_slope_deg": float(np.degrees(cross_slope)),
                    "cross_slope_positive_direction": "radially_outward",
                    "plane_normal": plane.normal.tolist(),
                    "plane_offset": plane.offset,
                    "plane_rmse_m": plane.rmse_m,
                    "plane_point_count": int(point_map.points_base_m.shape[0]),
                    "plane_inlier_count": int(
                        np.count_nonzero(plane.inlier_mask)
                    ),
                    "plane_inlier_ratio": float(np.mean(plane.inlier_mask)),
                    "valid_depth_fraction": point_map.valid_depth_fraction,
                    "rgb_crop_origin_uv": rgb_crop_origin_uv,
                    "depth_crop_origin_uv": depth_crop_origin_uv,
                    "telemetry_channel_names": metadata[
                        "telemetry_channel_names"
                    ],
                    "dominant_frequency_hz": [
                        [float(value) if np.isfinite(value) else None for value in row]
                        for row in dominant_frequency_hz
                    ],
                    "dominant_power": [
                        [float(value) if np.isfinite(value) else None for value in row]
                        for row in dominant_power
                    ],
                }
            )
        except (ValueError, OSError, KeyError) as error:
            feature_record["error"] = str(error)

        _write_json(stage2_directory / "features.json", feature_record)
        feature_records.append(feature_record)

    features_path = dataset_directory / "features.jsonl"
    if progress:
        progress(0.98, "Writing feature reports")
    features_path.write_text(
        "".join(json.dumps(record) + "\n" for record in feature_records),
        encoding="utf-8",
    )
    provenance["stage"] = "stage2_complete"
    provenance["stage2_completed_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["valid_feature_count"] = sum(
        bool(record["valid"]) for record in feature_records
    )
    _write_json(provenance_path, provenance)
    analysis_report_path = dataset_directory / "packets" / "analysis_report.json"
    if analysis_report_path.is_file():
        analysis_report = json.loads(
            analysis_report_path.read_text(encoding="utf-8")
        )
        analysis_report["status"] = "stage2_complete"
        analysis_report["stage2_valid_packet_count"] = sum(
            bool(record["valid"]) for record in feature_records
        )
        analysis_report["stage2_invalid_packets"] = [
            {
                "packet_id": record["packet_id"],
                "reason": record.get("error", "unknown_stage2_error"),
            }
            for record in feature_records
            if not record["valid"]
        ]
        _write_json(analysis_report_path, analysis_report)
    if progress:
        progress(1.0, "Feature extraction complete")
    return dataset_directory
