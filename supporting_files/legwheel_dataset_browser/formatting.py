"""Human-readable labels and metadata formatting for the packet browser."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .classification import TERRAIN_CLASSES
from .model import PacketData


def channel_unit(channel_name: str) -> str:
    """Return the physical unit associated with a telemetry channel."""
    if channel_name.endswith(".position"):
        return "rad"
    if channel_name.endswith(".velocity") or ".angular_velocity." in channel_name:
        return "rad/s"
    if channel_name.endswith(".effort"):
        return "N·m"
    if ".linear_acceleration." in channel_name:
        return "m/s²"
    return ""


def short_channel_name(channel_name: str) -> str:
    """Make stored channel names compact enough for small plots."""
    replacements = {
        "hip_joint.": "Hip ",
        "knee_joint.": "Knee ",
        "wheel_joint.": "Wheel ",
        "imu.angular_velocity.": "Gyro ",
        "imu.linear_acceleration.": "Accel ",
    }
    label = channel_name
    for prefix, replacement in replacements.items():
        if label.startswith(prefix):
            label = replacement + label[len(prefix):]
            break
    return label.replace("_", " ").title()


def format_timestamp_ns(timestamp_ns: Any) -> str:
    """Format nanoseconds as an exact integer plus an ISO UTC timestamp."""
    try:
        value = int(timestamp_ns)
    except (TypeError, ValueError):
        return "Unavailable"
    moment = datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
    return f"{value} ns ({moment.isoformat(timespec='milliseconds')})"


def _number(value: Any, unit: str = "", precision: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not np.isfinite(number):
        return "Unavailable"
    suffix = f" {unit}" if unit else ""
    return f"{number:.{precision}f}{suffix}"


def metadata_rows(packet: PacketData) -> list[tuple[str, str]]:
    """Select the packet metadata most useful during inspection."""
    metadata = packet.metadata
    features = packet.features or {}
    class_id = metadata.get("class_id")
    classification = "Unclassified"
    if type(class_id) is int and 0 <= class_id < len(TERRAIN_CLASSES):
        classification = f"{class_id} - {TERRAIN_CLASSES[class_id]}"
    return [
        ("Packet ID", str(metadata.get("packet_id", "Unavailable"))),
        ("Terrain class", classification),
        ("Classification method", metadata.get("classification_method") or "Unavailable"),
        ("Classification model", metadata.get("model_id") or "Unavailable"),
        ("Experiment", str(metadata.get("experiment_id", "Unavailable"))),
        ("Run", str(metadata.get("run_number", "Unavailable"))),
        ("Lap", str(metadata.get("lap_index", "Unavailable"))),
        ("Image time", format_timestamp_ns(metadata.get("image_time_ns"))),
        ("RGB time", format_timestamp_ns(metadata.get("rgb_time_ns"))),
        ("Depth time", format_timestamp_ns(metadata.get("depth_time_ns"))),
        ("RGB-depth delta", _number(metadata.get("sync_error_ns", 0.0), "ns", 0)),
        ("Contact delay", _number(metadata.get("contact_delay_ns"), "ns", 0)),
        (
            "θy image (motion)",
            _number(metadata.get("theta_y_image_motion_rad"), "rad"),
        ),
        (
            "θy image (kinematic)",
            _number(metadata.get("theta_y_image_kinematic_rad"), "rad"),
        ),
        ("θp RGB", _number(metadata.get("theta_p_rgb_rad"), "rad")),
        ("θp depth", _number(metadata.get("theta_p_depth_rad"), "rad")),
        (
            "IMU uncertainty",
            _number(
                features.get("IMU uncertainty", metadata.get("imu_uncertainty_deg")),
                "° (2σ)", 3,
            ),
        ),
        (
            "Patch-centre angle",
            _number(metadata.get("patch_centre_kinematic_angle_rad"), "rad"),
        ),
        (
            "Travel direction (motion)",
            _number(metadata.get("travel_direction_motion"), precision=0),
        ),
        (
            "Travel direction (kinematic)",
            _number(metadata.get("travel_direction_kinematic"), precision=0),
        ),
        ("Local window", _number(metadata.get("local_window_ms"), "ms", 3)),
        (
            "Stage 2",
            "Valid" if packet.stage2_available else features.get("error", "Not available"),
        ),
        ("Plane points", str(features.get("plane_point_count", "Unavailable"))),
        ("Plane inliers", str(features.get("plane_inlier_count", "Unavailable"))),
        ("Plane inlier ratio", _number(features.get("plane_inlier_ratio"), precision=3)),
        ("Plane RMSE", _number(features.get("plane_rmse_m"), "m", 6)),
        (
            "Valid depth fraction",
            _number(features.get("valid_depth_fraction"), precision=3),
        ),
    ]
