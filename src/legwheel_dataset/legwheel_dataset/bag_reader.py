"""Bounded-memory indexing and image extraction from ROS 2 MCAP bags."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from legwheel_kinematics.projection import CameraCalibration
from legwheel_encoder.config import TICKS_TOPIC
from legwheel_dataset.progress import ProgressCallback


RGB_IMAGE_TOPIC = "/legwheel_rgbd/color/image_raw"
RGB_INFO_TOPIC = "/legwheel_rgbd/color/camera_info"
DEPTH_IMAGE_TOPIC = "/legwheel_rgbd/depth/image_raw"
DEPTH_INFO_TOPIC = "/legwheel_rgbd/depth/camera_info"
IMU_TOPIC = "/legwheel_rgbd/gyro_accel/sample"
LEG_STATE_TOPIC = "/legwheel/joint_states"
WHEEL_STATE_TOPIC = "/wheel/wheel_state"
ROTATION_TICKS_TOPIC = TICKS_TOPIC
REQUESTED_TORQUE_TOPIC = "/wheel/requested_torque"
COMMANDED_TORQUE_TOPIC = "/wheel/commanded_torque"
OPTIONAL_INDEX_TOPICS = {COMMANDED_TORQUE_TOPIC}

INDEX_TOPICS = {
    RGB_IMAGE_TOPIC,
    RGB_INFO_TOPIC,
    DEPTH_IMAGE_TOPIC,
    DEPTH_INFO_TOPIC,
    IMU_TOPIC,
    LEG_STATE_TOPIC,
    WHEEL_STATE_TOPIC,
    ROTATION_TICKS_TOPIC,
    REQUESTED_TORQUE_TOPIC,
}
READABLE_INDEX_TOPICS = INDEX_TOPICS | OPTIONAL_INDEX_TOPICS


def message_stamp_ns(message, bag_time_ns: int) -> int:
    """Read a positive ROS header stamp, falling back only when absent."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(bag_time_ns)
    timestamp = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp if timestamp > 0 else int(bag_time_ns)


@dataclass(frozen=True)
class ImageMetadata:
    """Small image fields retained during the first, metadata-only pass."""

    timestamp_ns: int
    width: int
    height: int
    encoding: str
    frame_id: str = ""


@dataclass
class TelemetrySeries:
    """One topic represented as a regular set of named numeric columns."""

    channel_names: tuple[str, ...]
    times_ns: list[int] = field(default_factory=list)
    rows: list[list[float]] = field(default_factory=list)

    def append(self, timestamp_ns: int, row: list[float]) -> None:
        if len(row) != len(self.channel_names):
            raise ValueError("Telemetry row does not match its channel names.")
        self.times_ns.append(int(timestamp_ns))
        self.rows.append(row)

    def arrays(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return sorted data with duplicate timestamps retained only once."""
        if not self.times_ns:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, len(self.channel_names)), dtype=float),
            )
        times = np.asarray(self.times_ns, dtype=np.int64)
        values = np.asarray(self.rows, dtype=float)
        order = np.argsort(times, kind="stable")
        times = times[order]
        values = values[order]
        keep = np.concatenate(([True], np.diff(times) > 0))
        return times[keep], values[keep]


@dataclass
class CalibrationTimeline:
    """Camera calibration epochs selected by image sensor time."""

    times_ns: list[int] = field(default_factory=list)
    calibrations: list[CameraCalibration] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)

    def append(
        self,
        timestamp_ns: int,
        calibration: CameraCalibration,
        frame_id: str = "",
    ) -> None:
        if self.calibrations:
            previous = self.calibrations[-1]
            unchanged = (
                previous.width == calibration.width
                and previous.height == calibration.height
                and previous.distortion_model == calibration.distortion_model
                and np.array_equal(previous.K, calibration.K)
                and np.array_equal(previous.D, calibration.D)
                and self.frame_ids[-1] == str(frame_id)
            )
            if unchanged:
                return
        self.times_ns.append(int(timestamp_ns))
        self.calibrations.append(calibration)
        self.frame_ids.append(str(frame_id))

    def at(self, timestamp_ns: int) -> CameraCalibration:
        if not self.times_ns:
            raise ValueError("No CameraInfo messages were recorded.")
        index = int(np.searchsorted(self.times_ns, timestamp_ns, side="right")) - 1
        if index < 0:
            # Drivers can publish the first CameraInfo slightly after the first
            # image. It is safe to use that first epoch when no earlier one exists.
            index = 0
        return self.calibrations[index]

    def frame_id_at(self, timestamp_ns: int) -> str:
        """Return the frame associated with the selected calibration epoch."""
        if not self.times_ns:
            raise ValueError("No CameraInfo messages were recorded.")
        index = int(np.searchsorted(self.times_ns, timestamp_ns, side="right")) - 1
        if index < 0:
            index = 0
        return self.frame_ids[index]


@dataclass
class BagIndex:
    """All low-bandwidth data and image metadata required to form packets."""

    rgb_images: list[ImageMetadata] = field(default_factory=list)
    depth_images: list[ImageMetadata] = field(default_factory=list)
    rgb_calibrations: CalibrationTimeline = field(default_factory=CalibrationTimeline)
    depth_calibrations: CalibrationTimeline = field(default_factory=CalibrationTimeline)
    streams: dict[str, TelemetrySeries] = field(default_factory=dict)
    imu_frame_ids: set[str] = field(default_factory=set)


def _joint_value(message, name: str, field_name: str) -> float:
    try:
        index = list(message.name).index(name)
        values = getattr(message, field_name)
        return float(values[index])
    except (ValueError, IndexError):
        return float("nan")


def _joint_row(message, joint_names: tuple[str, ...]) -> list[float]:
    row = []
    for joint_name in joint_names:
        for field_name in ("position", "velocity", "effort"):
            row.append(_joint_value(message, joint_name, field_name))
    return row


def _open_reader(bag_directory: Path):
    """Open rosbag2 lazily so pure numerical modules remain importable."""
    try:
        import rosbag2_py
    except ImportError as error:
        raise RuntimeError(
            "rosbag2_py is required. Run this command in a sourced ROS 2 "
            "Jazzy environment."
        ) from error

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_directory),
            storage_id="mcap",
        ),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def _message_types(reader) -> dict[str, str]:
    return {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }


def _deserialize(serialized, type_name: str):
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError("ROS 2 Python message support is not available.") from error
    return deserialize_message(serialized, get_message(type_name))


def index_bag(
    bag_directory: Path, *, progress: ProgressCallback | None = None,
) -> BagIndex:
    """Perform a first pass without retaining high-bandwidth image data."""
    reader = _open_reader(bag_directory)
    message_types = _message_types(reader)
    missing = INDEX_TOPICS - set(message_types)
    if missing:
        raise ValueError(
            "Bag is missing required topics: " + ", ".join(sorted(missing))
        )

    leg_names = ("hip_joint", "knee_joint")
    leg_channels = tuple(
        f"{name}.{field_name}"
        for name in leg_names
        for field_name in ("position", "velocity", "effort")
    )
    index = BagIndex(
        streams={
            "leg": TelemetrySeries(leg_channels),
            "wheel": TelemetrySeries(
                (
                    "wheel_joint.position",
                    "wheel_joint.velocity",
                    "wheel_joint.effort",
                )
            ),
            "imu": TelemetrySeries(
                (
                    "imu.angular_velocity.x",
                    "imu.angular_velocity.y",
                    "imu.angular_velocity.z",
                    "imu.linear_acceleration.x",
                    "imu.linear_acceleration.y",
                    "imu.linear_acceleration.z",
                )
            ),
            "ticks": TelemetrySeries(("rotation_encoder.ticks",)),
            "torque_command": TelemetrySeries(("wheel.requested_torque",)),
        }
    )
    # Keep this separate from the stable, 15-channel dataset telemetry schema.
    # Older bags have no post-governor command topic and remain valid inputs.
    if COMMANDED_TORQUE_TOPIC in message_types:
        index.streams["governed_torque_command"] = TelemetrySeries(("wheel.commanded_torque",))

    message_count = reader.get_metadata().message_count if progress else 0
    messages_read = 0
    if progress:
        progress(0.0, "Reading bag")
    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        messages_read += 1
        if progress and messages_read % 100 == 0:
            progress(min(messages_read / max(message_count, 1), 1.0), "Reading bag")
        if topic not in READABLE_INDEX_TOPICS:
            continue
        message = _deserialize(serialized, message_types[topic])
        timestamp_ns = message_stamp_ns(message, bag_time_ns)
        if topic == RGB_IMAGE_TOPIC:
            index.rgb_images.append(
                ImageMetadata(
                    timestamp_ns,
                    int(message.width),
                    int(message.height),
                    str(message.encoding),
                    str(message.header.frame_id),
                )
            )
        elif topic == DEPTH_IMAGE_TOPIC:
            index.depth_images.append(
                ImageMetadata(
                    timestamp_ns,
                    int(message.width),
                    int(message.height),
                    str(message.encoding),
                    str(message.header.frame_id),
                )
            )
        elif topic == RGB_INFO_TOPIC:
            index.rgb_calibrations.append(
                timestamp_ns,
                CameraCalibration.from_message(message),
                str(message.header.frame_id),
            )
        elif topic == DEPTH_INFO_TOPIC:
            index.depth_calibrations.append(
                timestamp_ns,
                CameraCalibration.from_message(message),
                str(message.header.frame_id),
            )
        elif topic == LEG_STATE_TOPIC:
            index.streams["leg"].append(
                timestamp_ns,
                _joint_row(message, leg_names),
            )
        elif topic == WHEEL_STATE_TOPIC:
            index.streams["wheel"].append(
                timestamp_ns,
                _joint_row(message, ("wheel_joint",)),
            )
        elif topic == ROTATION_TICKS_TOPIC:
            index.streams["ticks"].append(
                timestamp_ns,
                [float(message.data)],
            )
        elif topic == REQUESTED_TORQUE_TOPIC:
            index.streams["torque_command"].append(
                timestamp_ns,
                [float(message.data)],
            )
        elif topic == COMMANDED_TORQUE_TOPIC:
            index.streams["governed_torque_command"].append(timestamp_ns, [float(message.data)])
        elif topic == IMU_TOPIC:
            # Keep the declared basis so offline pose estimation can reject a
            # body-frame stream instead of assuming it uses Gemini optical axes.
            index.imu_frame_ids.add(str(getattr(message.header, "frame_id", "")))
            index.streams["imu"].append(
                timestamp_ns,
                [
                    float(message.angular_velocity.x),
                    float(message.angular_velocity.y),
                    float(message.angular_velocity.z),
                    float(message.linear_acceleration.x),
                    float(message.linear_acceleration.y),
                    float(message.linear_acceleration.z),
                ],
            )
    if progress:
        progress(1.0, "Reading bag")
    return index


def extract_selected_images(
    bag_directory: Path,
    requested: dict[tuple[str, int], Callable],
    *,
    progress: ProgressCallback | None = None,
) -> set[tuple[str, int]]:
    """Perform a second pass and deliver only images selected for packets."""
    reader = _open_reader(bag_directory)
    message_types = _message_types(reader)
    found = set()
    image_topics = {RGB_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC}
    if progress:
        progress(0.0, "Recovering packet images")
    while reader.has_next() and len(found) < len(requested):
        topic, serialized, bag_time_ns = reader.read_next()
        if topic not in image_topics:
            continue
        message = _deserialize(serialized, message_types[topic])
        key = (topic, message_stamp_ns(message, bag_time_ns))
        callback = requested.get(key)
        if callback is not None and key not in found:
            callback(message)
            found.add(key)
            if progress:
                progress(len(found) / len(requested), "Recovering packet images")
    if progress:
        progress(1.0, "Recovering packet images")
    return found
