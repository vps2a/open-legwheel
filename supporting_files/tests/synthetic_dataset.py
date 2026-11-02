"""Small on-disk dataset fixture matching the production packet format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHANNEL_NAMES = [
    "hip_joint.position",
    "hip_joint.velocity",
    "hip_joint.effort",
    "knee_joint.position",
    "knee_joint.velocity",
    "knee_joint.effort",
    "wheel_joint.position",
    "wheel_joint.velocity",
    "wheel_joint.effort",
    "imu.angular_velocity.x",
    "imu.angular_velocity.y",
    "imu.angular_velocity.z",
    "imu.linear_acceleration.x",
    "imu.linear_acceleration.y",
    "imu.linear_acceleration.z",
]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def create_synthetic_dataset(root: Path) -> Path:
    """Create two chronological packets, one with Stage 2 and one without."""
    root.mkdir()
    _write_json(
        root / "provenance.json",
        {
            "stage": "stage2_complete",
            "processing_config": {
                "depth_scale_m": 0.001,
                "telemetry_resample_hz": 100.0,
            },
        },
    )

    manifest_lines = []
    feature_records = []
    for index, timestamp in enumerate((2_000_000_000, 3_000_000_000)):
        packet_id = f"packet_{index:06d}"
        packet_directory = root / "packets" / packet_id
        packet_directory.mkdir(parents=True)
        manifest_lines.append(
            json.dumps(
                {
                    "packet_id": packet_id,
                    "packet_path": f"packets/{packet_id}",
                    "image_time_ns": timestamp,
                }
            )
        )
        metadata = {
            "packet_id": packet_id,
            "experiment_id": "synthetic_experiment",
            "run_number": 4,
            "lap_index": 0,
            "image_time_ns": timestamp,
            "rgb_time_ns": timestamp - 1_000_000,
            "depth_time_ns": timestamp + 1_000_000,
            "sync_error_ns": 2_000_000,
            "contact_delay_ns": 500_000_000,
            "theta_y_image_motion_rad": 1.0 + index,
            "theta_y_image_kinematic_rad": -1.0 - index,
            "theta_p_rgb_rad": -0.1,
            "theta_p_depth_rad": -0.1,
            "patch_centre_kinematic_angle_rad": -1.3 - index,
            "travel_direction_motion": 1.0,
            "travel_direction_kinematic": -1.0,
            "local_window_ms": 175.0,
            "telemetry_channel_names": CHANNEL_NAMES,
        }
        _write_json(packet_directory / "metadata.json", metadata)

        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[:, :, 1] = 60 + index * 60
        depth = np.full((12, 16), 1000 + index * 1000, dtype=np.uint16)
        depth[0, 0] = 0
        np.save(packet_directory / "rgb.npy", rgb)
        np.save(packet_directory / "depth.npy", depth)

        mask = np.zeros((12, 16), dtype=bool)
        mask[3:10, 4:13] = True
        polygon = np.asarray([[4, 3], [12, 3], [12, 9], [4, 9]], dtype=float)
        inner = np.asarray([[0.7, -0.1, 0.0], [0.8, -0.1, 0.01]])
        outer = np.asarray([[0.7, 0.1, 0.0], [0.8, 0.1, 0.01]])
        np.savez_compressed(
            packet_directory / "geometry.npz",
            rgb_mask=np.packbits(mask, axis=None),
            rgb_mask_shape=np.asarray(mask.shape),
            depth_mask=np.packbits(mask, axis=None),
            depth_mask_shape=np.asarray(mask.shape),
            rgb_patch_polygon_pixels=polygon,
            depth_patch_polygon_pixels=polygon,
            patch_inner_points_base_m=inner,
            patch_outer_points_base_m=outer,
        )

        local_times = np.arange(-5, 6, dtype=np.int64) * 10_000_000
        spectral_times = np.arange(-50, 51, dtype=np.int64) * 10_000_000
        local_values = np.column_stack(
            [np.sin(np.arange(11) * 0.2 + channel) for channel in range(15)]
        )
        spectral_values = np.column_stack(
            [np.sin(np.arange(101) * 0.1 + channel) for channel in range(15)]
        )
        np.savez_compressed(
            packet_directory / "telemetry.npz",
            local_values=local_values,
            local_valid=np.ones_like(local_values, dtype=bool),
            local_relative_times_ns=local_times,
            spectral_values=spectral_values,
            spectral_valid=np.ones_like(spectral_values, dtype=bool),
            spectral_relative_times_ns=spectral_times,
        )

        if index == 0:
            stage2 = packet_directory / "stage2"
            stage2.mkdir()
            features = {
                "packet_id": packet_id,
                "valid": True,
                "along_slope_deg": 4.25,
                "cross_slope_deg": -1.5,
                "plane_normal": [0.0, 0.0, 1.0],
                "plane_offset": -0.01,
                "plane_rmse_m": 0.001,
                "plane_point_count": 4,
                "plane_inlier_count": 3,
                "plane_inlier_ratio": 0.75,
                "valid_depth_fraction": 0.95,
            }
            _write_json(stage2 / "features.json", features)
            feature_records.append(features)
            frequencies = np.linspace(0.0, 50.0, 51)
            power = np.ones((51, 15), dtype=float)
            np.savez_compressed(
                stage2 / "features.npz",
                plane_normal=np.asarray(features["plane_normal"]),
                plane_offset=features["plane_offset"],
                frequencies_hz=frequencies,
                spectral_power=power,
                dominant_frequency_hz=np.full((2, 15), 18.0),
                dominant_power=np.ones((2, 15)),
            )
            points = np.asarray(
                [
                    [0.70, -0.05, 0.01],
                    [0.72, 0.00, 0.01],
                    [0.74, 0.05, 0.01],
                    [0.76, 0.08, 0.03],
                ]
            )
            np.savez_compressed(
                stage2 / "point_map.npz",
                points_base_m=points,
                pixel_uv=np.asarray([[4, 3], [6, 5], [8, 7], [10, 9]]),
                depth_m=np.ones(4),
                ransac_inlier_mask=np.asarray([True, True, True, False]),
            )

    (root / "manifest.jsonl").write_text(
        "\n".join(reversed(manifest_lines)) + "\n",
        encoding="utf-8",
    )
    (root / "features.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in feature_records),
        encoding="utf-8",
    )
    return root
