"""Check angular packet spacing and telemetry bounded by terrain crossings."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from legwheel_dataset.bag_reader import (
    BagIndex,
    CalibrationTimeline,
    ImageMetadata,
    TelemetrySeries,
)
from legwheel_dataset.config import ProcessingConfig
from legwheel_dataset.patch_geometry import project_track_patch
from legwheel_dataset.processor import isolate_packets
from legwheel_dataset.run_analysis import CorrectedThetaTrajectory
from legwheel_dataset.synchronization import resample_channels
from legwheel_kinematics.imu_pitch import PitchTrajectory
from legwheel_kinematics.projection import CameraCalibration


INITIAL_ANGLE_RAD = 5.9


def accelerating_trajectory(direction):
    """Construct exact signed theta(t) = direction * (5.9 + t**2)."""
    ticks = np.arange(5, dtype=np.int64) * 1_000_000_000
    times = np.arange(401, dtype=np.int64) * 10_000_000
    angles = direction * (INITIAL_ANGLE_RAD + np.arange(5, dtype=float)**2)
    return CorrectedThetaTrajectory(
        tick_times_ns=ticks,
        raw_encoder_angles_rad=angles,
        theta_y_at_ticks_rad=angles,
        correction_factors=np.full(4, direction),
        predicted_interval_angles_rad=np.diff(direction * angles),
        encoder_interval_angles_rad=np.diff(angles),
        wheel_times_ns=times,
        wheel_velocity_rad_s=2.0 * times * 1e-9,
        radius_ratio=1.0,
        maximum_gap_ns=20_000_000,
        travel_direction=direction,
        rejected_reverse_transition_count=0,
    )


class PacketSamplingTest(unittest.TestCase):
    def test_patch_spacing_and_accelerating_local_windows_in_both_directions(self):
        for direction in (1.0, -1.0):
            with (
                self.subTest(direction=direction),
                tempfile.TemporaryDirectory() as root,
            ):
                self.check_packets(Path(root), direction)

    def check_packets(self, root, direction):
        trajectory = accelerating_trajectory(direction)
        times = trajectory.wheel_times_ns
        streams = {}
        for name, channels in (
            ("leg", (
                "hip.p", "hip.v", "hip.e", "knee_joint.position", "knee.v", "knee.e"
            )),
            ("wheel", ("wheel.p", "wheel.v", "wheel.e")),
            ("imu", tuple(f"imu.{index}" for index in range(6))),
        ):
            stream = TelemetrySeries(channels)
            for timestamp in times:
                stream.append(timestamp, [timestamp * 1e-9] * len(channels))
            streams[name] = stream
        # The first two frames are 150 ms apart, but their patches overlap.
        # Later, faster motion permits non-overlapping packets only 50 ms apart.
        image_times_s = [
            0.1, 0.25, 0.4, 0.45, 0.55, 0.65, 0.7, 0.8, 0.9, 1.0, 1.05, 3.95
        ]
        image_times = [int(round(time * 1e9)) for time in image_times_s]
        timeline = CalibrationTimeline()
        timeline.append(0, CameraCalibration(
            width=64, height=48,
            K=np.array([[46.18, 0.0, 32.42], [0.0, 46.17, 24.29], [0.0, 0.0, 1.0]]),
            D=np.zeros(5), distortion_model="plumb_bob",
        ), "camera_color_optical_frame")
        index = BagIndex(
            rgb_images=[ImageMetadata(time, 64, 48, "rgb8") for time in image_times],
            depth_images=[ImageMetadata(time, 64, 48, "16UC1") for time in image_times],
            rgb_calibrations=timeline, depth_calibrations=timeline, streams=streams,
        )
        pitch = PitchTrajectory(
            times_ns=times, theta_p_rad=np.full(times.size, 0.12),
            uncertainty_rad=np.zeros(times.size), valid=np.ones(times.size, dtype=bool),
            acceleration_rejected=np.zeros(times.size, dtype=bool), report={},
            _segment_ids=np.zeros(times.size, dtype=np.int64),
        )
        (root / "rosbag").mkdir()
        (root / "run_config.yaml").write_text("{}", encoding="utf-8")
        (root / "camera_driver_config_run_1_snapshot.yaml").write_text(
            "depth_registration: true\nalign_target_stream: COLOR\n", encoding="utf-8"
        )
        experiment = root / "experiment.yaml"
        experiment.write_text(json.dumps({
            "experiment_id": "sampling",
            "environment": {"terrain_and_light_ID": "TL001"},
            "rig_config": {"pivot_height_m": 0.353, "beam_radius_m": 0.83},
            "leg_config": {"hip_link_length_m": 0.2, "calf_link_length_m": 0.2},
            "wheel_config": {"wheel_radius_mm": 50.0, "wheel_width_mm": 100.0},
            "electronics_hardware": {
                "encoder_setup": {"theta_y_kinematic_sign": -direction},
                "camera": {"camera_config": {
                    "sensor_front_offset_m": 0.043,
                    "camera_beam_offset_m": 0.054,
                    "camera_height_m": 0.029,
                    "camera_front_angle_deg": 60.0,
                }},
            },
        }), encoding="utf-8")

        def project_candidate(*args):
            # This candidate is far enough from the last saved patch, but is
            # rejected. It must not become the reference for packet spacing.
            if math.isclose(args[1], -(INITIAL_ANGLE_RAD + 0.65**2), abs_tol=1e-12):
                raise ValueError("rgb_patch_not_fully_visible")
            return project_track_patch(*args)

        def provide_images(_directory, requests, *, progress=None):
            for (topic, _), callback in requests.items():
                rgb = topic.endswith("color/image_raw")
                array = np.zeros(
                    (48, 64, 3) if rgb else (48, 64),
                    dtype=np.uint8 if rgb else np.uint16,
                )
                callback(SimpleNamespace(
                    encoding="rgb8" if rgb else "16UC1", width=64, height=48,
                    is_bigendian=False, step=array.strides[0], data=array.tobytes(),
                ))
            if progress:
                progress(1.0, "Recovering packet images")
            return set(requests)

        output = root / "derived"
        progress_events = []
        with (
            patch("legwheel_dataset.processor.index_bag", return_value=index),
            patch("legwheel_dataset.processor._run_motion_analysis", return_value=(
                trajectory,
                {"corrected_motion": {"forward_recording": True},
                 "rate_limit_violations": [], "encoder_position": {}},
            )),
            patch("legwheel_dataset.processor._estimate_run_pitch", return_value=pitch),
            patch(
                "legwheel_dataset.processor.project_track_patch",
                side_effect=project_candidate,
            ),
            patch(
                "legwheel_dataset.processor.extract_selected_images",
                side_effect=provide_images,
            ),
        ):
            # A shorter FFT window must not reject a slow patch traversal.
            isolate_packets(
                root, ProcessingConfig(spectral_window_ms=20.0), experiment, output,
                progress=lambda fraction, label: progress_events.append((fraction, label)),
            )

        fractions = [fraction for fraction, _ in progress_events]
        self.assertEqual(fractions[-1], 1.0)
        self.assertEqual(fractions, sorted(fractions))
        provenance = json.loads((output / "provenance.json").read_text())
        self.assertEqual(provenance["terrain_and_light_ID"], "TL001")
        self.assertEqual(
            provenance["experiment_config"]["environment"]["terrain_and_light_ID"], "TL001",
        )
        self.assertFalse(provenance["requires_regeneration"])

        records = [
            json.loads(line)
            for line in (output / "manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [record["image_time_ns"] for record in records],
            [100_000_000, 400_000_000, 550_000_000, 700_000_000,
             800_000_000, 900_000_000, 1_000_000_000, 1_050_000_000],
        )
        centres = [
            direction * record["patch_centre_motion_angle_rad"] for record in records
        ]
        self.assertTrue(np.all(np.diff(centres) >= 0.1))
        self.assertLess(centres[0], 2 * math.pi)
        self.assertGreater(centres[-1], 2 * math.pi)
        previous_end = None
        for record in records:
            packet = output / record["packet_path"]
            metadata = json.loads((packet / "metadata.json").read_text())
            for name in ("class_id", "classification_method", "model_id"):
                self.assertIn(name, metadata)
                self.assertIsNone(metadata[name])
            centre = direction * metadata["patch_centre_motion_angle_rad"]
            expected_start_s = math.sqrt(centre - 0.05 - INITIAL_ANGLE_RAD)
            expected_end_s = math.sqrt(centre + 0.05 - INITIAL_ANGLE_RAD)
            start = metadata["local_start_time_ns"]
            end = metadata["local_end_time_ns"]
            contact = metadata["contact_time_ns"]
            self.assertAlmostEqual(start * 1e-9, expected_start_s, delta=1e-9)
            self.assertAlmostEqual(end * 1e-9, expected_end_s, delta=1e-9)
            self.assertGreater(contact - start, end - contact)
            self.assertAlmostEqual(metadata["local_window_ms"], (end - start) * 1e-6)
            if previous_end is not None:
                self.assertGreaterEqual(start, previous_end)
            previous_end = end
            with np.load(packet / "telemetry.npz") as telemetry:
                relative = telemetry["local_relative_times_ns"]
                self.assertEqual(int(relative[0]) + contact, start)
                self.assertEqual(int(relative[-1]) + contact, end)
                self.assertTrue(telemetry["local_valid"].all())
                # Every motor/IMU channel is a time ramp, so both interpolated
                # boundary values have an independent analytical expectation.
                np.testing.assert_allclose(
                    telemetry["local_values"][[0, -1]],
                    np.broadcast_to([[expected_start_s], [expected_end_s]], (2, 15)),
                    atol=1e-9, rtol=0.0,
                )
                np.testing.assert_array_equal(
                    telemetry["spectral_relative_times_ns"][[0, -1]],
                    [-10_000_000, 10_000_000],
                )
        report = json.loads((output / "packets" / "analysis_report.json").read_text())
        self.assertEqual(report["rejection_counts"], {
            "patch_overlaps_previous_packet": 2,
            "rgb_patch_not_fully_visible": 1,
            "target_angle_outside_corrected_theta_range": 1,
        })

    def test_crossing_times_include_run_endpoints_and_reject_unreached_angles(self):
        for direction in (1.0, -1.0):
            trajectory = accelerating_trajectory(direction)
            self.assertEqual(
                trajectory.crossing_time(direction * INITIAL_ANGLE_RAD), (0, 0.0)
            )
            self.assertEqual(
                trajectory.crossing_time(direction * (INITIAL_ANGLE_RAD + 16.0))[0],
                4_000_000_000,
            )
            for target in (INITIAL_ANGLE_RAD - 0.1, INITIAL_ANGLE_RAD + 16.1):
                with self.assertRaisesRegex(
                    ValueError, "outside_corrected_theta_range"
                ):
                    trajectory.crossing_time(direction * target)

    def test_resampling_preserves_asymmetric_endpoints_at_large_absolute_times(self):
        origin = 1_700_000_000_000_000_000
        times = origin + np.arange(11, dtype=np.int64) * 10_000_000
        values = (times - origin).astype(float) * 1e-9
        start = origin + 15_000_001
        centre = origin + 40_000_000
        end = origin + 55_000_003
        sampled, valid, relative = resample_channels(
            times, values, centre, start, end, rate_hz=100.0, maximum_gap_ms=20.0
        )
        np.testing.assert_array_equal(relative[[0, -1]], [start - centre, end - centre])
        np.testing.assert_allclose(sampled[[0, -1], 0], [0.015000001, 0.055000003])
        self.assertTrue(valid.all())

    def test_crossing_at_a_tick_finds_arrival_before_a_stationary_period(self):
        trajectory = replace(
            accelerating_trajectory(1.0),
            tick_times_ns=np.array([0, 1_000_000_000], dtype=np.int64),
            theta_y_at_ticks_rad=np.array([0.0, 1.0]),
            correction_factors=np.ones(1),
            wheel_times_ns=np.array([0, 500_000_000, 1_000_000_000], dtype=np.int64),
            wheel_velocity_rad_s=np.array([4.0, 0.0, 0.0]),
            maximum_gap_ns=500_000_000,
        )
        arrival, residual = trajectory.crossing_time(1.0)
        self.assertAlmostEqual(arrival * 1e-9, 0.5, delta=1e-8)
        self.assertAlmostEqual(residual, 0.0)

    def test_resampling_keeps_telemetry_gaps_invalid(self):
        times = np.array([0, 10_000_000, 100_000_000], dtype=np.int64)
        _, valid, _ = resample_channels(
            times, np.zeros(3), 50_000_000, 0, 100_000_000,
            rate_hz=100.0, maximum_gap_ms=20.0,
        )
        self.assertTrue(valid[[0, 1, -1]].all())
        self.assertFalse(valid[2:-1].any())
