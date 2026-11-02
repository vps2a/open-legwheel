"""Physical simulations for offline pitch; no ROS or recorded bag is required."""

import json
import math
import unittest
from dataclasses import replace

import numpy as np
from legwheel_kinematics.config import KinematicsConfig
from legwheel_kinematics.imu_pitch import GRAVITY_M_S2, estimate_pitch


def _geometry(camera_angle_deg=40.0):
    return KinematicsConfig(
        d1=0.353,
        a_b=0.83,
        h_b=0.0661,
        sensor_front_offset=0.043,
        camera_height=0.029,
        camera_beam_offset=0.054,
        camera_angle_deg=camera_angle_deg,
        Beam_radius=0.83,
        Wheel_width=0.12,
        wheel_radius_mm=110.0,
        hip_link_length_m=0.20,
        calf_link_length_m=0.20,
        theta_y_kinematic_sign=-1.0,
    )


def _simulate(seconds, pitch, yaw, geometry, pitch_bias=0.0, imu_to_camera=None):
    """Generate measurements by rotating world motion into the camera.

    World positions are differentiated numerically, independently of the
    estimator's centripetal/tangential correction formulas. This catches lever
    arm, lateral offset and acceleration-sign errors in that correction.
    """
    pitch = np.broadcast_to(pitch, seconds.shape)
    yaw = np.broadcast_to(yaw, seconds.shape)
    pitch_rate = np.gradient(pitch, seconds, edge_order=2)
    yaw_rate = np.gradient(yaw, seconds, edge_order=2)
    cosine, sine = np.cos(pitch), np.sin(pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    radial = geometry.a_b - geometry.sensor_front_offset
    height = geometry.h_b + geometry.camera_height
    lateral = geometry.camera_beam_offset
    horizontal_radius = radial * cosine - height * sine
    positions_world = np.column_stack(
        (
            horizontal_radius * cosine_yaw + lateral * sine_yaw,
            horizontal_radius * sine_yaw - lateral * cosine_yaw,
            radial * sine + height * cosine + geometry.d1,
        )
    )
    acceleration_world = np.gradient(
        np.gradient(positions_world, seconds, axis=0, edge_order=2),
        seconds,
        axis=0,
        edge_order=2,
    )
    acceleration_world[:, 2] += GRAVITY_M_S2
    angle = math.radians(geometry.camera_angle_deg)
    optical_to_beam = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -math.cos(angle), -math.sin(angle)],
            [0.0, -math.sin(angle), math.cos(angle)],
        ]
    )
    raw_to_optical = np.eye(3) if imu_to_camera is None else imu_to_camera
    gyro = np.empty((len(seconds), 3))
    acceleration = np.empty((len(seconds), 3))
    for sample in range(len(seconds)):
        beam_to_world = np.array(
            [
                [
                    cosine_yaw[sample] * cosine[sample],
                    -cosine_yaw[sample] * sine[sample],
                    sine_yaw[sample],
                ],
                [
                    sine_yaw[sample] * cosine[sample],
                    -sine_yaw[sample] * sine[sample],
                    -cosine_yaw[sample],
                ],
                [sine[sample], cosine[sample], 0.0],
            ]
        )
        optical_to_world = beam_to_world @ optical_to_beam
        angular_velocity_world = np.array(
            [
                sine_yaw[sample] * pitch_rate[sample],
                -cosine_yaw[sample] * pitch_rate[sample],
                yaw_rate[sample],
            ]
        )
        gyro_optical = optical_to_world.T @ angular_velocity_world
        gyro_optical += optical_to_beam.T @ np.array([0.0, 0.0, pitch_bias])
        gyro[sample] = raw_to_optical.T @ gyro_optical
        acceleration[sample] = (
            raw_to_optical.T @ optical_to_world.T @ acceleration_world[sample]
        )
    timestamps = np.rint(seconds * 1e9).astype(np.int64)
    return timestamps, gyro, acceleration


class ImuPitchTest(unittest.TestCase):
    def test_progress_is_monotonic_across_gaps_and_does_not_change_estimates(self):
        seconds = np.arange(0.0, 4.01, 0.01)
        seconds = seconds[(seconds < 1.8) | (seconds > 2.1)]
        geometry = _geometry()
        yaw = -0.4 * seconds
        timestamps, gyro, acceleration = _simulate(seconds, 0.1, yaw, geometry)
        arguments = (timestamps, gyro, acceleration, timestamps, yaw, geometry)
        reference = estimate_pitch(*arguments)
        events = []
        with_progress = estimate_pitch(
            *arguments, progress=lambda fraction, label: events.append((fraction, label)),
        )
        np.testing.assert_equal(with_progress.theta_p_rad, reference.theta_p_rad)
        np.testing.assert_equal(with_progress.uncertainty_rad, reference.uncertainty_rad)
        fractions = [fraction for fraction, _ in events]
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertEqual(fractions, sorted(fractions))

    def test_mounting_pitch_signs_and_10_or_15_second_circles(self):
        seconds = np.arange(0.0, 4.01, 0.01)
        for mount, pitch, revolution_s, direction in (
            (0.0, 0.15, 10.0, 1),
            (40.0, -0.12, 10.0, -1),
            (-25.0, 0.10, 15.0, -1),
            (90.0, -0.08, 15.0, 1),
        ):
            with self.subTest(mount=mount, pitch=pitch, revolution_s=revolution_s):
                geometry = _geometry(mount)
                yaw = direction * 2.0 * math.pi / revolution_s * seconds
                timestamps, gyro, acceleration = _simulate(
                    seconds, pitch, yaw, geometry
                )
                trajectory = estimate_pitch(
                    timestamps, gyro, acceleration, timestamps, yaw, geometry
                )
                self.assertAlmostEqual(
                    trajectory.sample(2_000_000_000), pitch, delta=math.radians(0.03)
                )
                self.assertGreater(int(trajectory.valid.sum()), 300)
                self.assertTrue(trajectory.report["circular_acceleration_compensated"])

    def test_tangential_acceleration_and_lateral_camera_offset(self):
        seconds = np.arange(0.0, 4.005, 0.005)
        geometry = replace(_geometry(57.0), camera_beam_offset=0.25)
        pitch = -0.18
        yaw = -0.20 * seconds - 0.15 * seconds**2
        timestamps, gyro, acceleration = _simulate(seconds, pitch, yaw, geometry)
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, geometry
        )
        errors = trajectory.theta_p_rad[trajectory.valid] - pitch
        self.assertGreater(len(errors), 500)
        self.assertLess(np.max(np.abs(errors)), math.radians(0.08))

    def test_stationary_bias_then_pitch_motion_and_impact_recovery(self):
        seconds = np.arange(0.0, 12.01, 0.01)
        geometry = _geometry()
        moving_time = np.clip(seconds - 2.0, 0.0, None)
        # Smoothly start circular motion after two recorded seconds of rest.
        yaw_rate = np.minimum(moving_time, 1.0) * -0.60
        yaw = np.cumsum(yaw_rate) * 0.01
        pitch = 0.10 + 0.04 * (1.0 - np.cos(0.40 * moving_time))
        bias = math.radians(1.0)
        timestamps, gyro, acceleration = _simulate(
            seconds, pitch, yaw, geometry, pitch_bias=bias
        )
        rng = np.random.default_rng(17)
        acceleration += rng.normal(0.0, 0.01, acceleration.shape)
        gyro += rng.normal(0.0, math.radians(0.04), gyro.shape)
        impact = (seconds >= 6.0) & (seconds <= 6.20)
        acceleration[impact] += [5.0, -12.0, 8.0]
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, geometry
        )
        self.assertGreater(trajectory.report["stationary_calibration_samples"], 50)
        self.assertAlmostEqual(
            trajectory.report["initial_gyro_bias_rad_s"], bias, delta=math.radians(0.02)
        )
        self.assertFalse(trajectory.valid[impact].any())
        self.assertTrue(trajectory.acceleration_rejected[impact].all())
        self.assertTrue(trajectory.available[impact].all())
        self.assertAlmostEqual(
            trajectory.sample(6_100_000_000), pitch[610], delta=math.radians(0.30)
        )
        with self.assertRaisesRegex(ValueError, "unreliable"):
            trajectory.sample(6_100_000_000, require_reliable=True)
        self.assertAlmostEqual(
            trajectory.sample(6_500_000_000), pitch[650], delta=math.radians(0.30)
        )
        retained_error = (
            trajectory.theta_p_rad[trajectory.valid] - pitch[trajectory.valid]
        )
        self.assertLess(np.max(np.abs(retained_error)), math.radians(1.0))
        json.dumps(trajectory.report, allow_nan=False)

    def test_slow_pitching_is_not_misidentified_as_stationary_bias(self):
        seconds = np.arange(0.0, 3.01, 0.01)
        pitch = 0.1 + math.radians(1.0) * seconds
        yaw = np.zeros_like(seconds)
        timestamps, gyro, acceleration = _simulate(seconds, pitch, yaw, _geometry())
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, _geometry()
        )
        self.assertEqual(trajectory.report["stationary_calibration_samples"], 0)
        self.assertAlmostEqual(
            trajectory.sample(2_000_000_000), pitch[200], delta=math.radians(0.1)
        )

    def test_gravity_initialization_accepts_pitch_change_explained_by_the_gyro(self):
        seconds = np.arange(0.0, 2.01, 0.01)
        pitch = 0.1 + math.radians(15.0) * seconds
        yaw = -0.6 * seconds
        geometry = _geometry(-28.0)
        timestamps, gyro, acceleration = _simulate(seconds, pitch, yaw, geometry)
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, geometry
        )
        self.assertEqual(trajectory.report["stationary_calibration_samples"], 0)
        self.assertAlmostEqual(
            trajectory.sample(600_000_000), pitch[60], delta=math.radians(0.5)
        )
        self.assertAlmostEqual(
            trajectory.sample(1_800_000_000), pitch[180], delta=math.radians(0.5)
        )

    def test_previous_stationary_bias_is_used_only_without_new_recorded_rest(self):
        seconds = np.arange(0.0, 2.01, 0.01)
        bias = math.radians(1.0)
        geometry = _geometry()
        yaw = -0.6 * seconds
        timestamps, gyro, acceleration = _simulate(
            seconds, 0.1, yaw, geometry, pitch_bias=bias
        )
        moving = estimate_pitch(
            timestamps,
            gyro,
            acceleration,
            timestamps,
            yaw,
            geometry,
            initial_gyro_bias_rad_s=bias,
            initial_gyro_bias_uncertainty_rad_s=math.radians(0.05),
        )
        self.assertEqual(
            moving.report["gyro_bias_source"], "previous_stationary_calibration"
        )
        self.assertIsNone(moving.report["stationary_calibration_end_time_ns"])
        self.assertAlmostEqual(
            moving.sample(1_000_000_000), 0.1, delta=math.radians(0.03)
        )

        yaw = np.zeros_like(seconds)
        timestamps, gyro, acceleration = _simulate(
            seconds, 0.1, yaw, geometry, pitch_bias=bias
        )
        resting = estimate_pitch(
            timestamps,
            gyro,
            acceleration,
            timestamps,
            yaw,
            geometry,
            initial_gyro_bias_rad_s=2.0 * bias,
            initial_gyro_bias_uncertainty_rad_s=math.radians(0.05),
        )
        self.assertEqual(resting.report["gyro_bias_source"], "stationary_measurements")
        self.assertIsNotNone(resting.report["stationary_calibration_end_time_ns"])
        self.assertAlmostEqual(resting.report["initial_gyro_bias_rad_s"], bias)
        for invalid_prior in (
            {"initial_gyro_bias_rad_s": bias},
            {"initial_gyro_bias_uncertainty_rad_s": 0.1},
            {
                "initial_gyro_bias_rad_s": bias,
                "initial_gyro_bias_uncertainty_rad_s": 0.0,
            },
        ):
            with self.assertRaises(ValueError):
                estimate_pitch(
                    timestamps,
                    gyro,
                    acceleration,
                    timestamps,
                    yaw,
                    geometry,
                    **invalid_prior,
                )

    def test_gap_and_nonfinite_row_are_not_interpolated_or_smoothed_across(self):
        seconds = np.concatenate(
            (np.arange(0.0, 1.01, 0.01), np.arange(1.30, 3.01, 0.01))
        )
        yaw = np.zeros_like(seconds)
        pitch = np.where(seconds <= 1.0, 0.1, -0.12)
        timestamps, gyro, acceleration = _simulate(seconds, pitch, yaw, _geometry())
        # Avoid interpreting the artificial position discontinuity as an impact;
        # these are separate stationary pieces of a missing-data recording.
        _, _, acceleration_first = _simulate(seconds, 0.1, yaw, _geometry())
        _, _, acceleration_second = _simulate(seconds, -0.12, yaw, _geometry())
        acceleration = np.where(
            (seconds <= 1.0)[:, None], acceleration_first, acceleration_second
        )
        gyro[:] = 0.0
        bad_index = int(np.argmin(np.abs(seconds - 2.0)))
        acceleration[bad_index, 0] = np.nan
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, _geometry()
        )
        for timestamp in (1_100_000_000, 2_000_000_000, 2_005_000_000):
            with (
                self.subTest(timestamp=timestamp),
                self.assertRaisesRegex(ValueError, "gap|initialization"),
            ):
                trajectory.sample(timestamp, require_reliable=False)
        self.assertAlmostEqual(
            trajectory.sample(800_000_000), 0.1, delta=math.radians(0.01)
        )
        self.assertAlmostEqual(
            trajectory.sample(2_800_000_000), -0.12, delta=math.radians(0.01)
        )
        self.assertEqual(trajectory.report["initialized_segments"], 3)
        with self.assertRaisesRegex(ValueError, "outside"):
            trajectory.sample(-1)
        with self.assertRaisesRegex(ValueError, "outside"):
            trajectory.uncertainty_at(4_000_000_000)

    def test_untrusted_initial_acceleration_does_not_define_orientation(self):
        seconds = np.arange(0.0, 2.01, 0.01)
        yaw = np.zeros_like(seconds)
        timestamps, gyro, acceleration = _simulate(seconds, 0.15, yaw, _geometry())
        acceleration[seconds < 0.5] = 0.0  # Free fall is not a gravity observation.
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, _geometry()
        )
        with self.assertRaisesRegex(ValueError, "initialization"):
            trajectory.sample(300_000_000, require_reliable=False)
        self.assertAlmostEqual(
            trajectory.sample(1_000_000_000), 0.15, delta=math.radians(0.01)
        )

    def test_calibrated_imu_rotation_and_epoch_timestamps(self):
        seconds = np.arange(0.0, 3.01, 0.01)
        yaw = -0.63 * seconds
        angle = math.radians(7.0)
        imu_rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        timestamps, gyro, acceleration = _simulate(
            seconds, -0.1, yaw, _geometry(), imu_to_camera=imu_rotation
        )
        relative = estimate_pitch(
            timestamps,
            gyro,
            acceleration,
            timestamps,
            yaw,
            _geometry(),
            imu_to_camera_rotation=imu_rotation,
        )
        epoch = np.int64(1_795_000_000_123_456_789)
        absolute = estimate_pitch(
            timestamps + epoch,
            gyro,
            acceleration,
            timestamps + epoch,
            yaw,
            _geometry(),
            imu_to_camera_rotation=imu_rotation,
        )
        np.testing.assert_array_equal(relative.valid, absolute.valid)
        np.testing.assert_allclose(
            relative.theta_p_rad, absolute.theta_p_rad, atol=1e-12, equal_nan=True
        )
        self.assertAlmostEqual(
            absolute.sample(int(epoch) + 1_234_567_890), -0.1, delta=math.radians(0.03)
        )
        self.assertLess(
            2.0 * absolute.uncertainty_at(int(epoch) + 1_234_567_890), math.radians(1.0)
        )

    def test_sensor_coverage_invalid_inputs_and_saturation_fail_closed(self):
        seconds = np.arange(0.0, 2.01, 0.01)
        yaw = np.zeros_like(seconds)
        timestamps, gyro, acceleration = _simulate(seconds, 0.0, yaw, _geometry())
        gyro[100] = math.radians(2000.0)
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps[20:-20], yaw[20:-20], _geometry()
        )
        for timestamp in (100_000_000, 1_000_000_000, 1_900_000_000):
            with self.assertRaises(ValueError):
                trajectory.sample(timestamp, require_reliable=False)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            estimate_pitch(
                timestamps[::-1], gyro, acceleration, timestamps, yaw, _geometry()
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            estimate_pitch(
                timestamps, gyro[:, :2], acceleration, timestamps, yaw, _geometry()
            )
        with self.assertRaisesRegex(ValueError, "rotation"):
            estimate_pitch(
                timestamps,
                gyro,
                acceleration,
                timestamps,
                yaw,
                _geometry(),
                imu_to_camera_rotation=np.diag([1, 1, -1]),
            )
        for keyword in (
            {"maximum_gap_s": 0.0},
            {"maximum_uncertainty_deg": float("nan")},
        ):
            with self.assertRaises(ValueError):
                estimate_pitch(
                    timestamps,
                    gyro,
                    acceleration,
                    timestamps,
                    yaw,
                    _geometry(),
                    **keyword,
                )

    def test_empty_duplicate_and_noninteger_timestamps_are_rejected(self):
        seconds = np.arange(0.0, 1.01, 0.01)
        yaw = np.zeros_like(seconds)
        timestamps, gyro, acceleration = _simulate(seconds, 0.0, yaw, _geometry())
        for bad_timestamps in (
            np.array([], dtype=np.int64),
            np.zeros_like(timestamps),
            timestamps.astype(float),
            timestamps[:, None],
        ):
            with self.assertRaises(ValueError):
                estimate_pitch(
                    bad_timestamps,
                    gyro,
                    acceleration,
                    timestamps,
                    yaw,
                    _geometry(),
                )

    def test_no_usable_gravity_produces_an_explicit_unavailable_trajectory(self):
        seconds = np.arange(0.0, 1.01, 0.01)
        timestamps = np.rint(seconds * 1e9).astype(np.int64)
        zeros = np.zeros((len(timestamps), 3))
        trajectory = estimate_pitch(
            timestamps,
            zeros,
            zeros,
            timestamps,
            np.zeros(len(timestamps)),
            _geometry(),
        )
        self.assertEqual(trajectory.report["status"], "no_pitch")
        self.assertFalse(trajectory.valid.any())
        self.assertTrue(np.isnan(trajectory.theta_p_rad).all())
        with self.assertRaisesRegex(ValueError, "initialization"):
            trajectory.sample(500_000_000, require_reliable=False)

    def test_high_uncertainty_is_retained_without_inventing_missing_estimates(self):
        seconds = np.arange(0.0, 2.01, 0.01)
        yaw = np.zeros_like(seconds)
        timestamps, gyro, acceleration = _simulate(seconds, 0.1, yaw, _geometry())
        trajectory = estimate_pitch(
            timestamps, gyro, acceleration, timestamps, yaw, _geometry(),
            maximum_uncertainty_deg=0.1,
        )
        self.assertFalse(trajectory.valid.any())
        self.assertTrue(trajectory.available.any())
        self.assertEqual(trajectory.report["status"], "available")
        self.assertAlmostEqual(trajectory.sample(1_000_000_000), 0.1)
        self.assertGreater(trajectory.uncertainty_at(1_000_000_000), 0.0)
        with self.assertRaisesRegex(ValueError, "unreliable"):
            trajectory.sample(1_000_000_000, require_reliable=True)
        with self.assertRaisesRegex(ValueError, "initialization"):
            trajectory.sample(0)


if __name__ == "__main__":
    unittest.main()
