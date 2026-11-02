"""Shared beam pitch estimation from timestamped camera IMU measurements.

The mechanism has one pitch degree of freedom, so a two-state estimator
(``pitch, pitch_gyro_bias``) is sufficient.  The gyro follows short movements;
gravity corrects its drift whenever the accelerometer is believable.  A backward
Kalman smoother uses the rest of the recording to improve those estimates.

This is deliberately independent of ROS and of the leg/knee geometry.  Its
reported uncertainty is a *model* confidence bound, not a guarantee of absolute
accuracy: camera mounting, timestamps and IMU calibration still need validation
against a measured reference before asserting a +/-1 degree dataset accuracy.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from legwheel_kinematics.config import KinematicsConfig
from legwheel_kinematics.transforms import transformation_matrices_from_config

PITCH_ESTIMATOR_VERSION = 3
GRAVITY_M_S2 = 9.80665

# These conservative defaults describe uncertainty, rather than claiming a
# factory calibration.  In particular, the floor prevents hundreds of highly
# correlated accelerometer readings from implying arbitrarily exact mounting.
_ANGLE_NOISE_RAD = math.radians(1.0)
_GYRO_NOISE_RAD_SQRT_S = math.radians(0.20)
_BIAS_WALK_RAD_S_SQRT_S = math.radians(0.03)
_MOUNTING_UNCERTAINTY_RAD = math.radians(0.25)
_INITIAL_GRAVITY_WINDOW_S = 0.20
_RECOVERY_TIME_S = 0.10
_ENCODER_HALF_WINDOW_S = 0.60


@dataclass
class PitchTrajectory:
    """Pitch at IMU acquisition times, with explicit unavailable intervals.

    ``uncertainty_rad`` is one standard deviation, including the conservative
    mounting floor. ``valid`` requires a two-standard-deviation bound below the
    requested limit and a recovered, accepted accelerometer observation.  A
    finite estimate during an impact remains available for packets and
    visualization. Confidence describes that estimate; it does not decide
    whether to keep it. Only genuinely unavailable intervals are omitted.
    """

    times_ns: NDArray[np.int64]
    theta_p_rad: NDArray[np.float64]
    uncertainty_rad: NDArray[np.float64]
    valid: NDArray[np.bool_]
    acceleration_rejected: NDArray[np.bool_]
    report: dict[str, Any]
    _segment_ids: NDArray[np.int64] = field(repr=False)

    @property
    def available(self) -> NDArray[np.bool_]:
        """Distinguish an uncertain estimate from one that does not exist.

        Gyro propagation can provide pitch through rejected accelerometer
        observations. Startup, missing measurements and separate time segments
        still have no supported estimate and must not be invented or bridged.
        """
        return (self._segment_ids >= 0) & np.isfinite(self.theta_p_rad)

    def _bracket(self, timestamp_ns: int) -> tuple[int, int, float]:
        """Find real bracketing measurements; never clamp or cross a gap."""
        timestamp_ns = int(timestamp_ns)
        upper = int(np.searchsorted(self.times_ns, timestamp_ns))
        if upper < len(self.times_ns) and self.times_ns[upper] == timestamp_ns:
            lower = upper
            fraction = 0.0
        else:
            lower = upper - 1
            if lower < 0 or upper >= len(self.times_ns):
                raise ValueError(
                    "IMU pitch is unavailable outside recorded IMU coverage."
                )
            fraction = (timestamp_ns - int(self.times_ns[lower])) / (
                int(self.times_ns[upper]) - int(self.times_ns[lower])
            )
        if (
            self._segment_ids[lower] < 0
            or self._segment_ids[lower] != self._segment_ids[upper]
            or not np.isfinite(self.theta_p_rad[[lower, upper]]).all()
        ):
            raise ValueError(
                "IMU pitch is unavailable across a data gap or before initialization."
            )
        return lower, upper, fraction

    def sample(self, timestamp_ns: int, require_reliable: bool = False) -> float:
        """Sample an existing estimate regardless of its confidence by default.

        Strict confidence gating remains an explicit diagnostic option. Packet
        generation and visualizers retain uncertain estimates and record their
        uncertainty instead; both modes still enforce actual data coverage.
        """
        lower, upper, fraction = self._bracket(timestamp_ns)
        if require_reliable and not (self.valid[lower] and self.valid[upper]):
            raise ValueError(
                "IMU pitch is unreliable at this timestamp (impact, recovery or uncertainty)."
            )
        return float(
            self.theta_p_rad[lower]
            + fraction * (self.theta_p_rad[upper] - self.theta_p_rad[lower])
        )

    def uncertainty_at(self, timestamp_ns: int) -> float:
        """Return one-standard-deviation uncertainty without relaxing gap checks."""
        lower, upper, fraction = self._bracket(timestamp_ns)
        return float(
            self.uncertainty_rad[lower]
            + fraction * (self.uncertainty_rad[upper] - self.uncertainty_rad[lower])
        )


def _timestamps(values: Any, name: str) -> NDArray[np.int64]:
    """Keep nanoseconds as integers: epoch-sized floats lose timing precision."""
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer nanosecond array.")
    values = values.astype(np.int64, copy=False)
    if values.size < 3 or np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{name} must contain at least three strictly increasing timestamps."
        )
    return values


def _segments(
    times_ns: NDArray[np.int64],
    usable: NDArray[np.bool_],
    maximum_gap_s: float,
) -> list[NDArray[np.int64]]:
    """Split before missing rows as well as before long time gaps."""
    indices = np.flatnonzero(usable)
    if not indices.size:
        return []
    boundaries = (
        np.flatnonzero(
            (np.diff(indices) != 1)
            | (np.diff(times_ns[indices]) > round(maximum_gap_s * 1e9))
        )
        + 1
    )
    return list(np.split(indices, boundaries))


def _encoder_motion(
    encoder_times_ns: NDArray[np.int64],
    encoder_angles_rad: NDArray[np.float64],
    imu_times_ns: NDArray[np.int64],
    *,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Differentiate a local encoder fit, rather than a quantized tick staircase.

    A weighted quadratic over 1.2 seconds preserves constant yaw speed and
    acceleration exactly.  One robust reweighting reduces the influence of
    isolated encoder outliers.  Both derivatives use this *same* fitted motion;
    differencing a noisy velocity again would amplify tick noise unnecessarily.
    Fits and interpolation stay inside each contiguous encoder segment.
    """
    median_period_s = float(np.median(np.diff(encoder_times_ns))) * 1e-9
    # Encoder messages can be slower than the IMU. Five missing nominal samples
    # or 250 ms is a discontinuity, rather than permission to bridge a long stop
    # in recording. Repeated, unchanged counts still establish stationary data.
    maximum_encoder_gap_s = max(0.25, 5.0 * median_period_s)
    yaw_rate = np.full(len(imu_times_ns), np.nan)
    yaw_acceleration = np.full(len(imu_times_ns), np.nan)
    for indices in _segments(
        encoder_times_ns, np.isfinite(encoder_angles_rad), maximum_encoder_gap_s
    ):
        if len(indices) < 3:
            continue
        times_ns = encoder_times_ns[indices]
        # Subtract while still integer, before converting a short time interval.
        times_s = (times_ns - times_ns[0]).astype(float) * 1e-9
        angles = encoder_angles_rad[indices]
        derivatives = np.empty((len(indices), 2))
        for sample, center_s in enumerate(times_s):
            if progress and sample % 100 == 0:
                progress(
                    (int(indices[sample]) + 1) / len(encoder_times_ns),
                    "Estimating IMU pitch: encoder motion",
                )
            first = int(np.searchsorted(times_s, center_s - _ENCODER_HALF_WINDOW_S))
            last = int(
                np.searchsorted(
                    times_s, center_s + _ENCODER_HALF_WINDOW_S, side="right"
                )
            )
            # Sparse data still needs three actual observations for a quadratic.
            first = max(0, min(first, len(indices) - 3))
            last = min(len(indices), max(last, first + 3))
            offsets = times_s[first:last] - center_s
            design = np.column_stack((np.ones(len(offsets)), offsets, offsets**2))
            # Smooth nonzero endpoint weights avoid ill-conditioned three-point
            # fits while retaining the local nature of a denser recording.
            weights = np.exp(-0.5 * (offsets / (_ENCODER_HALF_WINDOW_S / 2.0)) ** 2)
            local_angles = angles[first:last] - angles[sample]
            for _ in range(2):
                square_root_weights = np.sqrt(weights)
                coefficients = np.linalg.lstsq(
                    design * square_root_weights[:, None],
                    local_angles * square_root_weights,
                    rcond=None,
                )[0]
                residual = local_angles - design @ coefficients
                scale = max(1e-5, 1.4826 * float(np.median(np.abs(residual))))
                weights *= np.minimum(
                    1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-12)
                )
            derivatives[sample] = coefficients[1], 2.0 * coefficients[2]
        inside = (imu_times_ns >= times_ns[0]) & (imu_times_ns <= times_ns[-1])
        target_s = (imu_times_ns[inside] - times_ns[0]).astype(float) * 1e-9
        yaw_rate[inside] = np.interp(target_s, times_s, derivatives[:, 0])
        yaw_acceleration[inside] = np.interp(target_s, times_s, derivatives[:, 1])
    return yaw_rate, yaw_acceleration, maximum_encoder_gap_s


def _circular_acceleration(
    pitch_rad: float,
    yaw_rate_rad_s: float,
    yaw_acceleration_rad_s2: float,
    camera_position_beam_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return yaw-induced camera acceleration, expressed in the pitched beam.

    T12 puts beam +Z opposite the positive yaw tangent. Thus a positive lateral
    camera offset has a *negative* tangential coordinate in the base frame.
    Keeping this sign explicit also includes tangential acceleration at starts
    and stops. Pitch-induced accelerations and impacts are handled by rejection,
    rather than treating their direction as gravity.
    """
    cosine, sine = math.cos(pitch_rad), math.sin(pitch_rad)
    radial, height, lateral = camera_position_beam_m
    horizontal_radius = radial * cosine - height * sine
    radial_acceleration = (
        -(yaw_rate_rad_s**2) * horizontal_radius + yaw_acceleration_rad_s2 * lateral
    )
    tangential_acceleration = (
        yaw_acceleration_rad_s2 * horizontal_radius + yaw_rate_rad_s**2 * lateral
    )
    return np.array(
        [
            cosine * radial_acceleration,
            -sine * radial_acceleration,
            -tangential_acceleration,
        ]
    )


def _gravity_observation(
    acceleration_beam: NDArray[np.float64],
    predicted_pitch: float,
    yaw_rate: float,
    yaw_acceleration: float,
    camera_position: NDArray[np.float64],
) -> tuple[float, bool, float]:
    """Remove circular acceleration and assess whether gravity is believable."""
    gravity = acceleration_beam - _circular_acceleration(
        predicted_pitch, yaw_rate, yaw_acceleration, camera_position
    )
    magnitude_error = abs(float(np.linalg.norm(gravity)) - GRAVITY_M_S2)
    off_plane_acceleration = abs(float(gravity[2]))
    # A free fall, hard impact or unmodelled lateral acceleration must not turn
    # into an angle correction. The innovation test in the filter additionally
    # catches in-plane acceleration that happens to preserve acceleration norm.
    believable = (
        magnitude_error < 0.08 * GRAVITY_M_S2
        and off_plane_acceleration < 0.05 * GRAVITY_M_S2
    )
    angle = math.atan2(float(gravity[0]), float(gravity[1]))
    quality_error = max(magnitude_error, off_plane_acceleration) / GRAVITY_M_S2
    return angle, believable, quality_error


def _stationary_bias(
    times_ns: NDArray[np.int64],
    gyro_beam: NDArray[np.float64],
    acceleration_beam: NDArray[np.float64],
    encoder_times_ns: NDArray[np.int64],
    encoder_angles_rad: NDArray[np.float64],
    segments: list[NDArray[np.int64]],
    *,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[float, float, NDArray[np.bool_]]:
    """Find actual recorded rest, rather than assuming the first sample is rest.

    The central encoder, all three gyro axes, acceleration magnitude and its
    variation must agree over at least half a second. A robust median tolerates
    occasional vibrations in an otherwise stationary calibration interval.
    """
    stationary = np.zeros(len(times_ns), dtype=bool)
    for indices in segments:
        seconds = (times_ns[indices] - times_ns[indices[0]]).astype(float) * 1e-9
        for local_index, center_s in enumerate(seconds):
            if progress and local_index % 100 == 0:
                progress(
                    (int(indices[local_index]) + 1) / len(times_ns),
                    "Estimating IMU pitch: stationary calibration",
                )
            first = int(np.searchsorted(seconds, center_s - 0.30))
            last = int(np.searchsorted(seconds, center_s + 0.30, side="right"))
            window = indices[first:last]
            if (
                len(window) < 10
                or (times_ns[window[-1]] - times_ns[window[0]]) < 500_000_000
            ):
                continue
            acceleration = acceleration_beam[window]
            gyro = gyro_beam[window]
            window_s = (times_ns[window] - times_ns[window[0]]).astype(float) * 1e-9
            centered_s = window_s - np.mean(window_s)
            gravity_angles = np.unwrap(
                np.arctan2(acceleration[:, 0], acceleration[:, 1])
            )
            gravity_angle_rate = float(
                centered_s @ gravity_angles / (centered_s @ centered_s)
            )
            encoder_first = int(np.searchsorted(encoder_times_ns, times_ns[window[0]]))
            encoder_last = int(
                np.searchsorted(encoder_times_ns, times_ns[window[-1]], side="right")
            )
            encoder_window = encoder_angles_rad[encoder_first:encoder_last]
            if (
                len(encoder_window) >= 2
                and np.isfinite(encoder_window).all()
                and np.ptp(encoder_window) < 0.005
                and np.max(np.linalg.norm(gyro, axis=1)) < math.radians(3.0)
                and np.max(np.abs(np.linalg.norm(acceleration, axis=1) - GRAVITY_M_S2))
                < 0.04 * GRAVITY_M_S2
                and np.max(np.std(acceleration, axis=0)) < 0.015 * GRAVITY_M_S2
                and np.max(np.std(gyro, axis=0)) < math.radians(0.30)
                # A quiet, slowly pitching beam is not a stationary gyro. Its
                # gravity direction must agree with a negligible pitch rate.
                and abs(gravity_angle_rate) < math.radians(0.15)
            ):
                stationary[indices[local_index]] = True
    if not stationary.any():
        # The filter can still initialize gravity during steady movement, but
        # it must learn gyro bias instead of silently assuming a calibrated zero.
        return 0.0, math.radians(1.0), stationary
    values = gyro_beam[stationary, 2]
    bias = float(np.median(values))
    spread = 1.4826 * float(np.median(np.abs(values - bias)))
    # Keep a systematic floor: nearby samples are correlated, and the sensor
    # bias can change after the stationary interval as temperature changes.
    bias_uncertainty = max(math.radians(0.05), spread)
    return bias, bias_uncertainty, stationary


def estimate_pitch(
    imu_times_ns: Any,
    angular_velocity_rad_s: Any,
    linear_acceleration_m_s2: Any,
    encoder_times_ns: Any,
    encoder_angles_rad: Any,
    geometry: KinematicsConfig,
    *,
    maximum_gap_s: float = 0.05,
    maximum_uncertainty_deg: float = 1.0,
    imu_to_camera_rotation: Any = None,
    initial_gyro_bias_rad_s: float | None = None,
    initial_gyro_bias_uncertainty_rad_s: float | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> PitchTrajectory:
    """Fuse a full recorded IMU stream into the existing signed ``theta_p``.

    Encoder angles must already be unwrapped and converted to the kinematic yaw
    sign. ``imu_to_camera_rotation`` maps recorded IMU vectors into the camera
    optical frame; the default uses the Gemini 330 series' nominal matching
    right/down/forward axes. A supplied calibrated rotation must be a proper
    orthonormal matrix. The configured physical camera angle is then removed
    with exactly the same T45 rotation used for image projection.

    A streaming caller may preserve a previously measured stationary bias with
    the two optional prior values. Actual rest in this window takes precedence.
    This keeps a bounded live buffer from forgetting its recorded calibration.

    No estimate is invented before gravity initialization, across missing data,
    or outside either sensor's coverage. The original knee-based pitch is never
    consulted. Use ``PitchTrajectory.sample`` to enforce these constraints when
    selecting dataset frames.

    An optional progress callback receives a work fraction from 0 to 1 and the
    current phase. It only reports work; it never affects the numerical result.
    """
    if progress:
        progress(0.0, "Estimating IMU pitch: encoder motion")
    times = _timestamps(imu_times_ns, "imu_times_ns")
    encoder_times = _timestamps(encoder_times_ns, "encoder_times_ns")
    gyro = np.asarray(angular_velocity_rad_s, dtype=float)
    acceleration = np.asarray(linear_acceleration_m_s2, dtype=float)
    encoder_angles = np.asarray(encoder_angles_rad, dtype=float)
    if gyro.shape != (len(times), 3) or acceleration.shape != (len(times), 3):
        raise ValueError(
            "IMU angular velocity and acceleration must have shape (number of timestamps, 3)."
        )
    if encoder_angles.shape != encoder_times.shape:
        raise ValueError(
            "Encoder angles must have one value for each encoder timestamp."
        )
    if not math.isfinite(maximum_gap_s) or maximum_gap_s <= 0.0:
        raise ValueError("maximum_gap_s must be finite and positive.")
    if not math.isfinite(maximum_uncertainty_deg) or maximum_uncertainty_deg <= 0.0:
        raise ValueError("maximum_uncertainty_deg must be finite and positive.")
    if (initial_gyro_bias_rad_s is None) != (
        initial_gyro_bias_uncertainty_rad_s is None
    ):
        raise ValueError("Supply both gyro bias prior values, or neither.")
    if initial_gyro_bias_rad_s is not None and (
        not math.isfinite(initial_gyro_bias_rad_s)
        or not math.isfinite(initial_gyro_bias_uncertainty_rad_s)
        or initial_gyro_bias_uncertainty_rad_s <= 0.0
    ):
        raise ValueError("Gyro bias prior must be finite with positive uncertainty.")
    rotation = (
        np.eye(3)
        if imu_to_camera_rotation is None
        else np.asarray(imu_to_camera_rotation, dtype=float)
    )
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("imu_to_camera_rotation must be a finite proper 3x3 rotation.")

    # The optical-to-beam mount transform includes the front-angle adjustment
    # and the optical axis flip. Do not subtract an Euler camera angle later:
    # it would double-correct the mount or use the wrong rotation axis.
    optical_to_beam = transformation_matrices_from_config(geometry, theta_p=0.0)[-1][
        :3, :3
    ]
    imu_to_beam = optical_to_beam @ rotation
    gyro_beam = gyro @ imu_to_beam.T
    acceleration_beam = acceleration @ imu_to_beam.T
    camera_position = np.array(
        [
            geometry.a_b - geometry.sensor_front_offset,
            geometry.h_b + geometry.camera_height,
            geometry.camera_beam_offset,
        ]
    )
    if not np.isfinite(camera_position).all():
        raise ValueError("Camera lever-arm geometry must be finite.")
    yaw_rate, yaw_acceleration, encoder_maximum_gap_s = _encoder_motion(
        encoder_times, encoder_angles, times,
        progress=(lambda fraction, label: progress(0.20 * fraction, label))
        if progress else None,
    )
    usable = (
        np.isfinite(gyro_beam).all(axis=1)
        & np.isfinite(acceleration_beam).all(axis=1)
        & np.isfinite(yaw_rate)
        & np.isfinite(yaw_acceleration)
        # Saturated/implausible gyro data cannot safely bridge an impact.
        & (np.max(np.abs(gyro), axis=1) < math.radians(1000.0))
    )
    segments = _segments(times, usable, maximum_gap_s)
    if progress:
        progress(0.20, "Estimating IMU pitch: stationary calibration")
    bias, bias_uncertainty, stationary = _stationary_bias(
        times, gyro_beam, acceleration_beam, encoder_times, encoder_angles, segments,
        progress=(lambda fraction, label: progress(0.20 + 0.20 * fraction, label))
        if progress else None,
    )
    bias_source = "stationary_measurements" if stationary.any() else "estimated"
    if not stationary.any() and initial_gyro_bias_rad_s is not None:
        bias = float(initial_gyro_bias_rad_s)
        bias_uncertainty = float(initial_gyro_bias_uncertainty_rad_s)
        bias_source = "previous_stationary_calibration"
    pitch = np.full(len(times), np.nan)
    uncertainty = np.full(len(times), np.nan)
    rejected = np.ones(len(times), dtype=bool)
    recovered = np.zeros(len(times), dtype=bool)
    segment_ids = np.full(len(times), -1, dtype=np.int64)
    initialized_segments = 0

    # Count both passes so progress stays monotonic across missing-data segments.
    total_work = max(1, 2 * sum(len(indices) for indices in segments))
    completed_work = 0
    if progress:
        progress(0.40, "Estimating IMU pitch: filtering")

    for indices in segments:
        filtered_states = np.full((len(indices), 2), np.nan)
        filtered_covariances = np.full((len(indices), 2, 2), np.nan)
        predicted_states = np.full((len(indices), 2), np.nan)
        predicted_covariances = np.full((len(indices), 2, 2), np.nan)
        transitions = np.zeros((len(indices), 2, 2))
        state = None
        covariance = None
        initialization_start = None
        initialization_gyro_angle = 0.0
        initialization_residual_angles: list[float] = []
        accepted_since = None
        first_initialized = None

        for local_index, sample in enumerate(indices):
            if progress and local_index % 100 == 0:
                progress(
                    0.40 + 0.60 * (completed_work + local_index) / total_work,
                    "Estimating IMU pitch: filtering",
                )
            timestamp = int(times[sample])
            if state is None:
                # Circular compensation depends weakly on pitch itself. A few
                # fixed-point steps solve that small coupling before the first
                # reliable gravity observation, even if recording starts moving.
                initial_angle = math.atan2(
                    acceleration_beam[sample, 0], acceleration_beam[sample, 1]
                )
                for _ in range(4):
                    initial_angle, believable, _ = _gravity_observation(
                        acceleration_beam[sample],
                        initial_angle,
                        yaw_rate[sample],
                        yaw_acceleration[sample],
                        camera_position,
                    )
                if not believable:
                    initialization_start = None
                    initialization_gyro_angle = 0.0
                    initialization_residual_angles.clear()
                    continue
                if initialization_start is None:
                    initialization_start = timestamp
                else:
                    previous = indices[local_index - 1]
                    sample_period_s = (timestamp - int(times[previous])) * 1e-9
                    initialization_gyro_angle += sample_period_s * (
                        0.5 * (gyro_beam[previous, 2] + gyro_beam[sample, 2]) - bias
                    )
                initialization_residual_angles.append(
                    initial_angle - initialization_gyro_angle
                )
                elapsed_s = (timestamp - initialization_start) * 1e-9
                # Gravity must evolve consistently with the gyro over a short
                # history. Requiring a constant angle instead would prevent a
                # moving visualization window from initializing after its first
                # rest interval expires, even during clean, measurable pitching.
                if np.ptp(np.unwrap(initialization_residual_angles)) > math.radians(
                    2.0
                ):
                    initialization_start = timestamp
                    initialization_gyro_angle = 0.0
                    initialization_residual_angles = [initial_angle]
                    continue
                if elapsed_s < _INITIAL_GRAVITY_WINDOW_S:
                    continue
                state = np.array([initial_angle, bias])
                covariance = np.diag([_ANGLE_NOISE_RAD**2, bias_uncertainty**2])
                first_initialized = local_index
                initialized_segments += 1
                accepted_since = timestamp
                predicted_states[local_index] = state
                predicted_covariances[local_index] = covariance
                transitions[local_index] = np.eye(2)
            else:
                previous = indices[local_index - 1]
                elapsed_s = (timestamp - int(times[previous])) * 1e-9
                transition = np.array([[1.0, -elapsed_s], [0.0, 1.0]])
                # Trapezoidal gyro integration respects the full IMU sample
                # times; image cadence is deliberately absent from this step.
                measured_pitch_rate = 0.5 * (
                    gyro_beam[previous, 2] + gyro_beam[sample, 2]
                )
                state = transition @ state + np.array(
                    [elapsed_s * measured_pitch_rate, 0.0]
                )
                bias_diffusion = _BIAS_WALK_RAD_S_SQRT_S**2
                process_noise = np.array(
                    [
                        [
                            _GYRO_NOISE_RAD_SQRT_S**2 * elapsed_s
                            + bias_diffusion * elapsed_s**3 / 3.0,
                            -bias_diffusion * elapsed_s**2 / 2.0,
                        ],
                        [
                            -bias_diffusion * elapsed_s**2 / 2.0,
                            bias_diffusion * elapsed_s,
                        ],
                    ]
                )
                covariance = transition @ covariance @ transition.T + process_noise
                transitions[local_index] = transition
                predicted_states[local_index] = state
                predicted_covariances[local_index] = covariance

            observed_angle, believable, quality_error = _gravity_observation(
                acceleration_beam[sample],
                float(state[0]),
                yaw_rate[sample],
                yaw_acceleration[sample],
                camera_position,
            )
            # Wrap the innovation rather than the state: a harmless +/-pi angle
            # boundary must not produce an apparent full revolution correction.
            innovation = math.atan2(
                math.sin(observed_angle - state[0]), math.cos(observed_angle - state[0])
            )
            innovation_limit = max(
                math.radians(3.0), 3.0 * math.sqrt(float(covariance[0, 0]))
            )
            believable = believable and abs(innovation) < innovation_limit
            # A long rejected interval deliberately stays unavailable, even if
            # its gyro estimate looks smooth. We never force an inconsistent
            # accelerometer back into the filter merely to regain frame coverage.
            if believable:
                # Downweight small remaining nongravitational accelerations;
                # stronger departures are rejected by the checks above.
                observation_variance = _ANGLE_NOISE_RAD**2 * (
                    1.0 + (quality_error / 0.015) ** 2
                )
                gain = covariance[:, 0] / (covariance[0, 0] + observation_variance)
                state = state + gain * innovation
                # Joseph form retains a positive, symmetric covariance through
                # long recordings despite floating-point rounding.
                correction = np.eye(2) - np.outer(gain, [1.0, 0.0])
                covariance = (
                    correction @ covariance @ correction.T
                    + observation_variance * np.outer(gain, gain)
                )
                rejected[sample] = False
                if accepted_since is None:
                    accepted_since = timestamp
                recovered[sample] = (
                    timestamp - accepted_since
                ) * 1e-9 >= _RECOVERY_TIME_S
            else:
                accepted_since = None
            filtered_states[local_index] = state
            filtered_covariances[local_index] = covariance

        completed_work += len(indices)
        if first_initialized is None:
            completed_work += len(indices)
            continue
        # Rauch--Tung--Striebel smoothing is just a backward Kalman pass. It
        # stops at every missing-data boundary and never fabricates a reference
        # for samples preceding the trusted gravity initialization above.
        smoothed_states = filtered_states.copy()
        smoothed_covariances = filtered_covariances.copy()
        for local_index in range(len(indices) - 2, first_initialized - 1, -1):
            if progress and local_index % 100 == 0:
                progress(
                    0.40 + 0.60 * (
                        completed_work + len(indices) - 1 - local_index
                    ) / total_work,
                    "Estimating IMU pitch: smoothing",
                )
            next_index = local_index + 1
            smoother_gain = np.linalg.solve(
                predicted_covariances[next_index],
                transitions[next_index] @ filtered_covariances[local_index],
            ).T
            smoothed_states[local_index] += smoother_gain @ (
                smoothed_states[next_index] - predicted_states[next_index]
            )
            smoothed_covariances[local_index] += (
                smoother_gain
                @ (smoothed_covariances[next_index] - predicted_covariances[next_index])
                @ smoother_gain.T
            )
        supported = indices[first_initialized:]
        pitch[supported] = smoothed_states[first_initialized:, 0]
        variance = np.maximum(0.0, smoothed_covariances[first_initialized:, 0, 0])
        uncertainty[supported] = np.sqrt(variance + _MOUNTING_UNCERTAINTY_RAD**2)
        segment_ids[supported] = initialized_segments - 1
        completed_work += len(indices)

    if progress:
        progress(1.0, "IMU pitch estimation complete")

    valid = (
        recovered
        & ~rejected
        & np.isfinite(pitch)
        & (2.0 * uncertainty <= math.radians(maximum_uncertainty_deg))
    )
    report = {
        "estimator_version": PITCH_ESTIMATOR_VERSION,
        "method": "pitch_and_gyro_bias_kalman_filter_with_offline_RTS_smoothing",
        "imu_samples": len(times),
        "usable_samples": int(usable.sum()),
        "uninitialized_or_missing_samples": int(np.count_nonzero(~np.isfinite(pitch))),
        "initialized_segments": initialized_segments,
        "reliable_samples": int(valid.sum()),
        "available_samples": int(np.count_nonzero(np.isfinite(pitch))),
        "low_confidence_samples": int(np.count_nonzero(np.isfinite(pitch) & ~valid)),
        "status": "available" if np.isfinite(pitch).any() else "no_pitch",
        "uncertainty_rejects_packets": False,
        "acceleration_rejected_samples": int((rejected & usable).sum()),
        "stationary_calibration_samples": int(stationary.sum()),
        "stationary_calibration_end_time_ns": (
            int(times[stationary][-1]) if stationary.any() else None
        ),
        "gyro_bias_source": bias_source,
        "gyro_bias_walk_rad_s_sqrt_s": _BIAS_WALK_RAD_S_SQRT_S,
        "initial_gyro_bias_rad_s": bias,
        "initial_gyro_bias_uncertainty_rad_s": bias_uncertainty,
        "maximum_imu_gap_s": maximum_gap_s,
        "maximum_encoder_gap_s": encoder_maximum_gap_s,
        "maximum_two_sigma_uncertainty_deg": maximum_uncertainty_deg,
        "mounting_uncertainty_floor_deg": math.degrees(_MOUNTING_UNCERTAINTY_RAD),
        "encoder_smoothing_window_s": 2.0 * _ENCODER_HALF_WINDOW_S,
        "gravity_initialization_window_s": _INITIAL_GRAVITY_WINDOW_S,
        "impact_recovery_time_s": _RECOVERY_TIME_S,
        "camera_front_angle_deg": geometry.camera_angle_deg,
        "circular_acceleration_compensated": True,
        "imu_to_camera_rotation": rotation.tolist(),
        "imu_axes": "nominal_Gemini_330_optical_right_down_forward"
        if imu_to_camera_rotation is None
        else "supplied_calibrated_rotation",
        "confidence_note": "Model uncertainty includes a mounting floor; it does not guarantee absolute accuracy or correct timestamp alignment.",
    }
    return PitchTrajectory(
        times, pitch, uncertainty, valid, rejected, report, segment_ids
    )
