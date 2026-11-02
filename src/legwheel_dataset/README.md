# `legwheel_dataset`

`legwheel_dataset` is the offline-only ROS 2 package that converts one recorded
LegWheel MCAP run into self-contained, machine-learning-ready packets. It never
controls the robot and never changes the raw rosbag.

The package performs two processing stages:

1. **Stage 1 — packet isolation:** determine the valid constant-torque run,
   construct the continuous encoder-corrected rig angle, synchronize RGB and
   depth images, select the physical terrain patch, find when the wheel reaches
   that patch, and attach resampled telemetry.
2. **Stage 2 — feature extraction:** crop the images, reconstruct the selected
   depth pixels in 3D, robustly fit a plane, calculate two signed slopes, and
   calculate spectra for all 15 motor and IMU telemetry channels.

## Commands

Create or update the run's post-processing configuration interactively:

```bash
ros2 run legwheel_dataset create_config \
  --run-directory /path/to/experiment/run_1
```

Build both stages:

```bash
ros2 run legwheel_dataset build_dataset \
  --run-directory /path/to/experiment/run_1 \
  --processing-config /path/to/experiment/run_1/postprocess.yaml
```

`build_dataset` displays one continuous terminal progress bar from 0% to 100%,
including bag reading, motion/IMU analysis, packet isolation, image recovery,
feature extraction and report writing. The current phase appears beside the
bar. Percentages describe completed work, not predicted elapsed time; 100% is
shown only after the complete build succeeds. Progress is written to stderr,
while the final dataset path remains on stdout.

Run the stages separately:

```bash
ros2 run legwheel_dataset isolate_packets \
  --run-directory /path/to/experiment/run_1 \
  --processing-config /path/to/experiment/run_1/postprocess.yaml

ros2 run legwheel_dataset extract_features \
  --dataset /path/to/experiment/run_1/derived/CONFIGURATION_HASH
```

Inspect what a configuration will produce without writing packets:

```bash
ros2 run legwheel_dataset validate_config \
  --run-directory /path/to/experiment/run_1 \
  --processing-config /path/to/experiment/run_1/postprocess.yaml \
  --show-visualization
```

The validator reports the measured source rates, the patch-based local-window
rule, spectral Nyquist frequency and bin spacing, and an estimate of the
number of depth pixels in a fully visible patch. `--show-visualization` is
optional; without it the command only prints JSON.

When requested, the validator selects the synchronized RGB/depth pair closest
to the temporal midpoint of the steady-motion interval. It then loads those two
recorded images and opens a blocking Matplotlib window with four panels:

- the RGB and depth images at their actual pixel resolutions;
- an expanded RGB diagnostic and an expanded depth diagnostic that include
  projected points outside the camera boundary.

Each panel shows the inner and outer circular tracks, configured patch polygon,
patch centre and camera boundary. Titles and annotations report complete-patch
visibility, rejection reasons, sensor timestamps, knee angle, `theta_y` and
`theta_p`. The expanded views are intentionally allowed to extend beyond the
camera resolution; this makes an off-screen patch visible instead of clipping
the evidence needed to correct its configuration.

The JSON always prints before the window opens. If no synchronized pair can be
projected, the command keeps the JSON result and prints a yellow terminal
warning instead of opening an empty figure. The interactive `create_config`
workflow asks a separate question about opening this visualization after it
prints the validation report.

### Interactive configuration creator

`create_config` writes `RUN_DIRECTORY/postprocess.yaml`. For each algorithm
setting it shows the selected value and asks whether to change it. Enter `-h`
or `--help` either at that yes/no question or at the new-value prompt to see a
short explanation of the setting and its effect. Pressing Enter at the change
question keeps the displayed value.

Values are selected independently in this order:

1. the current run's existing `postprocess.yaml`;
2. a `postprocess.yaml` found in the three nearest lower-numbered sibling
   `run_N` directories, searched newest first;
3. the built-in default.

This field-by-field fallback means a partial current or previous file remains
useful. Missing or invalid fields fall through to the next source and produce a
terminal warning. The script shows the complete result and asks before saving.
Replacing an existing file requires explicit confirmation and uses an atomic
filesystem replacement, so an interrupted write cannot leave a partial YAML
file.

Generated files contain `schema_version` and a timezone-aware `created_utc`
timestamp without prompting, plus short comments for every editable setting.
The production configuration loader validates the timestamp but excludes it
from algorithm settings and the dataset hash. After saving, the script asks
whether to run the same recording-aware analysis as `validate_config` and
prints its JSON report if accepted. It then asks a separate question before
opening the blocking Matplotlib visualization.

## Required input

The recorded experiment configuration must include a valid
`environment.terrain_and_light_ID`, constant across the experiment's configs,
run snapshots and dataset provenance. This case-sensitive setup ID allows ASCII
letters, digits, underscores and hyphens, with at least one letter or digit.
Different experiments may share an ID. Individual builds reject missing or
conflicting IDs before analysis; use the batch builder to repair older metadata.

The `--run-directory` must contain:

```text
run_1/
├── rosbag/                                  # MCAP rosbag2 directory
├── run_config.yaml
├── metadata.json
├── experiment_config_run_1_snapshot.yaml
└── camera_driver_config_run_1_snapshot.yaml
```

The bag must contain:

- `/legwheel_rgbd/color/image_raw`
- `/legwheel_rgbd/color/camera_info`
- `/legwheel_rgbd/depth/image_raw`
- `/legwheel_rgbd/depth/camera_info`
- `/legwheel_rgbd/gyro_accel/sample`
- `/legwheel/joint_states`
- `/wheel/wheel_state`
- `/wheel/requested_torque`
- `/rotation_encoder/ticks`

Ticks use their original bag receipt timestamps because `std_msgs/msg/Int64`
has no header. Raw-run replay and the clock-driven visualization adapter use
the same timestamps and recording origin. Wheel-velocity refinement between
ticks is unchanged. `/rotation_encoder/joint_state` is not used for offline
position reconstruction.

This implementation deliberately requires the snapshotted camera configuration
to contain `depth_registration: true` and `align_target_stream: COLOR`.
Unregistered depth is rejected instead of silently using the wrong RGB/depth
extrinsic transform.

The experiment snapshot must also contain:

```yaml
electronics_hardware:
  encoder_setup:
    ticks_per_legwheel_revolution: 62.5
    # Keep -1 for this LegWheel geometry. Encoder-positive forward motion is
    # opposite to positive notebook/DH theta_y. Changing this mirrors all
    # camera projections, 3D points, and signed slope directions.
    theta_y_kinematic_sign: -1
  camera:
    camera_config:
      sensor_front_offset_m: 0.043  # Along the beam towards the pivot (-X).
      camera_beam_offset_m: 0.054   # Towards the camera-facing side (+Z).
      camera_height_m: 0.029
      camera_front_angle_deg: 40.0
```

This is a coordinate-convention mapping, not a post-processing tuning value.
Only `-1` and `1` are accepted. The current rig requires `-1` and should not be
changed unless the encoder mounting or kinematic-frame convention changes.
For an older recorded run whose snapshot predates this field, make a corrected
copy of the experiment YAML containing the value above and pass it with
`--experiment-config`; the MCAP itself does not need to be changed.

Camera offsets are translations in the beam-end frame before camera tilt:
`[-sensor_front_offset_m, camera_height_m, camera_beam_offset_m]`. To migrate
the former signed-X field `camera_beam_offset_m: -0.043`, replace it with
`sensor_front_offset_m: 0.043` and `camera_beam_offset_m: 0.054` in the corrected
full experiment YAML. Both offset fields are required for dataset processing.
Regenerate derived packets after correcting geometry because their projected
pixel coordinates are saved during packet isolation.

## Offline beam pitch from the camera IMU

Dataset analysis, packet generation and the raw-run 3D viewer estimate
`theta_p` from the recorded camera accelerometer and gyroscope. Knee angle
remains diagnostic/ML telemetry and is never a fallback for an unavailable
inertial pose. Live/bag-stream kinematics publishers use the same shared
estimator in `legwheel_kinematics`; knee geometry is not a pitch source in any
visualizer. Datasets saved before IMU pitch was introduced must be regenerated:
`visualise_stream` refuses their old poses and overlays rather than presenting
them as current inertial results.

The estimator rotates the Gemini 336's nominal optical-axis IMU measurements
into the beam frame using the existing camera transform. This includes
`camera_front_angle_deg` and the optical-axis flip: a level beam has zero
`theta_p` even when the physical camera is tilted. Camera image projection
still uses the actual configured mounting angle. Per-device IMU-to-color
misalignment is not recorded by the current bag format, so this uses the
manufacturer's nominal optical-axis alignment. Validate physical mounting and
calibration separately before claiming an absolute one-degree accuracy.

The complete IMU recording is processed before selecting steady wheel motion.
A pitch-and-gyro-bias filter uses quiet samples to estimate bias, rejects
acceleration disturbances, and applies an offline smoother within each
continuous segment. Smoothed central encoder motion and the camera lever arm
compensate for circular acceleration. Pitch is evaluated separately at each
RGB and depth acquisition timestamp. An impact can disable an accelerometer
correction while the gyro still supplies a finite pitch estimate. Those
low-confidence estimates are retained for packets, point clouds, frames and
image overlays. Only timestamps with no estimate at all, such as startup or a
genuine data gap, remain unavailable; no knee fallback or invented gap filling
is used. All other packet checks, such as image visibility, still apply.

Every new packet saves `imu_uncertainty_deg` in its metadata. Stage 2 exposes
the same scalar feature as **`"IMU uncertainty"`** in the packet's `features.json`
and the dataset's `features.jsonl`, and as `imu_uncertainty_deg` in `features.npz`.
It is the larger RGB/depth **two-sigma pitch uncertainty in degrees**. It is
descriptive and never invalidates a packet or its other features. The browser
shows it as **IMU uncertainty** with degree/2σ units, including Stage 1 metadata
before feature extraction. Unknown uncertainty in older packets is shown as
unavailable rather than zero.

`maximum_pitch_uncertainty_deg` defaults to `1.0` and now controls only a
diagnostic confidence flag; exceeding it does not skip packets or visualization
updates. Inspect `pitch_estimation` and `packets/run_pitch_estimation.npz` for
the separate `available` and diagnostic `valid` masks, uncertainty and
acceleration rejection. Packet metadata also retains each exposure's one-sigma
`theta_p_*_uncertainty_rad`. These are model confidence bounds and do not include
every calibration error. Estimator version and settings participate in the
dataset identity; regenerate packets to retain formerly rejected frames and
populate the new feature.

For new recordings, leave the mechanism genuinely still for 5–10 seconds after
the recorder reports ready and before confirming `START`. A recorded quiet
period after motion also helps. Shorter existing pauses can still be used;
the estimator must find quiet measurements rather than assuming the first
two seconds are motionless. Source timing faults and insufficient trustworthy
IMU data can leave frames unavailable; the report explains rejected pairs.

## Output layout

By default, output is placed in:

```text
RUN_DIRECTORY/derived/CONFIGURATION_HASH/
```

The hash covers the experiment configuration, run configuration, complete
post-processing configuration and pitch estimator version. An existing derived directory is never
overwritten.

```text
CONFIGURATION_HASH/
├── manifest.jsonl
├── features.jsonl                         # after Stage 2
├── provenance.json
├── quality_report.json
└── packets/
    ├── analysis_report.json
    ├── run_motion_analysis.npz
    ├── run_pitch_estimation.npz
    └── e<experiment>_r<run>_theta_<angle>_<hash>/
        ├── metadata.json
        ├── geometry.npz
        ├── telemetry.npz
        ├── rgb.npy                        # complete synchronized RGB image
        ├── depth.npy                      # complete synchronized depth image
        └── stage2/
            ├── rgb_patch.npy
            ├── rgb_patch_mask.npy
            ├── depth_patch.npy
            ├── depth_patch_mask.npy
            ├── point_map.npz
            ├── features.npz
            └── features.json
```

Packet names use the experiment ID, run number, unwrapped motion-coordinate
`theta_y` at image time rounded to three decimal places, and an eight-character
deterministic hash. Exact motion and kinematic values always remain in
`metadata.json`.

Every packet retains the complete RGB and depth images. It can therefore be
examined after the original MCAP has been moved or archived.

### Terrain and lighting setup identity

Each new dataset writes `terrain_and_light_ID` at the top level of
`provenance.json`, alongside the same value in
`experiment_config.environment.terrain_and_light_ID`. The ID participates in
the existing configuration hash, so changing it changes the normal output path.

When the batch builder adds or corrects an ID in an existing provenance file,
it preserves that dataset's identifier, folder, numerical files and packet
classifications, and sets `requires_regeneration: true` with an explanatory
`regeneration_reason`. A warning lists affected datasets before saving.
These outputs cannot be skipped as reusable completed datasets. The batch
builder creates a fresh dataset for selected runs and preserves the previous
output if a retry uses the same path. Unselected runs remain marked for later
regeneration. Stage 2 alone refuses flagged datasets; run the full build again.

New builds set `requires_regeneration: false`. Existing numerical results and
classifications remain in the preserved old dataset; labels are not transferred
automatically to the newly generated packets.

### Terrain classification

Each packet's `metadata.json`, `stage2/features.json` and corresponding row in
the dataset's `features.jsonl` carry the same three classification fields:

- `class_id`: `0` (Firm Terrain), `1` (Loose Terrain), `2` (Firm to Loose
  Transition), or `3` (Loose to Firm Transition);
- `classification_method`: `"manual"` or, for a future classifier, `"automatic"`;
- `model_id`: the future automatic classifier's identifier; currently `null`.

The class describes the highlighted physical terrain patch. New packets start
with all three fields set to `null` (unclassified). Stage 2 preserves labels
already assigned to Stage 1 metadata, including when feature extraction fails.
When extracting features from older Stage 1 packets, missing fields are added
to their metadata as well.

Use the standalone browser in `supporting_files` to assign manual classes.
It upgrades missing fields in older datasets when opening them and saves each
manual selection immediately to all applicable JSON files. Stage 1-only
packets are labelled in metadata until their feature files are generated.

## Stage 1, Pass 1: analysing the run

### 1. Find the constant-torque period

The configured positive torque magnitude comes from:

```text
run_config.yaml -> wheel_parameters.commanded_wheel_torque_nm
```

The actual command may be negative because of the motor direction convention.
The processor therefore compares magnitudes. With the default 90% criterion,
the accepted period begins at the first command satisfying:

```text
abs(requested_torque) >= 0.90 * commanded_wheel_torque_nm
```

It ends at the first later command below that threshold. Exact floating-point
equality is intentionally not used because the online ramp publishes discrete
intermediate commands and may not publish the nominal endpoint exactly.

### 2. Define `t0` and reject reverse encoder changes

Repeated central-encoder values are collapsed. `t0` is the first actual encoder
count change after the torque threshold is reached. This selects the analysis
interval without changing the position origin. Every angle uses the first
recorded tick in the complete run:

```text
theta_y_at_tick = (ticks - first_recorded_ticks) * 2*pi / ticks_per_legwheel_revolution
```

The first count may be nonzero when the encoder was started before recording.
The driver has already applied `invert_direction`; offline code must not apply
it again. `theta_y_kinematic_sign` subsequently maps the motion angle into the
geometry convention. Complete revolutions are preserved without wrapping.
The recording origin, calibration and bag timestamp source are saved in
`analysis_report.json` and dataset provenance. Regenerate previously derived
datasets to adopt this common origin; new dataset identifiers include the
encoder-position convention version.

Actual encoder differences are used, including a change of more than one tick.
Isolated changes opposite to the dominant run direction are treated as encoder
bounce and reported. A sustained reversal makes the future-patch assumption
invalid and prevents useful packet generation.

### 3. Construct continuous encoder-corrected `theta_y(t)`

The effective track radius is the wheel centreline. The wheel is mounted
inward from the beam end, so this is the midpoint of the inner and outer edges:

```text
r_b = beam_radius_m - wheel_width_m / 2
```

For every pair of accepted encoder changes `i` and `i+1`, the predicted rig
angle is:

```text
predicted_delta_theta_i = integral(
    wheel_velocity_rad_s *
    (wheel_radius_m + grouser_effective_height_m) / r_b
) dt
```

The interval correction factor is:

```text
k_i = encoder_delta_theta_i / predicted_delta_theta_i
```

Within that interval:

```text
theta_y(t) = theta_y(t_i) + k_i * integral(t_i to t)(
    wheel_velocity_rad_s *
    (wheel_radius_m + grouser_effective_height_m) / r_b
) dt
```

All signs are preserved. If the wheel and central encoder use opposite positive
directions, `k_i` is negative and corrects that convention. No arbitrary hard
limit is applied to `k_i`; its minimum, maximum and mean are written to the run
analysis report. A zero predicted interval or an unbridgeable wheel-telemetry
gap is an error.

Wheel velocity is linearly interpolated and integrated with the trapezoidal
rule. The resulting continuous function exactly matches the central encoder at
every accepted tick endpoint.

### 4. Convert motion coordinates and require forward travel

The corrected encoder trajectory is retained as `theta_y_motion(t)` because
future contact time must be solved in the same coordinate system as the
encoder. Before projection, angles and directions are converted into the
notebook/base convention:

```text
theta_y_kinematic = theta_y_kinematic_sign * theta_y_motion
direction_kinematic = theta_y_kinematic_sign * direction_motion
```

For this geometry, `theta_y_kinematic_sign = -1`. The verified camera optical
axis faces decreasing kinematic `theta_y`, so a recording is forward only when:

```text
direction_kinematic = -1
```

`validate_config` reports a reverse recording as invalid and skips its preview.
`isolate_packets` stops with `failed_reverse_recording` before extracting any
images or creating packets.

### 5. Report average speeds

The reported wheel speed is the time-weighted mean absolute velocity:

```text
omega_avg = integral(abs(wheel_velocity)) dt / run_duration
```

Minimum, maximum, median and standard deviation are also reported.

The corrected average rig angular speed is also reported:

```text
average_rig_speed = abs(theta_y(end) - theta_y(t0)) / run_duration
```

Average speeds are diagnostic. Local telemetry boundaries are calculated for
each patch from the corrected trajectory, as described below.

### 6. Measure acquisition rates

The processor calculates actual timestamp rates for the leg, wheel and IMU
streams that form the ML telemetry arrays. `packets/analysis_report.json`
records sample count, median rate, median period, 95th-percentile period, and
minimum/maximum interval rates.

Packet isolation stops if `telemetry_resample_hz` exceeds one of these three
streams' measured median rates by more than `source_rate_tolerance_fraction`.
Upsampling cannot create information that was not measured.

Central-encoder ticks are deliberately separate. They remain essential for
correcting `theta_y(t)` and finding terrain contact time, but are not resampled,
saved as ML telemetry, checked against `telemetry_resample_hz`, or sent through
the FFT. Their native rate is reported separately under
`motion_reference_effective_rate`. This allows the approximately 100 Hz motor
and IMU data to retain an FFT Nyquist frequency near 50 Hz even though the
encoder normally publishes near 20 Hz.

## Stage 1, Pass 2: creating packets

### 1. Synchronize images

RGB and depth images are paired one-to-one using the nearest unused depth
timestamp within `rgb_depth_max_delta_ms`. The packet image time is the integer
midpoint of the two sensor timestamps.

Only images at or after `t0` and before the last complete encoder interval are
eligible. The first eligible valid pair is kept. Each later candidate must
advance its patch centre by at least one full patch width in the motion
direction:

```text
direction_motion * (candidate_patch_motion - last_saved_patch_motion)
    >= alpha_sp_rad
```

This prevents overlap between successive patches while allowing small gaps
because images arrive at discrete times. Comparisons retain full revolutions
and do not wrap angles at 2*pi. Rejected candidates do not advance the saved
reference. There is no time-based saving limit.

### 2. Define the physical terrain patch

`alpha_off_rad` is a positive distance ahead. Contact timing is first defined
in corrected encoder-motion coordinates:

```text
patch_motion = theta_y_motion(image_time)
             + direction_motion * alpha_off_rad
```

The camera and terrain geometry are then converted together:

```text
camera_theta_kinematic = theta_y_kinematic_sign * theta_y_motion(camera_time)
patch_kinematic = theta_y_kinematic_sign * patch_motion
```

Keeping those two conventions separate is essential. For the current rig, a
forward encoder direction of `+1` becomes a kinematic direction of `-1`, which
places the future patch in front of the camera instead of across its camera
plane.

The radial limits are:

```text
inner_radius = beam_radius_m - wheel_width_m
outer_radius = beam_radius_m
```

The same radial limits are used for the patch, full-track image overlays, and
3D track markers. The width must be positive and smaller than the beam radius,
so the inner radius remains positive.

The angular limits are:

```text
patch_start = patch_kinematic - alpha_sp_rad / 2
patch_end   = patch_kinematic + alpha_sp_rad / 2
```

`track_point_count` sets the full-circle sampling density. Exact patch start and
end angles are always inserted, so the requested patch width is never rounded
to that grid.

The RGB and depth projections each use their own image timestamp, IMU-derived
`theta_p`, `CameraInfo`, image frame, and camera pose. A patch is accepted only
when its complete inner boundary, outer boundary and angular endpoints are in
front of the camera and inside both images. Partial patches are rejected and
listed by timestamp and reason in `packets/analysis_report.json`.

### 3. Find when the wheel reaches the photographed patch

The contact timestamp is the first future solution of:

```text
theta_y_motion(contact_time) = patch_motion
```

It is found by bracketing the target with corrected encoder intervals and then
bisection on continuous `theta_y(t)`. It is not snapped to a telemetry sample,
so no angular matching tolerance is required.

### 4. Attach telemetry

The local telemetry window covers the wheel's complete passage through the
patch. In motion coordinates, its entry and exit angles are:

```text
entry_motion = patch_motion - direction_motion * alpha_sp_rad / 2
exit_motion  = patch_motion + direction_motion * alpha_sp_rad / 2
```

The same corrected `theta_y(t)` crossing search finds each boundary timestamp.
`local_values` spans exactly from entry to exit, including interpolated endpoint
samples. A patch whose complete passage lies outside the corrected run is
rejected. Its duration and sample count can differ between packets in the same
run; downstream model preparation must pad or otherwise batch them deliberately.

Local times remain relative to `contact_time`, the crossing of the patch
centre. Entry-to-centre and centre-to-exit need not take equal times, so the
local window is not forced to be symmetric. Sampling uses a uniform time grid
near `telemetry_resample_hz`, with its spacing adjusted to include both exact
endpoints. Metadata saves `local_start_time_ns`, `local_end_time_ns`, and the
per-packet `local_window_ms`.

`spectral_values` retains the fixed `spectral_window_ms` duration centred on
`contact_time`. It is independent of local traversal duration, may be shorter
than a slow traversal, and may overlap other packets' spectral windows.

All 15 columns use the same grid and order stored in packet metadata:

- hip position, velocity and effort;
- knee position, velocity and effort;
- wheel position, velocity and effort;
- IMU angular velocity X/Y/Z;
- IMU linear acceleration X/Y/Z.

The encoder-motion and kinematic/base versions of `theta_y`, patch angle and
travel direction are saved explicitly in packet metadata. Unqualified legacy
spatial fields contain kinematic values; contact time and angular residual stay
in motion coordinates. Only the encoder's raw position/velocity/tick channels
are excluded from the resampled arrays.

The complete local window must remain inside the corrected run and have source
timestamp coverage without an excessive gap. Spectral cells outside the valid
run remain invalid and are subject to the Stage 2 valid-fraction rule. Ramp-up
and ramp-down telemetry are never introduced.

`telemetry.npz` saves values, per-cell validity masks and nanosecond times
relative to contact. `geometry.npz` saves bit-packed image masks, exact patch
polygons, 3D physical boundaries, and full-track overlay points.

## Stage 2: feature extraction

### RGB and depth patches

The complete images are loaded from the packet. Each bit-packed mask is
restored, and the smallest bounding crop containing the physical patch is
saved. Pixels outside the polygon are zeroed, and the separate crop mask must
be used so a genuine zero-valued pixel is not confused with padding.

### Depth point map

Every finite, positive depth pixel inside the mask is converted to metres.
Integer depth uses `depth_scale_m`; floating-point depth is treated as metres.
Pixels are undistorted using the selected depth `CameraInfo`, back-projected to
3D, and transformed from the registered camera frame into `legwheel_base`.

`point_map.npz` contains:

- `points_base_m`: `N x 3` metric points;
- `pixel_uv`: `N x 2` source depth-pixel coordinates;
- `depth_m`: the `N` original metric depth measurements;
- `ransac_inlier_mask`: which points support the accepted plane.

### RANSAC and plane refinement

RANSAC samples three points per iteration and counts points whose perpendicular
distance is at most `plane_inlier_threshold_m`. The best candidate has the most
inliers, with mean squared error as its tie-breaker. A fixed random seed makes
the result reproducible.

At least `minimum_plane_points` valid points and the same number of inliers are
required. The default is 40. The accepted inliers are refined by singular value
decomposition, and the normal is oriented so its base-frame Z component is
non-negative. `plane_rmse_m` is the root-mean-square inlier residual.

### Signed slopes

At kinematic patch-centre angle `theta_patch`, the radially outward unit vector
is:

```text
e_r = [cos(theta_patch), sin(theta_patch), 0]
```

The direction-of-travel tangent uses the converted kinematic direction:

```text
e_t = direction_kinematic * [-sin(theta_patch), cos(theta_patch), 0]
```

For upward-oriented plane normal `n`:

```text
along_slope = atan2(-dot(n, e_t), n_z)
cross_slope = atan2(-dot(n, e_r), n_z)
```

Positive along-slope means terrain rises in the direction of travel. Positive
cross-slope means terrain rises radially away from the rig centre. Both radians
and degrees are saved.

### FFT processing

All 15 motor and IMU telemetry channels receive a separate FFT. A channel must
first satisfy `minimum_spectral_valid_fraction`. Missing samples are filled only
for short, internal gaps allowed by `maximum_interpolation_gap_ms`; leading,
trailing and long gaps invalidate that channel.

Each accepted channel is linearly detrended, multiplied by a Hann window, and
converted to a one-sided power spectral density. Zero hertz is excluded from
peak detection. The two highest distinct local maxima are saved, ordered by
power, along with their powers and the complete spectrum. A missing second peak
is represented by `null` in JSON and `NaN` in NumPy output.

FFT properties are:

```text
Nyquist frequency = telemetry_resample_hz / 2
bin spacing       = telemetry_resample_hz / spectral_sample_count
```

The bin spacing is controlled by `spectral_window_ms`, not by the local patch
duration.

## Configuration reference

The installed defaults are:

```yaml
schema_version: "2.0"
rgb_depth_max_delta_ms: 10.0
spectral_window_ms: 1000.0
telemetry_resample_hz: 100.0
maximum_interpolation_gap_ms: 50.0
maximum_pitch_uncertainty_deg: 1.0
alpha_off_rad: 0.35
alpha_sp_rad: 0.10
grouser_effective_height_m: 0.0
minimum_target_torque_fraction: 0.90
source_rate_tolerance_fraction: 0.05
track_point_count: 360
depth_scale_m: 0.001
minimum_plane_points: 40
plane_ransac_iterations: 200
plane_inlier_threshold_m: 0.005
minimum_spectral_valid_fraction: 0.8
```

An interactively generated file additionally contains a quoted `created_utc`
ISO 8601 timestamp. This is file metadata, not an algorithm input, and therefore
does not change a derived dataset's configuration hash.

Older YAML files and saved dataset configurations may still contain
`maximum_packet_saving_frequency_hz`. It is accepted and ignored; patch width
now determines the minimum angular spacing. New configurations omit it.

`alpha_off_rad` and `alpha_sp_rad` are initial processing defaults, not measured
rig constants. Validate them against the actual camera view and desired physical
patch before generating a research dataset.

`grouser_effective_height_m` intentionally defaults to zero. A source-code TODO
marks that a geometry- and experiment-based estimator should replace this
manual correction in future.

## Foxglove visualization

Play the original bag with clock publication:

```bash
ros2 bag play /path/to/run_1/rosbag --clock
```

In another sourced terminal:

```bash
ros2 run legwheel_dataset visualise_stream \
  --dataset /path/to/run_1/derived/CONFIGURATION_HASH
```

The node follows `/clock`, including backward seeks, and publishes:

- `/legwheel/dataset/rgb_annotations`
- `/legwheel/dataset/depth_annotations`
- `/legwheel/dataset/geometry`
- `/legwheel/dataset/slope_text`
- derived moving transforms on `/tf`

The image annotations contain the selected patch and visible inner/outer track
points. The 3D markers contain the circular track, physical patch boundaries,
depth-plane inliers, fitted plane outline, plane normal, direction-of-travel
arrow, radially-outward arrow, and slope text.

Derived TF child names start with `dataset_` so they do not compete with frames
recorded in the original bag. The saved packet poses are interpolated at every
`/clock` update, so the frames move smoothly even though packet overlays only
change when playback reaches an accepted packet.

In the Foxglove 3D panel, set both **Fixed frame** and **Display frame** to
`legwheel_base`, turn **Sync timestamps** on, and enable the `dataset_*` entries
under **Transforms**. Frame labels can be disabled or their scale reduced if
they obscure the axes. `dataset_theta_y_link` rotates at a fixed origin and
`dataset_theta_p_link` shares that origin by construction; watch their axis
orientation or the downstream `dataset_beam_end_link` and
`dataset_camera_optical_frame` to see the translational motion clearly.

If the frames are visible but do not move, confirm outside Foxglove that the
transform changes while the bag plays:

```bash
ros2 run tf2_ros tf2_echo legwheel_base dataset_camera_optical_frame
```

The translation and rotation should update only during the dataset's analysed
steady-motion interval. The visualization intentionally publishes no new pose
before the first or after the last accepted packet.

## Dataset splitting

Runs may use pure raw torque or the optional raw-torque velocity governor.
Processing still selects the drive interval from `/wheel/requested_torque`;
this is a requested-drive plateau, not proof of constant motor torque. The
analysis report's `wheel_control` section preserves mode, cap and gains and
summarizes `/wheel/commanded_torque` when recorded. That optional stream is the
target successfully sent after governing; wheel-state effort is estimated motor
torque. Old recordings without the new configuration fields or command topic
remain supported. The existing 15-channel packet telemetry format is unchanged.

Every packet stores unwrapped motion and kinematic `theta_y` values plus a
`lap_index`. Repeated loops can observe the same physical terrain. Do not
randomly split packets from one run between training and evaluation; split by
complete run, experiment, or physical terrain setup to avoid leakage.
