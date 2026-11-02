# Welcome to the Open LegWheel Dataset Project!

This project aims to address the lack of traversability data for training future hybrid exploration rovers. It is an open-source project currently maintained by @vps2a (Filip Szczebak). If you have any suggestions or questions, please feel free to reach out at filip@szczebak.com!

![Main Picture](docs/readme_images/open_legwheel_main_pic.png)

# Dataset Location

The combined dataset is hosted on [Hugging Face](https://huggingface.co/datasets/vps2a/open_legwheel). As of November 2026, it contains data from one campaign, recorded at the Kąkolewo Space Robotics Testing Centre at Poznań University of Technology.

# CAD and Design Files

The CAD files are available on the free platform [Onshape](https://cad.onshape.com/documents/7945975a1e2987842bb7db33/w/577eb7f23b2fb39c5298f464/e/060e9628c73f0835c4b77241?renderMode=0&uiState=6ae3a08f8a82bae0df0d5337). Details of the motors, controllers, and camera used are stored in the experiment configuration files uploaded to Hugging Face.

# Data Analysis and ML Tools

## Batch Dataset Generator

After a run has been recorded as an MCAP rosbag, its telemetry must be correlated with camera images of the selected terrain patch ahead of the rover. Although individual runs can be processed as described in [this section](#build-learning-packets), this tool supports batch analysis of multiple runs and entire experiments. It generates correlated data packets within each run directory.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cd supporting_files
uv sync
uv run legwheel-batch-dataset-builder
```

## Dataset Browser

Correlated telemetry and image packets can then be viewed and manually classified with the dataset browser. It displays telemetry in both local and extended windows, along with the projected point cloud, image patches, and packet metadata. Manual classification mode allows you to classify the displayed packets.

```bash
cd supporting_files
uv sync
uv run python dataset_browser.py
```

![Screenshot of the main screen of the dataset browser](docs/readme_images/browser_main_sshot.png)

![Screenshot of the 3D screen of the dataset browser](docs/readme_images/browser_3d_sshot.png)

## Dataset Exporter

Once you are satisfied with the correlated data, you can export them in Parquet format using the dataset exporter. The tool combines the selected packets into one dataset and saves it, together with its metadata and a README file, in your chosen directory.

```bash
cd supporting_files
uv sync
uv run legwheel-dataset-exporter
```

![Screenshot of the exporter](docs/readme_images/batch_exporter_sshot.png)

## Terrain Classification Viewer and Analyser

The final tool in the current pipeline visualises the collected data and motor telemetry. It is tailored to building SVM and logistic regression classifiers, but it can also provide useful insight into the data and their trends. Contributions and ideas for improving this tool are especially welcome!

```bash
cd supporting_files
uv sync
uv run python -m ML_tools.telemetry_classification_visualiser_tool
```

![Screenshot of the band power view of the model builder](docs/readme_images/band_power_sshot.png)
![Screenshot of the standard deviation view of the model builder](docs/readme_images/std_dev_sshot.png)
![Screenshot of the model view of the model builder](docs/readme_images/model_sshot.png)

# Main Workspace

This repository is quite large, but it should be easy to navigate. I used LLMs to write what I consider an extensive README, which I hope will guide you through the operations and troubleshooting needed to get the project working on your machine (it worked on mine!).

This is a ROS 2 Jazzy workspace for operating and recording experiments with the Open LegWheel rig. It integrates MAB Robotics MD80 motor controllers, a serial
rotation encoder, an Orbbec Gemini 336 RGB-D camera, and a safety-gated
experiment runner that records MCAP rosbags.

> **Safety notice:** This repository can energise motors and command wheel
> torque. Start with the mechanism secured, the wheel clear of people and
> obstructions, conservative limits, and a tested hardware emergency-stop
> procedure. Software limits are not a substitute for MD80 firmware limits or
> physical safety measures.

## What is in the workspace

| Package | Purpose | Main command |
| --- | --- | --- |
| `legwheel_can` | Connects to MD80s through a CANdle USB-to-CAN adapter; provides passive zeroing, safety-gated leg control, wheel torque control, and terminal controllers. | `ros2 launch legwheel_can md80_legwheel_control.launch.py` |
| `legwheel_encoder` | Reads the central rotation encoder from a serial port and publishes ticks, angle, and velocity. | `ros2 launch legwheel_encoder rotation_encoder.launch.py experiment_config:=<file>` |
| `legwheel_rgbd` | Launches and configures the Orbbec driver. Experiments consume its raw RGB, depth, calibration, and IMU topics directly without an image relay. | `ros2 launch legwheel_rgbd gemini_336.launch.py` |
| `legwheel_experiments` | Validates an experiment, performs telemetry preflight, runs the leg/wheel sequence, and records an MCAP rosbag. | `ros2 run legwheel_experiments run_experiment --experiment-config <file>` |
| `legwheel_description` | Publishes the kinematic frame tree from the experiment geometry and derived `theta_p`. | `ros2 launch legwheel_description description.launch.py experiment_config:=<file>` |
| `legwheel_kinematics` | Implements the verified rig kinematics and publishes track annotations for Foxglove. | `ros2 launch legwheel_kinematics track_overlay.launch.py experiment_config:=<file>` |
| `legwheel_dataset` | Converts recorded MCAP runs into synchronized packets, terrain-slope labels, and telemetry spectra. | `ros2 run legwheel_dataset build_dataset --run-directory <run>` |

For an accumulated RGB-D cloud with animated tracks and coordinate frames, see
[the raw-run Foxglove 3D tutorial](#accumulated-rgb-d-and-tracks-in-foxglove-3d).

The workspace includes the MAB CANdle-SDK as a Git submodule. Orbbec's ROS 2
driver is downloaded through `dependencies.repos`.

## System requirements

This project is intended for **Ubuntu with ROS 2 Jazzy**. You need:

- a supported Linux computer with USB access;
- ROS 2 Jazzy Desktop or Base installed at `/opt/ros/jazzy`;
- a MAB CANdle USB adapter and correctly configured MD80 controllers;
- the central rotation encoder connected by USB serial;
- an Orbbec Gemini 336 and a USB 3 cable/port when camera data is required;
- sufficient disk space for MCAP recordings (raw RGB-D recordings can be large).

Install build tools and runtime dependencies once:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git libusb-1.0-0-dev \
  python3-vcstool python3-rosdep python3-serial python3-yaml \
  ros-jazzy-rosbag2-storage-mcap
```

Initialize rosdep once per computer. If it says it has already been initialized,
that is fine.

```bash
sudo rosdep init
rosdep update
```

## Installation

Clone the repository and enter its root:

```bash
git clone git@github.com:vps2a/esa-ssp-tactile-legwheel.git
cd esa-ssp-tactile-legwheel
```

Fetch the CANdle-SDK submodule and Orbbec ROS driver, install resolved ROS
dependencies, then build the workspace:

```bash
git submodule update --init --recursive
source /opt/ros/jazzy/setup.bash
vcs import . < dependencies.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Run the following in every new terminal before using the workspace:

```bash
cd /path/to/esa-ssp-tactile-legwheel
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Confirm that the workspace-built Orbbec driver is active, rather than a
system-installed copy:

```bash
ros2 pkg prefix orbbec_camera
```

The printed path should end in this workspace's `install/orbbec_camera`.

### USB permissions

Install Orbbec's udev rule once, then unplug and reconnect the camera:

```bash
cd src/third_party/OrbbecSDK_ROS2/orbbec_camera/scripts
sudo bash install_udev_rules.sh
```

Do not run ROS nodes as root. Ensure the CANdle and serial-encoder devices are
also accessible to the logged-in user. If the encoder appears as `/dev/ttyUSB0`,
the user commonly needs membership of the `dialout` group:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership.

## Configure hardware before enabling it

Review these files before the first real run:

| File | Configure |
| --- | --- |
| `src/legwheel_can/config/motor_ids.yaml` | MD80 CAN IDs for hip, knee, and wheel. |
| `src/legwheel_can/config/motor_limits.yaml` | Software leg position/velocity/torque limits and wheel torque limit. |
| `src/legwheel_can/config/motor_config.json` | Initial leg gains/zero values, wheel torque ramp time, default wheel torque, and provisional governor P/I defaults. |
| Experiment YAML: `electronics_hardware.encoder_setup` | All encoder settings: serial port, baud rate, polling rate, calibration, direction, and startup zeroing. |
| `src/legwheel_rgbd/config/gemini_336.yaml` | Orbbec stream selection, resolution/FPS, alignment, IMU, and filters. |
| `src/legwheel_experiments/legwheel_experiments/camera_topics.py` | Raw camera topics consumed by preflight, the retained runtime-watchdog implementation, and rosbag. |

The MD80 node requires `limits_provided: true` and valid software limits before
it will enable control. Configure conservative **firmware-level** MD80 current,
torque, velocity, temperature, and watchdog limits separately with MAB's
`candletool`; the ROS YAML limits do not replace them.

After changing installed-package configuration, either rebuild and source the
workspace, or pass the modified YAML file explicitly as a launch argument.

### Rotation-encoder calibration note

The experiment YAML is the only encoder configuration source. Use
`ticks_per_legwheel_revolution: 62.5` for this rig. The encoder launch requires
`experiment_config:=<file>` and has no separate calibration default. Restart the
encoder after changing its settings. The experiment runner checks the driver's
reported settings against its validated experiment config before allowing a run.

## Bring-up and verification

Bring hardware up one subsystem at a time. The following commands assume the
environment has been sourced as shown above.

### 1. MD80s: passive observation and zeroing

First check encoder feedback without allowing motor power:

```bash
ros2 launch legwheel_can md80_zero.launch.py
```

In another terminal:

```bash
ros2 topic echo /legwheel/joint_states
```

Move the relevant joint manually to its physical zero and call the matching
service. The zeroing node does not enable motors or command motion.

```bash
ros2 service call /legwheel/zero_hip_motor std_srvs/srv/Trigger {}
ros2 service call /legwheel/zero_knee_motor std_srvs/srv/Trigger {}
```

Stop the zeroing node before starting the control node; both publish
`/legwheel/joint_states`.

### 2. MD80 runtime monitoring and control

Start the runtime node in monitoring-only mode. This publishes state but keeps
the motor-control gate closed:

```bash
ros2 launch legwheel_can md80_legwheel_control.launch.py
```

Verify state:

```bash
ros2 topic echo /legwheel/joint_states
ros2 topic echo /wheel/wheel_state
```

To permit control later, restart it with the launch gate armed:

```bash
ros2 launch legwheel_can md80_legwheel_control.launch.py enable_control:=true
```

This still does **not** energise motors. A `true` message on
`/legwheel/motor_status` is also required, and the node refuses to enable if a
motor moved too far while disabled.

### 3. Central rotation encoder

Check the serial device name with `ls /dev/ttyUSB* /dev/ttyACM*`, update
the experiment YAML's `electronics_hardware.encoder_setup` section, then launch:

```bash
ros2 launch legwheel_encoder rotation_encoder.launch.py \
  experiment_config:=/path/to/experiment.yaml
ros2 topic echo /rotation_encoder/joint_state
ros2 topic echo /rotation_encoder/ticks
```

### 4. Gemini 336 RGB-D camera

Confirm that the camera is visible, then launch the project wrapper:

```bash
ros2 run orbbec_camera list_devices_node
ros2 launch legwheel_rgbd gemini_336.launch.py
```

Verify all required experiment-camera streams:

```bash
ros2 topic hz /legwheel_rgbd/color/image_raw
ros2 topic hz /legwheel_rgbd/depth/image_raw
ros2 topic hz /legwheel_rgbd/gyro_accel/sample
ros2 topic echo --once /legwheel_rgbd/depth/camera_info
```

These are the raw Orbbec driver topics. No LegWheel node republishes the images,
so their original content, rate, and `header.stamp` values go directly to the
experiment preflight checks and recorder. RGB and depth remain independent and are
paired later during dataset generation. Raw `CameraInfo` messages are recorded
at whatever rate the vendor driver publishes them.

Use `serial_number:=<camera-serial>` or `usb_port:=<port>` with the launch
command when more than one camera is connected. Start with point clouds disabled;
they can be enabled with `enable_point_cloud:=true` when needed.

To use a different RGB-D camera, map its five raw ROS topics in
`src/legwheel_experiments/legwheel_experiments/camera_topics.py`. The expected
interfaces are RGB and depth `sensor_msgs/msg/Image`, matching
`sensor_msgs/msg/CameraInfo`, and one combined `sensor_msgs/msg/Imu`. That single
mapping is imported by both the experiment runner and rosbag recorder. Replace
the Orbbec launch/configuration or start the alternative driver separately,
and update `CAMERA_DRIVER_CONFIG_PATH` in the same module so runs snapshot the
replacement configuration (or set it to `None`). Then rebuild and verify rates
and advancing header timestamps. If the API
publishes accelerometer and gyroscope separately, add a lightweight IMU adapter;
do not relay the high-bandwidth image topics merely to rename them. See
`src/legwheel_rgbd/README.md` for the complete integration checklist.

## Manual operation

Manual control is useful for a carefully supervised mechanism check. It is not
the experiment workflow.

1. Start the runtime node with `enable_control:=true`.
2. In a second terminal, start the leg command publisher:

   ```bash
   ros2 launch legwheel_can legwheel_controller.launch.py
   ```

3. Set knee impedance parameters, then enable only when the robot and area are
   ready:

   ```text
   knee set_zeropos 0.0
   knee set_spring 4.0
   knee set_damp 0.05
   e
   ```

   The hip mirrors the knee encoder position; only knee impedance settings are
   operator-tunable. Type `s` at any time to publish a stop request and disable
   connected motors.

4. For wheel torque testing, launch a third terminal:

   ```bash
   ros2 launch legwheel_can wheel_controller.launch.py torque_limit_nm:=2.0
   ```

   Keep this terminal focused. Hold `w` for one direction, `s` for the other,
   and use `+`/`-` to adjust the requested maximum torque in 0.1 Nm steps.
   Releasing the key ramps torque to zero. The C++ hardware node independently
   clamps torque to `wheel_torque_max_nm`.

The wheel remains in MD80 `RAW_TORQUE` mode. Recorded runs can optionally use
a C++ PI governor that reduces driving torque to regulate wheel rotation near
a per-run speed cap. Pure torque mode adds no software speed cap. The governor
never commands active braking: overspeed recovery depends on rolling resistance
and other external loads, and short excursions above the selected speed are possible.

## Running a recorded experiment

The experiment runner requires all four hardware data sources to be running:

- `/legwheel/joint_states` and `/wheel/wheel_state` from `legwheel_can`;
- `/rotation_encoder/joint_state` from `legwheel_encoder`;
- RGB, depth, and IMU streams from `legwheel_rgbd`.

It also requires the MD80 runtime node to subscribe to
`/legwheel/motor_status` and `/wheel/requested_torque`. Start the control node
with `enable_control:=true`, plus the encoder and camera launches, before
starting the experiment CLI.

Create an experiment YAML file, for example `experiments/demo.yaml`:

```yaml
schema_version: "1.0"
experiment_id: "demo"
datetime_of_creation: "2026-01-01T12:00:00+00:00"

rig_config:
  beam_radius_m: 0.83
  beam_total_length_m: 1.00
  pivot_height_m: 0.353
leg_config:
  hip_link_length_m: 0.20
  calf_link_length_m: 0.20
electronics_hardware:
  encoder_setup:
    port: "/dev/ttyUSB0"
    baudrate: 9600
    ticks_per_legwheel_revolution: 62.5
    publish_rate_hz: 100.0  # Serial polling limit; firmware controls sample rate.
    invert_direction: false
    zero_on_start: true
    # Keep -1 for this LegWheel geometry. The central encoder increases during
    # forward travel, while positive notebook/DH theta_y rotates the opposite
    # way. Changing this mirrors every camera projection and 3D slope direction.
    theta_y_kinematic_sign: -1
  camera:
    camera_config:
      sensor_front_offset_m: 0.043  # Positive along the beam towards the pivot.
      camera_beam_offset_m: 0.054   # Positive towards the camera-facing side.
      camera_height_m: 0.029
      camera_front_angle_deg: 40.0
wheel_config:
  wheel_radius_mm: 50.0
  wheel_width_mm: 20.0
environment:
  environment_description: "Describe the test surface and conditions."
```

`zero_on_start: true` subtracts the first accepted Arduino count when the
encoder driver starts. `false` preserves the Arduino count. Direction inversion
is applied afterwards. This is a software offset, not a hardware reset.
Offline analysis independently subtracts the first recorded tick of each run,
so driver startup and recording startup can occur at different times without
changing the common recording origin. All encoder settings are required for
new recordings; older experiment files need these fields added manually.

Replace the dimensional and hardware values with the measured, physically safe
values for the rig.
`theta_y_kinematic_sign` is a coordinate-convention mapping, not a tuning
parameter. Keep it at `-1` for the present LegWheel geometry: the central
encoder increases during forward motion, while the notebook/DH base frame uses
the opposite positive rotation. The experiment validator accepts only `-1` or
`1`; changing it mirrors camera projections, 3D points and signed slopes.
The runner validates the YAML structure and asks interactively for run length,
knee stiffness/zero/damping, wheel torque, and wheel-ramp duration. It then asks
for pure raw torque or raw torque with a speed cap. Capped runs require a wheel
rotation speed in rad/s, greater than zero and at most **4.5 rad/s**. This is
wheel-shaft speed, not rotation of the rig about the pivot.

For capped runs the operator can accept or override P/I defaults from the installed
`legwheel_can/config/motor_config.json`. The provisional defaults are P = 0.5 Nm s/rad
and I = 0.1 Nm/rad; they require tuning on the rig. I = 0 selects P-only control.
Every new `run_config.yaml` saves `control_mode`, `max_wheel_speed_rad_s` (null
for pure torque), `velocity_governor_kp`, and `velocity_governor_ki` alongside
the original torque/ramp settings. Pure mode saves the inactive default gains.
Editing JSON later cannot change the values already selected for that run.

The runner applies all wheel settings atomically to `/legwheel_can`, checks
readback and fresh `/wheel/governor_state` reports, and only then permits arming.
Wheel settings cannot change while motors are enabled. Missing velocity feedback
suppresses capped-mode drive immediately; a sustained loss or control fault
stops the run. Overspeed alone does not abort it. The wheel firmware mode is
`RAW_TORQUE` in both cases. An explicit zero-torque command always clears
governor history and wins over torque ramps.

Start it from the repository root:

```bash
ros2 run legwheel_experiments run_experiment \
  --experiment-config "$PWD/experiments/demo.yaml"
```

The runner will:

1. create `run_<n>` beside the experiment YAML and snapshot the experiment,
   camera-driver configuration, and raw camera-topic mapping;
2. hold a telemetry preflight until all motor and encoder streams are fresh,
   and RGB, depth, and IMU have each made three forward header-timestamp steps
   after an initial baseline sample;
3. require explicit `ARM` and `START` confirmations;
4. apply the leg settings, allow optional spring-zero adjustment, then start
   MCAP recording;
5. ramp wheel torque, use the central encoder to determine loop progress, ramp
   torque down, and enter safe mode.

The runtime camera watchdog is temporarily disabled while the reported IMU
latency is investigated. Its implementation remains in
`experiment_runner_node.py`, guarded by `_RUNTIME_CAMERA_WATCHDOG_ENABLED`, but
it does not create its 10 Hz timer or perform the final pre-run camera-health
check while that flag is false. Consequently, a camera stream that stalls after
preflight will not automatically abort an active experiment. Preflight still
requires RGB, depth, and IMU timestamps to advance before arming.

Use `Ctrl-C` to abort. The runner sends zero wheel torque, disables motors, and
attempts to finalise the rosbag in its cleanup path.

After normal completion, once the leg has straightened, motors are disabled,
and rosbag has stopped, the runner asks whether the run was valid and offers
optional post-run notes. It saves `run_valid` as a boolean and `post_run_notes`
as text in `run_config.yaml`, preserving the original `notes` and final run
parameters. Aborted/interrupted runs skip these prompts. Until reviewed,
`run_valid` is null; if notes are declined, `post_run_notes` is an empty string.

Recordings preserve `/wheel/requested_torque` (the experiment's request),
`/wheel/commanded_torque` (the final target successfully sent to the drive),
and `/wheel/governor_state` (settings, integral contribution, limiting and fault
status). Motor-estimated torque remains in `/wheel/wheel_state` effort. These
quantities are distinct; capped-mode torque need not remain constant during
the requested-torque plateau used by offline processing to select the run interval.

### Camera recording path

Normal runs record the Orbbec source topics directly. The former
`--record-raw-camera-topics` option and `/camera/...` relay topics have been
removed: recording both source and relayed images duplicated bandwidth while
the Python relay reduced the observed image rate substantially.

## Recorded data and Foxglove

Each run stores `run_config.yaml`, configuration snapshots, `metadata.json`, a
state marker, `rosbag_recorder.log`, and `rosbag/` containing an MCAP bag.
Recorded topics include motor state, wheel state, central encoder data,
independent RGB and depth streams, camera info, IMU, motor-enable state,
impedance commands, and requested wheel torque. Raw camera-info topics may
contain repeated calibration messages because rosbag preserves the driver
output without filtering. RGB/depth association is deliberately deferred to
post-processing and should use the preserved message header timestamps. Do not
infer a pair from adjacent MCAP records, rosbag receive times, or matching
message indices.

The camera topics recorded in every run are:

- `/legwheel_rgbd/color/image_raw`
- `/legwheel_rgbd/color/camera_info`
- `/legwheel_rgbd/depth/image_raw`
- `/legwheel_rgbd/depth/camera_info`
- `/legwheel_rgbd/gyro_accel/sample`

In Foxglove Desktop, choose **Open local file** and select the `.mcap` file under
`run_<n>/rosbag/`. Useful panels are:

| Panel | Topics or fields |
| --- | --- |
| Image | `/legwheel_rgbd/color/image_raw`, `/legwheel_rgbd/depth/image_raw` |
| Plot | `/legwheel/joint_states.position[0]`, `.position[1]`, `/wheel/wheel_state.velocity[0]`, `/rotation_encoder/joint_state.position[0]` |
| Plot | `/legwheel_rgbd/gyro_accel/sample/angular_velocity/*`, `/legwheel_rgbd/gyro_accel/sample/linear_acceleration/*` |
| Raw Messages | camera-info, joint-state, torque, and status topics |

Inspect the `name` array in a `JointState` Raw Messages panel to map each array
index to its joint name before interpreting a plot.

## Track projection and offline dataset generation

Two packages keep analysis code out of the motor-control and recording process:

| Package | Purpose |
| --- | --- |
| `legwheel_description` | Optional URDF representation of the notebook frame chain. |
| `legwheel_kinematics` | Estimates `theta_p` from camera IMU data, publishes measured-yaw/inertial-pitch TF and 3D track points, and projects the circular track into camera images. |
| `legwheel_dataset` | Reads MCAP directly, isolates synchronized packets, extracts RGB/depth patches, fits terrain planes, and calculates telemetry spectra. |

The experiment YAML must include the geometry shown in the example experiment
configuration above. `h_b` remains the measured constant `0.0661 m`. Both camera
offsets are explicit configuration values, measured in the **beam-end frame
before camera tilt**:

| Field | Positive direction | Current measurement |
| --- | --- | --- |
| `sensor_front_offset_m` | Along the beam towards the central pivot (local `-X`). | `0.043 m` |
| `camera_height_m` | Up from the beam-end mount (local `+Y`). | `0.029 m` |
| `camera_beam_offset_m` | Perpendicular to the beam, towards the camera-facing side (local `+Z`). | `0.054 m` |

The mount translation is `[-sensor_front_offset_m, camera_height_m,
camera_beam_offset_m]`. These values change camera position; the viewing
orientation still comes from `camera_front_angle_deg` and the optical-frame
rotation.

**Migrating an existing experiment YAML:** the old `camera_beam_offset_m`
represented signed local X. For example, replace `camera_beam_offset_m: -0.043`
with `sensor_front_offset_m: 0.043` and `camera_beam_offset_m: 0.054`. General
configuration loading requires both offset fields, so an old file cannot silently
use the new meaning. Equivalent migrated measurements preserve the camera pose.
Rebuild the affected packages, source the workspace, and restart geometry
publishers after updating code and configuration.

The wheel sits inward from the beam end: its outer track radius is
`beam_radius_m`, its inner radius is `beam_radius_m - wheel_width_m`, and its
centreline radius is `beam_radius_m - wheel_width_m / 2`.

All visualizers and offline analysis estimate `theta_p` from the camera IMU,
with gyro bias estimation, circular-acceleration compensation and smoothing.
The full camera rotation compensates for `camera_front_angle_deg` when interpreting
IMU vectors. Low-confidence poses remain available for packets and visualization;
their uncertainty is saved as a feature. Only timestamps with no estimate
(startup or genuine data gaps) are skipped. Knee angle remains telemetry and
is never a pitch fallback. See the
[offline IMU pitch documentation](src/legwheel_dataset/README.md#offline-beam-pitch-from-the-camera-imu)
for confidence limits and recording recommendations.

The standalone 3D view, image overlays and URDF share
`/legwheel/kinematic_joint_states`, produced by `kinematic_joint_state_node`.
Each launch starts this shared publisher by default. It requires the combined
camera IMU, `/rotation_encoder/ticks` and `/wheel/wheel_state`. Yaw is derived
from ticks using the experiment calibration and is refined by integrating measured wheel
velocity between encoder changes. Each interval is scaled to match its encoder endpoints,
using the same slip-correction calculation as dataset generation. All visualizers
share this implementation, retain both travel directions and use the first
recorded tick as zero, including dataset processing after steady-motion selection.
The encoder also supplies circular-acceleration correction for IMU pitch. Use
`wheel_topic:=...` to override the wheel topic and `imu_topic:=/camera/imu` for
legacy recordings. The source uses a bounded ten-second window and preserves detected
rest calibration as that window advances. A steady-clock timer runs at 5 Hz;
native IMU poses are published with their original timestamps and a 0.6-second
lookahead delay. No estimate is extrapolated into the unavailable end of a
stream. Dataset processing and `visualise_run` use the complete recording.

When several standalone views run together, launch only one pose publisher:
set `publish_kinematic_state:=false` on additional overlay/3D launches. If the
URDF already publishes the frame tree, also set `publish_tf:=false` on the 3D
marker launch. Image overlays hold only a bounded queue of image headers while
waiting for an available acquisition-time pose. Missing estimates are explicit
barriers; finite estimates during impacts are retained. Bag seeks
reset the streaming histories and their pitch calibration while preserving the
recording's encoder origin. Derived datasets made before this shared tick origin,
or without IMU pitch provenance, must be regenerated before running `visualise_stream`.

During bag playback, the shared publisher reads `/rotation_encoder/ticks`
directly from the bag because these `Int64` messages have no header timestamp.
It preserves each original bag receipt timestamp; `/clock` only selects which
samples have become available. The bag defaults to `rosbag/` beside the
experiment snapshot. Supply `encoder_bag:=/path/to/run/rosbag` when using an
experiment override elsewhere. A missing ticks topic or calibration is an
error; recorded joint-state angles are never a position fallback.

After a kinematics change, regenerate derived packets before inspecting dataset
overlays: those overlays contain saved pixel coordinates. The kinematics model
version is included in new dataset identifiers and provenance, giving corrected
geometry a different default output directory from older datasets.

### Verify the projection in Foxglove

For a recorded run, calculate annotations using simulated ROS time. Start the
two processing launches first in separate terminals, then start bag playback so
the beginning of the recording is not missed:

```bash
ros2 launch legwheel_kinematics track_overlay.launch.py \
  use_sim_time:=true \
  experiment_config:=/path/to/run_1/experiment_config_run_1_snapshot.yaml
```

```bash
ros2 launch legwheel_kinematics kinematics_visualization.launch.py \
  use_sim_time:=true \
  publish_kinematic_state:=false \
  experiment_config:=/path/to/run_1/experiment_config_run_1_snapshot.yaml
```

```bash
ros2 bag play /path/to/run_1/rosbag --clock
```

Connect Foxglove through the normal ROS bridge. In an Image panel, select the raw
RGB or depth image and add the corresponding annotation topic:

- `/legwheel/visualization/rgb_track`
- `/legwheel/visualization/depth_track`

The marker has the exact image timestamp. The overlay waits for IMU poses
bracketing that acquisition time and uses the same numerical projection library
as offline processing. Images without a supported pose have no annotation.

For the 3D view, add a **3D** panel, set its display frame to
`legwheel_base`, enable `/legwheel/visualization/kinematics_3d`, and enable the
transform frames and labels. The marker topic contains the inner and outer
circular track points, the six notebook frame origins, and a line joining the
kinematic chain. The node publishes the three camera/beam mounting transforms
on `/tf_static` and both measured yaw and IMU pitch on `/tf`. Markers and moving
transforms use the shared pose's original sensor timestamp. The direct launch
does not require TF to have been recorded in the raw bag. To add these markers
to an existing robot description, pass `publish_kinematic_state:=false
publish_tf:=false`; the URDF remains the sole frame-tree publisher.

### Accumulated RGB-D and tracks in Foxglove 3D

This viewer reads a **raw recorded run**, reconstructs a coloured point cloud
from depth and RGB images, and accumulates it around the stationary central
pivot. It displays the corrected inner/outer tracks, a red preview patch,
labelled coordinate axes, the kinematic chain, and a purple camera frustum.
Central rotation and beam pitch animate from recorded encoder and camera IMU data.
The camera frustum is drawn 0.20 m long as a direction/FOV aid.

No connected camera or motors are needed. The viewer publishes visualization
topics only and opens recordings read-only. It does not require ML packet
generation, steady wheel torque, recorded point clouds, or recorded TF.

#### 1. Build the two required workspace packages

Run on the **ROS 2 Jazzy computer**. If the recording is in Google Drive on a
different computer, copy or mount the complete run directory onto this one,
including its `rosbag` directory and YAML snapshots. Download cloud-only files
before starting. The run path is an argument, not hard-coded into the viewer.

```bash
cd /path/to/esa-ssp-tactile-legwheel
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src/legwheel_encoder src/legwheel_kinematics src/legwheel_dataset \
  --ignore-src -r -y
sudo apt install ros-jazzy-foxglove-bridge
colcon build --symlink-install \
  --packages-up-to legwheel_dataset
source install/setup.bash
```

Repeat the `cd` and both `source` commands in every terminal below. Use the
same `ROS_DOMAIN_ID` in all three terminals if you set one.

The normal run layout is:

```text
run_3/
  experiment_config_run_3_snapshot.yaml
  camera_driver_config_run_3_snapshot.yaml
  rosbag/
    metadata.yaml
    rosbag_0.mcap
```

Older runs with `gemini_336_run_3_snapshot.yaml` instead of the camera-driver
snapshot are supported too. Depth must have been registered to colour
(`depth_registration: true`, `align_target_stream: COLOR`). Native, unregistered
depth is rejected because its optical pose needs a different extrinsic model.

#### 2. Terminal A: start the Foxglove bridge

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml use_sim_time:=true
```

The bridge must use simulated time so its live connection follows the bag's
historical timestamps. It will wait for `/clock` until Terminal C starts.

#### 3. Terminal B: start the reconstruction viewer

Choose the run directory on the ROS computer; quote paths containing spaces:

```bash
RUN_DIR="/path/to/Experiment_Recordings/experiment_3/run_3"
ros2 run legwheel_dataset visualise_run --run-directory "$RUN_DIR"
```

Wait until the viewer prints **`READY`**. It first indexes calibration, joint
angles, and image metadata. Image payloads are read from MCAP on demand, rather
than retaining all full-resolution images in memory. Keep this terminal open
to see compatibility notes and any skipped-frame or point-limit warnings.

#### 4. Terminal C: play the original bag

Set the same run path in this terminal:

```bash
RUN_DIR="/path/to/Experiment_Recordings/experiment_3/run_3"
ros2 bag play "$RUN_DIR/rosbag" --clock --rate 0.5
```

Start at half speed while inspecting alignment. Use `--rate 1.0` for normal
speed. The player terminal prints its keyboard controls, including pause and
resume. Only one bag player should publish `/clock` for this viewer.

#### 5. Connect and configure Foxglove

1. Choose **Open connection → Foxglove WebSocket**. Use
   `ws://localhost:8765` if Foxglove runs on the ROS computer, or
   `ws://<ROS_COMPUTER_IP>:8765` from another computer.
2. Add a **3D** panel. Set both **Fixed frame** and **Display frame** to
   **`replay_base`**. Set **Follow mode** to **Off** to keep the inspection
   camera stationary while the rig frames move.
3. Enable **`/legwheel/replay/accumulated_cloud`**. Select **BGR (packed)**
   colour mode using its `rgb` field. Start with point size **2–3 pixels**.
   Set **Decay time to 0**: each message already contains the accumulated cloud,
   so Foxglove should display the latest message without adding another trail.
4. Enable **`/legwheel/replay/scene`**. Cyan is the inner track, orange is the
   outer track, red is the preview patch, and yellow joins the frame origins.
   Frame-axis colours are red for X, green for Y, and blue for Z. Labels are included in the
   markers; enabling Foxglove's transform labels as well is optional.
5. Add a **Raw Messages** panel for **`/legwheel/replay/status`** to see the
   processed-frame count, retained voxels, skipped frames, and whether the
   point limit has been reached.
6. Optionally add an **Image** panel for recorded RGB. Current recordings use
   `/legwheel_rgbd/color/image_raw`; older recordings use
   `/camera/rgbd/rgb/image_raw`. This provides an image reference beside the 3D
   reconstruction. Save the Foxglove layout once configured.

This workflow uses a WebSocket connection: `ros2 bag play` supplies its clock
and playback controls. Opening only the original MCAP as a local Foxglove file
does not run this reconstruction node or create its visualization topics.

The viewer uses an independent `replay_*` TF tree, so recorded transforms and
the existing dataset/kinematics visualizers cannot overwrite its frame names.
No original driver frame is silently reparented.

Official references: [ROS 2 bridge setup](https://docs.foxglove.dev/docs/getting-started/frameworks/ros2?modality=live)
and [3D panel settings](https://docs.foxglove.dev/docs/visualization/panels/3d).

#### Pause, restart, loop, and seek

While paused, the accumulated cloud remains visible. The viewer can also finish
rebuilding a seek while paused. It republishes its current snapshot periodically
so a newly connected Foxglove client can receive it.

To restart from the beginning, stop the bag player with Ctrl+C and run its
command again; the viewer can stay running. For continuous loops, add `--loop`
to the bag-play command. When `/clock` moves backward, the viewer clears all
future points and rebuilds accumulation from the beginning up to the new time.
Forward seeks include intervening frames too. Large seeks rebuild in batches;
watch the frame count catch up while the interface remains responsive. Seeking
must change the ROS player's `/clock`, rather than only a Foxglove panel's view.

#### Adjust detail, range, and memory use

For example:

```bash
ros2 run legwheel_dataset visualise_run \
  --run-directory "$RUN_DIR" \
  --pixel-stride 4 \
  --voxel-size-m 0.01 \
  --max-points 500000 \
  --cloud-rate-hz 5 \
  --maximum-depth-m 3.0
```

Restart the viewer after changing settings. Run `visualise_run --help` through
`ros2 run legwheel_dataset` for the full option list.

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-directory` | required | Run containing `rosbag/` and configuration snapshots. |
| `--experiment-config` | run snapshot | Optional geometry YAML override; the original stays untouched. |
| `--pixel-stride` | `4` | Sample every fourth row and column. Smaller values retain more detail. |
| `--voxel-size-m` | `0.01` | Retain the first measured point in each 1 cm cube. Smaller values preserve more detail and use more memory. |
| `--max-points` | `500000` | Hard limit on retained voxels. New voxels are ignored after the limit, with a warning. |
| `--cloud-rate-hz` | `5.0` | Maximum selected depth-frame rate, measured in bag receipt time. Does not interpolate missing camera frames. |
| `--minimum-depth-m` | `0.15` | Reject closer depth measurements and invalid zero depth. |
| `--maximum-depth-m` | `3.0` | Reject more distant depth measurements. |
| `--rgb-max-delta-ms` | `50.0` | Maximum acquisition-time separation for RGB colour. Unmatched depth still appears in grey. |
| `--maximum-pose-gap-s` | `0.5` | Reject interpolation across larger interior gaps in joint/encoder samples. |
| `--depth-scale-m` | `0.001` | Metres per integer depth unit. Floating-point depth is already treated as metres. |
| `--alpha-off-rad` | `0.35` | Preview patch offset ahead of the camera around the circle. |
| `--alpha-sp-rad` | `0.10` | Preview patch's full angular span. |

Reduce `--cloud-rate-hz`, increase `--pixel-stride`, or increase
`--voxel-size-m` if processing or the bridge becomes slow. Increase voxel size
or the point limit if the viewer reports that it stopped retaining new terrain.
The first-point-per-voxel rule deliberately avoids averaging away duplicated
surfaces caused by a calibration error.

#### Compatibility and interpretation

- Both the current `/legwheel_rgbd/...` image topics and legacy
  `/camera/rgbd/...` topics are detected automatically. A matching pair of
  `CameraInfo` topics, `/legwheel_rgbd/gyro_accel/sample` (or legacy
  `/camera/imu`), `/rotation_encoder/ticks`, and `/wheel/wheel_state`
  are required. The experiment snapshot must contain a positive
  `ticks_per_legwheel_revolution`; this rig uses `62.5`.
- Older camera fields are converted locally: `sensor_height_mm / 1000` becomes
  `camera_height_m`. When `sensor_front_offset_m` is absent, the former signed
  `camera_beam_offset_m` is negated into `sensor_front_offset_m`; the oldest
  snapshots instead use `sensor_front_offset_mm / 1000`. Both old layouts use
  the former fixed `0.054 m` as the perpendicular `camera_beam_offset_m`.
  Supplying **both new offset fields** overrides this migration.
  `theta_y_kinematic_sign` must be supplied explicitly (`-1` for this rig).
  Every camera migration is logged; use `--experiment-config` with a corrected full copy if an older
  installation used different measurements. Recordings are left untouched.
- Geometry uses the current shared model: inner radius `beam_radius_m -
  wheel_width_m`, outer radius `beam_radius_m`, and IMU-derived beam pitch.
  The first recorded tick defines zero yaw, preserved across playback seeks.
  Clouds are rejected when their poses fall outside recorded coverage or across gaps with no IMU pitch estimate.
  Finite pitch estimates remain displayed even when confidence is low.
- Yaw uses calibrated central-encoder ticks with wheel-odometry refinement, matching
  the dataset's interval correction without its steady-torque selection. Pauses,
  acceleration and reverse motion remain visible. Wheel velocity is integrated
  between encoder changes and scaled to match both endpoint angles, preventing
  accumulated wheel-slip drift. If wheel coverage has gaps beyond
  `--maximum-wheel-gap-s` (default `0.05`), or its integral cannot be normalized,
  the affected intervals retain measured central-encoder yaw without refinement.
  Wheel gaps alone do not reject otherwise available poses. The existing wheel
  refinement and dataset-selection settings are unchanged; dataset position now
  uses the same recording origin as the viewers.
- Depth is reconstructed using its calibration at acquisition time, then moved
  into `replay_base` using estimated IMU pitch and wheel-refined encoder yaw. RGB is projected
  separately using its own calibration and acquisition pose. It is not assumed
  that aligned RGB and depth have identical distortion coefficients.
- The cloud covers the valid camera image, rather than only an ML patch. Raw
  points are not fitted to the ground, registered with ICP, or corrected with
  SLAM. Tracks and the preview patch lie on modelled `z=0`; their discrepancy
  with measured terrain remains visible. The preview patch is a geometric
  reference, not a claim of future wheel contact during reverse motion.
- Sparse recordings update in steps. Grey regions indicate missing nearby RGB,
  not necessarily missing depth. Increasing the RGB tolerance can introduce
  colours from a different exposure, so it should be a deliberate tradeoff.

#### Troubleshooting

| Symptom | Check |
| --- | --- |
| No replay topics | Source the rebuilt workspace, start Terminal B, and wait for `READY`. Check the ROS domain and bridge connection. |
| `READY` but no animation | Start the player with `--clock`; inspect `ros2 topic echo /clock --once`. |
| Empty 3D panel | Select `replay_base` for both frame settings and enable both replay topics. |
| Axes appear frozen between cloud frames | Disable **Sync timestamps** in the 3D panel; the accumulated cloud updates less frequently than frame axes. |
| Cloud is a single colour | Select **BGR (packed)** / `rgb`; grey points can legitimately lack a close RGB exposure. |
| Cloud stops covering new terrain | Check `/legwheel/replay/status` and the point-limit warning; adjust voxel size or point limit. |
| Frames are skipped | Read Terminal B's specific error: missing pose support, invalid geometry, calibration dimensions, or optical frame mismatch. |
| Double surfaces or ground displaced from tracks | Check geometry, encoder scale/sign, camera offsets, timestamp alignment, and actual ground height. Accumulation makes these errors visible. |

The implementation is split into `run_replay.py` (read-only indexing,
interpolation, cloud construction and accumulation) and `run_replay_node.py`
(ROS messages, clock following and scene drawing). The core is testable without
a ROS installation. An optional real-recording test accepts
`LEGWHEEL_SAMPLE_RUN=/path/to/run_3` and uses the test-only `mcap-ros2-support`
package; it is not a runtime dependency of the viewer.

### Build learning packets

Learning packet generation is strictly offline. It determines the steady-torque run,
constructs continuous encoder-corrected `theta_y(t)`, isolates a configured
physical terrain patch without overlapping the previous saved patch, and
attaches local telemetry from wheel entry to exit plus a fixed spectral window.
Stage 2 saves RGB/depth crops, the complete 3D point map, a robustly fitted plane, signed
along/radial slopes, and spectra for 15 motor and IMU telemetry channels.
Encoder motion is converted through the experiment's
`theta_y_kinematic_sign` before any spatial projection. Recordings moving away
from the forward-facing camera are rejected before packet generation.

Validate the configuration against the actual recorded rates and camera view
before creating a dataset:

```bash
ros2 run legwheel_dataset create_config \
  --run-directory /path/to/run_1
```

The interactive creator writes `/path/to/run_1/postprocess.yaml`. It starts
with values from that file if it already exists. Otherwise, it falls back on a
field-by-field basis to a configuration from one of the three nearest earlier
sibling runs, then to built-in defaults. At either value prompt, enter `-h` or `--help` for a
short explanation. The creator validates every change, asks before atomically
saving, and offers to analyse the result against the recording immediately.
After printing that JSON report, it asks separately whether to open the
blocking midpoint visualization.

```bash
ros2 run legwheel_dataset validate_config \
  --run-directory /path/to/run_1 \
  --processing-config /path/to/run_1/postprocess.yaml \
  --show-visualization
```

The optional visualization selects the synchronized RGB/depth pair nearest the
steady-motion midpoint. A blocking Matplotlib window shows both recorded images
at their camera resolutions plus two expanded diagnostic views, with the full
track, configured patch, patch centre, camera boundary and visibility reasons.
Omit `--show-visualization` when only the JSON report is required.

```bash
ros2 run legwheel_dataset isolate_packets \
  --run-directory /path/to/run_1 \
  --processing-config /path/to/run_1/postprocess.yaml
```

Run Stage 2 on the directory printed by Stage 1:

```bash
ros2 run legwheel_dataset extract_features \
  --dataset /path/to/run_1/derived/CONFIGURATION_HASH
```

Or run both stages together:

```bash
ros2 run legwheel_dataset build_dataset \
  --run-directory /path/to/run_1 \
  --processing-config /path/to/run_1/postprocess.yaml
```

Algorithm settings can be overridden with `--processing-config`; the complete
defaults are installed from
`src/legwheel_dataset/config/postprocess_defaults.yaml`. Raw runs are never
modified or overwritten. Every packet retains its complete synchronized RGB
and depth images. `packets/analysis_report.json` records measured acquisition
rates, torque/motion analysis, correction-factor statistics, and every rejected
candidate with its reason.

The exact equations, file formats, configuration meanings, slope signs, and
Foxglove workflow are documented in
[`src/legwheel_dataset/README.md`](src/legwheel_dataset/README.md).

To visualize derived packets while the original bag supplies `/clock`:

```bash
ros2 bag play /path/to/run_1/rosbag --clock

ros2 run legwheel_dataset visualise_stream \
  --dataset /path/to/run_1/derived/CONFIGURATION_HASH
```

The derived TF frames use `dataset_` child names, preventing conflicts with TF
recorded in the raw bag.

#### Post-processing configuration reference

The default processing file is:

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

Packets are selected by angular coverage: their centres must advance by at
least `alpha_sp_rad`, so patches never overlap and small gaps are possible.
Older configuration files may contain `maximum_packet_saving_frequency_hz`;
this retired setting is accepted and ignored.

Each local telemetry window starts and ends at the corrected `theta_y(t)`
crossings of that patch's entry and exit boundaries. Its duration and sample
count can vary between packets, including within one run. Samples include both
exact endpoints and are timed relative to patch-centre contact, which need not
be halfway through the local window. The fixed FFT window remains centred on
contact and may overlap other packets' FFT windows.

`alpha_off_rad` is a positive distance ahead; the measured travel direction is
applied automatically in encoder-motion coordinates. The selected angle is
then converted into the notebook/base convention using
`theta_y_kinematic_sign`. `alpha_sp_rad` is the full angular width of the patch.
Exact endpoints are inserted regardless of `track_point_count`.

Central-encoder ticks remain the motion reference for correcting `theta_y` and
finding patch contact time, but are not part of the resampled ML telemetry or
FFT. The common grid therefore contains 15 channels from the approximately
100 Hz leg, wheel and IMU streams. Encoder rate statistics are still reported
separately in `packets/analysis_report.json`.


## Key topics

| Topic | Message type | Producer | Meaning |
| --- | --- | --- | --- |
| `/legwheel/joint_states` | `sensor_msgs/msg/JointState` | CAN node | Hip and knee state. |
| `/wheel/wheel_state` | `sensor_msgs/msg/JointState` | CAN node | Wheel position, velocity, and torque estimate. |
| `/rotation_encoder/joint_state` | `sensor_msgs/msg/JointState` | Encoder node | Central rig rotation angle and velocity. |
| `/rotation_encoder/ticks` | `std_msgs/msg/Int64` | Encoder node | Signed ticks after configured startup zeroing and direction inversion. Offline position reference, timed by bag receipt. |
| `/rotation_encoder/config` | `std_msgs/msg/String` | Encoder node | Retained JSON of effective encoder settings, checked by recording preflight. |
| `/legwheel_rgbd/color/image_raw` | `sensor_msgs/msg/Image` | Orbbec driver | Raw RGB image stream. |
| `/legwheel_rgbd/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Orbbec driver | Raw RGB calibration. |
| `/legwheel_rgbd/depth/image_raw` | `sensor_msgs/msg/Image` | Orbbec driver | Raw depth image stream. |
| `/legwheel_rgbd/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | Orbbec driver | Raw depth calibration. |
| `/legwheel_rgbd/gyro_accel/sample` | `sensor_msgs/msg/Imu` | Orbbec driver | Raw combined camera IMU stream. |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | Post-processing visualization | Dynamic and fixed frame transforms regenerated while replaying a bag. |
| `/legwheel/visualization/kinematics_3d` | `visualization_msgs/msg/MarkerArray` | Post-processing visualization | Inner/outer track points and moving kinematic-chain origins for Foxglove. |
| `/legwheel/replay/accumulated_cloud` | `sensor_msgs/msg/PointCloud2` | `visualise_run` | Accumulated measured XYZ/RGB points in stationary `replay_base`. |
| `/legwheel/replay/scene` | `visualization_msgs/msg/MarkerArray` | `visualise_run` | Tracks, preview patch, labelled frame axes, chain, and camera frustum. |
| `/legwheel/replay/status` | `std_msgs/msg/String` | `visualise_run` | Processed frames, retained voxels, skipped frames, and point-limit state. |
| `/legwheel/dataset/rgb_annotations` | `visualization_msgs/msg/ImageMarker` | Dataset visualization | Selected RGB patch and track boundaries synchronized to bag time. |
| `/legwheel/dataset/depth_annotations` | `visualization_msgs/msg/ImageMarker` | Dataset visualization | Selected depth patch and track boundaries synchronized to bag time. |
| `/legwheel/dataset/geometry` | `visualization_msgs/msg/MarkerArray` | Dataset visualization | Track, patch, point cloud, plane, frame directions, and slope label. |
| `/legwheel/dataset/slope_text` | `std_msgs/msg/String` | Dataset visualization | Current along-track and radially-outward slope values. |
| `/legwheel/motor_status` | `std_msgs/msg/Bool` | Controller/experiment | `true` permits the motor node to enable; `false` disables it. |
| `/wheel/requested_torque` | `std_msgs/msg/Float64` | Wheel controller/experiment | Requested wheel torque in Nm; it is clamped by the CAN node. |

## Troubleshooting

- **`Package ... not found`:** source `/opt/ros/jazzy/setup.bash`, then this
  workspace's `install/setup.bash`. If still absent, rebuild.
- **`orbbec_camera` missing or wrong version:** rerun `vcs import . <
  dependencies.repos`, build, source the workspace, and check `ros2 pkg prefix
  orbbec_camera`.
- **Camera cannot open:** install its udev rules, reconnect it, and use a USB 3
  cable/port. Low RGB-D rate should be investigated first on raw
  `/legwheel_rgbd/...` topics with Foxglove and rosbag recording stopped.
- **Camera stamps pause or move backward:** keep `enable_sync_host_time: false`
  with the Gemini driver's default global time domain. On a VM, verify the
  camera is attached to the guest at USB 3 speed with `lsusb -t`, and write
  bags to the guest's native filesystem rather than a shared folder.
- **Encoder cannot open its port:** check `dmesg -w` while reconnecting it,
  correct `port` in the YAML, and check `dialout` access.
- **Motors never enable:** verify `enable_control:=true`, valid limits, a
  `true` status message, fresh feedback, and that the shutdown-position
  interlock has not detected movement.
- **Experiment preflight waits forever:** use `ros2 topic hz` on every required
  stream above, then inspect `header.stamp` on RGB, depth, and IMU. The camera
  checks require three forward timestamp steps, not merely repeated callbacks.
  Images must have advanced within 2 seconds; IMU and motor/encoder state have
  tighter freshness limits of 0.5 seconds.
- **A run aborts for a stopped camera stream:** treat this as a data-integrity
  failure. Keep the generated aborted run and inspect the raw Orbbec rates and
  timestamps with rosbag and Foxglove stopped. If they degrade only while
  recording, inspect CPU and disk throughput and ensure the bag is written to a
  native Linux filesystem rather than a virtual-machine shared folder.

Package-specific implementation and hardware details are available in
[`src/legwheel_can/README.md`](src/legwheel_can/README.md),
[`src/legwheel_rgbd/README.md`](src/legwheel_rgbd/README.md), and
[`docs/orbbec_gemini_336_setup.md`](docs/orbbec_gemini_336_setup.md).
