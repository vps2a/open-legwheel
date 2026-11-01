# LegWheel Supporting Tools

This directory contains standalone analysis tools that do not participate in
online ROS processing or rosbag recording:

- `kinematics_calculator.ipynb` verifies the symbolic robot kinematics;
- `dataset_browser.py` is a desktop browser for inspecting derived packets
  and manually classifying their terrain patches.
- `batch_dataset_building_tool.py` builds datasets for selected runs across
  multiple recorded experiments through a desktop wizard.
- `dataset_exporter.py` assembles selected processed packets into a portable
  Parquet dataset, with band powers and temporal/spatial overlap analysis.
- `ML_tools/telemetry_classification_visualiser_tool.py` compares classified
  packet telemetry through STD histograms and motor/IMU band-power plots, and
  trains/evaluates logistic regression, linear SVM and RBF SVM models.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.12 (managed automatically by `uv`)
- VS Code with Microsoft’s **Python** and **Jupyter** extensions, if using the
  notebook

## Installation

From the repository root:

```bash
cd supporting_files
uv sync
```

This creates or updates `supporting_files/.venv` from the exact versions in
`uv.lock`. PySide6, NumPy and Matplotlib required by the dataset browser are
installed by this command as well.

To add or update a dependency during development, use `uv add` rather than
calling `pip` directly:

```bash
cd supporting_files
uv add <package-name>
```

## Telemetry classification visualiser

From the repository root:

```bash
cd supporting_files
uv sync
uv run python ML_tools/telemetry_classification_visualiser_tool.py
```

The installed equivalent is `uv run legwheel-telemetry-visualiser`. An export
can also be opened directly:

```bash
uv run python ML_tools/telemetry_classification_visualiser_tool.py \
  --dataset /path/to/Exported_Datasets/NAME_HASH
```

Choose **Open dataset folder…**, or select an exported packet with **Open Parquet
file…**. Both load **all** `packets/*.parquet` shards from that export, not just
one file. `runs.parquet` is not a packet table; selecting it locates the companion
packet folder. Only required numerical/grouping columns are loaded; embedded
images are not decoded. Loading runs in a cancellable background worker.

- The persistent terrain tree groups physical setups (e.g. `KK_01`) and offers
  individual terrain/light children (`KK_01_L1`). Parent checkboxes control all
  their children. Unknown IDs remain explicitly selectable.
- Class checkboxes and clickable plot legends control the same four global
  class selections. Colours stay consistent across all three tabs. Unlabelled
  and unsupported classes are counted separately and excluded from plots.
- **Motor standard deviations** displays six histograms: hip/knee/wheel
  velocity and effort. Each valid packet contributes one within-window STD.
  Choose spectral or local patch windows; the default is spectral. Computation
  uses the validity mask and finite samples, requires at least two samples, and
  uses population STD (`ddof=0`). Invalid measurements remain missing, not zero.
- **Histogram** switches between probability density (default) and packet
  counts. Bin count is configurable. Every class uses the same edges within a
  channel. Density is normalized separately for each visible class, in the
  original STD units, including when the horizontal axis is logarithmic.
- **Motor band powers** and **IMU band powers** each display six plots. They
  show the equal-packet arithmetic mean of saved absolute powers in each saved
  frequency band. Error bars describe **between-packet sample STD** (`ddof=1`),
  not uncertainty in the mean or confidence intervals. Type a multiplier from
  **0 to 3**, default **2**, and press Enter. A single valid value shows its
  mean without an error bar. Null/nonfinite/negative powers are excluded per
  measurement; the right panel shows actual contributor counts per channel.
- **Value scale** switches histogram x-axes and band-power y-axes between
  linear and logarithmic. Log histograms use logarithmically spaced common
  edges. Zero STD values and zero class/band means remain in the underlying
  statistics but are omitted from log display and counted in the summary.
  Power statistics and SD bars are calculated in linear units in both views.
  Lower SD limits at or below zero are truncated at a positive plot floor and
  marked with **▼**, with the truncation count shown in the summary.
- The right panel continuously reports selected packet/class/run/setup counts,
  valid measurements in the active tab, exclusions and input diagnostics. The
  Matplotlib toolbar supports zooming, panning and saving the active figure.

Every packet has equal weight, regardless of run length. Temporal/spatial
overlap and transition-adjacent packets are retained. The tool does not claim
that packets are independent. Band powers are read exactly as exported;
changing the STD window does not recalculate them. Different band layouts
within one dataset are rejected rather than silently averaged together.
Existing odd-length FFT exports can underweight their highest frequency bin;
this visualiser does not silently correct or rewrite that calculation.

All source files and labels remain unchanged. The **Data exploration** page is
separate from **Model training and comparison**: hiding a class or terrain in
the plots does not change training assignments. The dataset selector is shared.

### Model training and comparison

This page has three tabs and a persistent summary of eligible training,
validation and reserved test packets. Models use only class **0 (Firm)** and
**1 (Loose)**. Loose is the positive class. Every model receives the same feature
columns and eligible packets within a comparison.

1. **Data split:** assign entire runs to Training, Validation, Testing or
   Excluded. All runs initially start Excluded; the tool never silently makes
   a random packet split. **New-setup** mode keeps each physical terrain setup
   together across recordings and lighting. Use the setup's partition selector;
   individual run selectors are disabled. **New-run** mode permits different
   recordings of the same physical ground in different partitions. A setup
   selector can still assign all its runs together. Switching back to new-setup
   mode clears conflicting assignments within a setup. Runs with no binary
   labels are excluded. Counts in the tree are before filtering; the summary
   reports eligible counts and exclusions.
2. **Features and models:** choose a preset or expand the feature tree to select
   individual STD channels and motor/IMU channel-band pairs. Wheel speed and
   both slopes are independent options. The compact preset is the default
   (39 features with the sample export's five frequency bands). The spectral
   window supplies both STD and mean absolute wheel speed by default. Powers
   always come from the saved spectral windows. Band-power log transformation
   is on initially and is independent of the exploration plot's axis scale.
3. **Performance:** **Train all three models** fits the models on Training only
   and displays Validation results. Select Training to inspect fitting results.
   Reserved test packets are not scored until **Evaluate reserved test set**
   is pressed. Evaluation never refits a model. A test set is optional; training
   and validation must each contain both classes after filtering.

**Transition filtering** is enabled initially. A binary packet is excluded if
its closed FFT interval intersects a labelled class-2/3 packet's local patch
interval in the same run. Optional padding extends the transition interval on
both sides, initially by zero seconds. Transition annotations remain context
even though they are never classification targets. Missing required timing
in an assigned run produces an explicit error; disabling the filter is a
deliberate alternative for older exports. Only transitions present in the
export can be checked; this filter does not establish purity in unannotated
areas and does not compare clocks/angles across recordings.

Packets with any missing selected feature are excluded consistently for all
models. The summary separately counts incomplete rows and transition-adjacent
rows remaining after that check. Each packet has equal weight.

Each fitted pipeline contains training-only preprocessing and its classifier:

- Optional `log10` transformation affects power columns only. Each column's
  positive floor is 1% of its smallest positive **training** power, or 1e-12
  when all its training values are zero. Values below the floor are clipped.
- Standardization learns its means/scales from Training only. Validation/test
  statistics cannot change these fitted parameters.
- Logistic regression has fixed L2 regularization, C=1, and no tuning controls.
- Linear SVM exposes positive **C**, the margin-violation penalty. Its margin
  is learned rather than specified directly.
- RBF SVM exposes **C** and **gamma**. Automatic gamma uses scikit-learn's
  `scale` rule on the transformed training features; a positive numerical
  override is available. There is no automatic parameter search in this version.

Accuracy, TPR/recall, FPR, balanced accuracy, precision, F1 and ROC AUC are shown
for each model, with per-run metrics below. Confusion matrices show counts and
percentages within each actual class. ROC curves use decision scores; SVM
scores are not calibrated probabilities. A nonnegative score predicts Loose,
equivalent to probability >=0.5 for logistic regression. The operating points
are marked on ROC curves. Undefined rates/AUC display **N/A**, not zero.
Metrics do not include packet-level independence claims or confidence bounds.

Training runs in a background worker. Cancellation takes effect after the
current model operation. Changing features, parameters or assignments marks
previous results **outdated** and disables saving/test evaluation until models
are trained again. Models that fail to converge are reported rather than
silently accepted.

### Saving and reopening experiments

**Save experiment…** chooses a parent folder and creates a unique
`telemetry_experiment_ID/` directory. It saves the exact evaluated training-set
models; it never refits on combined training/validation data. A staged write
publishes the folder only after completion and never overwrites an earlier one.

- `logistic_regression.joblib`, `linear_svm.joblib`, `rbf_svm.joblib`: fitted
  preprocessing/classifiers and their ordered feature definitions.
- `experiment.json`: split assignments and exact eligible packet IDs, feature
  settings and band edges, dataset fingerprint/location, model settings,
  software versions, evaluation state and file checksums.
- `metrics.csv`, `metrics_by_run.csv`, `predictions.csv`: measured results and
  packet-level labels, predictions and decision scores. Reserved test scores
  are absent until test evaluation has been explicitly requested.
- CSV rates are fractions from 0 to 1; the GUI displays rates as percentages.
  Undefined CSV metric cells are empty.
- `figures/`: confusion matrices and ROC curves for evaluated partitions.
- `uv.lock`: the dependency lockfile when available beside the installed tool.

**Open experiment…** restores controls, fitted models and results without
calling any training method. It uses the currently open dataset if it matches,
otherwise loads the recorded location or asks for the relocated export. The
fingerprint covers loaded numerical features, labels, grouping and timing;
moving an export is allowed, while changed model inputs/labels are rejected.
The original scikit-learn version is required, and recorded file checksums
are verified before model loading. Joblib files contain Python objects: open
only your own experiments or files from a trusted source.

The code is separated by responsibility: `analysis.py` reads/caches packet
measurements, `features.py` defines inputs and split/filter rules, `training.py`
fits/evaluates models, and `experiments.py` saves/restores them. GUI modules
coordinate these operations without duplicating the numerical calculations.

## Dataset browser

### Starting the browser

From the repository root:

```bash
cd supporting_files
uv sync
uv run python dataset_browser.py
```

The installed console-script equivalent is:

```bash
cd supporting_files
uv run legwheel-dataset-browser
```

An explicit dataset may be supplied when scripting or testing:

```bash
uv run python dataset_browser.py \
  --dataset /path/to/run_1/derived/CONFIGURATION_HASH
```

The directory picker expects the configuration-hash directory itself. It must
contain `provenance.json`, `manifest.jsonl`, and `packets/`. The last valid
selection is remembered in the operating system's application settings. Use
**Change dataset…** at the top-right to select another one.

### Controls and views

- Click **Previous packet** and **Next packet**, or press the left/right arrow
  keys, to move through the chronologically ordered manifest. Navigation stops
  at the first and last packet.
- To open a packet directly, enter its 1-based position in the **Go to** field
  at the bottom and click **Jump to packet** or press **Enter**. For example,
  entering `64` opens the 64th packet in the sorted list. Invalid or
  out-of-range values leave the current packet unchanged and show an inline
  explanation.
- The title identifies the experiment and run. The lower navigation label shows
  the current packet number and packet ID.
- **Images and patches** displays the complete RGB and metric depth images with
  the physical patch highlighted. **Show cropped masked patches** switches both
  images to the smallest crop containing that patch.
- Depth uses fixed dataset-wide limits by default, so colours remain comparable
  between packets. Per-packet automatic scaling is available as an explicit
  checkbox.
- The bold along-travel and cross-track slopes are the primary Stage 2 outputs.
  Plane-fit diagnostics appear beside them.
- Motor and joint signals are plotted on the left; IMU signals are plotted on
  the right. The synchronized tabs select the local contact window, the longer
  spectral window, or the stored FFT power spectrum.
- **3D diagnostic** shows the depth point map, RANSAC inliers, physical patch
  boundaries, and fitted plane normal.
- **Metadata** lists timestamps, synchronization, angles, contact timing,
  travel directions, and plane-fit quality.

Stage 1-only packets remain inspectable. Their images, masks, metadata and
available telemetry are displayed, while slope, FFT and 3D views clearly state
that Stage 2 is unavailable.

### Manual terrain classification

The row above the plots contains four class buttons:

| Class | Highlighted terrain patch |
| --- | --- |
| 0 | Firm Terrain |
| 1 | Loose Terrain |
| 2 | Firm to Loose Transition |
| 3 | Loose to Firm Transition |

**Manual Classification Mode** is off when the browser starts and when another
dataset is opened. The buttons are disabled until it is enabled. Enabling it
shows one confirmation warning that subsequent selections will manually
overwrite previous classifications; enabling the mode alone changes no labels.

Click a class button or press **0**, **1**, **2** or **3** to save the current
packet's classification immediately and advance to the next packet. Number
shortcuts are ignored while typing in **Go to**, and holding a number does not
classify multiple packets. Manual mode remains enabled while navigating within
the current dataset. At the last packet, a selection saves and stays there.

An existing class is highlighted in light blue, including while manual mode is
off. The **Metadata** tab shows the class, classification method and model ID.
No automatic classifier is run by the browser.

### Data safety

Opening an older dataset adds missing `class_id`, `classification_method` and
`model_id` fields to every packet and existing feature record. Unclassified
fields are `null`; existing labels are preserved. This upgrade also runs with
manual mode off and requires write access to the selected dataset.

Each manual selection synchronizes `metadata.json`, `stage2/features.json` and
the matching dataset-wide `features.jsonl` row. A manual classification records
`classification_method: "manual"` and `model_id: null`. Stage 1-only packets
update metadata; Stage 2 later carries the label into feature files. Writes use
complete temporary files followed by replacement, with rollback if a save
fails. The browser only advances after the save succeeds.

Images, numerical arrays and the manifest remain unchanged. The remembered
dataset path is stored in the normal per-user Qt application settings outside
the dataset directory; manual mode is never remembered.

### Troubleshooting

If Qt cannot connect to the graphical display, start the program from a desktop
terminal in the Ubuntu VM rather than an SSH session. On Linux, missing Qt
system-library errors should be resolved through the distribution package
manager; do not install a second Python environment over `.venv`.

## Dataset exporter

This standalone tool needs no ROS installation, sourced workspace, or original
MCAP recordings. It only reads existing derived dataset folders. Source files,
including legacy classification metadata, are never upgraded or modified.

```bash
cd supporting_files
uv sync
uv run python dataset_exporter.py
```

The installed command is `uv run legwheel-dataset-exporter`. Optionally start
with `--directory /path/to/Experiment_Recordings` or a single configuration-hash
folder. The environment includes Hugging Face Datasets, PyArrow and Pillow for
typed Parquet output and lossless image encoding.

1. **Add source folder…** discovers processed datasets recursively, skipping raw
   bags and Python environments. Add more folders to combine collections.
2. Expand the experiment/run/processing-version tree and check individual
   packets or whole groups. A run or group checkbox selects all its exportable
   packets in one action, skipping blocked packets. Its checked/partial state
   reflects only exportable packets; groups with none have no checkbox.
   Group checkboxes include filtered packets. Select only one processing version per run.
   Search and class filters preserve existing selections; **Select visible** and
   **Deselect visible** act only on matching, usable packets. Unlabelled packets
   remain selectable. Selecting a packet displays its RGB patch and any blocking
   reason. The review panel shows class counts and an approximate input size.
3. Choose a dataset name and an existing destination folder. Adjust the uniform
   frequency band width (default **10 Hz**). Full RGB/depth frames and numerical
   point maps are separate options, both initially off. Patches, masks, telemetry,
   saved spectra/features and provenance are always included.
4. **Export selected packets** re-reads the selected sources, validates their
   FFT-window trajectory coverage, performs a second overlap-analysis pass and
   writes the export. Progress and cancellation remain available. Closing during
   work requests cancellation and waits for temporary output cleanup.

The output is `NAME_HASH/`, containing `packets/part-*.parquet`, `runs.parquet`,
`export_manifest.json`, and a generated `README.md`. Packet Parquet files include
Hugging Face feature metadata, with embedded PNG images and numerical arrays.
Depth PNGs preserve uint16 values and their metre scale. Missing numerical values
remain null, with telemetry validity masks retained. The companion run table holds
configurations and summary counts. The generated README describes every column
group, loading, units, calculations, and limitations. Its Hub configuration exposes
packet and run tables separately, with an explicitly **unassigned** split.
Choose a license and review the dataset card before publishing.

Band power is absolute power per channel, calculated as the sum of saved PSD bins
times frequency spacing. DC is excluded. Bins belong to `[lower, upper)` with
Nyquist included at the final edge. The exporter preserves original FFT resolution
and saves the actual band edges. Invalid/empty bands and bands beyond a source's
Nyquist range are null; mixed sample rates share one band layout. Dominant powers
remain PSD peak ordinates, distinct from integrated band powers.

Temporal overlap compares closed FFT time intervals. Spatial overlap compares
conservative wheel-travel arcs on the circular track, including repeated laps and
zero-angle wraparound. Boundary angles use linear interpolation between corrected
encoder ticks; conservative bounds include the surrounding ticks. This assumes
monotonic motion between ticks and does not bound unrecorded reversals or physical
calibration errors. Missing trajectory coverage, detected reversals, and intervals
between encoder movement events greater than **2 seconds** block affected packets.
Exactly 2 seconds is allowed. This exporter-only interpolation limit is independent
of the pitch estimator's message-gap threshold, and is saved as
`maximum_encoder_gap_ns` in the run table. Long intervals between count changes
can reflect slow motion; they do not establish missing sensor messages. No
extrapolation is used. Shared endpoints count as overlap.

Each packet stores symmetric overlap vectors referencing other **selected packets
in its run**, using stable export IDs and original 1-based packet numbers. Each run
stores the exact maximum temporally non-overlapping subset count and the separate
maximum spatially non-overlapping subset count. These are not statistical
independence estimates. All selected packets are exported; no subset is filtered.

`KK_01_L1` is preserved and separated into laboratory `KK`, physical terrain setup
`KK_01`, and lighting `L1`. Missing IDs produce null grouping fields; malformed or
conflicting IDs are reported during discovery. Splits remain unassigned so future
analysis can group the same physical setup across lighting changes.

The hash covers normalized exported content, labels, source configurations,
selection, settings, and exporter/schema version. It excludes output location,
selection order, display name, and export time. Existing output folders are not
overwritten. Exports are staged in a temporary sibling folder and published only
after completion; failures and cancellations remove that temporary folder.

## Running the kinematics notebook

1. Open an `.ipynb` file in VS Code.
2. Click **Select Kernel → Python Environments**.
3. Select `supporting_files/.venv/bin/python`.
4. Run cells with **Shift+Enter**.

No separate Jupyter server or manual environment activation is required.

## Development and checks

Run the standalone unit tests without requiring ROS:

```bash
cd supporting_files
uv run python -m unittest discover -s tests -v
```

Commit source files, `pyproject.toml`, and `uv.lock`. Keep `.venv/` and
`.ipynb_checkpoints/` out of Git.

Before committing, restart the kernel and run all cells to verify reproducibility.

This environment is intended for standalone calculations. Notebooks importing ROS packages require a compatible ROS environment.

## Batch dataset builder

Run this tool on Ubuntu with ROS 2 Jazzy and the built workspace sourced.
From the repository root:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cd supporting_files
uv sync
uv run python batch_dataset_building_tool.py
```

The installed command is `uv run legwheel-batch-dataset-builder`. Optionally
provide `--directory /path/to/Experiment_Recordings` to start with a directory
already loaded. The GUI uses the supporting-tools environment; each analysis
runs under `/usr/bin/python3` so it can load the sourced ROS libraries. For a
different ROS Python installation, pass `--ros-python /path/to/python3`.
Rebuild and source the workspace, including `legwheel_experiments` and
`legwheel_dataset`, after updating this repository so that the worker sees the
current validation and processing code.

1. Choose a directory containing experiment folders as immediate children.
   Experiment YAML files are recognized by their contents, including the
   experiment ID and geometry/hardware mappings, rather than their filenames.
2. Select experiments, then choose runs on a separate page for each experiment.
   Runs are ordered numerically. **Select all available** and **Clear all**
   controls are provided for both experiments and runs.
   Before run selection, a dialog collects missing terrain/light IDs and asks
   you to resolve conflicts. An ID is a case-sensitive string of ASCII letters,
   digits, underscores and hyphens, containing at least one letter or digit.
   Multiple experiments may share the same ID. No conflicting value is selected
   automatically; existing values are offered for an explicit choice.
3. Each run shows **Completed Datasets: N**. **Partial Datasets** and **Failed
   Attempts** appear only when their counts are nonzero. A completed dataset
   finished Stage 2; Stage 1-only or otherwise incomplete outputs are partial.
   Failed and cancelled attempts are counted as failures.
4. Invalid and unreviewed runs remain selectable with a warning. **Mark valid**
   immediately creates or updates `run_valid: true` in `run_config.yaml`,
   preserving its other values. This is saved even if no batch is started.
   Runs missing required inputs remain visible but cannot be selected.
5. Edit the shared settings, initially populated from the builder's defaults.
   All 16 numeric settings are available, with units and hover explanations.
   **Start batch** validates them and saves the same commented, timestamped
   `postprocess.yaml` in every selected run, replacing previous settings.
6. The tool processes one run at a time using its recorded experiment snapshot
   and the production packet/feature stages. Outputs use the normal
   `run_N/derived/CONFIGURATION_HASH/` location.

The progress screen shows overall 0–100% progress, the current experiment/run,
its own progress bar and processing phase, and an approximate remaining time.
The ETA initially says **Estimating…** and updates from observed processing
speed. Each run contributes equally to overall progress; longer runs can take
more time despite contributing the same share of the percentage.

**Save IDs** immediately updates `environment.terrain_and_light_ID` in each
experiment configuration and **all** its recorded run snapshots, including
unselected runs. It also synchronizes the top-level and embedded experiment IDs
in every existing dataset's `provenance.json`. Other configuration values and
packet files are preserved. Cancelling the ID dialog changes no files; cancelling
the batch afterwards does not undo IDs already saved.

The dialog warns which existing datasets need regeneration before saving an
added or corrected ID. Those provenance files retain their original dataset
identifiers and gain `requires_regeneration: true` and a reason. Run-selection
warnings identify affected runs. Selected runs are rebuilt when the batch starts;
unselected runs wait for a later batch. Previous datasets and their packet labels
are retained. New datasets have `requires_regeneration: false`; packet labels
are not automatically transferred from older datasets.

A completed dataset matching the selected settings and not marked for
regeneration is skipped and listed in
the final summary. Before retrying an incomplete or failed output at the normal
location, that folder is renamed with an `_attempt_TIMESTAMP_UNIQUEID` suffix.
The same preservation applies to completed datasets marked for regeneration if
the new build needs their output path.
Its files and any classifications are preserved. The tool never deletes an
earlier dataset or changes the raw recording.

Individual failures do not stop later runs. **Cancel batch** immediately kills
the active analysis process, preserves its partial output and records its
cancelled status in `batch_build_status.json`. Later runs are left unstarted;
completed runs remain available. Closing the window during a build performs
the same cancellation. If cancellation happens while settings are being saved,
only the settings files already written are changed.

The final table lists each run as completed, skipped, failed, cancelled or not
started, with its output path and explanatory message. **New batch** refreshes
dataset counts. **Show processing log** exposes detailed errors when needed.
