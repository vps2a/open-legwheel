"""Generate the schema and scientific conventions beside every export."""


def export_readme(name, options, packet_count):
    return f"""---
configs:
  - config_name: packets
    default: true
    data_files:
      - split: unassigned
        path: packets/*.parquet
  - config_name: runs
    data_files:
      - split: unassigned
        path: runs.parquet
---

# {name}

This export contains {packet_count} selected terrain packets. No packets have
been removed for overlap. Labels may be null. Training, validation and test
membership has not been assigned: the pool is explicitly named `unassigned`.
Choose and document a dataset license before publishing this directory.

## Loading

```python
from datasets import load_dataset

packets = load_dataset("parquet", data_files="packets/*.parquet")["train"]
runs = load_dataset("parquet", data_files="runs.parquet")["train"]
# The local generic Parquet loader calls its default pool "train"; that name
# does not establish a scientific train/test split. After uploading this folder:
# packets = load_dataset("your-account/your-dataset", "packets", split="unassigned")
```

The packet files include Hugging Face feature metadata. RGB, depth and masks
are embedded lossless PNG images; no source paths are needed. `depth_scale_m`
converts stored uint16 depth to metres. Mask PNG values are 0 (outside) and
255 (inside). Keep masks distinct from missing/invalid depth pixels.

## Packet columns

- `export_id` is the globally scoped reference used by overlap vectors.
  `packet_id` and `packet_number` retain the original ID and chronological,
  1-based source-manifest position. Numbers can have gaps after selection.
- `source_id` joins to `runs.parquet`. Experiment/run and processing IDs are
  also available directly. Only one processing version of each run is allowed.
- `class_id` uses 0: Firm Terrain; 1: Loose Terrain; 2: Firm to Loose Transition;
  3: Loose to Firm Transition. Null means unlabelled, never class zero.
  `classification_method` and `model_id` preserve annotation provenance.
- `terrain_and_light_ID` is preserved. For `KK_01_L1`, `laboratory_id` is `KK`,
  `terrain_setup_id` is `KK_01`, and `lighting_id` is `L1`. Missing IDs remain
  null. Keep the same physical setup together when assigning evaluation splits,
  including recordings made under different lighting.
- `local_values` and `spectral_values` have shape [time, channel]. Their
  corresponding boolean `*_valid` masks and integer `*_relative_times_ns` are
  preserved. Relative timestamps are centred on `contact_time_ns`. All absolute
  timestamps are integer nanoseconds. Original float64 precision is retained;
  nonfinite values become null, not zero. `channel_names` and `channel_units`
  define column order. Angular quantities use radians, effort N*m and linear
  acceleration m/s^2 (per-channel power units are the square of these units).
- `spectral_power` is the saved PSD [frequency, channel], in channel-units²/Hz.
  `frequencies_hz` gives its frequency grid. `dominant_frequency_hz` and
  `dominant_power` have shape [peak rank, channel]; dominant power is a PSD
  ordinate, not integrated band power.
- `band_power` has shape [band, channel], in channel-units². Bands have width
  {options.band_width_hz:g} Hz, with actual edges in `band_edges_hz`. Integration
  is the discrete sum of PSD bin values times FFT frequency spacing. DC is
  excluded. Bins belong to [lower, upper), with the Nyquist endpoint included
  in the final band. An odd-length FFT need not have a bin exactly at Nyquist.
  There is no interpolation, zero-padding, spectral recomputation or relative
  normalization. Invalid/empty bands and bands extending beyond a source's
  Nyquist frequency are null. Bands share edges across this export, even when
  source sample rates differ. Inspect per-row coverage before comparing them.
- `rgb_patch`, `depth_patch`, their masks and `*_crop_origin_uv` preserve the
  selected image patches. `geometry` contains projected tracks and physical
  patch boundaries. Full-image calibration is retained in `metadata_json`.
- Plane slopes, plane normal/offset, fit metrics and IMU uncertainty are
  accessible as numerical columns. `imu_uncertainty_deg` is the saved two-sigma
  pitch uncertainty. `metadata_json` and `features_json` preserve additional
  original fields. Image/depth-derived features are supporting measurements,
  not inputs to the telemetry-only benchmark unless explicitly declared.
- Full RGB/depth frame columns included: {options.include_full_frames}.
  Numerical `point_map` included: {options.include_point_maps}.

## Overlap and maximum subset counts

FFT time bounds are contact time plus the first/last saved spectral timestamps.
Intervals are closed: shared endpoints count as overlap. References are symmetric,
exclude self, and include only other selected packets from the same run.
`temporal_overlap_ids`/`spatial_overlap_ids` reference `export_id`; the matching
`*_overlap_packet_numbers` retain the source packet numbers in the same order.

FFT boundary angles are interpolated between corrected encoder ticks using
relative integer time differences. Conservative lower/upper unwrapped angle
bounds include the surrounding tick positions. They assume monotonic movement
between ticks; unrecorded between-tick reversals and calibration errors are not
bounded by this method. Detected reversals, out-of-range windows and movement-event
intervals greater than 2 seconds block export. Exactly 2 seconds is allowed. This
is an exporter-only interpolation limit, saved as `maximum_encoder_gap_ns` in the
run table; it is independent of the pitch estimator's message-gap threshold.
Sparse count changes can reflect slow motion and do not establish a recording
dropout. Original message-gap status cannot be inferred from these event times.
No extrapolation or gap filling is performed.

Spatial overlap compares conservative wheel-travel arcs modulo 2*pi, including
repeated laps and arcs crossing zero. Coverage of a whole revolution overlaps
every packet. This is the track traversed during the FFT window, not a camera
footprint or the full robot footprint. Angular comparisons use a 1e-12 rad
tolerance. Spatial comparisons are only within each run, not across runs with
potentially different angular origins.

Each run records the exact maximum temporally non-overlapping subset count
(earliest-finish interval scheduling) and exact maximum spatially non-overlapping
subset count (circular-arc scheduling by trying each anchor). These are separate
optimizations, may describe different subsets, and are **not estimates of
statistically independent sample counts**. No subset is filtered or labelled
as preferred. Counts describe the exported selection, not all source packets.

## Reproducibility

`runs.parquet` preserves source configurations/provenance in `provenance_json`,
with machine-local source paths and processing creation times omitted.
`export_manifest.json` records settings, class counts, selected references,
run summaries and Parquet SHA-256 checksums. The directory hash covers normalized
exported Arrow records, settings, run configuration and exporter version. It is
independent of selection order, destination, display name and export time.
It is not a checksum of Parquet compression bytes. The dependency lockfile and
exporter version identify the serializer used to calculate it.
"""
