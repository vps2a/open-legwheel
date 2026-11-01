"""Save exact evaluated models and restore experiments without refitting.

Joblib files are Python objects: open only experiments from trusted sources.
JSON, CSVs and checksums make the recorded split/results independently auditable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .analysis import TelemetryDataset
from .evaluation_plots import comparison_figure
from .features import (
    FEATURE_SCHEMA_VERSION,
    PARTITIONS,
    ConfigurationError,
    ExperimentConfig,
    FeatureSettings,
    FeatureSpec,
    prepare_data,
    require_trainable,
)
from .training import (
    MODEL_NAMES,
    Experiment,
    FittedModel,
    evaluate_model,
    per_run_metrics,
)

EXPERIMENT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_experiment(experiment: Experiment, destination: Path) -> Path:
    """Stage a complete new folder; never overwrite a previous experiment."""
    destination = Path(destination).expanduser().resolve()
    if not destination.is_dir():
        raise ValueError("Choose an existing destination folder.")
    target = destination / f"telemetry_experiment_{uuid.uuid4().hex[:12]}"
    stage = Path(tempfile.mkdtemp(prefix=".telemetry_experiment_", dir=destination))
    dataset, data = experiment.dataset, experiment.prepared
    try:
        for name, model in experiment.models.items():
            joblib.dump(model, stage / f"{name}.joblib", compress=3)
        metric_rows, prediction_rows, run_rows = [], [], []
        for role, evaluations in experiment.evaluations.items():
            run_rows.extend(per_run_metrics(experiment, role))
            for name, result in evaluations.items():
                metric_rows.append(
                    {
                        "model": name,
                        "partition": role,
                        "packets": len(result.indices),
                        **result.metrics,
                    }
                )
                for index, prediction, score in zip(
                    result.indices, result.predictions, result.scores
                ):
                    prediction_rows.append(
                        {
                            "model": name,
                            "partition": role,
                            "export_id": dataset.packet_ids[index],
                            "source_id": dataset.source_ids[index],
                            "terrain_setup_id": dataset.terrain_keys[index][0],
                            "actual_class": int(dataset.class_ids[index]),
                            "predicted_class": int(prediction),
                            "decision_score_loose": float(score),
                        }
                    )
        _write_csv(stage / "metrics.csv", metric_rows)
        _write_csv(stage / "predictions.csv", prediction_rows)
        _write_csv(stage / "metrics_by_run.csv", run_rows)
        (stage / "figures").mkdir()
        for role in experiment.evaluations:
            figure = comparison_figure(experiment, role)
            FigureCanvasAgg(figure).print_png(stage / "figures" / f"{role}.png")
            figure.clear()
        lockfile = Path(__file__).resolve().parent.parent / "uv.lock"
        if lockfile.is_file():
            shutil.copyfile(lockfile, stage / "uv.lock")
        manifest = {
            "experiment_version": EXPERIMENT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "created_utc": experiment.created_utc,
            "dataset_path": str(dataset.root),
            "dataset_fingerprint": dataset.fingerprint,
            "config": asdict(experiment.config),
            "feature_names": list(data.feature_names),
            "feature_definitions": [
                {
                    "name": feature.name,
                    "input_unit": feature.input_unit,
                    "band_edges_hz": dataset.band_edges_hz[feature.band].tolist()
                    if feature.kind == "power"
                    else None,
                }
                for feature in experiment.config.features.features
            ],
            "band_edges_hz": dataset.band_edges_hz.tolist(),
            "class_mapping": {"0": "Firm", "1": "Loose"},
            "positive_class": 1,
            "decision_score_threshold": 0.0,
            "evaluated_partitions": list(experiment.evaluations),
            "partition_packet_ids": {
                role: [dataset.packet_ids[i] for i in np.flatnonzero(data.masks[role])]
                for role in PARTITIONS
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "model_parameters": {
                name: model.pipeline.named_steps["classifier"].get_params()
                for name, model in experiment.models.items()
            },
            "files": {
                str(path.relative_to(stage)): _sha256(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        (stage / "experiment.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage)
        raise
    return target


def read_experiment_manifest(path: Path) -> tuple[Path, dict]:
    """Read metadata before locating the dataset or deserializing models."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    try:
        manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        if (
            manifest["experiment_version"] != EXPERIMENT_VERSION
            or manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION
        ):
            raise ValueError(
                "This experiment uses an unsupported format or feature calculation version."
            )
        for key in (
            "dataset_path",
            "dataset_fingerprint",
            "config",
            "environment",
            "files",
        ):
            if key not in manifest:
                raise ValueError(f"Experiment metadata is missing {key}.")
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read saved experiment: {error}") from error
    return root, manifest


def _config_from_json(document: dict) -> ExperimentConfig:
    values = dict(document)
    features = dict(values.pop("features"))
    features["features"] = tuple(FeatureSpec(**item) for item in features["features"])
    return ExperimentConfig(features=FeatureSettings(**features), **values)


def load_experiment(path: Path, dataset: TelemetryDataset) -> Experiment:
    """Verify identity and compatibility, then restore exact fitted pipelines.

    No fit method is called. Recomputed displays use the saved models, the same
    inputs and only partitions that were already evaluated in the saved file.
    """
    root, manifest = read_experiment_manifest(path)
    if dataset.fingerprint != manifest["dataset_fingerprint"]:
        raise ConfigurationError(
            "This dataset does not match the saved experiment. Locate the original export."
        )
    if manifest["environment"]["scikit_learn"] != sklearn.__version__:
        raise ConfigurationError(
            "The saved model needs its original scikit-learn version. See its uv.lock/environment metadata."
        )
    for relative, expected in manifest["files"].items():
        candidate = (root / relative).resolve()
        if (
            not candidate.is_relative_to(root)
            or not candidate.is_file()
            or _sha256(candidate) != expected
        ):
            raise ValueError(f"Saved experiment file is missing or changed: {relative}")
    config = _config_from_json(manifest["config"])
    data = prepare_data(dataset, config)
    require_trainable(dataset, data)
    for role in PARTITIONS:
        actual = [dataset.packet_ids[i] for i in np.flatnonzero(data.masks[role])]
        if actual != manifest["partition_packet_ids"][role]:
            raise ConfigurationError(
                "The saved split no longer produces the same eligible packets."
            )
    models = {}
    for name in MODEL_NAMES:
        model = joblib.load(root / f"{name}.joblib")
        if (
            not isinstance(model, FittedModel)
            or model.name != name
            or model.features != config.features
            or model.feature_names != data.feature_names
            or not np.array_equal(model.band_edges_hz, dataset.band_edges_hz)
        ):
            raise ValueError(
                f"The saved {name} does not match the experiment feature definitions."
            )
        models[name] = model
    roles = manifest["evaluated_partitions"]
    if not {"training", "validation"}.issubset(roles) or set(roles) - set(PARTITIONS):
        raise ValueError("The saved evaluated partitions are invalid.")
    evaluations = {
        role: {
            name: evaluate_model(model, dataset, data, role)
            for name, model in models.items()
        }
        for role in roles
    }
    return Experiment(
        dataset, config, data, models, evaluations, manifest["created_utc"]
    )
