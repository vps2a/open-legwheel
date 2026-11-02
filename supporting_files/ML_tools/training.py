"""Training-only preprocessing and comparable binary model evaluation.

The GUI calls these functions from a worker. They contain no Qt code and never
write to the dataset. Test evaluation is a separate, explicit operation.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .analysis import TelemetryDataset
from .features import (
    ConfigurationError,
    ExperimentConfig,
    FeatureSettings,
    PreparedData,
    prepare_data,
    require_trainable,
)

MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
}
METRIC_NAMES = {
    "accuracy": "Accuracy",
    "tpr": "TPR / recall",
    "fpr": "FPR",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "f1": "F1",
    "roc_auc": "ROC AUC",
}


class TrainingCancelled(Exception):
    """Cancellation is checked between model fits/evaluations."""


class PowerLogTransformer(TransformerMixin, BaseEstimator):
    """Log selected power columns, learning a positive floor from training only.

    Positive powers below the floor and zeros are clipped to that floor. STD,
    speed and slope columns are passed through unchanged. The floor is 1% of
    the smallest positive training value in the column, or 1e-12 if all zero.
    """

    def __init__(self, power_columns: tuple[int, ...] = ()):
        self.power_columns = power_columns

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                "Model inputs must be a finite two-dimensional feature matrix."
            )
        self.n_features_in_ = values.shape[1]
        floors = []
        for column in self.power_columns:
            samples = values[:, column]
            if np.any(samples < 0):
                raise ValueError("Band powers cannot be negative.")
            positive = samples[samples > 0]
            floor = float(positive.min()) * 0.01 if positive.size else 1e-12
            floors.append(max(floor, np.finfo(float).tiny))
        self.floors_ = np.asarray(floors)
        return self

    def transform(self, X):
        check_is_fitted(self, "floors_")
        values = np.array(X, dtype=float, copy=True)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("The feature matrix does not match the trained model.")
        for column, floor in zip(self.power_columns, self.floors_):
            if np.any(values[:, column] < 0):
                raise ValueError("Band powers cannot be negative.")
            values[:, column] = np.log10(np.maximum(values[:, column], floor))
        return values


@dataclass(frozen=True)
class FittedModel:
    """The portable model file also carries its ordered feature contract."""

    name: str
    pipeline: Pipeline
    features: FeatureSettings
    feature_names: tuple[str, ...]
    band_edges_hz: np.ndarray

    def decision_function(self, raw_features: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipeline.decision_function(raw_features), dtype=float)

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        # A shared zero score threshold means P(loose)>=0.5 for logistic
        # regression and a nonnegative signed decision score for either SVM.
        return (self.decision_function(raw_features) >= 0).astype(int)


@dataclass(frozen=True)
class Evaluation:
    indices: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    metrics: dict[str, float | None]
    confusion: np.ndarray
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray


@dataclass(frozen=True)
class Experiment:
    dataset: TelemetryDataset
    config: ExperimentConfig
    prepared: PreparedData
    models: dict[str, FittedModel]
    evaluations: dict[str, dict[str, Evaluation]]
    created_utc: str


def classification_metrics(labels, predictions, scores) -> dict[str, float | None]:
    """Loose is positive. Undefined rates stay None, never a misleading zero."""
    labels, predictions, scores = (
        np.asarray(labels),
        np.asarray(predictions),
        np.asarray(scores),
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    tpr, fpr = divide(tp, tp + fn), divide(fp, fp + tn)
    return {
        "accuracy": divide(tp + tn, len(labels)),
        "tpr": tpr,
        "fpr": fpr,
        "balanced_accuracy": (tpr + 1 - fpr) / 2
        if tpr is not None and fpr is not None
        else None,
        "precision": divide(tp, tp + fp),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(labels, scores))
        if set(labels) == {0, 1}
        else None,
    }


def evaluate_model(
    model: FittedModel, dataset: TelemetryDataset, data: PreparedData, role: str
) -> Evaluation:
    indices = np.flatnonzero(data.masks[role])
    values, labels = data.matrix[indices], dataset.class_ids[indices]
    scores = model.decision_function(values)
    predictions = (scores >= 0).astype(int)
    fpr, tpr = np.array([]), np.array([])
    if set(labels) == {0, 1}:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return Evaluation(
        indices,
        predictions,
        scores,
        classification_metrics(labels, predictions, scores),
        confusion_matrix(labels, predictions, labels=[0, 1]),
        fpr,
        tpr,
    )


def train_experiment(
    dataset: TelemetryDataset,
    config: ExperimentConfig,
    progress=None,
    check_cancel=None,
) -> Experiment:
    config = deepcopy(config)  # Later GUI edits cannot mutate the recorded split.
    data = prepare_data(dataset, config)
    require_trainable(dataset, data)
    classifiers = {
        "logistic_regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000),
        "linear_svm": SVC(kernel="linear", C=config.linear_c, max_iter=100000),
        "rbf_svm": SVC(
            kernel="rbf", C=config.rbf_c, gamma=config.rbf_gamma, max_iter=100000
        ),
    }
    models = {}
    evaluations = {"training": {}, "validation": {}}
    for name, classifier in classifiers.items():
        if check_cancel:
            check_cancel()
        if progress:
            progress(f"Training {MODEL_NAMES[name]}…")
        pipeline = Pipeline(
            [
                (
                    "power_log",
                    PowerLogTransformer(
                        data.power_columns if config.features.log_powers else ()
                    ),
                ),
                ("standardize", StandardScaler()),
                ("classifier", classifier),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(
                data.matrix[data.masks["training"]],
                dataset.class_ids[data.masks["training"]],
            )
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ConfigurationError(
                f"{MODEL_NAMES[name]} did not converge. Try a smaller C or different features."
            )
        if check_cancel:
            check_cancel()
        model = FittedModel(
            name,
            pipeline,
            config.features,
            data.feature_names,
            dataset.band_edges_hz.copy(),
        )
        models[name] = model
        for role, results in evaluations.items():
            results[name] = evaluate_model(model, dataset, data, role)
    # Reserved test packets have not been passed to any fitted model here.
    return Experiment(
        dataset, config, data, models, evaluations, datetime.now(UTC).isoformat()
    )


def evaluate_test(
    experiment: Experiment, progress=None, check_cancel=None
) -> Experiment:
    if not np.any(experiment.prepared.masks["testing"]):
        raise ConfigurationError(
            "Assign at least one eligible run/setup to Testing first."
        )
    results = {}
    for name, model in experiment.models.items():
        if check_cancel:
            check_cancel()
        if progress:
            progress(f"Evaluating {MODEL_NAMES[name]} on the reserved test set…")
        results[name] = evaluate_model(
            model, experiment.dataset, experiment.prepared, "testing"
        )
    return replace(
        experiment, evaluations={**experiment.evaluations, "testing": results}
    )


def per_run_metrics(experiment: Experiment, role: str):
    """Yield comparable per-run rows for the GUI and CSV export."""
    dataset = experiment.dataset
    for name, evaluation in experiment.evaluations[role].items():
        sources = np.asarray(dataset.source_ids)[evaluation.indices]
        for source in sorted(set(sources)):
            keep = sources == source
            yield {
                "model": name,
                "partition": role,
                "source_id": source,
                "run": dataset.run_names[source],
                "packets": int(keep.sum()),
                **classification_metrics(
                    dataset.class_ids[evaluation.indices[keep]],
                    evaluation.predictions[keep],
                    evaluation.scores[keep],
                ),
            }
