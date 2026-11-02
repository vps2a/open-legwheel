"""Evaluation figures shared by the desktop viewer and saved experiments."""

import numpy as np
from matplotlib.figure import Figure

from .training import MODEL_NAMES, Experiment

MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73")


def comparison_figure(
    experiment: Experiment, role: str, figure: Figure | None = None
) -> Figure:
    figure = (
        figure if figure is not None else Figure(figsize=(11, 7), layout="constrained")
    )
    figure.clear()
    grid = figure.add_gridspec(2, 3, height_ratios=(1, 1.2))
    results = experiment.evaluations[role]
    for column, name in enumerate(MODEL_NAMES):
        axis = figure.add_subplot(grid[0, column])
        counts = results[name].confusion
        totals = counts.sum(axis=1, keepdims=True)
        fractions = np.divide(counts, totals, out=np.zeros((2, 2)), where=totals > 0)
        axis.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for col in range(2):
                percentage = f"{fractions[row, col]:.0%}" if totals[row, 0] else "N/A"
                axis.text(
                    col,
                    row,
                    f"{counts[row, col]}\n({percentage})",
                    ha="center",
                    va="center",
                    color="white" if fractions[row, col] > 0.55 else "#222222",
                    fontsize=10,
                )
        axis.set_xticks([0, 1], ["Firm", "Loose"])
        axis.set_yticks([0, 1], ["Firm", "Loose"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_title(MODEL_NAMES[name], fontsize=11)
    roc = figure.add_subplot(grid[1, :])
    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        result = results[name]
        if result.roc_fpr.size:
            roc.plot(
                result.roc_fpr,
                result.roc_tpr,
                color=color,
                label=f"{MODEL_NAMES[name]} — AUC {result.metrics['roc_auc']:.3f}",
            )
            roc.plot(
                result.metrics["fpr"],
                result.metrics["tpr"],
                "o",
                color=color,
                markersize=5,
            )
    roc.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    roc.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="False positive rate (firm called loose)",
        ylabel="True positive rate (loose detected)",
        title=f"{role.title()} ROC — dots mark the fixed decision thresholds",
    )
    roc.grid(alpha=0.2)
    if any(result.roc_fpr.size for result in results.values()):
        roc.legend(loc="lower right", fontsize=8)
    else:
        roc.text(
            0.5,
            0.5,
            "ROC needs both actual classes",
            ha="center",
            transform=roc.transAxes,
        )
    return figure
