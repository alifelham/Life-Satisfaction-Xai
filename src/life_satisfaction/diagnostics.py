from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve


def save_learning_curve(
    estimator: Any,
    X,
    y,
    path: str | Path,
    *,
    cv: int = 5,
    n_jobs: int = -1,
) -> None:
    """Generate the same five-fold learning-curve analysis used in the notebook."""
    sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_scores.mean(axis=1), marker="o", label="Training score")
    ax.plot(sizes, test_scores.mean(axis=1), marker="o", label="Validation score")
    ax.fill_between(
        sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    ax.fill_between(
        sizes,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set(xlabel="Training examples", ylabel="Score", title="Learning curve")
    ax.legend()
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)


def save_calibration_curves(
    classifiers: Mapping[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    path: str | Path,
    *,
    n_bins: int = 10,
) -> None:
    """Generate the study reliability-diagram analysis using scikit-learn."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for name, classifier in classifiers.items():
        fitted = clone(classifier).fit(X_train, y_train)
        probability = fitted.predict_proba(X_test)[:, 1]
        fraction, mean_probability = calibration_curve(
            y_test, probability, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_probability, fraction, marker="o", label=name)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
        title="Calibration curves",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)


def save_error_analysis(
    confusion_matrices: Mapping[str, list[list[int]] | np.ndarray],
    path: str | Path,
) -> None:
    """Plot false-positive and false-negative counts for the evaluated models."""
    names = list(confusion_matrices)
    matrices = [np.asarray(confusion_matrices[name]) for name in names]
    false_positives = [int(matrix[0, 1]) for matrix in matrices]
    false_negatives = [int(matrix[1, 0]) for matrix in matrices]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(names, false_positives, label="False positives")
    ax.bar(names, false_negatives, bottom=false_positives, label="False negatives")
    ax.set(xlabel="Model", ylabel="Count", title="Prediction error analysis")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
