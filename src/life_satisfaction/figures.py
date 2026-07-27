from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def _target(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_target_distribution(y: pd.Series, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x=y, ax=ax)
    ax.set(
        title="Life-satisfaction target distribution",
        xlabel="Class (0 = content, 1 = discontent)",
        ylabel="Count",
    )
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_missing_value_counts(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
) -> None:
    missing = frame.isna().sum().sort_values(ascending=False)
    missing = missing.loc[missing > 0]
    fig, ax = plt.subplots(figsize=(max(8, 0.2 * len(missing)), 6))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_xticks([])
    else:
        missing.plot.bar(ax=ax)
    ax.set(title=title, xlabel="Variable", ylabel="Missing values")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_missingness_matrix(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
) -> None:
    matrix = frame.isna().to_numpy(dtype=np.uint8).T
    fig, ax = plt.subplots(figsize=(12, max(6, min(24, 0.12 * frame.shape[1]))))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set(title=title, xlabel="Observations", ylabel="Variables")
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_correlation_heatmap(frame: pd.DataFrame, path: str | Path) -> None:
    correlation = frame.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    fig, ax = plt.subplots(figsize=(24, 24))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="seismic",
        center=0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title("Pearson correlation matrix")
    ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_outlier_boxplot(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    variables = ["D2", "D6", "D8"]
    missing = [name for name in variables if name not in frame.columns]
    if missing:
        raise KeyError(f"Outlier-plot variables are missing: {missing}")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=frame.loc[:, variables], ax=ax)
    ax.set(title=title, xlabel="Variables", ylabel="Values")
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_class_distribution(y, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=y, ax=ax)
    ax.set(title=title, xlabel="Class", ylabel="Count")
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_rfecv_curve(rfecv, path: str | Path) -> None:
    scores = rfecv.cv_results_["mean_test_score"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(1, len(scores) + 1), scores, linewidth=2)
    ax.set(
        title="Recursive Feature Elimination with Cross-Validation",
        xlabel="Number of features selected",
        ylabel="Mean ROC-AUC",
    )
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)


def save_feature_importance(
    feature_names: list[str],
    importances: list[float],
    path: str | Path,
) -> None:
    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    table = table.sort_values("importance", ascending=True)
    height = max(7, min(30, 0.28 * len(table)))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(table["feature"], table["importance"])
    ax.set(
        title="RFECV estimator feature importance",
        xlabel="Importance",
        ylabel="Feature",
    )
    fig.tight_layout()
    fig.savefig(_target(path), dpi=300)
    plt.close(fig)
