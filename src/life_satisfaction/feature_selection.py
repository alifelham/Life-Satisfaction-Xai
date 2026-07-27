from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from .figures import save_class_distribution


@dataclass
class SelectedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    selected_features: list[str]
    rfecv: RFECV
    stage_one_features: list[str]
    stage_one_importances: list[float]
    importance_selected_features: list[str]


def _smote_tomek(
    sampling_strategy: float,
    random_state: int,
) -> SMOTETomek:
    kwargs = {
        "sampling_strategy": sampling_strategy,
        "random_state": random_state,
    }
    # n_jobs existed in the publication-era imbalanced-learn release but was
    # removed later. Use it only where accepted.
    if "n_jobs" in inspect.signature(SMOTETomek).parameters:
        kwargs["n_jobs"] = -1
    return SMOTETomek(**kwargs)


def balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    smote_tomek_strategy: float = 0.4,
    undersampling_strategy: float = 1.0,
    smote_random_state: int = 21,
    undersampling_random_state: int | None = None,
    figure_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the notebook's SMOTE–Tomek then random-undersampling sequence."""

    over = _smote_tomek(smote_tomek_strategy, smote_random_state)
    X_balanced, y_balanced = over.fit_resample(X_train, y_train)
    if figure_dir is not None:
        save_class_distribution(
            y_balanced,
            Path(figure_dir) / "smote_tomek_distribution.png",
            "Class distribution after SMOTE–Tomek",
        )

    under = RandomUnderSampler(
        sampling_strategy=undersampling_strategy,
        random_state=undersampling_random_state,
    )
    X_balanced, y_balanced = under.fit_resample(X_balanced, y_balanced)
    if figure_dir is not None:
        save_class_distribution(
            y_balanced,
            Path(figure_dir) / "balanced_distribution.png",
            "Class distribution after random undersampling",
        )
    return (
        pd.DataFrame(X_balanced, columns=X_train.columns),
        pd.Series(y_balanced, name=y_train.name),
    )


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 21,
    importance_threshold: float = 0.0094,
    final_features: list[str] | None = None,
) -> SelectedData:
    """Run the notebook's two-stage RFECV + importance-threshold selection."""

    estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(folds),
        scoring=scoring,
        n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)

    stage_one = list(X_train.columns[rfecv.support_])
    importances = [float(value) for value in rfecv.estimator_.feature_importances_]
    importance_selected = [
        name
        for name, importance in zip(stage_one, importances)
        if importance >= importance_threshold
    ]
    selected = list(final_features) if final_features is not None else importance_selected
    if not selected:
        raise RuntimeError("Feature selection produced an empty feature set.")
    missing = [name for name in selected if name not in X_train.columns]
    if missing:
        raise KeyError(f"Final study variables are missing from the feature matrix: {missing}")

    return SelectedData(
        X_train=X_train.loc[:, selected].copy(),
        X_test=X_test.loc[:, selected].copy(),
        y_train=y_train.reset_index(drop=True),
        selected_features=selected,
        rfecv=rfecv,
        stage_one_features=stage_one,
        stage_one_importances=importances,
        importance_selected_features=importance_selected,
    )
