from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from .constants import EXCLUDED_COLUMNS, TARGET_COLUMN, TARGET_MAPPING
from .figures import (
    save_correlation_heatmap,
    save_missing_value_counts,
    save_missingness_matrix,
    save_outlier_boxplot,
)
from .mappings import CATEGORY_MAPPINGS


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_full: pd.DataFrame
    y_full: pd.Series
    imputer: IterativeImputer
    dropped_missing_columns: list[str]
    dropped_constant_columns: list[str]
    dropped_correlated_columns: list[str]
    target_filtered_rows: int
    post_missing_feature_count: int


def construct_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's binary target mapping and remove other responses."""

    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Required target column {TARGET_COLUMN!r} is missing.")
    result = frame.copy()
    result[TARGET_COLUMN] = result[TARGET_COLUMN].map(TARGET_MAPPING).fillna(-1).astype(int)
    return result.loc[result[TARGET_COLUMN] > -1].copy()


def remove_sparse_columns(
    frame: pd.DataFrame,
    threshold_percent: int = 18,
) -> tuple[pd.DataFrame, list[str]]:
    if not 0 <= threshold_percent <= 100:
        raise ValueError("threshold_percent must be between 0 and 100.")
    minimum_non_null = int(((100 - threshold_percent) / 100) * frame.shape[0])
    kept = frame.dropna(axis=1, thresh=minimum_non_null)
    dropped = [column for column in frame.columns if column not in kept.columns]
    return kept, dropped


def encode_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply all 95 hand-authored encodings recovered from the notebook."""

    encoded = frame.copy()
    for column, mapping in CATEGORY_MAPPINGS.items():
        if column in encoded.columns:
            encoded[column] = encoded[column].map(
                lambda value, column_mapping=mapping: column_mapping.get(value, value)
            )

    unresolved = list(encoded.select_dtypes(include=["object", "category"]).columns)
    if unresolved:
        previews = {
            column: encoded[column].dropna().astype(str).unique()[:5].tolist()
            for column in unresolved
        }
        raise ValueError(
            "Unencoded categorical columns remain after publication mappings: "
            f"{previews}"
        )
    return encoded


def correlation_columns(dataset: pd.DataFrame, threshold: float) -> list[str]:
    """Mirror the notebook's upper-triangle correlation removal rule."""

    correlated: set[str] = set()
    matrix = dataset.corr()
    for i in range(len(matrix.columns)):
        for j in range(i):
            if abs(matrix.iloc[i, j]) > threshold:
                correlated.add(matrix.columns[i])
    return sorted(correlated)


def cap_training_outliers(
    frame: pd.DataFrame,
    standard_deviations: float = 2.0,
) -> pd.DataFrame:
    """Replace training values beyond mean ± N SD with the column median."""

    capped = frame.copy()
    for column in capped.columns:
        mean = capped[column].mean()
        standard_deviation = capped[column].std()
        median = capped[column].median()
        upper = mean + standard_deviations * standard_deviation
        lower = mean - standard_deviations * standard_deviation
        capped[column] = capped[column].mask(capped[column] > upper, median)
        capped[column] = capped[column].mask(capped[column] < lower, median)
    return capped


def prepare_data(
    raw: pd.DataFrame,
    *,
    test_size: float = 0.2,
    split_seed: int = 20,
    imputer_seed: int = 21,
    missing_threshold_percent: int = 18,
    correlation_threshold: float = 0.8,
    outlier_standard_deviations: float = 2.0,
    figure_dir: str | Path | None = None,
) -> PreparedData:
    """Execute preprocessing in the same order as the paper notebook."""

    target_ready = construct_target(raw)
    y = target_ready[TARGET_COLUMN].copy()

    missing_exclusions = [column for column in EXCLUDED_COLUMNS if column not in target_ready.columns]
    if missing_exclusions:
        raise KeyError(f"Expected excluded columns are missing: {missing_exclusions}")

    X = target_ready.drop(columns=EXCLUDED_COLUMNS)
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        save_missing_value_counts(
            X,
            figure_dir / "missing_values_before.png",
            "Missing values before column filtering",
        )
        save_missingness_matrix(
            X,
            figure_dir / "missingness_matrix_before.png",
            "Missingness before column filtering",
        )
    X, dropped_missing = remove_sparse_columns(X, missing_threshold_percent)
    if figure_dir is not None:
        save_missing_value_counts(
            X,
            figure_dir / "missing_values_after.png",
            "Missing values after column filtering",
        )
        save_missingness_matrix(
            X,
            figure_dir / "missingness_matrix_after.png",
            "Missingness after column filtering",
        )
    post_missing_feature_count = X.shape[1]

    before_duplicates = len(X)
    X = X.drop_duplicates().copy()
    if len(X) != before_duplicates:
        raise ValueError(
            "Duplicate removal changed the row count and would misalign features with the target."
        )

    X = encode_categories(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
    )

    # Zero-variance and correlation exclusions use the full feature matrix.
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    constants = [
        column
        for column, keep in zip(X.columns, selector.get_support())
        if not keep
    ]
    X = X.drop(columns=constants)
    X_train = X_train.drop(columns=constants)
    X_test = X_test.drop(columns=constants)
    if figure_dir is not None:
        save_correlation_heatmap(X_train, figure_dir / "correlation_heatmap.png")

    correlated = correlation_columns(X, correlation_threshold)
    X = X.drop(columns=correlated)
    X_train = X_train.drop(columns=correlated)
    X_test = X_test.drop(columns=correlated)

    if figure_dir is not None:
        save_outlier_boxplot(
            X_train,
            figure_dir / "outliers_before.png",
            "Selected variables before outlier replacement",
        )
    X_train = cap_training_outliers(X_train, outlier_standard_deviations)
    if figure_dir is not None:
        save_outlier_boxplot(
            X_train,
            figure_dir / "outliers_after.png",
            "Selected variables after outlier replacement",
        )

    columns = list(X_train.columns)
    imputer = IterativeImputer(max_iter=10, random_state=imputer_seed)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=columns,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=columns,
    )
    X_full_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=columns,
    )

    return PreparedData(
        X_train=X_train_imputed,
        X_test=X_test_imputed,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_full=X_full_imputed,
        y_full=y.reset_index(drop=True),
        imputer=imputer,
        dropped_missing_columns=dropped_missing,
        dropped_constant_columns=constants,
        dropped_correlated_columns=correlated,
        target_filtered_rows=len(target_ready),
        post_missing_feature_count=post_missing_feature_count,
    )
