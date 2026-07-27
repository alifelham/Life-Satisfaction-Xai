import numpy as np
import pandas as pd
import pytest

from life_satisfaction import feature_selection
from life_satisfaction.feature_selection import select_features
from life_satisfaction.preprocessing import encode_categories, remove_sparse_columns


def test_sparse_column_rule_uses_publication_threshold():
    frame = pd.DataFrame({
        "kept": range(100),
        "dropped": [1] * 81 + [None] * 19,
    })
    reduced, dropped = remove_sparse_columns(frame, threshold_percent=18)
    assert list(reduced.columns) == ["kept"]
    assert dropped == ["dropped"]


def test_unknown_category_fails_instead_of_silent_encoding():
    frame = pd.DataFrame({"gender": ["Unknown category"]})
    with pytest.raises(ValueError, match="Unencoded categorical columns"):
        encode_categories(frame)


def test_feature_selection_uses_the_study_variable_order(monkeypatch):
    class FittedEstimator:
        feature_importances_ = np.array([0.02, 0.01, 0.005])

    class FakeRFECV:
        def __init__(self, **kwargs):
            self.n_features_ = 3
            self.support_ = np.array([True, True, True])
            self.estimator_ = FittedEstimator()
            self.cv_results_ = {"mean_test_score": np.array([0.5, 0.6, 0.7])}

        def fit(self, X, y):
            return self

    monkeypatch.setattr(feature_selection, "RFECV", FakeRFECV)
    X_train = pd.DataFrame({"a": [0, 1], "b": [1, 0], "c": [0, 0]})
    X_test = pd.DataFrame({"a": [1], "b": [0], "c": [1]})
    selected = select_features(
        X_train,
        X_test,
        pd.Series([0, 1]),
        final_features=["b", "a"],
    )

    assert selected.selected_features == ["b", "a"]
    assert list(selected.X_train.columns) == ["b", "a"]
    assert selected.importance_selected_features == ["a", "b"]
