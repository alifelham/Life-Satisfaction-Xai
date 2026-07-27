import numpy as np
import pandas as pd

from life_satisfaction import analysis


def test_decision_tree_cross_validation_runs_both_notebook_assessments(monkeypatch):
    calls = []

    def fake_cross_val_score(estimator, X, y, *, cv, scoring):
        calls.append((type(estimator).__name__, cv.n_splits, scoring))
        return np.array([0.5, 0.75])

    monkeypatch.setattr(analysis, "cross_val_score", fake_cross_val_score)
    result = analysis.decision_tree_cross_validation(
        pd.DataFrame({"x": [0, 1]}),
        pd.Series([0, 1]),
    )

    assert list(result) == [
        "after_random_forest_oob",
        "after_class_weighted_random_forest",
    ]
    assert calls == [
        ("DecisionTreeClassifier", 20, "f1_macro"),
        ("DecisionTreeClassifier", 20, "accuracy"),
        ("DecisionTreeClassifier", 20, "f1_macro"),
        ("DecisionTreeClassifier", 20, "accuracy"),
    ]
    assert result["after_random_forest_oob"]["f1_macro"]["mean"] == 0.625
