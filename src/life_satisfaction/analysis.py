from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .models import xgboost_parameter_search


def decision_tree_cross_validation(X, y) -> dict[str, Any]:
    """Run the two 20-fold decision-tree assessments in cells 102 and 106."""

    assessments: dict[str, Any] = {}
    labels = (
        "after_random_forest_oob",
        "after_class_weighted_random_forest",
    )
    for label in labels:
        folds = StratifiedKFold(n_splits=20)
        estimator = DecisionTreeClassifier()
        f1_scores = cross_val_score(
            estimator,
            X,
            y,
            cv=folds,
            scoring="f1_macro",
        )
        accuracy_scores = cross_val_score(
            estimator,
            X,
            y,
            cv=folds,
            scoring="accuracy",
        )
        assessments[label] = {
            "estimator": "DecisionTreeClassifier",
            "folds": 20,
            "f1_macro": _score_summary(f1_scores),
            "accuracy": _score_summary(accuracy_scores),
        }
    return assessments


def run_xgboost_search(X, y) -> dict[str, Any]:
    """Run the five-candidate randomized XGBoost search in cells 117–123."""

    search = xgboost_parameter_search()
    search.fit(X, y)
    candidates = []
    for index, parameters in enumerate(search.cv_results_["params"]):
        candidates.append(
            {
                "parameters": parameters,
                "mean_test_roc_auc": float(search.cv_results_["mean_test_score"][index]),
                "std_test_roc_auc": float(search.cv_results_["std_test_score"][index]),
                "rank": int(search.cv_results_["rank_test_score"][index]),
            }
        )
    return {
        "search": "RandomizedSearchCV",
        "iterations": 5,
        "folds": 5,
        "scoring": "roc_auc",
        "best_parameters": search.best_params_,
        "best_roc_auc": float(search.best_score_),
        "candidates": candidates,
    }


def _score_summary(scores: np.ndarray) -> dict[str, Any]:
    return {
        "scores": [float(value) for value in scores],
        "mean": float(np.mean(scores)),
        "standard_deviation": float(np.std(scores)),
    }
