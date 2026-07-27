from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _lightgbm_module():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "LightGBM is required for the publication pipeline. Install requirements-paper.txt."
        ) from exc
    return lightgbm


def _xgboost_module():
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "XGBoost is required for the publication pipeline. Install requirements-paper.txt."
        ) from exc
    return xgboost


def lightgbm_model() -> Any:
    """Return the exact LightGBM configuration shown in notebook cell 91."""
    lightgbm = _lightgbm_module()
    return lightgbm.LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        metric="binary_logloss",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
    )


def tuned_xgboost_model() -> Any:
    """Return the tuned XGBoost configuration executed in cells 124–126."""
    xgboost = _xgboost_module()
    return xgboost.XGBClassifier(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.4,
        gamma=0.3,
        learning_rate=0.1,
        max_bin=256,
        max_depth=12,
        min_child_weight=1,
        n_estimators=600,
        n_jobs=0,
        num_parallel_tree=1,
        random_state=21,
        reg_alpha=0,
        reg_lambda=1,
    )


def xgboost_parameter_search() -> RandomizedSearchCV:
    """Return the five-candidate XGBoost search executed in cells 117–123."""
    xgboost = _xgboost_module()
    parameters = {
        "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
        "booster": ["gbtree", "gblinear", "dart"],
    }
    return RandomizedSearchCV(
        xgboost.XGBClassifier(),
        param_distributions=parameters,
        n_iter=5,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=3,
    )


def _adaboost(stump: DecisionTreeClassifier | None = None) -> AdaBoostClassifier:
    """Support publication-era and current scikit-learn parameter names."""
    if stump is None:
        stump = DecisionTreeClassifier(max_depth=1)
    kwargs: dict[str, Any] = {
        "n_estimators": 600,
        "random_state": 21,
        "learning_rate": 1,
    }
    parameter = (
        "estimator"
        if "estimator" in inspect.signature(AdaBoostClassifier).parameters
        else "base_estimator"
    )
    kwargs[parameter] = stump
    return AdaBoostClassifier(**kwargs)


def _bagging_svc() -> BaggingClassifier:
    kwargs: dict[str, Any] = {"n_estimators": 10, "random_state": 21, "n_jobs": -1}
    parameter = (
        "estimator"
        if "estimator" in inspect.signature(BaggingClassifier).parameters
        else "base_estimator"
    )
    kwargs[parameter] = SVC()
    return BaggingClassifier(**kwargs)


def publication_models() -> OrderedDict[str, Any]:
    """Models and parameters executed in notebook cells 88–126."""
    return OrderedDict(
        [
            ("svc", SVC(class_weight={0: 1, 1: 1.4})),
            ("lightgbm", lightgbm_model()),
            ("naive_bayes", GaussianNB()),
            ("decision_tree", DecisionTreeClassifier(random_state=21)),
            # The notebook declares a parameter dictionary but does not pass it here.
            ("random_forest_oob", RandomForestClassifier(oob_score=True)),
            (
                "gradient_boosting",
                GradientBoostingClassifier(
                    n_estimators=500,
                    learning_rate=1,
                    max_depth=1,
                    random_state=21,
                ),
            ),
            ("adaboost", _adaboost()),
            ("logistic_regression", LogisticRegression(solver="liblinear", penalty="l2")),
            ("xgboost", tuned_xgboost_model()),
        ]
    )


def supplementary_models() -> OrderedDict[str, Any]:
    """Additional executed notebook models outside the primary benchmark table."""
    return OrderedDict(
        [
            ("svc_standard_scaled", make_pipeline(StandardScaler(), SVC(gamma="auto"))),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("bagging_svc", _bagging_svc()),
            (
                "random_forest_class_weighted",
                RandomForestClassifier(class_weight={0: 10, 1: 0.1}),
            ),
        ]
    )


def publication_ensemble(*, learning_curve_variant: bool = False) -> VotingClassifier:
    """Return the soft-voting ensemble from cells 141 and 150.

    Cell 141 uses random_state=21 for the RF component. Cell 150 uses 20 in
    the learning-curve configuration.
    """
    rf_seed = 20 if learning_curve_variant else 21
    estimators = [
        (
            "RF",
            RandomForestClassifier(
                class_weight={0: 5, 1: 0.1},
                n_estimators=400,
                random_state=rf_seed,
            ),
        ),
        ("gb", GradientBoostingClassifier()),
        ("lgb", lightgbm_model()),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def learning_curve_models() -> OrderedDict[str, Any]:
    """Estimators used by notebook learning-curve cells 79, 80, and 150."""
    return OrderedDict(
        [
            ("logistic_regression", LogisticRegression(solver="liblinear", penalty="l2")),
            ("xgboost", tuned_xgboost_model()),
            ("lightgbm", lightgbm_model()),
            ("ensemble", publication_ensemble(learning_curve_variant=True)),
        ]
    )


def calibration_models(*, include_ensemble: bool) -> OrderedDict[str, Any]:
    """Estimators used by notebook calibration cells 148 and 151."""
    xgboost = _xgboost_module()
    models: OrderedDict[str, Any] = OrderedDict(
        [
            ("Gradient Boost", GradientBoostingClassifier()),
            ("Ada Boost", AdaBoostClassifier()),
            ("Logistic Regression", LogisticRegression()),
            ("Random Forest", RandomForestClassifier()),
            ("XGBoost", xgboost.XGBClassifier()),
            ("Decision Tree", DecisionTreeClassifier()),
        ]
    )
    if include_ensemble:
        models["Ensemble"] = publication_ensemble(learning_curve_variant=True)
    return models


# Stable public aliases.
benchmark_models = publication_models
