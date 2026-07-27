from types import SimpleNamespace

from sklearn.ensemble import BaggingClassifier

from life_satisfaction.models import (
    benchmark_models,
    publication_ensemble,
    supplementary_models,
    xgboost_parameter_search,
)


class ParameterRecorder:
    def __init__(self, **parameters):
        self.parameters = parameters

    def get_params(self, deep=True):
        return self.parameters


def _replace_native_estimators(monkeypatch):
    monkeypatch.setattr(
        "life_satisfaction.models._lightgbm_module",
        lambda: SimpleNamespace(LGBMClassifier=ParameterRecorder),
    )
    monkeypatch.setattr(
        "life_satisfaction.models._xgboost_module",
        lambda: SimpleNamespace(XGBClassifier=ParameterRecorder),
    )


def test_model_registry_and_ensemble(monkeypatch):
    _replace_native_estimators(monkeypatch)
    models = benchmark_models()
    assert list(models) == [
        "svc",
        "lightgbm",
        "naive_bayes",
        "decision_tree",
        "random_forest_oob",
        "gradient_boosting",
        "adaboost",
        "logistic_regression",
        "xgboost",
    ]
    assert models["svc"].class_weight == {0: 1, 1: 1.4}
    assert models["random_forest_oob"].oob_score is True
    assert models["random_forest_oob"].random_state is None
    gradient_boosting = models["gradient_boosting"].get_params()
    assert gradient_boosting["learning_rate"] == 1
    assert gradient_boosting["max_depth"] == 1
    assert gradient_boosting["n_estimators"] == 500
    assert gradient_boosting["random_state"] == 21
    assert models["logistic_regression"].solver == "liblinear"
    assert models["logistic_regression"].penalty == "l2"
    xgboost = models["xgboost"].get_params()
    assert xgboost["colsample_bytree"] == 0.4
    assert xgboost["gamma"] == 0.3
    assert xgboost["max_depth"] == 12
    assert xgboost["n_estimators"] == 600
    assert xgboost["random_state"] == 21

    ensemble = publication_ensemble()
    assert ensemble.voting == "soft"
    assert [name for name, _ in ensemble.estimators] == ["RF", "gb", "lgb"]
    random_forest = ensemble.estimators[0][1]
    assert random_forest.class_weight == {0: 5, 1: 0.1}
    assert random_forest.n_estimators == 400
    assert random_forest.random_state == 21


def test_supplementary_models_match_analysis_notebook():
    models = supplementary_models()
    assert list(models) == [
        "svc_standard_scaled",
        "knn",
        "bagging_svc",
        "random_forest_class_weighted",
    ]
    assert models["knn"].n_neighbors == 5
    assert isinstance(models["bagging_svc"], BaggingClassifier)
    assert models["bagging_svc"].n_estimators == 10
    assert models["bagging_svc"].random_state == 21
    assert models["random_forest_class_weighted"].class_weight == {0: 10, 1: 0.1}


def test_xgboost_search_matches_analysis_notebook(monkeypatch):
    _replace_native_estimators(monkeypatch)
    search = xgboost_parameter_search()
    assert search.n_iter == 5
    assert search.scoring == "roc_auc"
    assert search.cv == 5
    assert search.random_state is None
    assert search.param_distributions["booster"] == ["gbtree", "gblinear", "dart"]
    assert search.param_distributions["max_depth"] == [3, 4, 5, 6, 8, 10, 12, 15]
