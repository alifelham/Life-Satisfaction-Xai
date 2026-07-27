import json
from pathlib import Path

from life_satisfaction.constants import POST_IMPUTATION_COLUMNS, SELECTED_FEATURES


def test_selected_features_are_unique_and_present_upstream():
    assert len(SELECTED_FEATURES) == len(set(SELECTED_FEATURES)) == 27
    assert set(SELECTED_FEATURES).issubset(POST_IMPUTATION_COLUMNS)


def test_reference_metrics_have_required_models_and_fields():
    payload = json.loads(Path("results/reference/published_metrics.json").read_text())
    assert {"random_forest_oob", "gradient_boosting", "xgboost", "ensemble"}.issubset(payload)
    for model in payload.values():
        assert {"accuracy", "macro_f1", "macro_precision", "macro_recall"}.issubset(model)
