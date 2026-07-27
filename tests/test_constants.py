from life_satisfaction.constants import SELECTED_FEATURES


def test_paper_feature_set_has_27_features():
    assert len(SELECTED_FEATURES) == 27
    assert SELECTED_FEATURES[0] == "age"
    assert SELECTED_FEATURES[-1] == "M8"
