from life_satisfaction.app import INPUTS
from life_satisfaction.constants import SELECTED_FEATURES


def test_demo_input_order_matches_publication_features():
    assert [name for name, *_ in INPUTS] == SELECTED_FEATURES
    assert len(INPUTS) == 27
