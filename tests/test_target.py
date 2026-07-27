import pandas as pd

from life_satisfaction.preprocessing import construct_target


def test_publication_target_mapping():
    frame = pd.DataFrame({"A1": ["Content", "Very content", "Discontent", "Very discontent", "Neither nor"]})
    result = construct_target(frame)
    assert result["A1"].tolist() == [0, 0, 1, 1]
