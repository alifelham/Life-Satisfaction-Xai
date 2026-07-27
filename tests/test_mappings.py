import pandas as pd

from life_satisfaction.preprocessing import encode_categories


def test_representative_publication_encodings():
    frame = pd.DataFrame({
        "gender": ["Woman", "Man"],
        "A2": ["Very poor", "Very well"],
        "C1": ["No", "Yes"],
        "job": ["Holds an ordinary or supported job", "Doesn't hold an ordinary or supported job"],
    })
    encoded = encode_categories(frame)
    assert encoded.to_dict("list") == {
        "gender": [0, 1], "A2": [0, 3], "C1": [0, 1], "job": [0, 1]
    }
