import numpy as np
import pandas as pd

from life_satisfaction.statistics import notebook_wilcoxon_tests


def test_notebook_wilcoxon_result_shape():
    y = pd.Series([0, 0, 1, 1, 0, 1])
    predictions = {
        "svc": np.array([0, 1, 1, 0, 0, 1]),
        "random_forest_oob": np.array([0, 0, 1, 0, 0, 1]),
        "ensemble": np.array([0, 0, 1, 1, 0, 1]),
    }
    result = notebook_wilcoxon_tests(y, predictions)
    assert set(result["comparisons"]) == {
        "svc_vs_random_forest",
        "random_forest_vs_ensemble",
    }
