from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import save_json

LIME_CASES = (
    {"name": "case_1", "row": 25, "num_features": 20},
    {"name": "case_2", "row": -50, "num_features": 29},
)


def generate_lime_cases(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Generate the two LIME case analyses used in the study."""

    try:
        from lime import lime_tabular
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LIME is not installed. Install the `explain` extra or requirements-paper.txt."
        ) from exc

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=list(X_train.columns),
        class_names=["Content", "Discontent"],
        mode="classification",
    )

    records: list[dict[str, Any]] = []
    for case in LIME_CASES:
        row_position = int(case["row"])
        instance = X_test.iloc[row_position]
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=int(case["num_features"]),
        )
        name = str(case["name"])
        html_path = target / f"{name}.html"
        png_path = target / f"{name}.png"
        explanation.save_to_file(str(html_path))
        figure = explanation.as_pyplot_figure()
        figure.set_figwidth(10)
        figure.set_figheight(6)
        figure.tight_layout()
        figure.savefig(png_path, dpi=300)
        plt.close(figure)

        record = {
            "name": name,
            "row_position": row_position,
            "true_class": int(y_test.iloc[row_position]),
            "predicted_class": int(model.predict(instance.to_frame().T)[0]),
            "predicted_probabilities": [
                float(value)
                for value in model.predict_proba(instance.to_frame().T)[0]
            ],
            "terms": [
                {"condition": condition, "weight": float(weight)}
                for condition, weight in explanation.as_list()
            ],
            "html": html_path.name,
            "figure": png_path.name,
        }
        records.append(record)

    save_json(records, target / "lime_cases.json")
    return records
