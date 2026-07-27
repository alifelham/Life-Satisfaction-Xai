from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .constants import SELECTED_FEATURES

# Encoded ranges mirror the paper notebook's demonstration surface.
# They are not raw survey response encoders and are not validated for deployment.
FEATURE_SPECS = [
    ("age", "Age", 16, 64),
    ("A2", "General health rating", 0, 3),
    ("C1", "Long-term physical health problem or disability", 0, 1),
    ("D2", "Frequency of depression", 0, 4),
    ("D4", "Relaxed and handles stress well", 0, 4),
    ("D6", "Can be tense", 0, 4),
    ("D8", "Worries a lot", 0, 4),
    ("D10", "Emotionally stable", 0, 4),
    ("D11", "Does not give up until a task is completed", 0, 4),
    ("D15", "Prepares and implements plans", 0, 4),
    ("D16", "Gets nervous easily", 0, 4),
    ("D17", "Is easily distracted", 0, 4),
    ("E1", "Height in centimetres", 100, 230),
    ("E2", "Weight in kilograms", 20, 250),
    ("E5_a", "Consulted another practitioner or therapist", 0, 1),
    ("E17", "Primary confidant category", -1, 9),
    ("job", "Holds a job", 0, 1),
    ("F15", "Job satisfaction", 0, 9),
    ("G1", "Has a spouse or partner", 0, 1),
    ("J2", "Time with other relatives", 0, 6),
    ("J4", "Time with acquaintances", 0, 6),
    ("J9", "Cinema, concert, or theatre frequency", 0, 6),
    ("J14", "Print-newspaper reading frequency", 0, 6),
    ("J17", "Trips abroad in the past year", 0, 30),
    ("M2", "Primary income-source category", 0, 13),
    ("M6", "Medicine and supplement expenditure", 0, 100000),
    ("M8", "Current financial situation", 0, 4),
]


def encoded_frame(values: Sequence[float]) -> pd.DataFrame:
    if len(values) != len(SELECTED_FEATURES):
        raise ValueError(f"Expected {len(SELECTED_FEATURES)} values, received {len(values)}")
    return pd.DataFrame([dict(zip(SELECTED_FEATURES, values))], columns=SELECTED_FEATURES)


def predict_encoded(model: Any, values: Sequence[float]) -> dict[str, Any]:
    frame = encoded_frame(values)
    prediction = int(model.predict(frame)[0])
    output: dict[str, Any] = {
        "predicted_class": "Content" if prediction == 0 else "Discontent",
        "encoded_class": prediction,
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        output["probabilities"] = {
            "Content": float(probabilities[0]),
            "Discontent": float(probabilities[1]),
        }
    return output


def load_model(path: str | Path) -> Any:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Model not found: {source}. Run the pipeline execution command first."
        )
    return joblib.load(source)


def build_gradio_app(model_path: str | Path):
    """Build the notebook's encoded-input demonstration as an optional app."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional interface
        raise RuntimeError("Install the optional app dependencies with pip install -e '.[app]'") from exc

    if [name for name, *_ in FEATURE_SPECS] != SELECTED_FEATURES:
        raise RuntimeError("Demonstration input order no longer matches the paper feature schema.")

    model = load_model(model_path)
    inputs = [
        gr.Slider(minimum=minimum, maximum=maximum, step=1, label=f"{name}: {label}")
        for name, label, minimum, maximum in FEATURE_SPECS
    ]
    return gr.Interface(
        fn=lambda *values: predict_encoded(model, values),
        inputs=inputs,
        outputs=gr.JSON(label="Research-model output"),
        title="Life-satisfaction research demonstrator",
        description=(
            "Inputs use the encoded survey-variable scales defined by the study. This interface "
            "is not a clinical, psychological, or individual decision tool."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the paper encoded-input demonstration.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/generated/best_model.pkl"),
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_gradio_app(args.model).launch(share=args.share, show_error=True)


if __name__ == "__main__":
    main()
