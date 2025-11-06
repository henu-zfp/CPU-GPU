"""Predict CPU execution time using the trained Random Forest model."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import random_forest  # noqa: F401  (ensure pickle can locate the model class)

FEATURE_ORDER = [
    "feat_float_ops",
    "feat_control_statements",
    "feat_loops",
    "feat_function_calls",
    "input_data_size_bytes",
]
MODEL_FILENAME = "cpu_model.pkl"


def prompt_for_features() -> List[float]:
    """Interactively prompt the user for feature values in the required order."""
    values: List[float] = []
    print("Please enter the following feature values for CPU prediction (one per prompt):")
    for feature_name in FEATURE_ORDER:
        while True:
            try:
                raw = input(f"  {feature_name}: ")
                value = float(raw)
                values.append(value)
                break
            except ValueError:
                print("    Invalid number, please try again.")
    return values


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_path}. Run cpu-train.py first."
        )
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def main() -> None:
    base_dir = Path(__file__).parent
    model_path = base_dir / MODEL_FILENAME
    model = load_model(model_path)
    features = prompt_for_features()
    prediction = model.predict([features])[0]
    print(f"Predicted CPU execution time: {prediction:.4f} ms")


if __name__ == "__main__":
    main()
