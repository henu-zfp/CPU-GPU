"""Train a Random Forest model to predict CPU execution time."""
from __future__ import annotations

import csv
import pickle
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from random_forest import RandomForestRegressor

FEATURE_COLUMNS = [
    "feat_float_ops",
    "feat_control_statements",
    "feat_loops",
    "feat_function_calls",
    "input_data_size_bytes",
]
TARGET_COLUMN = "cpu_time_ms"
MODEL_FILENAME = "cpu_model.pkl"
RANDOM_STATE = 13900
VALIDATION_FRACTION = 0.2


def load_dataset(csv_path: Path) -> Tuple[List[List[float]], List[float]]:
    """Load the CPU dataset from ``cpu.csv`` and split features/target."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Dataset has no header row")
        missing_columns = [
            column for column in FEATURE_COLUMNS + [TARGET_COLUMN]
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(missing_columns)
            )

        features: List[List[float]] = []
        target: List[float] = []
        for row in reader:
            try:
                feature_row = [float(row[column]) for column in FEATURE_COLUMNS]
                target_value = float(row[TARGET_COLUMN])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row encountered: {row}") from exc
            features.append(feature_row)
            target.append(target_value)

    if not features:
        raise ValueError("Dataset is empty")

    return features, target


def build_model() -> RandomForestRegressor:
    """Configure the Random Forest regressor with sensible defaults."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )


def train_validation_split(
    features: List[List[float]],
    target: List[float],
    test_fraction: float,
    seed: int,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = random.Random(seed)
    indices = list(range(len(features)))
    rng.shuffle(indices)
    test_count = max(1, int(len(features) * test_fraction))
    if test_count >= len(features):
        test_count = max(1, len(features) - 1)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train = [features[i] for i in train_indices]
    X_valid = [features[i] for i in test_indices]
    y_train = [target[i] for i in train_indices]
    y_valid = [target[i] for i in test_indices]
    return X_train, X_valid, y_train, y_valid


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((value - mean_true) ** 2 for value in y_true)
    ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0
    return 1 - (ss_res / ss_tot)


def main() -> None:
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "cpu.csv"
    features, target = load_dataset(dataset_path)

    X_train, X_valid, y_train, y_valid = train_validation_split(
        features, target, test_fraction=VALIDATION_FRACTION, seed=RANDOM_STATE
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)
    print("Training complete for CPU model")
    print(f"Validation MAE: {mae:.4f} ms")
    print(f"Validation R^2: {r2:.4f}")

    final_model = build_model()
    final_model.fit(features, target)

    model_path = base_dir / MODEL_FILENAME
    with model_path.open("wb") as model_file:
        pickle.dump(final_model, model_file)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
