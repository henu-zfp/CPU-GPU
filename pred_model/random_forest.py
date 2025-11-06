"""Lightweight Random Forest Regressor implementation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

Number = float


def _variance(values: Sequence[Number]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


@dataclass
class _Leaf:
    value: Number


@dataclass
class _DecisionNode:
    feature_index: int
    threshold: Number
    left: "_TreeNode"
    right: "_TreeNode"


_TreeNode = _Leaf | _DecisionNode


class _DecisionTreeRegressor:
    """Very small decision tree used as the base estimator."""

    def __init__(
        self,
        max_depth: int,
        min_samples_split: int,
        max_features: int,
        random_state: random.Random,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self._rng = random_state
        self._root: _TreeNode | None = None

    def fit(self, features: Sequence[Sequence[Number]], target: Sequence[Number]) -> None:
        self._root = self._build_tree(features, target, depth=0)

    def predict_row(self, row: Sequence[Number]) -> Number:
        if self._root is None:
            raise RuntimeError("Decision tree has not been trained")
        node = self._root
        while isinstance(node, _DecisionNode):
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _build_tree(
        self,
        features: Sequence[Sequence[Number]],
        target: Sequence[Number],
        depth: int,
    ) -> _TreeNode:
        if depth >= self.max_depth or len(target) < self.min_samples_split:
            return _Leaf(sum(target) / len(target))

        n_features = len(features[0])
        feature_indices = list(range(n_features))
        self._rng.shuffle(feature_indices)
        feature_indices = feature_indices[: self.max_features]

        best_feature = None
        best_threshold = None
        best_score = math.inf
        best_left_indices: List[int] = []
        best_right_indices: List[int] = []

        for feature_index in feature_indices:
            feature_values = [row[feature_index] for row in features]
            unique_values = sorted(set(feature_values))
            if len(unique_values) <= 1:
                continue

            candidate_thresholds: List[Number] = []
            # Evaluate up to 16 thresholds per feature.
            if len(unique_values) <= 32:
                for i in range(len(unique_values) - 1):
                    candidate_thresholds.append(
                        (unique_values[i] + unique_values[i + 1]) / 2.0
                    )
            else:
                step = len(unique_values) / 16
                for i in range(1, 16):
                    idx = min(int(i * step), len(unique_values) - 1)
                    candidate_thresholds.append(unique_values[idx])

            for threshold in candidate_thresholds:
                left_indices = [
                    idx for idx, value in enumerate(feature_values) if value <= threshold
                ]
                right_indices = [
                    idx for idx, value in enumerate(feature_values) if value > threshold
                ]
                if not left_indices or not right_indices:
                    continue

                left_target = [target[idx] for idx in left_indices]
                right_target = [target[idx] for idx in right_indices]
                score = (
                    len(left_target) * _variance(left_target)
                    + len(right_target) * _variance(right_target)
                ) / len(target)

                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_feature is None or best_threshold is None:
            return _Leaf(sum(target) / len(target))

        left_features = [features[i] for i in best_left_indices]
        left_target = [target[i] for i in best_left_indices]
        right_features = [features[i] for i in best_right_indices]
        right_target = [target[i] for i in best_right_indices]

        left_child = self._build_tree(left_features, left_target, depth + 1)
        right_child = self._build_tree(right_features, right_target, depth + 1)
        return _DecisionNode(best_feature, best_threshold, left_child, right_child)


class RandomForestRegressor:
    """A compact Random Forest Regressor with an sklearn-like interface."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_split: int = 4,
        max_features: str | int | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._rng = random.Random(random_state)
        self._trees: List[_DecisionTreeRegressor] = []

    def fit(self, features: Sequence[Sequence[Number]], target: Sequence[Number]) -> None:
        if not features:
            raise ValueError("Cannot train RandomForestRegressor with no data")
        n_features = len(features[0])
        if isinstance(self.max_features, int):
            max_features = max(1, min(self.max_features, n_features))
        elif self.max_features in ("sqrt", None):
            max_features = max(1, int(math.sqrt(n_features)))
        elif self.max_features == "log2":
            max_features = max(1, int(math.log2(n_features)))
        else:
            raise ValueError(f"Unsupported max_features value: {self.max_features}")

        self._trees = []
        for _ in range(self.n_estimators):
            bootstrap_features, bootstrap_target = self._bootstrap_sample(features, target)
            tree_rng = random.Random(self._rng.random())
            tree = _DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=tree_rng,
            )
            tree.fit(bootstrap_features, bootstrap_target)
            self._trees.append(tree)

    def predict(self, features: Sequence[Sequence[Number]]) -> List[Number]:
        if not self._trees:
            raise RuntimeError("RandomForestRegressor has not been trained")
        predictions: List[Number] = []
        for row in features:
            row_predictions = [tree.predict_row(row) for tree in self._trees]
            predictions.append(sum(row_predictions) / len(row_predictions))
        return predictions

    def _bootstrap_sample(
        self,
        features: Sequence[Sequence[Number]],
        target: Sequence[Number],
    ) -> tuple[List[List[Number]], List[Number]]:
        n_samples = len(features)
        indices = [self._rng.randrange(0, n_samples) for _ in range(n_samples)]
        bootstrap_features = [list(features[idx]) for idx in indices]
        bootstrap_target = [target[idx] for idx in indices]
        return bootstrap_features, bootstrap_target
