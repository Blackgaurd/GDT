from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .classifier import DecisionTree


def linear(depth: int, optimal_depth: int) -> float:
    return 1 - depth / optimal_depth


def tanh(depth: int, optimal_depth: int) -> float:
    return math.tanh(linear(depth, optimal_depth))


def sigmoid(depth: int, optimal_depth: int) -> float:
    return 1 / (1 + math.exp(depth - optimal_depth)) - 0.5


def negative_quadratic(depth: int, optimal_depth: int) -> float:
    return -((depth - optimal_depth) ** 2)


class FitnessEvaluator:
    def __init__(
        self, a1: float, a2: float, f2_func: Callable[[int, int], float]
    ) -> None:
        self.a1 = a1
        self.a2 = a2
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.f2_func = f2_func

    def _init(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y

    def accuracy(self, individual: DecisionTree) -> float:
        predictions = np.array(individual.classify_many(self.X), dtype=int)
        return np.sum(predictions == self.Y) / len(self.Y)

    def __call__(self, individual: DecisionTree) -> float:
        f1 = self.accuracy(individual)
        # TODO: update f2 to non-linear function that intersects y=0 at x=optimal_depth
        f2 = self.f2_func(individual.depth, individual.optimal_depth)

        return self.a1 * f1 + self.a2 * f2
