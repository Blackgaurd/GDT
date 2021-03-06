from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .classifier import DecisionTree


class Selector:
    def __init__(self, *args, **kwargs):
        self.do_crossover = True

    def __call__(
        self, population: List[DecisionTree], fitness: List[float], rounds: int
    ):
        pass


class Tournament(Selector):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(
        self, population: List[DecisionTree], fitness: List[float], rounds: int
    ):
        for _ in range(rounds):
            inds = random.sample(
                range(len(population)),
                self.k,
            )
            ind = max(inds, key=lambda i: fitness[i])
            p1 = population[ind]

            inds = random.sample(
                range(len(population)),
                self.k,
            )
            ind = max(inds, key=lambda i: fitness[i])
            p2 = population[ind]

            yield p1, p2


class Elitism(Selector):
    def __init__(self):
        super().__init__()
        self.do_crossover = False

    def __call__(
        self, population: List[DecisionTree], fitness: List[float], rounds: int
    ):
        fp = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
        for i in range(0, rounds * 2, 2):
            yield fp[i][1], fp[i + 1][1]
