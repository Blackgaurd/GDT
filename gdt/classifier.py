import copy
import random
import statistics
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygraphviz as pgv

from .selection import Selector
from .fitness import FitnessEvaluator

data_loaded = 0


def load_features(
    trainX: np.ndarray,
    trainY: np.ndarray,
    feature_names: Optional[List[str]] = None,
    label_names: Optional[Dict[int, str]] = None,
) -> None:
    global features, feature_vals, labels, data_loaded, _feature_names, _label_names

    # features and feature values
    features = list(range(trainX.shape[1]))  #! watch for this
    feature_vals = [list(set(trainX[:, i])) for i in features]

    # unique labels
    labels = list(set(trainY))

    data_loaded = 1

    if feature_names is None or label_names is None:
        return

    # feature and label names
    _feature_names = feature_names
    _label_names = label_names

    data_loaded = 2


class Node:
    __slots__ = "split_val", "feature", "label", "depth"

    def __init__(
        self, split_val: float = None, feature: int = None, label: int = None
    ) -> None:
        self.split_val = split_val
        self.feature = feature
        self.label = label
        self.depth = 0

    @classmethod
    def leaf(cls, label: int):
        return cls(label=label)

    @classmethod
    def split(cls, feature: int, split_val: float):
        return cls(feature=feature, split_val=split_val)

    def is_leaf(self) -> bool:
        return self.label is not None

    def __str__(self) -> str:
        if self.is_leaf():
            return f"Label: {_label_names[self.label]}"
        return f"Feature: {_feature_names[self.feature]}, Split: {self.split_val}"


class DecisionTree:
    __slots__ = "nodes", "depth", "optimal_depth"

    def __init__(self, optimal_depth: int) -> None:
        # store nodes in list for fast random access
        # lists are 1 indexed

        self.optimal_depth = optimal_depth
        l = 2 ** (optimal_depth + 1)

        self.nodes: List[Optional[Node]] = [None for _ in range(l)]

        # root has depth 0
        root_feature = random.choice(features)
        self.nodes[1] = Node.split(
            root_feature, random.choice(feature_vals[root_feature])
        )
        self.depth = 0

    @classmethod
    def generate_random(cls, optimal_depth: int, split_p: float):
        assert 0 <= split_p < 1
        ret = cls(optimal_depth)

        # generate tree with bfs
        q = deque([1])
        while q:
            cur = q.popleft()

            # loop twice for left and right child
            for _ in range(2):
                if (
                    ret.nodes[cur].depth + 2 <= optimal_depth
                    and random.random() <= split_p
                ):
                    next_pos = ret.add_node(cur, "split")
                    q.append(next_pos)
                else:
                    ret.add_node(cur, "leaf")

        return ret

    def add_node(self, node_ind: int, node_type: str) -> int:
        if self.nodes[node_ind * 2] is None:
            next_pos = node_ind * 2
        else:
            next_pos = node_ind * 2 + 1

        if node_type == "leaf":
            new_node = (
                Node.leaf(
                    random.choice(
                        [i for i in labels if i != self.nodes[next_pos - 1].label]
                    )
                )
                if next_pos % 2 == 1 and self.nodes[next_pos - 1].is_leaf()
                else Node.leaf(random.choice(labels))
            )

        elif node_type == "split":
            feature = random.choice(features)
            new_node = Node.split(feature, random.choice(feature_vals[feature]))

        self.nodes[next_pos] = new_node
        self.nodes[next_pos].depth = self.nodes[node_ind].depth + 1
        self.depth = max(self.depth, self.nodes[next_pos].depth)
        return next_pos

    def extend_nodes(self) -> None:
        self.nodes.extend([None for _ in range(len(self.nodes))])

    def classify_one(self, X: np.ndarray) -> int:
        cur = 1
        while True:
            node = self.nodes[cur]
            if node.label is not None:
                return node.label

            cur *= 2
            if X[node.feature] > node.split_val:
                cur += 1

    def classify_many(self, X: np.ndarray) -> List[int]:
        return [self.classify_one(x) for x in X]

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        predictions = np.array(self.classify_many(X), dtype=int)
        return np.mean(predictions == Y)

    def visualize(self, filename: str) -> None:
        if data_loaded != 2:
            raise Exception("Feature and label names not loaded")

        g = pgv.AGraph(directed=True)
        g.node_attr["shape"] = "box"
        for i, node in enumerate(self.nodes):
            if node is None:
                continue
            g.add_node(i, label=str(node))
        q = deque([1])
        while q:
            cur = q.popleft()
            if self.nodes[cur].is_leaf():
                continue
            g.add_edge(cur, cur * 2)
            g.add_edge(cur, cur * 2 + 1)
            q.append(cur * 2)
            q.append(cur * 2 + 1)
        g.layout(prog="dot")
        if filename.endswith(".png"):
            g.draw(filename)
        elif filename.endswith(".dot"):
            g.write(filename)
        else:
            raise ValueError("Unsupported file type")


def qlog2(x: int) -> int:
    return (x - 1).bit_length() - 1


def crossover_v2(
    p1: DecisionTree, p2: DecisionTree
) -> Tuple[DecisionTree, DecisionTree]:
    def replace(
        source: DecisionTree, source_ind: int, dest: DecisionTree, dest_ind: int
    ) -> None:
        q = deque([(source_ind, dest_ind)])
        while q:
            si, di = q.popleft()
            if di >= len(dest.nodes):
                dest.extend_nodes()
            dest.nodes[di] = source.nodes[si]
            dest.nodes[di].depth = dest.nodes[di // 2].depth + 1
            dest.depth = max(dest.depth, dest.nodes[di].depth)
            if not source.nodes[si].is_leaf():
                q.append((si * 2, di * 2))
                q.append((si * 2 + 1, di * 2 + 1))

        # del is slightly faster than .pop()
        while dest.nodes[-1] is None:
            del dest.nodes[-1]

        # clean unused nodes
        reachable = [False for i in range(len(dest.nodes))]
        q = deque([dest_ind])
        reachable[dest_ind] = True
        while q:
            cur = q.popleft()
            if not reachable[cur]:
                dest.nodes[cur] = None
            elif not dest.nodes[cur].is_leaf() and cur * 2 < len(dest.nodes):
                reachable[cur * 2] = True
                reachable[cur * 2 + 1] = True
            if cur * 2 < len(dest.nodes):
                q.append(cur * 2)
                q.append(cur * 2 + 1)

        while dest.nodes[-1] is None:
            del dest.nodes[-1]

        dest.depth = qlog2(len(dest.nodes))

    # start from 2 to avoid root node
    p1_inds = [i for i in range(2, len(p1.nodes)) if p1.nodes[i] is not None]
    p2_inds = [i for i in range(2, len(p2.nodes)) if p2.nodes[i] is not None]

    c1 = copy.deepcopy(p1)
    replace(p2, random.choice(p2_inds), c1, random.choice(p1_inds))
    c2 = copy.deepcopy(p2)
    replace(p1, random.choice(p1_inds), c2, random.choice(p2_inds))
    return c1, c2


def mutate(tree: DecisionTree) -> None:
    valid = [i for i in range(len(tree.nodes)) if tree.nodes[i] is not None]
    ind = random.choice(valid)
    if tree.nodes[ind].is_leaf():
        other_ind = ind + 1 if ind % 2 == 0 else ind - 1
        if tree.nodes[other_ind].is_leaf():
            tree.nodes[ind] = Node.leaf(
                random.choice([x for x in labels if x != tree.nodes[other_ind].label])
            )
            return

        tree.nodes[ind] = Node.leaf(random.choice(labels))
        return

    feature = random.choice(features)
    tree.nodes[ind] = Node.split(feature, random.choice(feature_vals[feature]))


class GDTClassifier:
    def __init__(
        self,
        population_size: int,
        split_probability: float,
        selectors: List[Tuple[Selector, float]],
        crossover_probability: float,
        mutation_probability: float,
        optimal_depth: int,
        fitness_evaluator: FitnessEvaluator,
    ) -> None:
        self.population_size = population_size
        self.split_p = split_probability
        self.cross_p = crossover_probability
        self.mut_p = mutation_probability
        self.optimal_depth = optimal_depth
        self.fitness_eval = fitness_evaluator
        self.population = None

        # make all the weights in selectors sum up to population_size
        half_pop = population_size // 2
        selection_sum = sum(s[1] for s in selectors)
        self.selectors = [
            [s[0], round(s[1] / selection_sum * half_pop)] for s in selectors
        ]

        # |diff| < half_pop
        diff = half_pop - sum(s[1] for s in self.selectors)
        if diff > 0:
            for i in range(diff):
                self.selectors[i][1] += 1
        else:
            for i in range(diff):
                self.selectors[i][1] -= 1

        # assert sum(s[1] for s in self.selectors) == population_size

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        generations: int,
        valX: np.ndarray = None,
        valY: np.ndarray = None,
        verbose: bool = False,
        feature_names: List[str] = None,
        label_names: List[str] = None,
    ) -> None:
        if data_loaded == 0:
            load_features(X, Y, feature_names, label_names)

        n = self.population_size

        if self.population is None:
            # initiate population
            self.population = [
                DecisionTree.generate_random(self.optimal_depth, self.split_p)
                for _ in range(n)
            ]

        train_eval = copy.copy(self.fitness_eval)
        train_eval._init(X, Y)

        val_eval = None
        if validation := (valX is not None and valY is not None):
            val_eval = copy.copy(self.fitness_eval)
            val_eval._init(valX, valY)

        # fit history
        history = []

        # main loop
        for gen in range(1, generations + 1):
            fitnesses = [train_eval(tree) for tree in self.population]

            if verbose and gen % verbose == 0:
                history.append(
                    self._verbose(
                        gen, generations, train_eval, validation, val_eval, fitnesses
                    )
                )

            new_pop = []

            # selection + crossover
            for selector, rounds in self.selectors:
                for p1, p2 in selector(self.population, fitnesses, rounds):
                    if selector.do_crossover:
                        new_pop.extend(crossover_v2(p1, p2))
                    else:
                        new_pop.extend((p1, p2))

            # mutation
            for i in random.sample(range(n), int(n * self.mut_p)):
                mutate(new_pop[i])

            self.population = new_pop

        # sort by fitness
        fitnesses = [train_eval(tree) for tree in self.population]
        fp = sorted(zip(fitnesses, self.population), key=lambda x: x[0], reverse=True)
        self.population = [fp[i][1] for i in range(n)]

        return history

    def _verbose(
        self,
        gen: int,
        generations: int,
        train_eval: FitnessEvaluator,
        validation: bool,
        val_eval: FitnessEvaluator,
        fitnesses: List[float],
    ) -> None:
        accuracies = tuple(train_eval.accuracy(tree) for tree in self.population)
        tree_gen = tuple(tree.depth for tree in self.population)
        max_depth = max(tree_gen)
        min_depth = min(tree_gen)
        avg_depth = statistics.mean(tree_gen)
        median_depth = statistics.median(tree_gen)
        std_depth = statistics.stdev(tree_gen)

        if validation:
            val_accuracies = tuple(val_eval.accuracy(tree) for tree in self.population)
            avg_val_accuracy = statistics.mean(val_accuracies)

        avg_fitness = statistics.mean(fitnesses)
        avg_accuracy = statistics.mean(accuracies)

        print(f"Generation {gen}/{generations}")
        print(f"Average fitness: {avg_fitness:.4f}")
        print(f"Average accuracy: {avg_accuracy:.4f}")
        if validation:
            print(f"Average validation accuracy: {avg_val_accuracy:.4f}")
        print("Tree depths:")
        print(f"- max    : {max_depth}")
        print(f"- min    : {min_depth}")
        print(f"- average: {avg_depth:.2f}")
        print(f"- median : {median_depth:.2f}")
        print(f"- std    : {std_depth:.2f}")
        print()

        return {
            "generation": gen,
            "avg_fitness": avg_fitness,
            "avg_accuracy": avg_accuracy,
            "max_depth": max_depth,
            "min_depth": min_depth,
            "avg_depth": avg_depth,
            "median_depth": median_depth,
            "std_depth": std_depth,
        }

    def predict(self, X: np.ndarray, top_k: float = 1) -> List[int]:
        assert 0 < top_k <= 1

        predictions = [
            tree.classify_many(X)
            for tree in self.population[: int(top_k * len(self.population))]
        ]

        return [statistics.mode(individidual) for individidual in zip(*predictions)]
