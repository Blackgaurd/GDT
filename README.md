# Genetic Decision Tree

A proof of concept implementation of using a genetic algorithm to optimize decision trees for data c;assification. Used in my extended essay ([paper here](https://github.com/Blackgaurd/extended-essay)).

## Installation

```bash
pip install git+https://github.com/Blackgaurd/GDT.git
```

## Example

```python
from gdt.classifier import GDTClassifier
from gdt.data.load import load_iris
from gdt.fitness import FitnessEvaluator, negative_quadratic
from gdt.selection import Elitism, Tournament

(X, Y), (X_test, Y_test) = load_iris().data()

kwargs = {
    "population_size": 100,
    "split_probability": 0.5,
    "selectors": [(Tournament(4), 0.8), (Elitism(), 0.2)],
    "crossover_probability": 0.7,
    "mutation_probability": 0.1,
    "optimal_depth": 5,
    "fitness_evaluator": FitnessEvaluator(0.8, 0.1, negative_quadratic),
}

clf = GDTClassifier(**kwargs)
clf.fit(X, Y, 50, verbose=10)
```
