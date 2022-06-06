from edt.classifier import EDTClassifier
from edt.selection import Tournament, Elitism
from edt.fitness import FitnessEvaluator, negative_quadratic
from edt.data.load import load_titanic

tmp = load_titanic()
(X, Y), (X_test, y_test) = tmp.data()

kwargs = {
    "population_size": 400,
    "split_probability": 0.5,
    "selectors": [(Tournament(10), 0.8), (Elitism(), 0.2)],
    "crossover_probability": 0.8,
    "mutation_probability": 0.1,
    "optimal_depth": 5,
    "fitness_evaluator": FitnessEvaluator(0.8, 0.5, negative_quadratic),
}
clf = EDTClassifier(**kwargs)
clf.fit(X, Y, 200, verbose=10)
