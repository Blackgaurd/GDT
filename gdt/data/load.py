import os
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd

DIR = Path(os.path.dirname(os.path.realpath(__file__)))


@dataclass
class Data:
    trainX: np.ndarray
    trainY: np.ndarray
    testX: np.ndarray
    testY: np.ndarray
    feature_names: list
    label_names: dict

    def data(
        self,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return (self.trainX, self.trainY), (self.testX, self.testY)


# data cache decorator
def data_cache(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        file_name = DIR / "cache" / f"{func.__name__}.pkl"
        if not os.path.isfile(file_name):
            data = func(*args, **kwargs)
            pkl.dump(data, open(file_name, "wb"))
        else:
            print(f"Loading {func.__name__} from cache")
            data = pkl.load(open(file_name, "rb"))
        return data

    return wrapper


@data_cache
def load_titanic():
    train = pd.read_csv(DIR / "titanic" / "train.csv")
    test = pd.read_csv(DIR / "titanic" / "test.csv")

    # drop unnecessary columns
    trainY = train["Survived"]
    train = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    # fill missing data
    train["Age"].fillna(int(train["Age"].mean()), inplace=True)
    train["Embarked"].fillna("S", inplace=True)

    # replace non-numerical values with numbers
    train["Sex"].replace({"male": 0, "female": 1}, inplace=True)
    train["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

    # to numpy
    trainX = train.to_numpy().astype(float)
    trainY = trainY.to_numpy().astype(int)

    # repeat steps with test data
    test = test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]
    test["Age"].fillna(int(test["Age"].mean()), inplace=True)
    test["Embarked"].fillna("S", inplace=True)
    test["Sex"].replace({"male": 0, "female": 1}, inplace=True)
    test["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
    test = test.to_numpy().astype(float)

    feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]
    label_names = {0: "Did not survive", 1: "Survived"}

    return Data(trainX, trainY, test, None, feature_names, label_names)


@data_cache
def load_iris():
    train = pd.read_csv(DIR / "isris" / "train.csv")

    train["Species"].replace(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}, inplace=True
    )

    feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    label_names = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    trainX = (
        train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
        .to_numpy()
        .astype(float)
    )
    trainY = train["Species"].to_numpy().astype(int)

    return Data(trainX, trainY, None, None, feature_names, label_names)


@data_cache
def load_mushroom(replace_missing=True):
    train = pd.read_csv(DIR / "mushroom" / "train.csv")
    train["poisonous"].replace({"e": 0, "p": 1}, inplace=True)

    trainX = train.loc[:, train.columns != "poisonous"]
    if replace_missing:
        trainX["stalk-root"].replace(
            "?", trainX["stalk-root"].value_counts().idxmax(), inplace=True
        )
    features = trainX.columns.tolist()
    values = [
        {"b": 0, "c": 1, "x": 2, "f": 3, "k": 4, "s": 5},
        {"f": 0, "g": 1, "y": 2, "s": 3},
        {
            "n": 0,
            "b": 1,
            "c": 2,
            "g": 3,
            "r": 4,
            "p": 5,
            "u": 6,
            "e": 7,
            "w": 8,
            "y": 9,
        },
        {"f": 0, "t": 1},
        {"a": 0, "l": 1, "c": 2, "y": 3, "f": 4, "m": 5, "n": 6, "p": 7, "s": 8},
        {"f": 0, "a": 1, "n": 2, "d": 3},
        {"c": 0, "w": 1, "d": 2},
        {"b": 0, "n": 1},
        {
            "k": 0,
            "n": 1,
            "b": 2,
            "h": 3,
            "g": 4,
            "r": 5,
            "o": 6,
            "p": 7,
            "u": 8,
            "e": 9,
            "w": 10,
            "y": 11,
        },
        {"e": 0, "t": 1},
        {"b": 0, "c": 1, "u": 2, "e": 3, "z": 4, "r": 5, "?": 6},
        {"f": 0, "y": 1, "k": 2, "s": 3},
        {"f": 0, "y": 1, "k": 2, "s": 3},
        {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8},
        {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8},
        {"p": 0, "u": 1},
        {"n": 0, "o": 1, "w": 2, "y": 3},
        {"n": 0, "o": 1, "t": 2},
        {"c": 0, "e": 1, "f": 2, "l": 3, "n": 4, "p": 5, "s": 6, "z": 7},
        {"k": 0, "n": 1, "b": 2, "h": 3, "r": 4, "o": 5, "u": 6, "w": 7, "y": 8},
        {"a": 0, "c": 1, "n": 2, "s": 3, "v": 4, "y": 5},
        {"g": 0, "l": 1, "m": 2, "p": 3, "u": 4, "w": 5, "d": 6},
    ]
    for feature, value in zip(features, values):
        trainX[feature].replace(value, inplace=True)

    feature_names = trainX.columns.tolist()
    label_names = {1: "poisonous", 0: "edible"}

    trainX = trainX.to_numpy().astype(float)
    trainY = train["poisonous"].to_numpy().astype(int)

    return Data(trainX, trainY, None, None, feature_names, label_names)
