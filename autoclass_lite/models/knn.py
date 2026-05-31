import numpy as np
from collections import Counter
from .base import BaseModel

class KNNClassifier(BaseModel): 
    """
    K-Nearest Neighbors classifier. 
    Stores training data at fit time and classifies by majority vote among the k nearest neighbors at predict time.
    """
    _DISTANCES = {
        "euclidean": lambda X, x: np.sqrt(np.sum((X - x) ** 2, axis=1)),
        "manhattan": lambda X, x: np.sum(np.abs(X - x), axis=1)
    }

    def __init__(self, k: int=3, distance: str="euclidean") -> None:
        """
        k is the number of neighbors to consider when voting. distance is the metric to use ('euclidean' or 'manhattan').
        """
        if distance not in self._DISTANCES:
            raise ValueError(f"Unknown distance '{distance}'. Choose from {list(self._DISTANCES.keys())}")
        self.distance = distance
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Memorize training data. KNN does no computation here.
        """
        self._X_train = X
        self._y_train = y

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For each sample, find the k nearest training points and return the majority label.
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict()")
        predictions = []
        for x in X:
            distance = self._DISTANCES[self.distance](self._X_train, x)
            k_indices = np.argsort(distance)[:self.k]
            k_labels = self._y_train[k_indices]
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(predictions)

    def get_params(self) -> dict:
        """
        Return hyperparameters for this model.
        """
        return {"k": self.k, "distance": self.distance}