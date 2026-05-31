from ..models.logistic import LogisticRegression
from ..models.base import BaseModel
from ..models.knn import KNNClassifier
from ..models.naive_bayes import GaussianNaiveBayes
from ..models.tree import DecisionTreeClassifier


class ModelFactory:
    """
    Factory that instantiates model objects from name strings (Factory pattern).
    """
    _registry = {
        "knn":                 KNNClassifier,
        "logistic_regression": LogisticRegression,
        "naive_bayes":         GaussianNaiveBayes,
        "decision_tree":       DecisionTreeClassifier,
    }

    @classmethod
    def create(cls, name:str, **kwargs) -> BaseModel:
        """
        Instantiate a model by name, passing kwargs to its constructor.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown model '{name}'. Choose from {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod 
    def available_models(cls) -> list:
        """
        Return the list of registered model names.
        """
        return list(cls._registry.keys())
