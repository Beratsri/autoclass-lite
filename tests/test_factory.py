import pytest
from autoclass_lite.automl.factory import ModelFactory
from autoclass_lite.models.knn import KNNClassifier
from autoclass_lite.models.logistic import LogisticRegression
from autoclass_lite.models.naive_bayes import GaussianNaiveBayes
from autoclass_lite.models.tree import DecisionTreeClassifier
from autoclass_lite.models.base import BaseModel


def test_factory_creates_knn():
    model = ModelFactory.create("knn", k=5)
    assert isinstance(model, KNNClassifier)
    assert model.k == 5


def test_factory_creates_logistic_regression():
    model = ModelFactory.create("logistic_regression", learning_rate=0.01)
    assert isinstance(model, LogisticRegression)


def test_factory_creates_naive_bayes():
    model = ModelFactory.create("naive_bayes")
    assert isinstance(model, GaussianNaiveBayes)


def test_factory_creates_decision_tree():
    model = ModelFactory.create("decision_tree", max_depth=4)
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 4


def test_factory_all_models_are_base_model_subclasses():
    """Every registered model must inherit from BaseModel."""
    for name in ModelFactory.available_models():
        model = ModelFactory.create(name)
        assert isinstance(model, BaseModel), f"{name} is not a BaseModel subclass"


def test_factory_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelFactory.create("random_forest")


def test_factory_available_models_returns_list():
    models = ModelFactory.available_models()
    assert isinstance(models, list)
    assert len(models) >= 4


def test_factory_available_models_contains_expected():
    models = ModelFactory.available_models()
    for expected in ["knn", "logistic_regression", "naive_bayes", "decision_tree"]:
        assert expected in models
