"""Unit tests for all model classes."""

import numpy as np
import pytest

from autoclass_lite.models.knn import KNNClassifier
from autoclass_lite.models.logistic import LogisticRegression
from autoclass_lite.models.naive_bayes import GaussianNaiveBayes
from autoclass_lite.models.tree import DecisionTreeClassifier
from autoclass_lite.metrics.classification import accuracy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3],
                  [5, 6], [5, 7], [6, 6], [6, 7]] * 3, dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
    return X, y


# ---------------------------------------------------------------------------
# Generic model tests (fit / predict / accuracy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    KNNClassifier(k=3),
    GaussianNaiveBayes(),
    LogisticRegression(learning_rate=0.1),
    DecisionTreeClassifier(max_depth=3),
])
def test_model_fit_predict_above_threshold(model, sample_data):
    X, y = sample_data
    model.fit(X, y)
    assert accuracy(y, model.predict(X)) > 0.8


def test_knn_manhattan_distance(sample_data):
    X, y = sample_data
    model = KNNClassifier(k=3, distance="manhattan")
    model.fit(X, y)
    assert accuracy(y, model.predict(X)) > 0.8


def test_knn_invalid_distance():
    with pytest.raises(ValueError, match="Unknown distance"):
        KNNClassifier(distance="invalid")


def test_decision_tree_get_params():
    model = DecisionTreeClassifier(max_depth=4, min_samples_split=3)
    assert model.get_params() == {"max_depth": 4, "min_samples_split": 3}


def test_fit_predict_template_method(sample_data):
    X, y = sample_data
    model = KNNClassifier(k=3)
    preds = model.fit_predict(X, y, X)
    assert accuracy(y, preds) > 0.8


def test_logistic_predict_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1.0, 2.0]])
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        model.predict(X)


def test_logistic_get_params():
    model = LogisticRegression(learning_rate=0.05, n_iterations=500)
    params = model.get_params()
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["n_iterations"] == 500
