import numpy as np
from autoclass_lite.metrics.classification import accuracy, precision, recall, f1_score, evaluate
import pytest

@pytest.fixture
def partial_preds():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    return y_true, y_pred

def test_accuracy_perfect():
    y_true = y_pred = np.array([0, 0, 1, 1])
    assert round(accuracy(y_true, y_pred), 4) == 1

def test_accuracy_partial(partial_preds):
    y_true, y_pred = partial_preds
    assert round(accuracy(y_true, y_pred), 4) == 0.75

def test_precision_perfect():
    y_true = y_pred = np.array([0, 0, 1, 1])
    assert round(precision(y_true, y_pred), 4) == 1

def test_precision_partial(partial_preds):
    y_true, y_pred = partial_preds

    assert round(precision(y_true, y_pred), 3) == 0.833

def test_recall_perfect():
    y_true = y_pred = np.array([0, 0, 1, 1])
    assert round(recall(y_true, y_pred), 4) == 1

def test_precision_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert precision(y_true, y_pred) == 0.0

def test_recall_partial(partial_preds):
    y_true, y_pred = partial_preds
    assert round(recall(y_true, y_pred), 3) == 0.75

def test_f1_perfect():
    y_true = y_pred = np.array([0, 0, 1, 1])
    assert round(f1_score(y_true, y_pred), 4) == 1

def test_f1_partial(partial_preds):
    y_true, y_pred = partial_preds
    assert round(f1_score(y_true, y_pred), 3) == 0.733

def test_evaluate_returns_dict(partial_preds):
    y_true, y_pred = partial_preds
    result = evaluate(y_true, y_pred, [accuracy, f1_score])
    assert "accuracy" in result
    assert "f1_score" in result

def test_evaluate_keys_match_function_names(partial_preds):
    y_true, y_pred = partial_preds
    result = evaluate(y_true, y_pred, [accuracy, f1_score])
    assert list(result.keys()) == ["accuracy", "f1_score"]
