import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pure function. Fraction of predictions that match the true labels.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pure function. Macro-averaged precision across all classes.
    """
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        scores.append(TP / (TP + FP) if (TP + FP) > 0 else 0.0)
    return float(np.mean(scores))

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pure function. Macro-averaged recall across all classes.
    """
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        TP = np.sum((y_pred == c) & (y_true == c))
        FN = np.sum((y_pred != c) & (y_true == c))
        scores.append(TP / (TP + FN) if (TP + FN) > 0 else 0.0)
    return float(np.mean(scores))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pure function. Macro-averaged F1 score (harmonic mean of precision and recall).
    """
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))
        p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(scores))

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, metrics: list) -> dict:
    """
    Higher-order function. Applies a list of metric functions to predictions and returns a results dict.
    """
    return {fn.__name__: fn(y_true, y_pred) for fn in metrics}
