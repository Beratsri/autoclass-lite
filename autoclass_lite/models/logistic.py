"""
Logistic Regression classifier.

Uses One-vs-Rest (OvR) for multiclass classification
with vanilla gradient descent.
"""

import numpy as np
from .base import BaseModel


class LogisticRegression(BaseModel):
    """Multiclass Logistic Regression using One-vs-Rest and gradient descent.

    Trains one binary classifier per class using the sigmoid function
    and batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates. Default ``0.01``.
    n_iterations : int
        Number of gradient descent steps per binary classifier. Default ``1000``.

    Examples
    --------
    >>> import numpy as np
    >>> from autoclass_lite.models.logistic import LogisticRegression
    >>> X = np.array([[1, 2], [2, 3], [5, 6], [6, 7]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> model = LogisticRegression(learning_rate=0.1)
    >>> model.fit(X, y)
    >>> model.predict(X)
    array([0, 0, 1, 1])
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ) -> None:
        """
        Parameters
        ----------
        learning_rate : float
            Gradient descent step size. Default ``0.01``.
        n_iterations : int
            Number of training iterations. Default ``1000``.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self._weights: np.ndarray | None = None
        self._biases: np.ndarray | None = None
        self._classes: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _fit_binary(self, X: np.ndarray, y_binary: np.ndarray):
        """Train a single binary logistic classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y_binary : np.ndarray, shape (n_samples,)
            Binary labels (0/1) for this class.

        Returns
        -------
        tuple[np.ndarray, float]
            Learned weight vector and bias term.
        """
        w = np.zeros(X.shape[1])
        b = 0.0
        n = len(X)

        for _ in range(self.n_iterations):
            z = X @ w + b
            y_pred = self._sigmoid(z)
            error = y_pred - y_binary

            dw = (1.0 / n) * (X.T @ error)
            db = (1.0 / n) * np.sum(error)

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

        return w, b

    # ------------------------------------------------------------------
    # Public API (BaseModel interface)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train one binary classifier per class using One-vs-Rest.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Class labels (any hashable type).
        """
        self._classes = np.unique(y)
        weights, biases = [], []

        for c in self._classes:
            y_binary = (y == c).astype(float)
            w, b = self._fit_binary(X, y_binary)
            weights.append(w)
            biases.append(b)

        self._weights = np.array(weights)
        self._biases = np.array(biases)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the class with the highest OvR probability for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if self._weights is None:
            raise RuntimeError("Call fit() before predict()")

        probs = np.column_stack([
            self._sigmoid(X @ self._weights[i] + self._biases[i])
            for i in range(len(self._classes))
        ])
        indices = np.argmax(probs, axis=1)
        return self._classes[indices]

    def get_params(self) -> dict:
        """Return hyperparameters needed to recreate this model.

        Returns
        -------
        dict
            Keys: ``learning_rate``, ``n_iterations``.
        """
        return {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
        }
