import numpy as np
from .base import BaseModel

class GaussianNaiveBayes(BaseModel):
    """
    Gaussian Naive Bayes classifier. 
    Models each feature per class as a 
    normal distribution and classifies by maximum 
    log posterior probability.
    """
    def __init__(self) -> None:
        """No hyperparameters. State is set during fit."""
        self._classes = None
        self._log_priors = None
        self._means = None
        self._variances = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute per-class log priors, feature means, and feature variances from training data.
        """
        self._classes = np.unique(y)
        log_priors = []
        means = []
        variances = []
        for c in self._classes:
            X_c = X[y == c]
            log_priors.append(np.log(len(X_c) / len(X)))
            means.append(np.mean(X_c, axis=0))
            variances.append(np.var(X_c, axis=0))
        self._log_priors = np.array(log_priors)
        self._means = np.array(means)
        self._variances = np.array(variances)
    
    def _log_gaussian(self, x, mean, variance):
        return -0.5 * np.log(2 * np.pi * (variance + 1e-9)) - ((x - mean) ** 2) / (2 * (variance + 1e-9))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For each sample, compute log posterior for every class and
        return the class with the highest score.
        """
        if self._classes is None:
            raise RuntimeError("Call fit() before predict()")
        predictions = []
        for x in X:
            scores = []
            for i,c in enumerate(self._classes):
                log_posterior = self._log_priors[i] + np.sum(self._log_gaussian(x, self._means[i], self._variances[i]))
                scores.append(log_posterior)
            predictions.append(self._classes[np.argmax(scores)])
        return np.array(predictions)


    def get_params(self) -> dict:
        """
        No hyperparameters for this model.
        """
        return {}