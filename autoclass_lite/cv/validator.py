import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .splitter import KFoldSplitter, BaseSplitter
from ..metrics.classification import evaluate

class CrossValidator:
    """
    Runs k-fold cross-validation in parallel using ThreadPoolExecutor.
    """
    def __init__(self, splitter: BaseSplitter, metrics: list) -> None:
        """
        splitter provides fold indices, metrics is the list of metric functions to evaluate.
        """
        self.splitter = splitter
        self.metrics = metrics

    def _evaluate_fold(self, model, X, y, train_idx, test_idx) -> dict:
        fold_model = copy.deepcopy(model)
        fold_model.fit(X[train_idx], y[train_idx])
        predictions = fold_model.predict(X[test_idx])
        return evaluate(y[test_idx], predictions, self.metrics)
    
    def validate(self, model, X:np.ndarray, y:np.ndarray) -> dict:
        """
        Fit and evaluate the model on each fold in parallel. Returns mean scores across all folds.
        """
        folds = self.splitter.split(X)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._evaluate_fold, model, X, y, train_idx, test_idx) 
                       for train_idx, test_idx in folds]
            results = [f.result() for f in futures]
        return {metric: np.mean([r[metric] for r in results]) for metric in results[0]}
