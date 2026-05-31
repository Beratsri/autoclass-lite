import numpy as np
from collections import Counter
from .base import BaseModel

class _Node:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    @property
    def is_leaf(self):
        return self.value is not None
    
class DecisionTreeClassifier(BaseModel):
    """
    Decision tree classifier that recursively splits data by Gini impurity. 
    Recursion is the core algorithmic technique.
    """
    def __init__(self, max_depth: int=5, min_samples_split: int = 2) -> None:
        """
        max_depth limits tree depth, 
        min_samples_split is the minimum samples required to attempt a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root = None

    def _gini(self, y: np.ndarray) -> float:
        _ , counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X:np.ndarray, y: np.ndarray):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                n = len(y)
                weighted = (np.sum(left_mask)/n) * self._gini(y[left_mask]) + \
                        (np.sum(right_mask)/n) * self._gini(y[right_mask])
                if weighted < best_gini:
                    best_gini = weighted
                    best_feature = j
                    best_threshold = t
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the tree by splitting on the best feature until stopping conditions are met."""
        node = _Node()
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        feat, thresh = self._best_split(X, y)
        if feat is None:          # no valid split found → make a leaf
            node.value = Counter(y).most_common(1)[0][0]
            return node
        node.feature_index, node.threshold = feat, thresh
        left_mask = X[:, node.feature_index] <= node.threshold
        node.left  = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node


    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree from training data.
        """
        self._root = self._grow_tree(X, y)

    def _traverse(self, x, node):
        if node.is_leaf: return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Traverse the tree for each sample 
        and return predicted labels.
        """
        return np.array([self._traverse(x, self._root) for x in X])

    def get_params(self) -> dict:
        """
        Return max_depth and min_samples_split.
        """
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split}
        