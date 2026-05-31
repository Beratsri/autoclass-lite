import numpy as np
from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    """Abstract base class for all data splitters."""

    @abstractmethod
    def split(self, X: np.ndarray) -> list:
        """Return a list of (train_indices, test_indices) tuples."""
        pass
    
class KFoldSplitter(BaseSplitter):
    """
    Generates train/test index pairs for k-fold cross-validation.
    """
    def __init__(self, n_splits: int=5, shuffle: bool=True, random_state: int=None) -> None:
        """
        n_splits is the number of folds, shuffle randomizes sample order before splitting.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray) -> list:
        """
        Return a list of (train_indices, test_indices) tuples, one per fold.
        """
        result = []
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(indices)
        chunks = np.array_split(indices, self.n_splits)
        for i in range(self.n_splits):
            test = chunks[i]
            train = np.concatenate([chunks[j] for j in range(self.n_splits) if j != i])
            result.append((train, test))
        return result
    

