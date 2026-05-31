from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all classifiers (Strategy pattern).

    Subclasses must implement fit and predict. The fit_predict sequence
    is defined here once and inherited by all models (Template Method pattern).
    """

    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Takes a 2D numpy array 
        X (samples x features) and
        a 1D array y (labels). 
        Returns None

        Parameters
        -----------
        X: samples features
        y: labels

        Returns
        ---------

        """
        pass

    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Takes a 2D numpy array,
        returns a 1D numpy array 
        of predicted labels.

        Parameters
        ----------
        X: 2D numpy array

        Returns
        --------
        y: Predicted labels
        """
        pass

    def fit_predict(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray) -> np.ndarray:
        """Fit on training data then predict on test data (Template Method skeleton)."""
        self.fit(X_train, y_train)
        return self.predict(X_test)
    
    def get_params(self) -> dict:
        """Return the model's hyperparameters. Subclasses override this."""
        return {}