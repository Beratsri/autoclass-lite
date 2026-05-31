"""
AutoML orchestrators: SimpleAutoML and GridAutoML.

SimpleAutoML trains every registered model with K-fold cross-validation in
parallel and returns a ranked leaderboard. GridAutoML extends it with an
exhaustive hyperparameter grid search: all (model, params) combinations are
evaluated concurrently via ThreadPoolExecutor, and results are memoized so
duplicate configurations are never re-evaluated (dynamic programming).
"""

import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .factory import ModelFactory
from .observers import Observer
from ..cv.splitter import KFoldSplitter
from ..cv.validator import CrossValidator
from ..metrics.classification import accuracy, precision, recall, f1_score


class SimpleAutoML:
    """Train all registered models with cross-validation and rank them.

    Internally uses :class:`~autoclass_lite.cv.validator.CrossValidator`
    (which runs K folds in parallel via ``ThreadPoolExecutor``) and an
    Observer list for progress events.

    Class-level ``DEFAULT_CONFIGS`` defines which models are trained and
    with what default hyperparameters.  To add a new model you only need
    to register it in :class:`~autoclass_lite.automl.factory.ModelFactory`
    **and** add one entry here — ``fit()`` itself never needs to change
    (Open/Closed Principle).
    """

    # ------------------------------------------------------------------
    # Default model configurations (Open/Closed Principle)
    # Adding a new model requires only a new entry here — fit() is closed
    # for modification but open for extension via this registry.
    # ------------------------------------------------------------------
    DEFAULT_CONFIGS: dict = {
        "knn":                  {"k": 3},
        "logistic_regression":  {},
        "naive_bayes":          {},
        "decision_tree":        {},
    }

    def __init__(
        self,
        cv_splits: int = 5,
        metric: str = "accuracy",
        random_state: int | None = None,
    ) -> None:
        self.cv_splits = cv_splits
        self.metric = metric
        self.random_state = random_state
        self._observers: list[Observer] = []
        self._leaderboard: list[dict] = []
        self._best_model = None

    # ------------------------------------------------------------------
    # Observer support
    # ------------------------------------------------------------------

    def add_observer(self, observer: Observer) -> None:
        """Register an observer to receive progress events.

        Parameters
        ----------
        observer : Observer
            Any object implementing :class:`~autoclass_lite.automl.observers.Observer`.
        """
        self._observers.append(observer)

    def _notify(self, event: str, data: dict) -> None:
        """Broadcast an event to all registered observers."""
        for obs in self._observers:
            obs.update(event, data)

    # ------------------------------------------------------------------
    # Core fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleAutoML":
        """Train all models with cross-validation and rank them.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        SimpleAutoML
            *self* — enables method chaining.
        """
        self._leaderboard = []        # reset so fit() is idempotent
        self._best_model  = None
        self._notify("fit_start", {})
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        splitter = KFoldSplitter(n_splits=self.cv_splits, random_state=self.random_state)
        validator = CrossValidator(splitter, metrics=[accuracy, precision, recall, f1_score])

        for name, kwargs in self.DEFAULT_CONFIGS.items():
            model = ModelFactory.create(name, **kwargs)
            scores = validator.validate(model, X, y)
            self._notify("model_done", {"model": name, "scores": scores})
            self._leaderboard.append(
                {"model": name, "params": kwargs, **{k: float(v) for k, v in scores.items()}}
            )

        self._leaderboard.sort(key=lambda r: r[self.metric], reverse=True)

        best = self._leaderboard[0]
        self._best_model = ModelFactory.create(best["model"], **best["params"])
        self._best_model.fit(X, y)

        self._notify("fit_done", {"best_model": best["model"]})
        return self

    def leaderboard(self) -> list:
        """Return the ranked list of model results.

        Returns
        -------
        list[dict]
            Each entry has keys ``model``, ``params``, and one key per metric.
        """
        return self._leaderboard

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best model found during :meth:`fit`.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self._best_model is None:
            raise RuntimeError("Call fit() before predict()")
        X = np.asarray(X, dtype=float)
        return self._best_model.predict(X)

    def summary(self) -> None:
        """Print the leaderboard as a formatted table."""
        if not self._leaderboard:
            print("No results yet. Call fit() first.")
            return
        cols = ["model", "accuracy", "precision", "recall", "f1_score"]
        widths = [22, 10, 11, 10, 10]
        header = "".join(c.ljust(w) for c, w in zip(cols, widths))
        print(header)
        print("-" * len(header))
        for row in self._leaderboard:
            line = row["model"].ljust(widths[0])
            for col, w in zip(cols[1:], widths[1:]):
                line += f"{row[col]:.4f}".ljust(w)
            print(line)


class GridAutoML(SimpleAutoML):
    """Extend SimpleAutoML with concurrent exhaustive hyperparameter grid search.

    All ``(model, params)`` combinations in *param_grid* are submitted to a
    ``ThreadPoolExecutor`` at once and evaluated in parallel. A ``threading.Lock``
    protects the memoization cache so duplicate configurations are never
    re-evaluated across threads (dynamic programming / memoization).

    Concurrency is two-layered:
    - **Outer** (here): all grid configurations run simultaneously.
    - **Inner** (:class:`~autoclass_lite.cv.validator.CrossValidator`): each
      configuration's K folds also run in parallel.

    Parameters
    ----------
    param_grid : dict or None
        Maps model name → list of param dicts to try. If ``None`` a
        sensible default grid is used.
    cv_splits : int
        Number of cross-validation folds. Default ``5``.
    metric : str
        Primary ranking metric. Default ``"accuracy"``.
    random_state : int or None
        Fold splitter seed. Default ``None``.

    Examples
    --------
    >>> grid = GridAutoML(param_grid={
    ...     "knn": [{"k": 3}, {"k": 5}],
    ...     "decision_tree": [{"max_depth": 3}, {"max_depth": 5}],
    ... })
    >>> grid.fit(X_train, y_train)
    >>> grid.summary()
    """

    def __init__(
        self,
        param_grid: dict | None = None,
        cv_splits: int = 5,
        metric: str = "accuracy",
        random_state: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        param_grid : dict or None
            Maps model name → list of param dicts. Uses a default grid if None.
        cv_splits : int
            Number of cross-validation folds. Default ``5``.
        metric : str
            Metric used to pick the best config per model. Default ``"accuracy"``.
        random_state : int or None
            Seed for fold shuffling. Default ``None``.
        """
        super().__init__(cv_splits, metric, random_state)
        self._cache: dict = {}
        self._cache_lock = threading.Lock()

        if param_grid is not None:
            self.param_grid = param_grid
        else:
            self.param_grid = {
                "knn": [{"k": 1}, {"k": 3}, {"k": 5}, {"k": 7}],
                "logistic_regression": [
                    {"learning_rate": 0.01},
                    {"learning_rate": 0.1},
                    {"learning_rate": 0.5},
                ],
                "naive_bayes": [{}],
                "decision_tree": [
                    {"max_depth": 3},
                    {"max_depth": 5},
                    {"max_depth": 7},
                ],
            }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridAutoML":
        """Evaluate all grid configurations concurrently, memoize, and rank.

        All ``(model, params)`` pairs are submitted to a ``ThreadPoolExecutor``
        at once. A ``threading.Lock`` ensures cache reads/writes are thread-safe.
        After all futures complete, the best config per model is kept and the
        leaderboard is sorted by *metric*.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        GridAutoML
            *self* — enables method chaining.
        """
        self._leaderboard = []
        self._best_model  = None
        self._cache       = {}        # clear memoization cache on re-fit
        self._notify("fit_start", {})
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        splitter  = KFoldSplitter(n_splits=self.cv_splits, random_state=self.random_state)
        validator = CrossValidator(splitter, metrics=[accuracy, precision, recall, f1_score])

        def _evaluate(name: str, params: dict) -> tuple[str, dict, dict]:
            """Evaluate one (model, params) config; use cache if already done."""
            cache_key = (name, frozenset(params.items()))
            with self._cache_lock:
                if cache_key in self._cache:
                    return name, params, self._cache[cache_key]

            scores = validator.validate(ModelFactory.create(name, **params), X, y)

            with self._cache_lock:
                self._cache[cache_key] = scores

            return name, params, scores

        # Submit every (model, params) combination at once
        all_tasks = [
            (name, params)
            for name, configs in self.param_grid.items()
            for params in configs
        ]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_evaluate, name, params)
                       for name, params in all_tasks]
            results = [f.result() for f in futures]

        # Keep only the best config per model name
        best_per_model: dict[str, tuple[dict, dict]] = {}
        for name, params, scores in results:
            current_best = best_per_model.get(name)
            if current_best is None or scores[self.metric] > current_best[1][self.metric]:
                best_per_model[name] = (params, scores)

        for name, (params, scores) in best_per_model.items():
            self._notify("model_done", {"model": name, "params": params, "scores": scores})
            self._leaderboard.append({
                "model": name,
                "params": params,
                **{k: float(v) for k, v in scores.items()},
            })

        self._leaderboard.sort(key=lambda r: r[self.metric], reverse=True)

        best = self._leaderboard[0]
        self._best_model = ModelFactory.create(best["model"], **best["params"])
        self._best_model.fit(X, y)

        self._notify("fit_done", {"best_model": best["model"]})
        return self
