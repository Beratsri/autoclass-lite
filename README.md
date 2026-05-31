# autoclass-lite

A lightweight AutoClass library for classification, built from scratch using only NumPy. Trains multiple models with cross-validation, ranks them by performance, and returns the best one — all in a single `fit()` call.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from autoclass_lite import SimpleAutoML, GridAutoML

# Train all models and get a ranked leaderboard
automl = SimpleAutoML()
automl.fit(X_train, y_train)
automl.summary()

# Predict using the best model
predictions = automl.predict(X_test)

# Grid search over hyperparameters
grid = GridAutoML()
grid.fit(X_train, y_train)
grid.summary()
```

## Features

- 4 classifiers implemented from scratch: KNN, Naive Bayes, Logistic Regression, Decision Tree
- K-Fold cross-validation with parallel fold evaluation
- Hyperparameter grid search with memoization cache (dynamic programming)
- Observer-based progress reporting
- Accepts NumPy arrays, Python lists, and Pandas DataFrames

## Usage Examples

### SimpleAutoML

```python
from autoclass_lite import SimpleAutoML

automl = SimpleAutoML(
    cv_splits=5,        # number of cross-validation folds
    metric="accuracy",  # metric used to rank models
    random_state=42
)
automl.fit(X_train, y_train)
automl.summary()
preds = automl.predict(X_test)
```

### GridAutoML

```python
from autoclass_lite import GridAutoML

param_grid = {
    "logistic_regression": [
        {"learning_rate": 0.01},
        {"learning_rate": 0.1},
    ],
    "knn": [{"k": 3}, {"k": 5}, {"k": 7}],
    "decision_tree": [{"max_depth": 3}, {"max_depth": 5}],
}

grid = GridAutoML(param_grid=param_grid, metric="f1_score")
grid.fit(X_train, y_train)
grid.summary()
```

### Logistic Regression

```python
from autoclass_lite.models.logistic import LogisticRegression

model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Metrics

```python
from autoclass_lite.metrics.classification import accuracy, precision, recall, f1_score, evaluate

acc = accuracy(y_true, y_pred)

results = evaluate(y_true, y_pred, metrics=[accuracy, precision, recall, f1_score])
```

### Observer (progress reporting)

```python
from autoclass_lite.automl.observers import ConsoleObserver

automl = SimpleAutoML()
automl.add_observer(ConsoleObserver())
automl.fit(X_train, y_train)
```

## Project Structure

```
autoclass_lite/
├── __init__.py               # Public API: SimpleAutoML, GridAutoML
├── automl/
│   ├── orchestrator.py       # SimpleAutoML, GridAutoML (Facade pattern)
│   ├── factory.py            # ModelFactory (Factory pattern)
│   └── observers.py          # Observer, ConsoleObserver (Observer pattern)
├── models/
│   ├── base.py               # BaseModel ABC (Strategy + Template Method)
│   ├── logistic.py           # LogisticRegression
│   ├── knn.py                # KNNClassifier
│   ├── naive_bayes.py        # GaussianNaiveBayes
│   └── tree.py               # DecisionTreeClassifier (recursive)
├── cv/
│   ├── splitter.py           # KFoldSplitter
│   └── validator.py          # CrossValidator (ThreadPoolExecutor)
└── metrics/
    └── classification.py     # Pure functions + evaluate() HOF
tests/                        # pytest test suite
pyproject.toml
README.md
```

---

## Learning Outcomes

### 1. Object-Oriented Programming (OOP)

- **Encapsulation:** All internal state uses private attributes (`_weights`, `_biases`, `_cache`, etc.). Users interact only through the public API.
- **Abstract Base Class:** `BaseModel` in [autoclass_lite/models/base.py](autoclass_lite/models/base.py) defines the interface all classifiers must implement.
- **Inheritance:** `KNNClassifier`, `GaussianNaiveBayes`, `LogisticRegression`, `DecisionTreeClassifier` all extend `BaseModel`. `GridAutoML` extends `SimpleAutoML`.
- **Polymorphism:** `CrossValidator` calls `.fit()` and `.predict()` on any `BaseModel` subclass without knowing its concrete type.

### 2. Functional Programming

- **Pure functions:** `accuracy`, `precision`, `recall`, `f1_score` in [autoclass_lite/metrics/classification.py](autoclass_lite/metrics/classification.py) take arrays and return a value with no side effects.
- **Higher-order function:** `evaluate(y_true, y_pred, metrics: list)` takes a list of metric functions and applies them, returning a results dict.
- **Lambdas as data:** `KNNClassifier._DISTANCES` is a class-level dict mapping distance names to lambda functions, selected at runtime.

### 3. Concurrency

`CrossValidator` in [autoclass_lite/cv/validator.py](autoclass_lite/cv/validator.py) uses `ThreadPoolExecutor` to evaluate all K folds in parallel. Each fold gets a `copy.deepcopy` of the model to ensure thread safety.

```python
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(self._evaluate_fold, ...) for fold in folds]
```

### 4. Recursion / Dynamic Programming

- **Recursion:** `DecisionTreeClassifier` in [autoclass_lite/models/tree.py](autoclass_lite/models/tree.py) uses `_grow_tree()` to recursively split nodes and `_traverse()` to recursively walk the tree during prediction.
- **Dynamic Programming (Memoization):** `GridAutoML` in [autoclass_lite/automl/orchestrator.py](autoclass_lite/automl/orchestrator.py) maintains a `_cache` dict. Every `(model_name, params)` combination is evaluated at most once — repeated configurations are looked up from cache instead of re-running cross-validation.

```python
cache_key = (name, frozenset(params.items()))
if cache_key in self._cache:
    scores = self._cache[cache_key]
else:
    scores = validator.validate(model, X, y)
    self._cache[cache_key] = scores
```

### 5. SOLID Principles

- **Single Responsibility:** Each class does exactly one thing. `KFoldSplitter` only splits indices. `CrossValidator` only runs folds. `ModelFactory` only creates models.
- **Open/Closed:** New models can be added by extending `BaseModel` and registering in `ModelFactory` without touching existing code. `SimpleAutoML.DEFAULT_CONFIGS` allows adding new default configurations without modifying `fit()`.
- **Liskov Substitution:** Any `BaseModel` subclass can replace another anywhere in the system.
- **Interface Segregation:** `BaseModel` only requires `fit` and `predict`. Optional `get_params` has a default implementation in the base class.
- **Dependency Inversion:** `CrossValidator` depends on `BaseModel` (abstraction), not any concrete classifier.

### 6. Architectural & Design Patterns

**Architecture:** Layered pipeline — models → cross-validation → metrics → AutoML orchestration. Each layer depends only on the layer below it.

**Factory Pattern** — [autoclass_lite/automl/factory.py](autoclass_lite/automl/factory.py)

`ModelFactory.create(name, **kwargs)` instantiates any registered model by name. Adding a new model requires only one line in the registry dict.

**Strategy Pattern** — [autoclass_lite/models/base.py](autoclass_lite/models/base.py), [autoclass_lite/models/knn.py](autoclass_lite/models/knn.py)

- `BaseModel` defines the classifier interface; each model is a concrete strategy swappable at runtime by `ModelFactory`.
- `KNNClassifier._DISTANCES` selects the distance function at construction time — `euclidean` or `manhattan` — without any `if/else` in the prediction loop.

**Observer Pattern** — [autoclass_lite/automl/observers.py](autoclass_lite/automl/observers.py)

`SimpleAutoML` notifies all registered observers on `fit_start`, `model_done`, and `fit_done` events without coupling to any specific logger or UI.

**Template Method Pattern** — [autoclass_lite/models/base.py](autoclass_lite/models/base.py)

`BaseModel.fit_predict()` defines the skeleton (fit then predict). Subclasses implement the steps; the sequence is inherited and never duplicated.

**Facade Pattern** — [autoclass_lite/automl/orchestrator.py](autoclass_lite/automl/orchestrator.py)

`SimpleAutoML` exposes a single `fit() / predict() / summary()` interface that internally coordinates model creation, cross-validation, and ranking.
