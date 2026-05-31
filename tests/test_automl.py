import numpy as np
import pytest

from autoclass_lite import SimpleAutoML, GridAutoML
from autoclass_lite.automl.observers import Observer, ConsoleObserver


@pytest.fixture
def sample_data():
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3],
                  [5, 6], [5, 7], [6, 6], [6, 7]] * 3, dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
    return X, y


# ---------------------------------------------------------------------------
# SimpleAutoML — basic functionality
# ---------------------------------------------------------------------------

def test_simple_automl_fit_returns_self(sample_data):
    X, y = sample_data
    automl = SimpleAutoML()
    assert automl.fit(X, y) is automl


def test_simple_automl_leaderboard_length(sample_data):
    X, y = sample_data
    automl = SimpleAutoML()
    automl.fit(X, y)
    assert len(automl.leaderboard()) == 4


def test_simple_automl_leaderboard_sorted(sample_data):
    X, y = sample_data
    automl = SimpleAutoML()
    automl.fit(X, y)
    leaderboard = automl.leaderboard()
    assert leaderboard[0]["accuracy"] >= leaderboard[-1]["accuracy"]


def test_simple_automl_leaderboard_has_required_keys(sample_data):
    X, y = sample_data
    automl = SimpleAutoML()
    automl.fit(X, y)
    for row in automl.leaderboard():
        assert "model" in row
        assert "accuracy" in row
        assert "f1_score" in row


def test_simple_automl_predict_shape(sample_data):
    X, y = sample_data
    automl = SimpleAutoML()
    automl.fit(X, y)
    assert len(automl.predict(X)) == len(X)


def test_simple_automl_predict_before_fit_raises(sample_data):
    X, _ = sample_data
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        SimpleAutoML().predict(X)


def test_simple_automl_fit_is_idempotent(sample_data):
    """Calling fit() twice should reset and produce a fresh leaderboard."""
    X, y = sample_data
    automl = SimpleAutoML(cv_splits=3, random_state=0)
    automl.fit(X, y)
    lb1 = [r["accuracy"] for r in automl.leaderboard()]
    automl.fit(X, y)
    lb2 = [r["accuracy"] for r in automl.leaderboard()]
    assert lb1 == lb2


def test_simple_automl_summary_runs_without_error(sample_data, capsys):
    X, y = sample_data
    automl = SimpleAutoML()
    automl.fit(X, y)
    automl.summary()          # should not raise
    captured = capsys.readouterr()
    assert "accuracy" in captured.out


def test_simple_automl_summary_before_fit(capsys):
    automl = SimpleAutoML()
    automl.summary()
    captured = capsys.readouterr()
    assert "fit()" in captured.out


def test_simple_automl_accepts_lists(sample_data):
    X, y = sample_data
    automl = SimpleAutoML(cv_splits=3)
    automl.fit(X.tolist(), y.tolist())
    preds = automl.predict(X.tolist())
    assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# SimpleAutoML — Observer pattern
# ---------------------------------------------------------------------------

class _CapturingObserver(Observer):
    """Test observer that records all received events."""
    def __init__(self):
        self.events = []

    def update(self, event: str, data: dict) -> None:
        self.events.append((event, data))


def test_observer_receives_fit_start(sample_data):
    X, y = sample_data
    obs = _CapturingObserver()
    automl = SimpleAutoML(cv_splits=3)
    automl.add_observer(obs)
    automl.fit(X, y)
    event_names = [e for e, _ in obs.events]
    assert "fit_start" in event_names


def test_observer_receives_model_done_for_each_model(sample_data):
    X, y = sample_data
    obs = _CapturingObserver()
    automl = SimpleAutoML(cv_splits=3)
    automl.add_observer(obs)
    automl.fit(X, y)
    model_done_events = [e for e, _ in obs.events if e == "model_done"]
    assert len(model_done_events) == 4   # one per DEFAULT_CONFIGS entry


def test_observer_receives_fit_done(sample_data):
    X, y = sample_data
    obs = _CapturingObserver()
    automl = SimpleAutoML(cv_splits=3)
    automl.add_observer(obs)
    automl.fit(X, y)
    event_names = [e for e, _ in obs.events]
    assert "fit_done" in event_names


def test_console_observer_fit_start(capsys):
    obs = ConsoleObserver()
    obs.update("fit_start", {})
    captured = capsys.readouterr()
    assert "AutoML" in captured.out


def test_console_observer_fit_done(capsys):
    obs = ConsoleObserver()
    obs.update("fit_done", {"best_model": "knn"})
    captured = capsys.readouterr()
    assert "knn" in captured.out


# ---------------------------------------------------------------------------
# GridAutoML
# ---------------------------------------------------------------------------

def test_grid_automl_fit_returns_self(sample_data):
    X, y = sample_data
    automl = GridAutoML(cv_splits=3)
    assert automl.fit(X, y) is automl


def test_grid_automl_leaderboard_length(sample_data):
    X, y = sample_data
    automl = GridAutoML()
    automl.fit(X, y)
    assert len(automl.leaderboard()) == 4


def test_grid_automl_leaderboard_sorted(sample_data):
    X, y = sample_data
    automl = GridAutoML(cv_splits=3)
    automl.fit(X, y)
    lb = automl.leaderboard()
    assert lb[0]["accuracy"] >= lb[-1]["accuracy"]


def test_grid_automl_cache_populated(sample_data):
    X, y = sample_data
    automl = GridAutoML()
    automl.fit(X, y)
    assert bool(automl._cache)


def test_grid_automl_cache_cleared_on_refit(sample_data):
    X, y = sample_data
    automl = GridAutoML(cv_splits=3)
    automl.fit(X, y)
    old_cache_size = len(automl._cache)
    automl.fit(X, y)
    assert len(automl._cache) == old_cache_size   # same configs → same size


def test_grid_automl_predict_shape(sample_data):
    X, y = sample_data
    automl = GridAutoML(cv_splits=3)
    automl.fit(X, y)
    assert len(automl.predict(X)) == len(X)


def test_grid_automl_custom_param_grid(sample_data):
    X, y = sample_data
    grid = GridAutoML(
        param_grid={
            "knn": [{"k": 3}, {"k": 5}],
            "naive_bayes": [{}],
        },
        cv_splits=3,
    )
    grid.fit(X, y)
    assert len(grid.leaderboard()) == 2


def test_grid_automl_f1_metric(sample_data):
    X, y = sample_data
    automl = GridAutoML(
        param_grid={"knn": [{"k": 3}], "naive_bayes": [{}]},
        metric="f1_score",
        cv_splits=3,
    )
    automl.fit(X, y)
    lb = automl.leaderboard()
    assert lb[0]["f1_score"] >= lb[-1]["f1_score"]
