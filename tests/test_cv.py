import numpy as np
import pytest
from autoclass_lite.cv.splitter import KFoldSplitter
from autoclass_lite.cv.validator import CrossValidator
from autoclass_lite.metrics.classification import accuracy
from autoclass_lite.models.knn import KNNClassifier


# ---------------------------------------------------------------------------
# KFoldSplitter tests
# ---------------------------------------------------------------------------

def test_kfold_returns_correct_number_of_folds():
    splitter = KFoldSplitter(n_splits=5)
    idx = np.arange(100)
    folds = splitter.split(idx)
    assert len(folds) == 5


def test_kfold_test_indices_cover_all_samples():
    splitter = KFoldSplitter(n_splits=5)
    idx = np.arange(100)
    all_test = np.concatenate([test for _, test in splitter.split(idx)])
    assert sorted(all_test) == list(range(100))


def test_kfold_train_and_test_are_disjoint():
    """Train and test indices for each fold must not overlap."""
    splitter = KFoldSplitter(n_splits=5)
    for train, test in splitter.split(np.arange(50)):
        assert len(set(train) & set(test)) == 0


def test_kfold_no_shuffle_is_deterministic():
    """Without shuffle, two calls must return identical folds."""
    splitter = KFoldSplitter(n_splits=3, shuffle=False)
    folds1 = splitter.split(np.arange(30))
    folds2 = splitter.split(np.arange(30))
    for (tr1, te1), (tr2, te2) in zip(folds1, folds2):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


def test_kfold_random_state_reproducible():
    """Same random_state → same fold order."""
    splitter = KFoldSplitter(n_splits=3, random_state=42)
    folds1 = splitter.split(np.arange(30))
    folds2 = KFoldSplitter(n_splits=3, random_state=42).split(np.arange(30))
    for (tr1, te1), (tr2, te2) in zip(folds1, folds2):
        np.testing.assert_array_equal(tr1, tr2)


def test_kfold_different_random_states_differ():
    """Different random states should (almost certainly) produce different folds."""
    folds1 = KFoldSplitter(n_splits=5, random_state=1).split(np.arange(50))
    folds2 = KFoldSplitter(n_splits=5, random_state=99).split(np.arange(50))
    all_same = all(
        np.array_equal(te1, te2) for (_, te1), (_, te2) in zip(folds1, folds2)
    )
    assert not all_same


def test_kfold_3_splits():
    splitter = KFoldSplitter(n_splits=3)
    folds = splitter.split(np.arange(90))
    assert len(folds) == 3


# ---------------------------------------------------------------------------
# CrossValidator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3],
                  [5, 6], [5, 7], [6, 6], [6, 7]] * 3, dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
    return X, y


def test_cross_validator_returns_dict(sample_data):
    X, y = sample_data
    splitter = KFoldSplitter(n_splits=3, random_state=0)
    validator = CrossValidator(splitter, metrics=[accuracy])
    scores = validator.validate(KNNClassifier(k=3), X, y)
    assert isinstance(scores, dict)


def test_cross_validator_contains_accuracy_key(sample_data):
    X, y = sample_data
    splitter = KFoldSplitter(n_splits=3, random_state=0)
    validator = CrossValidator(splitter, metrics=[accuracy])
    scores = validator.validate(KNNClassifier(k=3), X, y)
    assert "accuracy" in scores


def test_cross_validator_accuracy_in_range(sample_data):
    X, y = sample_data
    splitter = KFoldSplitter(n_splits=3, random_state=0)
    validator = CrossValidator(splitter, metrics=[accuracy])
    scores = validator.validate(KNNClassifier(k=3), X, y)
    assert 0.0 <= scores["accuracy"] <= 1.0


def test_cross_validator_does_not_mutate_original_model(sample_data):
    """CrossValidator uses deepcopy — original model should stay unfitted."""
    X, y = sample_data
    model = KNNClassifier(k=3)
    splitter = KFoldSplitter(n_splits=3, random_state=0)
    validator = CrossValidator(splitter, metrics=[accuracy])
    validator.validate(model, X, y)
    # The original model's training data must remain unchanged
    assert model._X_train is None
