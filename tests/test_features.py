from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import (  # noqa: e402
    CheckedLogisticRegression,
    checked_numeric_matrix,
    normalise_topic_weights,
    vectorizer_stopwords,
)


def test_normalise_topic_weights_preserves_zero_rows():
    x = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    got = normalise_topic_weights(x)
    assert got[0].sum() == pytest.approx(1.0)
    assert np.array_equal(got[1], np.zeros(3))


def test_normalise_topic_weights_rejects_signed_input():
    with pytest.raises(ValueError, match="non-negative"):
        normalise_topic_weights([[1.0, -0.1]])


def test_checked_logistic_regression_returns_finite_predictions():
    x = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array([0, 0, 1, 1])
    model = CheckedLogisticRegression(random_state=42).fit(x, y)
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.decision_function(x)).all()
    assert np.isfinite(model.predict_proba(x)).all()


def test_checked_logistic_regression_rejects_nonfinite_input():
    x = np.array([[0.0], [np.inf], [0.9], [1.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(FloatingPointError, match="design matrix"):
        CheckedLogisticRegression(random_state=42).fit(x, y)


def test_checked_numeric_matrix_rejects_nonfinite_output():
    with pytest.raises(FloatingPointError, match="test matrix"):
        checked_numeric_matrix(lambda: [[1.0, np.nan]], "test matrix")


def test_checked_numeric_matrix_enforces_nonnegative_output():
    with pytest.raises(FloatingPointError, match="negative"):
        checked_numeric_matrix(
            lambda: [[1.0, -0.1]], "test matrix", nonnegative=True
        )


def test_english_stopwords_match_vectorizer_tokenisation():
    stop = set(vectorizer_stopwords("en"))
    assert "hadn" in stop
    assert "mightn" in stop
