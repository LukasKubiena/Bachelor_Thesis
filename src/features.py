"""shared feature definitions"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

_SENT_SPLIT = re.compile(r"[.!?]+")


def vectorizer_stopwords(lang: str) -> list[str]:
    """return stopwords consistent with countvectorizer's tokenisation"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        import stopwordsiso

    raw = set(stopwordsiso.stopwords(lang))
    analyzer = CountVectorizer(lowercase=True).build_analyzer()
    analysed = {token for word in raw for token in analyzer(word)}
    return sorted(raw | analysed)


def _assert_finite_array(values, label: str) -> None:
    """fail loudly when a dense or sparse numerical object is non-finite"""
    data = values.data if sparse.issparse(values) else np.asarray(values)
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{label} contains NaN or infinite values")


def checked_numeric_matrix(operation, label: str, *, nonnegative: bool = False) -> np.ndarray:
    """run a matrix-producing operation and enforce finite output"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"sklearn\.(decomposition\._nmf|utils\.extmath)",
            )
            matrix = np.asarray(operation(), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional, got {matrix.shape}")
    _assert_finite_array(matrix, label)
    if nonnegative and (matrix < -1e-12).any():
        raise FloatingPointError(f"{label} contains negative values")
    return matrix


class CheckedLogisticRegression(LogisticRegression):
    """logistic regression with explicit numerical postconditions"""

    @staticmethod
    def _checked_call(operation, label: str):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r"sklearn\.(linear_model\._linear_loss|utils\.extmath)",
                )
                result = operation()
        if hasattr(result, "coef_"):
            _assert_finite_array(result.coef_, "logistic-regression coefficients")
            _assert_finite_array(result.intercept_, "logistic-regression intercept")
        else:
            _assert_finite_array(result, label)
        return result

    def fit(self, X, y, sample_weight=None):
        _assert_finite_array(X, "logistic-regression design matrix")
        self._checked_call(
            lambda: super(CheckedLogisticRegression, self).fit(
                X, y, sample_weight=sample_weight
            ),
            "fitted logistic-regression estimator",
        )
        return self

    def decision_function(self, X):
        _assert_finite_array(X, "logistic-regression prediction matrix")
        return self._checked_call(
            lambda: super(CheckedLogisticRegression, self).decision_function(X),
            "logistic-regression decision scores",
        )

    def predict_proba(self, X):
        _assert_finite_array(X, "logistic-regression prediction matrix")
        return self._checked_call(
            lambda: super(CheckedLogisticRegression, self).predict_proba(X),
            "logistic-regression probabilities",
        )


def surface_features(texts: pd.Series) -> pd.DataFrame:
    """four surface features: two of length, two of complexity. no content words"""
    rows = []
    for t in texts.astype(str):
        words = t.split()
        n_w = len(words)
        n_s = max(1, len([s for s in _SENT_SPLIT.split(t) if s.strip()]))
        rows.append({
            "log_tokens": np.log1p(n_w),                                       # length
            "n_sentences": n_s,                                                # length
            "tokens_per_sentence": n_w / n_s,                                  # complexity
            "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        })
    return pd.DataFrame(rows, index=texts.index)


def word_count_feature(word_count) -> np.ndarray:
    """log word count on its own, as a deliberately minimal length baseline"""
    return np.c_[np.log1p(np.asarray(word_count, dtype=float))]


def validation_groups(df: pd.DataFrame) -> np.ndarray:
    """use merged parallel/near-duplicate families for cross-validation"""
    if "cv_group" in df.columns and df["cv_group"].notna().all():
        return df["cv_group"].astype(str).to_numpy()
    if "pair_id" in df.columns and df["pair_id"].notna().all():
        return df["pair_id"].astype(str).to_numpy()
    raise ValueError(
        "No validation grouping found. Run src/01c_near_duplicates.py after "
        "src/01_load_data.py."
    )


def normalise_topic_weights(matrix) -> np.ndarray:
    """convert non-negative topic loadings into per-document relative weights"""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"topic matrix must be 2-dimensional, got shape {mat.shape}")
    if not np.isfinite(mat).all():
        raise ValueError("topic matrix contains NaN or infinite values")
    if (mat < 0).any():
        raise ValueError("topic weights must be non-negative for L1 normalisation")
    row_sum = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)


# compatibility alias; new code uses surface_features
length_features = surface_features
