"""tests for the functions in stats_utils.py.

the statistical functions produce many of the thesis results. the tests cover
known failure modes in the cmh covariance and confidence intervals.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import normalize_level  # noqa: E402
from stats_utils import (
    ci_coverage_check,  # noqa: E402
    bootstrap_v,
    cmh_by_level,
    cramers_v,
    cramers_v_ncx2_ci,
    cramers_v_uncorrected,
    epsilon_squared_kruskal,
    minimum_detectable_v,
    permutation_p,
    standardised_residuals,
    theils_u,
)


# ---------------------------------------------------------------------------
# cramer's v
# ---------------------------------------------------------------------------


def test_cramers_v_perfect_association():
    """A block-diagonal table is a perfect association: V must be ~1."""
    ct = pd.DataFrame(np.diag([100, 100, 100]))
    v, _ = cramers_v(ct)
    assert v == pytest.approx(1.0, abs=0.01)


def test_cramers_v_independence_is_near_zero():
    """A table built from independent margins has V at or near 0, and the
    Bergsma correction should clip it to exactly 0 rather than reporting noise."""
    rng = np.random.default_rng(0)
    a = rng.choice(["A1", "A2", "B1"], 4000)
    b = rng.choice(list(range(5)), 4000)
    ct = pd.crosstab(pd.Series(a), pd.Series(b))
    v, p = cramers_v(ct)
    assert v < 0.05
    assert p > 0.01


def test_cramers_v_known_2x2():
    """Hand-checkable case. For a 2x2 table V equals the phi coefficient,
    which for this table is |ad - bc| / sqrt(row/col product marginals)."""
    ct = pd.DataFrame([[40, 10], [10, 40]])
    a, b, c, d = 40, 10, 10, 40
    phi = abs(a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    assert cramers_v_uncorrected(ct) == pytest.approx(phi, abs=1e-6)


def test_bias_correction_shrinks_estimate():
    """The corrected V must never exceed the uncorrected one."""
    rng = np.random.default_rng(1)
    a = rng.choice(list("ABCDEF"), 300)
    b = rng.choice(list(range(15)), 300)
    ct = pd.crosstab(pd.Series(a), pd.Series(b))
    assert cramers_v(ct)[0] <= cramers_v_uncorrected(ct) + 1e-12


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_primary_ci_contains_point_estimate_in_regular_table():
    """Basic guard for the noncentral-chi-square interval."""
    rng = np.random.default_rng(2)
    a = rng.choice(["A1", "A2", "B1", "B2", "C1", "C2"], 500)
    b = rng.choice(list(range(15)), 500)
    r = bootstrap_v(a, b, n_boot=200, seed=7)
    assert r["ci_lo"] <= r["v"] <= r["ci_hi"]


@pytest.mark.slow
def test_ncx2_ci_has_reasonable_simulated_coverage():
    """measure coverage against an analytically known v.

    the bound is loose because this is a small and fast simulation. the full
    corpus-specific simulation is written by step 03d.
    """
    r = ci_coverage_check(n=509, tilt=0.30, n_rep=40, n_boot=80, seed=11)
    assert 0.80 <= r["coverage_ncx2"] <= 1.0, r


@pytest.mark.slow
def test_ncx2_interval_beats_bootstrap_alternatives_near_the_null():
    """Near the null the resampling intervals can sit above the true V.

    Noncentral inversion should retain zero far more often.
    """
    r = ci_coverage_check(n=600, tilt=0.02, n_rep=40, n_boot=80, seed=12)
    assert r["coverage_ncx2"] > r["coverage_normal"], r
    assert r["coverage_ncx2"] > r["coverage_percentile"], r


def test_ncx2_ci_includes_zero_for_nonsignificant_table():
    ct = pd.DataFrame([[45, 42, 47], [44, 48, 44]])
    _, p = cramers_v(ct)
    ci = cramers_v_ncx2_ci(ct)
    assert p > 0.05
    assert ci["ci_lo"] == 0.0


def test_bootstrap_ci_bounds_are_valid_range():
    rng = np.random.default_rng(3)
    a = rng.choice(["A1", "A2"], 400)
    b = rng.choice([0, 1, 2], 400)
    r = bootstrap_v(a, b, n_boot=200, seed=7)
    assert 0.0 <= r["ci_lo"] <= r["ci_hi"] <= 1.0


def test_bootstrap_bias_is_positive_for_sparse_table():
    """Documents the pathology that motivated the choice of interval: V is a
    convex function of chi-square, so resampling inflates it."""
    rng = np.random.default_rng(4)
    a = rng.choice(["A1", "A2", "B1", "B2", "C1", "C2"], 400)
    b = rng.choice(list(range(15)), 400)
    r = bootstrap_v(a, b, n_boot=300, seed=11)
    assert r["boot_bias"] > 0


# ---------------------------------------------------------------------------
# permutation
# ---------------------------------------------------------------------------


def test_permutation_p_detects_real_association():
    n = 600
    a = np.array(["A1"] * (n // 2) + ["C2"] * (n // 2))
    b = np.array([0] * (n // 2) + [1] * (n // 2))
    r = permutation_p(a, b, n_perm=200, seed=5)
    assert r["perm_p"] < 0.01


def test_permutation_p_is_uniformish_under_null():
    rng = np.random.default_rng(6)
    a = rng.choice(["A1", "A2", "B1"], 600)
    b = rng.choice([0, 1, 2, 3], 600)
    r = permutation_p(a, b, n_perm=200, seed=6)
    assert r["perm_p"] > 0.02


def test_permutation_p_never_exactly_zero():
    """The add-one estimator must never report p = 0, which is what the
    chi-square asymptotic underflowed to in the original draft."""
    n = 400
    a = np.array(["A1"] * (n // 2) + ["C2"] * (n // 2))
    b = np.array([0] * (n // 2) + [1] * (n // 2))
    r = permutation_p(a, b, n_perm=50, seed=1)
    assert r["perm_p"] > 0


# ---------------------------------------------------------------------------
# theil's u
# ---------------------------------------------------------------------------


def test_theils_u_self_is_one():
    x = np.random.default_rng(7).choice(list("ABC"), 300)
    assert theils_u(x, x) == pytest.approx(1.0, abs=1e-9)


def test_theils_u_independent_is_near_zero():
    rng = np.random.default_rng(8)
    x = rng.choice(list("ABC"), 3000)
    y = rng.choice([0, 1, 2, 3], 3000)
    assert theils_u(x, y) < 0.02


def test_theils_u_is_asymmetric():
    """The whole reason for using it: a coarse variable can determine a fine one
    without the reverse holding."""
    x = np.array(["A"] * 100 + ["B"] * 100)          # 2 categories
    y = np.array([0] * 50 + [1] * 50 + [2] * 100)    # 3 categories, nested in x
    assert theils_u(x, y) == pytest.approx(1.0, abs=1e-9)
    assert theils_u(y, x) < 1.0


# ---------------------------------------------------------------------------
# cmh: checked against statsmodels on the 2x2xk case
# ---------------------------------------------------------------------------


def test_cmh_matches_statsmodels_on_2x2xk():
    """Regression test for the covariance bug. The generalised CMH implemented
    here must reduce exactly to the standard CMH when there are two topics."""
    from statsmodels.stats.contingency_tables import StratifiedTable

    rng = np.random.default_rng(0)
    rows = []
    for k in range(4):
        for _ in range(300):
            t = int(rng.integers(0, 2))
            lv = "A1" if rng.random() < 0.3 + 0.15 * t else "A2"
            rows.append({"cefr_level": lv, "topic": t, "dataset": f"s{k}"})
    d = pd.DataFrame(rows)

    mine = cmh_by_level(d)
    mine_stat = float(mine[mine.level == "A1"]["cmh_chi2"].iloc[0])

    tabs = [
        pd.crosstab(s["cefr_level"] == "A1", s["topic"])
        .reindex(index=[False, True], columns=[0, 1])
        .fillna(0)
        .to_numpy()
        for _, s in d.groupby("dataset")
    ]
    ref = StratifiedTable(np.stack(tabs, axis=2)).test_null_odds().statistic
    assert mine_stat == pytest.approx(float(ref), rel=1e-9)


def test_cmh_handles_strata_with_different_topic_sets():
    """Regression test: strata that do not contain every topic must not crash
    the pooling step, which was the original failure."""
    rows = []
    rng = np.random.default_rng(1)
    for k, topics in enumerate([[0, 1, 2], [0, 1], [1, 2, 3]]):
        for _ in range(200):
            rows.append({
                "cefr_level": rng.choice(["A1", "A2", "B1"]),
                "topic": int(rng.choice(topics)),
                "dataset": f"s{k}",
            })
    res = cmh_by_level(pd.DataFrame(rows))
    assert not res.empty
    assert (res["df"] >= 1).all()


def test_cmh_does_not_reject_under_conditional_independence():
    rng = np.random.default_rng(2)
    rows = []
    for k in range(3):
        for _ in range(800):
            rows.append({
                "cefr_level": rng.choice(["A1", "A2", "B1"]),
                "topic": int(rng.integers(0, 4)),
                "dataset": f"s{k}",
            })
    res = cmh_by_level(pd.DataFrame(rows))
    assert (res["p"] > 0.001).all()


# ---------------------------------------------------------------------------
# residuals, effect sizes, power
# ---------------------------------------------------------------------------


def test_standardised_residuals_sum_structure():
    """Under independence residuals should be small; a planted cell should
    stand out sharply."""
    ct = pd.DataFrame([[100, 100], [100, 100]])
    assert np.allclose(standardised_residuals(ct).to_numpy(), 0.0, atol=1e-9)
    ct2 = pd.DataFrame([[200, 20], [20, 200]])
    assert standardised_residuals(ct2).to_numpy()[0, 0] > 5


def test_epsilon_squared_zero_under_no_effect():
    rng = np.random.default_rng(9)
    ranks = rng.integers(0, 6, 2000).astype(float)
    groups = rng.integers(0, 5, 2000)
    eps2, _, p = epsilon_squared_kruskal(ranks, groups)
    assert eps2 < 0.01
    assert p > 0.01


def test_epsilon_squared_high_under_strong_effect():
    ranks = np.array([0.0] * 500 + [5.0] * 500)
    groups = np.array([0] * 500 + [1] * 500)
    eps2, _, p = epsilon_squared_kruskal(ranks, groups)
    assert eps2 > 0.9
    assert p < 1e-10


def test_power_simulation_holds_nominal_size():
    """The Monte Carlo critical value must give a type I error at alpha. This
    is the check that caught the degenerate power results produced when the
    clipped, bias-corrected V was used as the test statistic."""
    res = minimum_detectable_v(
        level_counts=np.array([80, 2600, 2400, 400, 130, 100]),
        topic_counts=np.full(15, 380),
        n=1000, n_sim=200, seed=3,
    )
    assert res["type_i_error_check"] == pytest.approx(0.05, abs=0.03)
    assert 0.0 < res["min_detectable_v"] < 1.0


def test_power_increases_with_sample_size():
    """More data must make smaller effects detectable."""
    kw = dict(level_counts=np.array([500, 500]), topic_counts=np.full(4, 250),
              n_sim=200, seed=4)
    small = minimum_detectable_v(n=200, **kw)["min_detectable_v"]
    large = minimum_detectable_v(n=2000, **kw)["min_detectable_v"]
    assert large < small


# ---------------------------------------------------------------------------
# length-quartile bins (script 03c)
# ---------------------------------------------------------------------------


def test_length_quartile_bins_four_on_spread_data():
    from stats_utils import length_quartile_bins

    s = pd.Series(np.arange(100))
    bins = length_quartile_bins(s)
    assert set(bins) == {"Q1", "Q2", "Q3", "Q4"}
    assert bins.value_counts().min() >= 20


def test_length_quartile_bins_collapses_on_ties():
    from stats_utils import length_quartile_bins

    s = pd.Series([10] * 40 + [20] * 40)
    bins = length_quartile_bins(s)
    assert bins.nunique() == 2
    assert set(bins) <= {"Q1", "Q2"}


def test_length_quartile_bins_single_value():
    from stats_utils import length_quartile_bins

    bins = length_quartile_bins(pd.Series([42] * 12))
    assert (bins == "Q1").all()


# ---------------------------------------------------------------------------
# label normalisation (config)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("a1", "A1"), ("A1+", "A1"), (" A1 ", "A1"), ("B2", "B2"), ("c1", "C1"),
    ("A", None), ("", None), (None, None), ("X9", None), ("B1+", "B1"),
])
def test_normalize_level(raw, expected):
    assert normalize_level(raw) == expected


def test_drift_guard_ignores_ungrouped_rows():
    """'ungrouped' contains the substring 'grouped'; the guard must not
    pick the leakage-check row as the reportable topic-mixture F1."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_manifest",
        Path(__file__).resolve().parents[1] / "src" / "09_build_manifest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest = {
        "baselines": {"all": [
            {"model": "topic mixture", "cv": "ungrouped", "macro_f1": 0.229},
            {"model": "topic mixture", "cv": "grouped", "macro_f1": 0.2593},
        ]},
        "length_benchmark": {"models": [
            {"features": "topic only", "cv": "ungrouped", "macro_f1": 0.2568},
            {"features": "topic only", "cv": "grouped (pair_id)", "macro_f1": 0.2593},
        ]},
        "robustness": {"k_sweep": [
            {"n_topics": 15, "measure": "topic_mixture_macro_f1", "value": 0.259},
        ]},
    }
    assert mod.collect_macro_f1_drift(manifest) == []
