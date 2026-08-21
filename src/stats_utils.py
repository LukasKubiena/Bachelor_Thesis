"""statistics used by more than one analysis step.

cramér's v is bias-corrected for sparse tables. permutation tests provide the
reported p-values. adjusted mutual information supports comparisons across
topic counts, and theil's u keeps the direction of association explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2_contingency, kruskal, ncx2
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# effect sizes
# ---------------------------------------------------------------------------


def cramers_v(confusion: pd.DataFrame) -> tuple[float, float]:
    """Bias-corrected Cramer's V (Bergsma, 2013) and the chi-square p-value.

    correction=False turns off Yates' continuity correction, which scipy
    applies to 2x2 tables by default. Yates is meant to improve a p-value, not
    an effect size, so it would shrink V for the wrong reasons. None of my
    tables are 2x2, so this doesn't change any reported number, but it would be
    wrong if the function were reused elsewhere.
    """
    chi2, p, _, _ = chi2_contingency(confusion, correction=False)
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    v = np.sqrt(phi2_corr / denom) if denom > 0 else 0.0
    return float(v), float(p)


def cramers_v_uncorrected(confusion: pd.DataFrame) -> float:
    """Cramer's V without the bias correction. Only used inside simulations.

    The uncorrected version is needed there for two reasons. The correction is a
    finite-sample adjustment, so applying it to a population table (as in the
    power simulation) would report a real association as zero. And the
    corrected V clips at zero, so on sparse tables its null distribution piles
    up on 0 and a percentile critical value stops working.
    """
    chi2, _, _, _ = chi2_contingency(confusion, correction=False)
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    denom = min(r - 1, k - 1)
    return float(np.sqrt((chi2 / n) / denom)) if denom > 0 and n > 0 else 0.0


def cramers_v_from_arrays(a, b) -> float:
    """Convenience wrapper for the resampling loops."""
    ct = pd.crosstab(pd.Series(np.asarray(a)), pd.Series(np.asarray(b)))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    return cramers_v(ct)[0]


def theils_u(x, y) -> float:
    """Theil's U: what share of the uncertainty about x is removed by knowing y.

    Unlike mutual information this is asymmetric, so theils_u(level, topic)
    answers "how much does topic tell me about level" rather than the other way
    round. Returns 0 if y says nothing about x, 1 if y determines x.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ct = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy().astype(float)
    n = ct.sum()
    px = ct.sum(axis=1) / n
    h_x = -np.sum(px[px > 0] * np.log(px[px > 0]))
    if h_x == 0:
        return 0.0
    # h(x|y)
    pxy = ct / n
    py = ct.sum(axis=0) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / py[None, :], 1.0)
        h_x_given_y = -np.sum(pxy[pxy > 0] * np.log(ratio[pxy > 0]))
    return float((h_x - h_x_given_y) / h_x)


def epsilon_squared_kruskal(level_ranks, groups) -> tuple[float, float, float]:
    """Kruskal-Wallis on CEFR rank grouped by topic, with epsilon squared.

    Cramer's V treats the CEFR levels as unordered, so this is reported alongside
    it. Epsilon squared is the share of variance in level rank explained by
    topic. It only needs level to be ordered, not topic, which fits the data.

    Returns (epsilon_squared, H, p).
    """
    level_ranks = np.asarray(level_ranks, dtype=float)
    groups = np.asarray(groups)
    samples = [level_ranks[groups == g] for g in np.unique(groups)]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2:
        return np.nan, np.nan, np.nan
    h, p = kruskal(*samples)
    n = len(level_ranks)
    k = len(samples)
    eps2 = (h - k + 1) / (n - k) if n > k else np.nan
    return float(max(0.0, eps2)), float(h), float(p)


def information_measures(a, b) -> dict:
    """AMI, plain NMI, and Theil's U in both directions."""
    return {
        "ami": float(adjusted_mutual_info_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
        "theils_u_level_given_topic": theils_u(a, b),
        "theils_u_topic_given_level": theils_u(b, a),
    }


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


def cramers_v_ncx2_ci(confusion: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Approximate CI for population Cramer's V by inverting noncentral chi².

    For a contingency table, Pearson's statistic is asymptotically noncentral
    chi-square under an alternative, with noncentrality ``lambda = n * phi²``.
    Since ``V = phi / sqrt(min(r-1, c-1))``, inverting that distribution gives
    a bounded interval for V. Unlike the resample-and-recentre intervals below,
    it includes zero for the near-null news tables and is coherent with the
    chi-square/permutation evidence. Simulation in 03d checks its coverage for
    the observed table structures rather than only balanced synthetic margins.
    """
    ct = pd.DataFrame(confusion).copy()
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"ci_lo": 0.0, "ci_hi": 0.0,
                "ci_method": "noncentral chi-square inversion"}
    stat, _, dfree, _ = chi2_contingency(ct, correction=False)
    n = float(ct.to_numpy().sum())
    scale = min(ct.shape[0] - 1, ct.shape[1] - 1)

    def lambda_at_cdf(target: float) -> float:
        fn = lambda lam: float(ncx2.cdf(stat, dfree, lam) - target)
        # increasing lambda shifts mass to the right, so the cdf decreases. if
        # the target already lies above the central cdf, the bound is zero.
        if fn(0.0) <= 0:
            return 0.0
        hi = max(1.0, float(stat + dfree))
        while fn(hi) > 0 and hi < 1e10:
            hi *= 2.0
        if fn(hi) > 0:
            raise RuntimeError("failed to bracket noncentrality parameter")
        return float(brentq(fn, 0.0, hi))

    lam_lo = lambda_at_cdf(1 - alpha / 2)
    lam_hi = lambda_at_cdf(alpha / 2)
    return {
        "ci_lo": float(np.sqrt(lam_lo / (n * scale))),
        "ci_hi": float(np.sqrt(lam_hi / (n * scale))),
        "ci_method": "noncentral chi-square inversion",
    }


def bootstrap_v(a, b, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Bias-corrected V with a primary noncentral-chi² interval.

    Bootstrap draws are retained for a standard-error and sensitivity
    diagnostics. They are not the headline interval: resampling sparse tables
    pushes corrected V upward, causing percentile intervals to fail near zero,
    while a symmetric normal interval can exclude zero even when independence
    is not rejected. ``ci_lo``/``ci_hi`` therefore use noncentral chi-square
    inversion; all bootstrap alternatives are returned under explicit names.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    obs = cramers_v_from_arrays(a, b)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = cramers_v_from_arrays(a[idx], b[idx])
    good = boots[~np.isnan(boots)]
    q_lo, q_hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = float(np.std(good, ddof=1))
    z = 1.959963984540054  # two-sided 95%
    normal_lo = float(np.clip(obs - z * se, 0.0, 1.0))
    normal_hi = float(np.clip(obs + z * se, 0.0, 1.0))
    ncx = cramers_v_ncx2_ci(pd.crosstab(pd.Series(a), pd.Series(b)), alpha=alpha)
    return {
        "v": float(obs),
        "ci_lo": ncx["ci_lo"],
        "ci_hi": ncx["ci_hi"],
        "ci_lo_bootstrap_normal": normal_lo,
        "ci_hi_bootstrap_normal": normal_hi,
        "ci_lo_percentile": float(q_lo),
        "ci_hi_percentile": float(q_hi),
        "ci_lo_basic": float(max(0.0, 2 * obs - q_hi)),
        "ci_hi_basic": float(max(0.0, 2 * obs - q_lo)),
        "boot_bias": float(good.mean() - obs),
        "boot_se": se,
        "boot_sd": se,
        "n_boot_used": int(len(good)),
        "ci_method": ncx["ci_method"],
    }



def ci_coverage_check(n: int, tilt: float | None = None, n_lev: int = 6,
                      n_top: int = 15, n_rep: int = 1000, n_boot: int = 120,
                      seed: int = 42, joint: np.ndarray | None = None,
                      n_bootstrap_rep: int | None = None) -> dict:
    """Simulated coverage against a known joint distribution.

    Pass ``joint`` to reproduce an observed table structure (or its independence
    model). ``tilt`` remains available for small regression tests. The true
    population V is calculated analytically from the generating probabilities.
    """
    rng = np.random.default_rng(seed)
    if joint is None:
        if tilt is None:
            raise ValueError("pass either joint or tilt")
        p = np.full((n_lev, n_top), 1.0 / n_top)
        p[np.arange(n_lev), np.arange(n_lev) % n_top] += tilt
        p /= p.sum(axis=1, keepdims=True)
        p /= n_lev
    else:
        p = np.asarray(joint, dtype=float)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
            raise ValueError(f"joint must be a 2D contingency distribution, got {p.shape}")
        if (p < 0).any() or not np.isfinite(p).all() or p.sum() <= 0:
            raise ValueError("joint probabilities must be finite and non-negative")
        p = p / p.sum()
        n_lev, n_top = p.shape
    pi = p.sum(axis=1, keepdims=True)
    pj = p.sum(axis=0, keepdims=True)
    phi2 = (((p - pi * pj) ** 2) / (pi * pj)).sum()
    true_v = float(np.sqrt(phi2 / min(n_lev - 1, n_top - 1)))

    if n_bootstrap_rep is None:
        n_bootstrap_rep = n_rep
    if not 0 <= n_bootstrap_rep <= n_rep:
        raise ValueError("n_bootstrap_rep must be between zero and n_rep")

    flat = p.ravel()
    hits = {"ncx2": 0, "normal": 0, "percentile": 0, "basic": 0}
    for i in range(n_rep):
        idx = rng.choice(flat.size, size=n, p=flat)
        a, b = (idx // n_top).astype(str), idx % n_top
        ct = pd.crosstab(pd.Series(a), pd.Series(b))
        ncx = cramers_v_ncx2_ci(ct)
        hits["ncx2"] += ncx["ci_lo"] <= true_v <= ncx["ci_hi"]
        if i < n_bootstrap_rep:
            r = bootstrap_v(a, b, n_boot=n_boot,
                            seed=int(rng.integers(1_000_000)))
            hits["normal"] += r["ci_lo_bootstrap_normal"] <= true_v <= r["ci_hi_bootstrap_normal"]
            hits["percentile"] += r["ci_lo_percentile"] <= true_v <= r["ci_hi_percentile"]
            hits["basic"] += r["ci_lo_basic"] <= true_v <= r["ci_hi_basic"]
    return {
        "n": n, "tilt": tilt, "true_v": true_v,
        "n_rep_ncx2": n_rep, "n_rep_bootstrap": n_bootstrap_rep,
        "coverage_ncx2": hits["ncx2"] / n_rep,
        **{f"coverage_{k}": hits[k] / n_bootstrap_rep
           if n_bootstrap_rep else np.nan
           for k in ("normal", "percentile", "basic")},
    }


def permutation_p(a, b, n_perm: int = 5000, seed: int = 42) -> dict:
    """Permutation p-value for the null that a and b are independent.

    Shuffles a, recomputes V, and counts how often the shuffled V reaches the
    observed one. If that never happens I report p < 1/n_perm, which is the
    lowest the test can actually resolve. The chi-square p underflows to
    0.00e+00 on the full dataset, which isn't something I can put in a table.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a).copy()
    b = np.asarray(b)
    obs = cramers_v_from_arrays(a, b)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(a)
        if cramers_v_from_arrays(a, b) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)  # add-one, never reports exactly zero
    return {
        "v": float(obs),
        "perm_p": float(p),
        "perm_p_label": f"< {1 / n_perm:.1e}" if count == 0 else f"{p:.2e}",
        "n_perm": n_perm,
    }


def subsample_v(a, b, n_sub: int, n_rep: int = 200, seed: int = 42) -> dict:
    """spread of v after subsampling each corpus to the same size.

    this checks whether the corpus ranking is mainly driven by sample size.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n < n_sub:
        return {"n_sub": n_sub, "mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan}
    vals = []
    for _ in range(n_rep):
        idx = rng.choice(n, n_sub, replace=False)
        v = cramers_v_from_arrays(a[idx], b[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "n_sub": n_sub,
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "lo": float(lo),
        "hi": float(hi),
    }


# ---------------------------------------------------------------------------
# power / equivalence for the null results
# ---------------------------------------------------------------------------


def minimum_detectable_v(
    level_counts: np.ndarray,
    topic_counts: np.ndarray,
    n: int,
    power: float = 0.80,
    alpha: float = 0.05,
    n_sim: int = 400,
    seed: int = 42,
) -> dict:
    """Scenario-specific detection point for one family of alternatives.

    A non-significant result alone does not establish equivalence. This
    diagnostic explores what size of effect would be detected under the particular
    rotating-level tilt below. Power is not determined by V and sample size
    alone, so the result must not be presented as a universal MDE or bound.

    How it works: build a table with the observed margins, add an association
    of increasing strength by tilting the level distribution inside each topic,
    draw n documents from it, and test. Returns the smallest injected V that
    gets rejected at least `power` of the time.
    """
    rng = np.random.default_rng(seed)
    p_level = np.asarray(level_counts, dtype=float)
    p_level = p_level / p_level.sum()
    p_topic = np.asarray(topic_counts, dtype=float)
    p_topic = p_topic / p_topic.sum()
    n_lv, n_tp = len(p_level), len(p_topic)

    # test statistic is the uncorrected v, and the critical value comes from
    # simulation rather than a chi-square table. the corrected v piles up on 0
    # for sparse tables so a percentile cutoff stops working, and the
    # chi-square approximation is too generous at ~6 expected per cell, which
    # would overstate power. type_i_error_check below confirms this holds .05.
    def stat(counts: np.ndarray) -> float:
        ct = pd.DataFrame(counts)
        ct = ct.loc[ct.sum(1) > 0, ct.sum(0) > 0]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return 0.0
        return cramers_v_uncorrected(ct)

    null_joint = np.outer(p_level, p_topic)
    null_stats = np.array([
        stat(rng.multinomial(n, null_joint.ravel()).reshape(n_lv, n_tp))
        for _ in range(n_sim)
    ])
    crit = float(np.percentile(null_stats, 100 * (1 - alpha)))

    def simulate(joint: np.ndarray) -> tuple[float, float]:
        """Returns (empirical power, true V of the table I generated from).

        Uncorrected V here, because the generating table is a population and
        not a sample, so the finite-sample correction doesn't apply.
        """
        true_v = cramers_v_uncorrected(pd.DataFrame(joint * n))
        hits = sum(
            stat(rng.multinomial(n, joint.ravel()).reshape(n_lv, n_tp)) > crit
            for _ in range(n_sim)
        )
        return hits / n_sim, true_v

    # size check: under independence the rejection rate must sit at alpha.
    null_power = float(np.mean(null_stats > crit))

    for strength in np.arange(0.02, 0.90, 0.02):
        # tilt: within each topic, shift mass towards one level, cycling which,
        # which produces a monotone family of alternatives with increasing v.
        joint = np.outer(p_level, p_topic).copy()
        for j in range(n_tp):
            target = j % n_lv
            col = joint[:, j].copy()
            tilted = (1 - strength) * col
            tilted[target] += strength * col.sum()
            joint[:, j] = tilted
        joint = joint / joint.sum()
        emp_power, true_v = simulate(joint)
        if emp_power >= power:
            return {
                "min_detectable_v": float(true_v),
                "interpretation": "scenario-specific detection point under rotating-level tilt",
                "empirical_power_at_min": float(emp_power),
                "critical_v": crit,
                "type_i_error_check": float(null_power),
                "power": power,
                "alpha": alpha,
                "n": int(n),
            }
    return {"min_detectable_v": np.nan,
            "interpretation": "scenario-specific detection point under rotating-level tilt",
            "empirical_power_at_min": np.nan,
            "critical_v": crit, "type_i_error_check": float(null_power),
            "power": power, "alpha": alpha, "n": int(n)}


# ---------------------------------------------------------------------------
# contingency table diagnostics
# ---------------------------------------------------------------------------


def standardised_residuals(ct: pd.DataFrame) -> pd.DataFrame:
    """Adjusted Pearson residuals, i.e. which topics go with which levels.

    Under independence these are roughly standard normal, so |r| > 2 is worth
    mentioning and |r| > 3 is a strong cell.
    """
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    row_p = obs.sum(axis=1) / n
    col_p = obs.sum(axis=0) / n
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    denom = np.sqrt(expected * np.outer(1 - row_p, 1 - col_p))
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(denom > 0, (obs - expected) / denom, 0.0)
    return pd.DataFrame(resid, index=ct.index, columns=ct.columns)


def expected_counts(ct: pd.DataFrame) -> pd.DataFrame:
    obs = ct.to_numpy(dtype=float)
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / obs.sum()
    return pd.DataFrame(exp, index=ct.index, columns=ct.columns)


def cmh_by_level(df: pd.DataFrame, level_col="cefr_level", topic_col="topic",
                 stratum_col="dataset") -> pd.DataFrame:
    """Cochran-Mantel-Haenszel: is topic still tied to level within corpora?

    The overall V mixes two things together, since topics differ between
    corpora and corpora also differ in which levels they contain. Stratifying
    by corpus and pooling separates them.

    I run it once per CEFR level (that level against all the others, by topic,
    stratified by corpus), which makes it easy to report which levels survive
    the conditioning. The stratum_col argument also lets me stratify by length
    bin instead of corpus, which is what 03c does.
    """
    from scipy.stats import chi2 as chi2_dist

    # every stratum has to use the same topic list, otherwise the per-stratum
    # vectors come out different lengths and can't be added together.
    all_topics = sorted(df[topic_col].unique())
    rows = []
    for level in sorted(df[level_col].unique()):
        strata_terms = []
        for _, sub in df.groupby(stratum_col):
            if sub[level_col].nunique() < 2:
                continue  # stratum carries no contrast for this level
            ct = (
                pd.crosstab(sub[level_col] == level, sub[topic_col])
                .reindex(index=[False, True], columns=all_topics)
                .fillna(0.0)
            )
            obs = ct.to_numpy(dtype=float)
            n_k = obs.sum()
            n1 = obs[1, :].sum()
            if n_k < 2 or n1 == 0 or n1 == n_k:
                continue
            colsum = obs.sum(axis=0)
            exp = n1 * colsum / n_k
            # hypergeometric covariance of the top-row counts, dropping the
            # last topic so the matrix isn't singular by construction:
            #   var(n_1j)      =  n1*n2*c_j*(n - c_j) / (n^2 (n-1))
            #   cov(n_1j,n_1l) = -n1*n2*c_j*c_l       / (n^2 (n-1))
            # i.e. n1*n2/(n^2(n-1)) * (n*diag(c) - outer(c,c)).
            # the n inside the bracket is required
            # and the statistic came out n times too big (chi2 of ~200,000).
            d = (obs[1, :] - exp)[:-1]
            c = colsum[:-1]
            n2 = n_k - n1
            cov = (n1 * n2 / (n_k ** 2 * (n_k - 1))) * (
                n_k * np.diag(c) - np.outer(c, c)
            )
            strata_terms.append((d, cov))
        if len(strata_terms) < 2:
            continue
        d_sum = np.sum(np.stack([t[0] for t in strata_terms]), axis=0)
        cov_sum = np.sum(np.stack([t[1] for t in strata_terms]), axis=0)

        # the pooled covariance can be rank deficient if a topic is missing
        # from some strata, so i use a pseudo-inverse with an explicit cutoff.
        # numpy's default (1e-15) turns a near-zero singular value into a huge
        # number and the result starts depending on floating point noise.
        # i return the condition number too, so it's checkable: on my data it
        # sits around 10-300 and the statistic is the same to six decimals for
        # any cutoff between 1e-15 and 1e-6.
        sv = np.linalg.svd(cov_sum, compute_uv=False)
        if sv.size == 0 or sv[0] <= 0:
            continue
        rcond = 1e-10
        dfree = int(np.sum(sv > rcond * sv[0]))
        if dfree < 1:
            continue
        # the matmul can throw overflow warnings even when the answer is fine
        # (this happened on the english data, and the value was stable). so i
        # silence the warning but still fail loudly if the result isn't finite.
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                stat = float(d_sum @ np.linalg.pinv(cov_sum, rcond=rcond) @ d_sum)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(stat) or stat < 0:
            raise ValueError(
                f"CMH statistic for level {level!r} is not finite (stat={stat!r}). "
                "The pooled covariance is too ill-conditioned to invert."
            )
        p = float(chi2_dist.sf(stat, dfree))
        rows.append({"level": level, "cmh_chi2": stat, "df": dfree, "p": p,
                     "n_strata": len(strata_terms),
                     "cond_number": float(sv[0] / sv[sv > rcond * sv[0]][-1])})
    return pd.DataFrame(rows)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    from statsmodels.stats.multitest import multipletests

    _, adj, _, _ = multipletests(pvals, method="holm")
    return [float(x) for x in adj]


def length_quartile_bins(word_count, n_bins: int = 4) -> pd.Series:
    """Within-sample quartile labels (Q1..Qk) for a length column.

    Ties can collapse the number of bins (`duplicates='drop'`). That is
    reported rather than forced: inventing extra boundaries on a discrete
    length distribution would make the strata look more even than they are.
    Documents with identical length therefore share a bin, which is the
    honest assignment.
    """
    s = pd.Series(word_count)
    if len(s) == 0:
        return s.astype(str)
    if s.nunique(dropna=True) < 2:
        return pd.Series(["Q1"] * len(s), index=s.index)
    codes = pd.qcut(s, q=n_bins, duplicates="drop", labels=False)
    return pd.Series([f"Q{int(c) + 1}" for c in codes], index=s.index)


# ---------------------------------------------------------------------------
# classification metrics (shared by 04, 06, 07)
# ---------------------------------------------------------------------------

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from config import LEVEL_ORDER


def adjacent_accuracy(y_true, y_pred) -> float:
    """Share of predictions within one CEFR level of the truth."""
    idx = {level: i for i, level in enumerate(LEVEL_ORDER)}
    t = np.array([idx[l] for l in y_true])
    p = np.array([idx[l] for l in y_pred])
    return float(np.mean(np.abs(t - p) <= 1))


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    labels = [l for l in LEVEL_ORDER if l in set(y_true) or l in set(y_pred)]
    if len(labels) < 2:
        return float("nan")
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels))


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
    }


def cramers_v_ci(levels, topics, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Compatibility wrapper: returns (v, lo, hi, boots_placeholder)."""
    res = bootstrap_v(levels, topics, n_boot=n_boot, seed=seed, alpha=alpha)
    return res["v"], res["ci_lo"], res["ci_hi"], np.array([res["v"]])
