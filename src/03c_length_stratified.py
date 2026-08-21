"""Step 3c: does topic still carry level information within length strata?

The thesis asks how far the topic–level association in learner data overlaps
with text length. Script 06 provides a complementary prediction-side analysis.
Here, documents are binned by word-count
quartile *within each corpus*, recompute Cramér's V inside each bin, and run
the CMH test of topic against level stratified by those
bins.

A null here is not automatically support for the claim. Conditioning on length
shrinks the variance of level inside each stratum (length predicts level), so
power drops. Every bin therefore reports a scenario-specific 80% detection
point for the simulated alternative used here. It is not a universal MDE.

Quartiles are coarse and residual confounding within a bin is possible. This
is evidence, not proof.

Run:
    python src/03c_length_stratified.py
    python src/03c_length_stratified.py --lang en
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr

from config import (
    LEVEL_ORDER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths,
    topic_outputs_problem,
)
from stats_utils import (
    cramers_v_from_arrays,
    bootstrap_v,
    cmh_by_level,
    cramers_v,
    holm,
    length_quartile_bins,
    minimum_detectable_v,
    permutation_p,
)

N_BOOT = 2000
N_PERM = 5000
N_POWER_SIM = 400
LEARNER_LIKE = {"merlin_de", "icle500_en", "cefr_asag_en"}
CONTROL_LIKE = {"deplain_apa_doc", "apa_lha"}


def fmt_v(d: dict) -> str:
    return f"{d['v']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"



def placebo_attenuation(sub: pd.DataFrame, n_draws: int = 200,
                        seed: int = RANDOM_SEED) -> dict:
    """compare length bins with random bins of the same sizes.

    smaller bins reduce power and make the tables sparser. random reassignment
    shows how much the statistic changes from binning alone. this is a
    descriptive comparison, not a causal length effect.
    """
    rng = np.random.default_rng(seed)
    bins = sub["length_bin"].to_numpy()
    levels = sub["cefr_level"].to_numpy()
    topics = sub["topic"].to_numpy()
    draws = []
    for _ in range(n_draws):
        shuffled = rng.permutation(bins)
        vs = []
        for b in pd.unique(shuffled):
            m = shuffled == b
            if len(np.unique(levels[m])) < 2 or len(np.unique(topics[m])) < 2:
                continue
            vs.append(cramers_v_from_arrays(levels[m], topics[m]))
        if vs:
            draws.append(float(np.mean(vs)))
    if not draws:
        return {"status": "skipped: no usable placebo draws"}
    draws = np.asarray(draws)
    return {
        "status": "ok",
        "n_draws": int(len(draws)),
        "mean_within_bin_v": float(draws.mean()),
        "sd": float(draws.std(ddof=1)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
    }


def try_mnlogit_lrt(sub: pd.DataFrame) -> dict:
    """Unregularised LRT: level ~ length  vs  level ~ length + topic.

    Dropped (not reported) on non-convergence or separation. A regularised
    model would make the likelihood-ratio test invalid, so that path is not
    used as a fallback.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import MNLogit
    except ImportError as exc:
        return {"status": f"skipped: statsmodels unavailable ({exc})"}

    present = [l for l in LEVEL_ORDER if l in set(sub["cefr_level"])]
    if len(present) < 2:
        return {"status": "skipped: fewer than two levels"}
    y = sub["cefr_level"].map({l: i for i, l in enumerate(present)}).to_numpy(dtype=int)
    log_len = np.log1p(sub["word_count"].to_numpy(dtype=float))
    topic_dummies = pd.get_dummies(sub["topic"], prefix="t", drop_first=True, dtype=float)
    if topic_dummies.shape[1] < 1:
        return {"status": "skipped: fewer than two topics"}
    X0 = sm.add_constant(pd.DataFrame({"log_len": log_len}), has_constant="add")
    X1 = sm.add_constant(
        pd.concat([pd.DataFrame({"log_len": log_len}), topic_dummies.reset_index(drop=True)], axis=1),
        has_constant="add",
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0 = MNLogit(y, X0.to_numpy(dtype=float)).fit(disp=False, maxiter=300)
            m1 = MNLogit(y, X1.to_numpy(dtype=float)).fit(disp=False, maxiter=300)
    except Exception as exc:
        return {"status": f"dropped: {type(exc).__name__}: {exc}"}

    def _converged(fit) -> bool:
        ret = getattr(fit, "mle_retvals", None) or {}
        return bool(ret.get("converged", False)) and np.isfinite(getattr(fit, "llf", np.nan))

    if not _converged(m0) or not _converged(m1):
        return {"status": "dropped: did not converge (or hit separation)"}
    lr = float(2 * (m1.llf - m0.llf))
    dfree = int(round(m1.df_model - m0.df_model))
    if dfree < 1 or not np.isfinite(lr) or lr < 0:
        return {"status": f"dropped: invalid LRT (lr={lr!r}, df={dfree})"}
    return {
        "status": "ok",
        "lr_stat": lr,
        "df": dfree,
        "p": float(chi2.sf(lr, dfree)),
        "llf_length": float(m0.llf),
        "llf_length_topic": float(m1.llf),
    }


def interpret_corpus(name: str, overall_v: float, bin_rows: list[dict],
                     cmh: pd.DataFrame, underpowered: bool,
                     mean_decile_v: float = float("nan")) -> list[str]:
    """Print how much of the association is left after controlling for length.

    No yes/no verdict on purpose. My first version used the rule
    mean_within_bin_V < 0.5 * unstratified_V, and on MERLIN that came out at
    .192 against a cutoff of .178 and said the association survives. Deciding
    something categorical on three hundredths isn't a result.

    So it prints three things instead:

    - how far V drops as the bins get tighter (none -> quartile -> decile)
    - whether the bins actually control for length. If level still correlates
      with word count inside a bin, the within-bin V is not length-adjusted.
    - whether sparse, smaller within-bin tables make the bias correction more
      influential, especially for deciles.

    Those mechanisms operate in different directions but are not quantified
    well enough to form mathematical bounds. Script 06 is the complementary
    prediction-side analysis because it uses length continuously.
    """
    lines = []
    vs = [r["v"] for r in bin_rows if np.isfinite(r["v"])]
    mean_v = float(np.mean(vs)) if vs else float("nan")
    any_reject = bool(len(cmh) and (cmh["p_holm"] < 0.05).any()) if len(cmh) else False
    resid = [abs(r.get("residual_rho_level_length", np.nan)) for r in bin_rows]
    resid = [r for r in resid if np.isfinite(r)]
    max_resid = float(np.max(resid)) if resid else float("nan")

    lines.append(f"  Unstratified V = {overall_v:.3f}; mean within-QUARTILE V = "
                 f"{mean_v:.3f} ({len(vs)} bins)"
                 + (f"; mean within-DECILE V = {mean_decile_v:.3f}."
                    if np.isfinite(mean_decile_v) else "."))
    if np.isfinite(mean_v) and overall_v > 0:
        lines.append(f"  Change after length stratification: "
                     f"{1 - mean_v / overall_v:.0%} at quartile level"
                     + (f", {1 - mean_decile_v / overall_v:.0%} at decile level."
                        if np.isfinite(mean_decile_v) else "."))
    if np.isfinite(max_resid):
        lines.append(f"  Largest residual rho(level, word count) inside a bin: "
                     f"{max_resid:.3f}.")
        if max_resid > 0.30:
            lines.append("  This is LARGE: the bins do not fully control for length,")
            lines.append("  so the within-bin V overstates any residual topic signal.")
            lines.append("  Treat both summaries as diagnostics rather than adjusted effects;")
            lines.append("  finer bins reduce residual variation but increase sparse-table bias.")
    if underpowered:
        lines.append("  POWER CAUTION: in at least one bin the observed V sits")
        lines.append("  below the scenario-specific 80% detection point. A non-rejection")
        lines.append("  in that bin is uninterpretable on its own.")
    lines.append(f"  Length-stratified CMH rejects for at least one level: {any_reject}.")

    if overall_v < 0.10:
        lines.append("  Reading: the unstratified association is already near")
        lines.append("  zero, so this corpus cannot support (or refute) the")
        lines.append("  learner-data collinearity claim.")
    else:
        lines.append("  Reading: report the quartile and decile summaries and the CMH")
        lines.append("  result together. Do not convert this into a yes/no claim")
        lines.append("  about whether topic 'survives' length control, and do not")
        lines.append("  read a surviving within-bin V as contradicting the")
        lines.append("  classifier decomposition: coarse bins under-control for")
        lines.append("  length, whereas the classifier conditions on it continuously.")
        lines.append("  Cross-check against the per-metric deltas in script 06,")
        lines.append("  which separate the bulk classes (weighted F1, QWK) from the")
        lines.append("  rare ones (macro F1).")
    if name in CONTROL_LIKE:
        lines.append("  (This is a parallel news control: A2/B1 is a professional")
        lines.append("  simplification target. Treat its result as a comparison of label")
        lines.append("  and production structure, not as confirmation of a predetermined null.)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-placebo", type=int, default=200,
                    help="random-bin draws for the placebo attenuation control")
    args = ap.parse_args()
    lang = args.lang
    p = paths(lang)
    if problem := topic_outputs_problem(lang):
        sys.exit(problem)
    df = pd.read_csv(p["with_topics_csv"])
    if "word_count" not in df.columns:
        sys.exit("texts_*_with_topics.csv has no word_count column.")
    rank = {l: i for i, l in enumerate(LEVEL_ORDER)}
    df["level_rank"] = df["cefr_level"].map(rank)

    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    say("=" * 78)
    say(f"LENGTH-STRATIFIED ASSOCIATION ({lang}), n = {len(df):,}")
    say(f"bootstrap draws = {args.n_boot:,}, permutations = {args.n_perm:,}")
    say("=" * 78)
    say("")
    say("CAUTION 1. Conditioning on length reduces the variance in level within")
    say("each stratum, since length predicts level so strongly. Power drops.")
    say("A scenario-specific detection point is reported per bin so that a")
    say("non-rejection is not mistaken for evidence of no association.")
    say("")
    say("CAUTION 2. Quartiles are coarse and residual confounding within a bin")
    say("is possible. This is evidence, not proof.")
    say("")

    bin_rows: list[dict] = []
    cmh_rows: list[dict] = []
    lrt_rows: list[dict] = []
    corpus_summaries: list[dict] = []

    for name, sub in df.groupby("dataset"):
        sub = sub.copy().reset_index(drop=True)
        say("-" * 78)
        say(f"{name}  n = {len(sub):,}")
        rho, rho_p = spearmanr(sub["level_rank"], sub["word_count"])
        say(f"  Spearman(level rank, word count) = {rho:.3f}, p = {rho_p:.2e}")

        overall = bootstrap_v(sub["cefr_level"], sub["topic"], n_boot=args.n_boot,
                              seed=RANDOM_SEED)
        overall_perm = permutation_p(sub["cefr_level"], sub["topic"],
                                     n_perm=args.n_perm, seed=RANDOM_SEED)
        say(f"  Unstratified topic vs level: V = {fmt_v(overall)}  "
            f"perm p {overall_perm['perm_p_label']}")

        sub["length_bin"] = length_quartile_bins(sub["word_count"])
        bin_counts = sub["length_bin"].value_counts().sort_index()
        say("  Length-bin sizes: " + ", ".join(f"{k}={v}" for k, v in bin_counts.items()))
        if bin_counts.shape[0] < 4:
            say("  NOTE: fewer than four bins; word-count ties collapsed the quartiles.")

        this_bins: list[dict] = []
        underpowered = False
        for bname, bsub in sub.groupby("length_bin"):
            n_levels = int(bsub["cefr_level"].nunique())
            n_topics = int(bsub["topic"].nunique())
            level_mix = bsub["cefr_level"].value_counts().to_dict()
            wc = bsub["word_count"]
            row = {
                "corpus": name, "length_bin": bname, "n": int(len(bsub)),
                "n_levels": n_levels, "n_topics": n_topics,
                "word_count_min": int(wc.min()), "word_count_max": int(wc.max()),
                "word_count_median": float(wc.median()),
                "level_counts": json.dumps(level_mix, sort_keys=True),
            }
            # validity check on the stratification itself. a length bin only
            # controls for length to the extent that length no longer varies
            # with level inside it. if this residual correlation is still large,
            # the within-bin v is not a length-free estimate and must not be
            # read as one. on merlin the quartiles leave residual correlations
            # as high as 0.53, which is why the graded reading below refuses to
            # treat a surviving within-bin v as clean evidence of residual
            # topic signal.
            if bsub["cefr_level"].nunique() > 1:
                r_res, p_res = spearmanr(bsub["level_rank"], bsub["word_count"])
            else:
                r_res, p_res = np.nan, np.nan
            row["residual_rho_level_length"] = float(r_res) if np.isfinite(r_res) else np.nan
            row["residual_rho_p"] = float(p_res) if np.isfinite(p_res) else np.nan

            if n_levels < 2 or n_topics < 2:
                row.update({
                    "v": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                    "perm_p": np.nan, "perm_p_label": "skipped (<2 levels or topics)",
                    "min_detectable_v_80pct": np.nan, "power_sim_type_i_error": np.nan,
                    "note": "no level (or topic) contrast inside this bin",
                })
                say(f"  {bname}: n={len(bsub):>4}  skipped "
                    f"({n_levels} levels, {n_topics} topics); mix={level_mix}")
                this_bins.append(row)
                bin_rows.append(row)
                continue
            b = bootstrap_v(bsub["cefr_level"], bsub["topic"], n_boot=args.n_boot,
                            seed=RANDOM_SEED)
            perm = permutation_p(bsub["cefr_level"], bsub["topic"],
                                 n_perm=args.n_perm, seed=RANDOM_SEED)
            ct = pd.crosstab(bsub["cefr_level"], bsub["topic"])
            power = minimum_detectable_v(
                ct.sum(axis=1).to_numpy(), ct.sum(axis=0).to_numpy(),
                n=len(bsub), n_sim=N_POWER_SIM, seed=RANDOM_SEED,
            )
            mde = power["min_detectable_v"]
            if np.isfinite(mde) and b["v"] < mde:
                underpowered = True
            row.update({
                "v": b["v"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "perm_p": perm["perm_p"], "perm_p_label": perm["perm_p_label"],
                "min_detectable_v_80pct": mde,
                "power_sim_type_i_error": power["type_i_error_check"],
                "note": "",
            })
            flag = ""
            if np.isfinite(mde) and b["v"] < mde:
                flag = "  [below scenario-specific 80% point]"
            resid_txt = (f"  residual rho={r_res:+.3f}" if np.isfinite(r_res) else "")
            say(f"  {bname}: n={len(bsub):>4}  V = {fmt_v(b)}  "
                f"perm p {perm['perm_p_label']}  "
                f"scenario80={mde:.3f}  levels={n_levels}{resid_txt}{flag}")
            this_bins.append(row)
            bin_rows.append(row)

        # finer stratification. quartiles are coarse; if the attenuation of v
        # continues as the bins tighten, that is consistent with overlap between
        # length and the topic association. this is reported as a
        # trend rather than a test, because within-decile n is around 100 and the
        # bergsma correction can be influential at that size, so the decile
        # figure is a sensitivity trend rather than an adjusted effect.
        decile_v = []
        try:
            dec = pd.qcut(sub["word_count"], 10, labels=False, duplicates="drop")
            for _, dsub in sub.groupby(dec):
                if dsub["cefr_level"].nunique() < 2 or dsub["topic"].nunique() < 2:
                    continue
                decile_v.append(cramers_v(pd.crosstab(dsub["cefr_level"],
                                                      dsub["topic"]))[0])
        except (ValueError, IndexError):
            pass
        mean_decile_v = float(np.mean(decile_v)) if decile_v else float("nan")
        if decile_v:
            say(f"  Finer control: mean within-DECILE V = {mean_decile_v:.3f} "
                f"({len(decile_v)} deciles, n~{len(sub)//10} each). "
                "Read as a sparse-table sensitivity trend, not a bound.")

        say("  CMH: topic vs level, stratified by length bin")
        cmh = cmh_by_level(sub, stratum_col="length_bin")
        if cmh.empty:
            say("    not computable (fewer than two length bins with a level contrast).")
            cmh = cmh.assign(p_holm=pd.Series(dtype=float), corpus=name)
        else:
            cmh = cmh.copy()
            cmh["p_holm"] = holm(list(cmh["p"]))
            cmh["corpus"] = name
            for r in cmh.itertuples():
                flag = "REJECTED" if r.p_holm < 0.05 else "not rejected"
                say(f"    level {r.level}: chi2 = {r.cmh_chi2:8.1f}  df = {r.df:>2}  "
                    f"p = {r.p:.2e}  (Holm {r.p_holm:.2e})  -> {flag}")
            cmh_rows.extend(cmh.to_dict(orient="records"))

        lrt = try_mnlogit_lrt(sub)
        lrt["corpus"] = name
        lrt_rows.append(lrt)
        if lrt["status"] == "ok":
            say(f"  MNLogit LRT (level ~ length  vs  length + topic): "
                f"LR = {lrt['lr_stat']:.2f}  df = {lrt['df']}  p = {lrt['p']:.2e}")
        else:
            say(f"  MNLogit LRT: {lrt['status']}")

        for msg in interpret_corpus(name, overall["v"], this_bins, cmh, underpowered,
                                    mean_decile_v=mean_decile_v):
            say(msg)

        vs = [r["v"] for r in this_bins if np.isfinite(r.get("v", np.nan))]

        # placebo control: same bin sizes, random membership. anything the real
        # length bins attenuate beyond this is the contrast with random binning;
        # it is not a causal amount attributable to length.
        placebo = placebo_attenuation(sub, n_draws=args.n_placebo)
        obs_mean_v = float(np.mean(vs)) if vs else float("nan")
        if placebo["status"] == "ok" and np.isfinite(obs_mean_v) and overall["v"] > 0:
            att_real = 100 * (1 - obs_mean_v / overall["v"])
            att_null = 100 * (1 - placebo["mean_within_bin_v"] / overall["v"])
            say(f"  Change in V under length quartiles: {att_real:.0f}%")
            say(f"  Change under RANDOM bins of the same sizes: {att_null:+.0f}% "
                f"(sd {100 * placebo['sd'] / overall['v']:.1f}pp, "
                f"{placebo['n_draws']} draws)")
            say(f"  Difference beyond the random-binning placebo: "
                f"{att_real - att_null:.0f} percentage points.")
            if abs(att_null) > 5:
                say("  NOTE: random binning itself changes V materially. Interpret the")
                say("  observed-minus-placebo contrast, not the raw change alone.")
            placebo["attenuation_observed_pct"] = att_real
            placebo["attenuation_placebo_pct"] = att_null
            placebo["attenuation_beyond_placebo_pct"] = att_real - att_null
        else:
            say(f"  Placebo control: {placebo['status']}")

        corpus_summaries.append({
            "placebo_length_bins": placebo,
            "corpus": name, "n": int(len(sub)),
            "spearman_length_level": float(rho),
            "unstratified_v": overall["v"],
            "mean_within_bin_v": float(np.mean(vs)) if vs else None,
            "cmh_any_holm_reject": bool(len(cmh) and (cmh["p_holm"] < 0.05).any())
            if len(cmh) and "p_holm" in cmh.columns else False,
            "underpowered_bin": underpowered,
            "lrt": lrt,
        })
        say("")

    out_csv = RESULTS_DIR / f"length_stratified_{lang}.csv"
    pd.DataFrame(bin_rows).to_csv(out_csv, index=False)
    cmh_df = pd.DataFrame(cmh_rows)
    cmh_path = RESULTS_DIR / f"length_stratified_cmh_{lang}.csv"
    if len(cmh_df):
        cmh_df.to_csv(cmh_path, index=False)

    txt = RESULTS_DIR / f"length_stratified_{lang}.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _json_safe(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))

    payload = {
        "lang": lang, "n_documents": int(len(df)),
        "by_corpus": corpus_summaries,
        "bins": bin_rows, "cmh": cmh_rows, "lrt": lrt_rows,
        "cautions": [
            "Conditioning on length reduces level variance within strata; report MDE.",
            "Quartiles are coarse; residual confounding within a bin is possible.",
        ],
    }
    (RESULTS_DIR / f"length_stratified_{lang}.json").write_text(
        json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8"
    )
    print(f"Saved {out_csv}")
    print(f"Saved {txt}")
    if len(cmh_df):
        print(f"Saved {cmh_path}")


if __name__ == "__main__":
    main()
