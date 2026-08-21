"""step 3b: topic-level association"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from config import LEVEL_ORDER, OPEN_HF_DATASETS, RESULTS_DIR, paths, topic_outputs_problem
from stats_utils import (
    bootstrap_v,
    cmh_by_level,
    cramers_v,
    epsilon_squared_kruskal,
    expected_counts,
    holm,
    information_measures,
    minimum_detectable_v,
    permutation_p,
    standardised_residuals,
    subsample_v,
)

N_BOOT = 2000
N_PERM = 5000
N_SUBSAMPLE_REPS = 200


def topic_labels(lang: str) -> dict:
    """number -> 'n: kw/kw/kw', plus the hand-written label when available"""
    p = paths(lang)
    words = pd.read_csv(p["topic_words_csv"]).set_index("topic")["top_words"]
    labels = {t: "/".join(str(w).split(", ")[:3]) for t, w in words.items()}
    summary_path = RESULTS_DIR / f"topic_summary_{lang}.csv"
    interpreted = {}
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        if "interpreted_label" in s.columns:
            interpreted = {
                int(r.topic): str(r.interpreted_label).strip()
                for r in s.itertuples()
                if isinstance(r.interpreted_label, str) and r.interpreted_label.strip()
            }
    return labels, interpreted, words


def fmt_v(d: dict) -> str:
    return f"{d['v']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    lang = args.lang
    p = paths(lang)

    if problem := topic_outputs_problem(lang):
        sys.exit(problem)
    df = pd.read_csv(p["with_topics_csv"])
    levels = [l for l in LEVEL_ORDER if l in set(df["cefr_level"])]
    rank = {l: i for i, l in enumerate(LEVEL_ORDER)}
    df["level_rank"] = df["cefr_level"].map(rank)

    kw_label, interpreted, top_words = topic_labels(lang)

    def name_topic(t: int) -> str:
        t = int(t)
        base = f"{t}: {kw_label.get(t, '')}"
        return f"{base} ({interpreted[t]})" if t in interpreted else base

    out: dict = {"lang": lang, "n_documents": int(len(df)),
                 "n_topics": int(df["topic"].nunique()), "levels": levels}
    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    say("=" * 78)
    say(f"EXTENDED ASSOCIATION ANALYSIS ({lang}), n = {len(df):,} documents")
    say(f"bootstrap diagnostic draws = {args.n_boot:,}, permutations = {args.n_perm:,}")
    say("=" * 78)

    # 1. overall association
    say("\n--- 1. Topic vs CEFR level, all data ---")
    b_all = bootstrap_v(df["cefr_level"], df["topic"], n_boot=args.n_boot)
    perm_all = permutation_p(df["cefr_level"], df["topic"], n_perm=args.n_perm)
    info_all = information_measures(df["cefr_level"], df["topic"])
    eps2, h_stat, kw_p = epsilon_squared_kruskal(df["level_rank"], df["topic"])

    say(f"Cramer's V (bias-corrected): {fmt_v(b_all)}   bootstrap SD {b_all['boot_sd']:.3f}")
    say(f"Permutation p: {perm_all['perm_p_label']}  ({args.n_perm:,} permutations)")
    say(f"Adjusted mutual information: {info_all['ami']:.3f}   (raw NMI {info_all['nmi']:.3f})")
    say(f"Theil's U (level | topic):   {info_all['theils_u_level_given_topic']:.3f}"
        "   <- share of level entropy resolved by knowing the topic")
    say(f"Theil's U (topic | level):   {info_all['theils_u_topic_given_level']:.3f}")
    say(f"Kruskal-Wallis on level rank by topic: H = {h_stat:.1f}, "
        f"epsilon^2 = {eps2:.3f}, p = {kw_p:.2e}")
    say("  epsilon^2 is the ordinal-aware complement to V: the share of variance in")
    say("  CEFR rank explained by topic membership, without assuming topics are ordered.")

    out["overall"] = {**b_all, **perm_all, **info_all,
                      "epsilon_squared": eps2, "kruskal_h": h_stat, "kruskal_p": kw_p}

    # 2. topic and source corpus
    say("\n--- 2. Topic vs source corpus (the competing explanation) ---")
    b_src = bootstrap_v(df["dataset"], df["topic"], n_boot=args.n_boot)
    perm_src = permutation_p(df["dataset"], df["topic"], n_perm=args.n_perm)
    say(f"Cramer's V: {fmt_v(b_src)}   permutation p {perm_src['perm_p_label']}")
    say("Compare against the level association above. Non-overlap is a descriptive")
    say("contrast, not a formal test of the difference between the two V values.")
    overlap = not (b_src["ci_lo"] > b_all["ci_hi"] or b_all["ci_lo"] > b_src["ci_hi"])
    say(f"CIs overlap: {overlap}")
    out["source"] = {**b_src, **perm_src, "ci_overlaps_level": bool(overlap)}

    # 3. within-corpus estimates
    say("\n--- 3. Topic vs level WITHIN each corpus ---")
    sizes = df.groupby("dataset").size()
    n_match = int(sizes.min())
    say(f"Matched-n analyses subsample every corpus to n = {n_match} "
        f"(the smallest corpus), {N_SUBSAMPLE_REPS} repetitions.")

    per_corpus = []
    raw_ps = []
    for name, sub in df.groupby("dataset"):
        if sub["cefr_level"].nunique() < 2 or sub["topic"].nunique() < 2:
            say(f"{name}: skipped, not enough level or topic variety")
            continue
        b = bootstrap_v(sub["cefr_level"], sub["topic"], n_boot=args.n_boot)
        perm = permutation_p(sub["cefr_level"], sub["topic"], n_perm=args.n_perm)
        matched = subsample_v(sub["cefr_level"], sub["topic"], n_match,
                              n_rep=N_SUBSAMPLE_REPS)
        ct = pd.crosstab(sub["cefr_level"], sub["topic"])
        _, chi_p = cramers_v(ct)
        e2, _, _ = epsilon_squared_kruskal(sub["level_rank"], sub["topic"])
        power = minimum_detectable_v(
            ct.sum(axis=1).to_numpy(), ct.sum(axis=0).to_numpy(), n=len(sub)
        )
        per_corpus.append({
            "corpus": name, "n": len(sub),
            "n_levels": int(sub["cefr_level"].nunique()),
            "v": b["v"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
            "ci_method": b["ci_method"],
            "boot_sd": b["boot_sd"],
            "perm_p": perm["perm_p"], "perm_p_label": perm["perm_p_label"],
            "chi2_p": chi_p,
            "v_matched_n_mean": matched["mean"], "v_matched_n_sd": matched["sd"],
            "v_matched_n_lo": matched["lo"], "v_matched_n_hi": matched["hi"],
            "epsilon_squared": e2,
            "min_detectable_v_80pct": power["min_detectable_v"],
            "power_sim_type_i_error": power["type_i_error_check"],
            "boot_bias": b["boot_bias"],
            "ci_lo_bootstrap_normal": b["ci_lo_bootstrap_normal"],
            "ci_hi_bootstrap_normal": b["ci_hi_bootstrap_normal"],
            "ci_lo_percentile": b["ci_lo_percentile"],
            "ci_hi_percentile": b["ci_hi_percentile"],
        })
        raw_ps.append(perm["perm_p"])

    corpus_df = pd.DataFrame(per_corpus).sort_values("v", ascending=False)
    corpus_df["perm_p_holm"] = holm(list(corpus_df["perm_p"]))
    corpus_df.to_csv(RESULTS_DIR / f"cramers_v_by_corpus_{lang}.csv", index=False)

    say("")
    for r in corpus_df.itertuples():
        say(f"{r.corpus:<18} n={r.n:>5}  V = {r.v:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]  "
            f"perm p {r.perm_p_label} (Holm {r.perm_p_holm:.3f})")
        say(f"{'':<18}         at matched n={n_match}: V = {r.v_matched_n_mean:.3f} "
            f"+/- {r.v_matched_n_sd:.3f}   eps^2 = {r.epsilon_squared:.3f}")
        detected = "observed V is above this scenario point" if r.v > r.min_detectable_v_80pct \
            else "observed V is below this scenario point"
        say(f"{'':<18}         scenario-specific 80% detection point: "
            f"V = {r.min_detectable_v_80pct:.3f} "
            f"({detected})")
        say(f"{'':<18}         [power sim size check: type I error = "
            f"{r.power_sim_type_i_error:.3f}, nominal 0.05; "
            f"bootstrap bias = {r.boot_bias:+.3f}]")
    say("")
    say("Reading the power column: it applies only to the rotating-level tilt family")
    say("implemented in minimum_detectable_v; it is not a universal threshold for")
    say("all contingency tables having the same V.")
    say("CIs use approximate noncentral-chi-square inversion. Bootstrap-normal,")
    say("percentile and basic alternatives remain in the JSON/CSV as diagnostics.")
    out["by_corpus"] = corpus_df.to_dict(orient="records")
    out["matched_n"] = n_match

    # matched-n check for the learner > reference > news pattern
    from scipy.stats import spearmanr

    ranked_raw = list(corpus_df.sort_values("v", ascending=False)["corpus"])
    ranked_matched = list(corpus_df.sort_values("v_matched_n_mean", ascending=False)["corpus"])
    rho, _ = spearmanr(corpus_df["v"], corpus_df["v_matched_n_mean"])
    say(f"\nCorpus ranking, raw n:      {' > '.join(ranked_raw)}")
    say(f"Corpus ranking, matched n:  {' > '.join(ranked_matched)}")
    say(f"Spearman rho between raw and matched-n V: {rho:.3f}")

    parallel = {"deplain_apa_doc", "apa_lha"}
    m = corpus_df.set_index("corpus")["v_matched_n_mean"]
    grouping_holds = None
    if "merlin_de" in m.index and parallel & set(m.index):
        top = m.get("merlin_de", np.nan)
        news_max = m[[c for c in m.index if c in parallel]].max()
        elg = m.get("elg_cefr_de", np.nan)
        grouping_holds = bool(top > elg > news_max) if not np.isnan(elg) else bool(top > news_max)
        say(f"Grouping at matched n (learner > reference > parallel news): {grouping_holds}")
        say(f"  merlin {top:.3f} > elg {elg:.3f} > max(news) {news_max:.3f}")
    say("Note: the smallest corpus is subsampled to its own size, so its matched-n")
    say("SD is 0 by construction. That is expected, not an error.")

    out["ranking_raw"] = ranked_raw
    out["ranking_matched_n"] = ranked_matched
    out["ranking_spearman_rho"] = float(rho)
    out["grouping_holds_at_matched_n"] = grouping_holds

    # 4. corpus-stratified cmh
    say("\n--- 4. Cochran-Mantel-Haenszel: topic vs level, conditioning on corpus ---")
    say("Null: within corpora, topic carries no information about level.")
    cmh = cmh_by_level(df)
    if not cmh.empty:
        cmh["p_holm"] = holm(list(cmh["p"]))
        for r in cmh.itertuples():
            flag = "REJECTED" if r.p_holm < 0.05 else "not rejected"
            say(f"  level {r.level}: CMH chi2 = {r.cmh_chi2:8.1f}  df = {r.df:>2}  "
                f"p = {r.p:.2e}  (Holm {r.p_holm:.2e})  -> {flag}")
        cmh.to_csv(RESULTS_DIR / f"cmh_{lang}.csv", index=False)
        say("")
        say("For rejected levels, topic and level remain associated within the")
        say("corpus strata. This rules out corpus composition as the sole")
        say("explanation, but does not identify a causal topic effect.")
        out["cmh"] = cmh.to_dict(orient="records")
    else:
        say("  CMH not computable on this table shape.")

    # 5. topic-level residuals
    say("\n--- 5. Standardised residuals (which topics go with which levels) ---")

    def residual_report(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        ct = pd.crosstab(frame["cefr_level"], frame["topic"]).reindex(
            [l for l in LEVEL_ORDER if l in set(frame["cefr_level"])]
        )
        resid = standardised_residuals(ct)
        exp = expected_counts(ct)
        resid.to_csv(RESULTS_DIR / f"residuals_level_topic_{tag}.csv")
        recs = []
        for lv in resid.index:
            for tp in resid.columns:
                recs.append({
                    "level": lv,
                    "topic": int(tp),
                    "topic_label": name_topic(tp),
                    "observed": int(ct.loc[lv, tp]),
                    "expected": round(float(exp.loc[lv, tp]), 1),
                    "obs_over_exp": round(float(ct.loc[lv, tp] / exp.loc[lv, tp]), 2)
                    if exp.loc[lv, tp] > 0 else np.nan,
                    "std_residual": round(float(resid.loc[lv, tp]), 2),
                })
        return pd.DataFrame(recs)

    tbl = residual_report(df, lang)
    tbl_sorted = tbl.reindex(tbl["std_residual"].abs().sort_values(ascending=False).index)
    tbl_sorted.to_csv(RESULTS_DIR / f"top_associations_{lang}.csv", index=False)

    strong = tbl_sorted[tbl_sorted["std_residual"].abs() > 3]
    say(f"Cells with |standardised residual| > 3: {len(strong)} of {len(tbl)}")
    say(f"Cells with |standardised residual| > 2: {(tbl['std_residual'].abs() > 2).sum()}")
    say("\nStrongest positive associations (level over-represented in topic):")
    pos = tbl_sorted[tbl_sorted["std_residual"] > 0].head(12)
    for r in pos.itertuples():
        say(f"  {r.level}  topic {r.topic_label:<45.45}  "
            f"obs {r.observed:>4} vs exp {r.expected:>6.1f}  "
            f"({r.obs_over_exp:>4.1f}x)  resid {r.std_residual:+.1f}")
    say("\nStrongest negative associations (level under-represented in topic):")
    neg = tbl_sorted[tbl_sorted["std_residual"] < 0].head(6)
    for r in neg.itertuples():
        say(f"  {r.level}  topic {r.topic_label:<45.45}  "
            f"obs {r.observed:>4} vs exp {r.expected:>6.1f}  resid {r.std_residual:+.1f}")

    out["n_cells_resid_gt3"] = int(len(strong))
    out["n_cells_resid_gt2"] = int((tbl["std_residual"].abs() > 2).sum())
    out["top_associations"] = pos.head(10).to_dict(orient="records")

    # merlin alone
    merlin = df[df["dataset"] == "merlin_de"]
    if len(merlin) > 0 and merlin["cefr_level"].nunique() > 1:
        say("\n--- 5b. Same, MERLIN only (learner corpus) ---")
        mt = residual_report(merlin, f"merlin_{lang}")
        mt_sorted = mt.reindex(mt["std_residual"].abs().sort_values(ascending=False).index)
        mt_sorted.to_csv(RESULTS_DIR / f"top_associations_merlin_{lang}.csv", index=False)
        for r in mt_sorted[mt_sorted["std_residual"] > 0].head(10).itertuples():
            say(f"  {r.level}  topic {r.topic_label:<45.45}  "
                f"obs {r.observed:>4} vs exp {r.expected:>6.1f}  "
                f"({r.obs_over_exp:>4.1f}x)  resid {r.std_residual:+.1f}")
        out["top_associations_merlin"] = (
            mt_sorted[mt_sorted["std_residual"] > 0].head(10).to_dict(orient="records")
        )

    # 6. assignment-confidence sensitivity
    if "topic_strength" in df.columns:
        say("\n--- 6. Sensitivity to topic assignment confidence ---")
        med = df["topic_strength"].median()
        conf = df[df["topic_strength"] >= med]
        b_conf = bootstrap_v(conf["cefr_level"], conf["topic"], n_boot=args.n_boot)
        say(f"topic_strength: median {med:.3f}, "
            f"IQR [{df['topic_strength'].quantile(.25):.3f}, "
            f"{df['topic_strength'].quantile(.75):.3f}]")
        say(f"All documents          (n={len(df):>5}): V = {fmt_v(b_all)}")
        say(f"Top half by confidence (n={len(conf):>5}): V = {fmt_v(b_conf)}")
        say("This is a selection sensitivity check: the subset has a different")
        say("composition, so a higher V does not by itself show that low-strength")
        say("documents dilute the association.")
        out["confident_half"] = b_conf

    # outputs
    txt = RESULTS_DIR / f"association_extended_{lang}.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")

    def _json_safe(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))

    (RESULTS_DIR / f"association_extended_{lang}.json").write_text(
        json.dumps(out, indent=2, default=_json_safe), encoding="utf-8"
    )
    print(f"\nSaved {txt}")
    print(f"Saved {RESULTS_DIR / f'association_extended_{lang}.json'}")
    print(f"Saved {RESULTS_DIR / f'cramers_v_by_corpus_{lang}.csv'}")
    print(f"Saved {RESULTS_DIR / f'top_associations_{lang}.csv'}")


if __name__ == "__main__":
    main()
