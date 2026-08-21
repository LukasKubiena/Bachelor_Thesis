"""step 6: length, surface, and topic benchmarks"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    CheckedLogisticRegression,
    normalise_topic_weights,
    surface_features,
    validation_groups,
    word_count_feature,
)
from config import (
    LEVEL_ORDER,
    MOST_FREQUENT_WEIGHTED_F1,
    OPEN_HF_DATASETS,
    RANDOM_SEED,
    REFERENCE_LADDER,
    RESULTS_DIR,
    XLMR_WEIGHTED_F1,
    paths,
    topic_outputs_problem,
)

N_SPLITS = 5
N_REPEATS = 5
N_PERM_NULL = 40
N_BOOT = 600
# bootstrap cis for thesis rows only
CI_FEATURE_SETS = ("word count only", "surface features", "topic only", "surface + topic")

# imperial et al. values from config.py



# metrics


def adjacent_accuracy(y_true, y_pred) -> float:
    idx = {l: i for i, l in enumerate(LEVEL_ORDER)}
    t = np.array([idx[l] for l in y_true])
    p = np.array([idx[l] for l in y_pred])
    return float(np.mean(np.abs(t - p) <= 1))


def qwk(y_true, y_pred) -> float:
    """quadratic weighted kappa. cefr is ordinal, and this is also the metric"""
    idx = {l: i for i, l in enumerate(LEVEL_ORDER)}
    t = [idx[l] for l in y_true]
    p = [idx[l] for l in y_pred]
    return float(cohen_kappa_score(t, p, weights="quadratic"))


def all_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(np.mean(np.asarray(y_pred) == np.asarray(y_true))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
        "qwk": qwk(y_true, y_pred),
    }


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=N_BOOT, seed=RANDOM_SEED):
    """percentile interval over the pooled out-of-fold predictions"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y_true[i])) < 2:
            continue
        vals.append(metric_fn(y_true[i], y_pred[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# model runner


def make_lr():
    return make_pipeline(
        StandardScaler(),
        CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
    )


def run_cv(X, y, groups, clf, grouped: bool, seed: int = RANDOM_SEED):
    if grouped:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(clf, X, y, cv=cv)
    return y_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-perm", type=int, default=N_PERM_NULL)
    args = ap.parse_args()
    lang = args.lang
    p = paths(lang)
    tag = lang if args.dataset is None else f"{lang}_{args.dataset}"

    if problem := topic_outputs_problem(lang):
        sys.exit(problem)
    df = pd.read_csv(p["with_topics_csv"])
    doc_topic = np.load(p["doc_topic_matrix"])
    assert len(df) == doc_topic.shape[0], "table and topic matrix out of sync, rerun step 2"

    if args.dataset:
        keep = (df["dataset"] == args.dataset).to_numpy()
        if not keep.any():
            sys.exit(f"No rows for '{args.dataset}'. Have: {sorted(df['dataset'].unique())}")
        df, doc_topic = df[keep].reset_index(drop=True), doc_topic[keep]

    # minimum class size for stratified cv
    counts = df["cefr_level"].value_counts()
    rare = counts[counts < N_SPLITS].index.tolist()
    if rare:
        print(f"Dropping levels with fewer than {N_SPLITS} texts: {rare}")
        keep = (~df["cefr_level"].isin(rare)).to_numpy()
        df, doc_topic = df[keep].reset_index(drop=True), doc_topic[keep]

    y = df["cefr_level"].to_numpy()
    groups = validation_groups(df)
    levels = [l for l in LEVEL_ORDER if l in set(y)]

    print("=" * 78)
    print(f"LENGTH VS TOPIC BENCHMARK ({tag}), n = {len(df):,}")
    print("=" * 78)
    support = df["cefr_level"].value_counts().reindex(levels)
    print("\nClass support:")
    print(support.to_string())
    print(f"\nGrouping variable: {df['cv_group'].nunique():,} unique cv_group values "
          f"over {len(df):,} documents")
    n_paired = int((df.groupby("cv_group").size() > 1).sum())
    if n_paired:
        print(f"  {n_paired:,} groups contain more than one document, i.e. parallel "
              "or near-duplicate versions that must not be split across folds.")
    else:
        print("  No group contains more than one document in this analysis subset, "
              "so document-level leakage between folds is not "
              "possible here. The 'leakage' table below is therefore reporting "
              "fold-assignment noise between two different partitions, NOT a "
              "leak, and must be described that way in the text.")

    # length-level association
    from scipy.stats import spearmanr

    rank = {l: i for i, l in enumerate(LEVEL_ORDER)}
    lr_rho, lr_p = spearmanr(df["cefr_level"].map(rank), df["word_count"])
    print(f"\nSpearman(level rank, word count) = {lr_rho:.3f}, p = {lr_p:.2e}")
    print("This is the within-sample evidence for the length-level association.")
    desc = (df.groupby(["dataset", "cefr_level"])["word_count"]
              .agg(["count", "mean", "median",
                    lambda s: s.quantile(.25), lambda s: s.quantile(.75)]))
    desc.columns = ["n", "mean", "median", "q25", "q75"]
    desc.to_csv(RESULTS_DIR / f"length_vs_level_{tag}.csv")

    # feature sets
    Xsurface = surface_features(df["text"]).to_numpy()
    Xtop_raw = doc_topic
    Xtop = normalise_topic_weights(doc_topic)
    Xboth = np.hstack([Xsurface, Xtop])
    Xboth_raw = np.hstack([Xsurface, Xtop_raw])
    n_zero_topic_rows = int(np.sum(Xtop_raw.sum(axis=1) == 0))
    print(f"All-zero topic rows after filtering: {n_zero_topic_rows}")

    feature_sets = {
        "majority class (floor)": (np.zeros((len(y), 1)), DummyClassifier(strategy="most_frequent")),
        # word count separate from complexity features
        "word count only": (word_count_feature(df["word_count"]), make_lr()),
        "surface features": (Xsurface, make_lr()),
        "topic only": (Xtop, make_lr()),
        "surface + topic": (Xboth, make_lr()),
        "topic only (class-weighted)": (
            Xtop,
            make_pipeline(StandardScaler(),
                          CheckedLogisticRegression(max_iter=5000, class_weight="balanced",
                                             random_state=RANDOM_SEED)),
        ),
        "topic only (gradient boosting)": (Xtop, HistGradientBoostingClassifier(
            random_state=RANDOM_SEED)),
        # raw-loading sensitivity only
        "topic only (raw loadings)": (Xtop_raw, make_lr()),
        "surface + topic (raw loadings)": (Xboth_raw, make_lr()),
    }

    # grouped and ungrouped splits
    rows = []
    preds = {}
    for grouped in (False, True):
        label = "grouped (cv_group)" if grouped else "ungrouped"
        for name, (X, clf) in feature_sets.items():
            y_pred = run_cv(X, y, groups, clf, grouped=grouped)
            m = all_metrics(y, y_pred)
            m.update({"features": name, "cv": label})
            rows.append(m)
            preds[(label, name)] = y_pred

    res = pd.DataFrame(rows)[
        ["cv", "features", "accuracy", "macro_f1", "weighted_f1", "adjacent_accuracy", "qwk"]
    ]
    print("\n" + "=" * 78)
    print("RESULTS (5-fold cross-validation)")
    print("=" * 78)
    for label in ["ungrouped", "grouped (cv_group)"]:
        print(f"\n--- {label} ---")
        print(res[res["cv"] == label].drop(columns="cv")
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    leak = (res[res.cv == "ungrouped"].set_index("features")["weighted_f1"]
            - res[res.cv == "grouped (cv_group)"].set_index("features")["weighted_f1"])
    label = ("Leakage from ungrouped folds" if n_paired
             else "Fold-assignment noise (NOT leakage: no parallel documents exist)")
    print(f"\n{label} (ungrouped minus grouped, weighted F1):")
    print(leak.round(4).to_string())

    # raw vs row-normalised loadings
    g = res[res.cv == "grouped (cv_group)"].set_index("features")
    if {"topic only", "topic only (raw loadings)"} <= set(g.index):
        print("\n--- Raw-loading sensitivity (raw minus L1-normalised main model) ---")
        for raw, main in [("topic only (raw loadings)", "topic only"),
                          ("surface + topic (raw loadings)", "surface + topic")]:
            print(f"  {main:<22} wF1 {g.loc[raw,'weighted_f1'] - g.loc[main,'weighted_f1']:+.3f}   "
                  f"QWK {g.loc[raw,'qwk'] - g.loc[main,'qwk']:+.3f}")
        print("  Similar predictive scores do not prove that raw magnitude contains")
        print("  no length information; normalised weights remain the main construct.")

    res.to_csv(RESULTS_DIR / f"length_benchmark_{tag}.csv", index=False)

    # grouped-score uncertainty
    print("\n--- Bootstrap 95% CIs on the grouped results ---")
    ci_rows = []
    for name in CI_FEATURE_SETS:
        yp = preds[("grouped (cv_group)", name)]
        wlo, whi = bootstrap_metric_ci(y, yp, lambda a, b: f1_score(a, b, average="weighted"))
        mlo, mhi = bootstrap_metric_ci(y, yp, lambda a, b: f1_score(a, b, average="macro"))
        ci_rows.append({"features": name,
                        "weighted_f1": f1_score(y, yp, average="weighted"),
                        "weighted_f1_ci": f"[{wlo:.3f}, {whi:.3f}]",
                        "macro_f1": f1_score(y, yp, average="macro"),
                        "macro_f1_ci": f"[{mlo:.3f}, {mhi:.3f}]"})
    ci_df = pd.DataFrame(ci_rows)
    print(ci_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # topic gain over surface features
    print("\n--- Incremental contribution of topic over surface features ---")
    print(f"Repeated CV: {N_SPLITS} folds x {N_REPEATS} seeds, grouped on cv_group.")
    deltas = {"macro_f1": [], "weighted_f1": [], "qwk": []}
    per_seed = []
    for r in range(N_REPEATS):
        seed = RANDOM_SEED + r
        ys = run_cv(Xsurface, y, groups, make_lr(), grouped=True, seed=seed)
        yb = run_cv(Xboth, y, groups, make_lr(), grouped=True, seed=seed)
        ms, mb = all_metrics(y, ys), all_metrics(y, yb)
        for k in deltas:
            deltas[k].append(mb[k] - ms[k])
        per_seed.append({"seed": seed, **{f"surface_{k}": ms[k] for k in deltas},
                         **{f"surface_topic_{k}": mb[k] for k in deltas}})
    from scipy.stats import wilcoxon

    inc = {}
    for k, vals in deltas.items():
        vals = np.asarray(vals)
        try:
            _, wp = wilcoxon(vals)
        except ValueError:
            wp = np.nan
        inc[k] = {"mean_delta": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                  "min": float(vals.min()), "max": float(vals.max()),
                  "wilcoxon_p": float(wp)}
        print(f"  {k:<12} surface -> surface+topic: "
              f"{vals.mean():+.4f} +/- {vals.std(ddof=1):.4f}  "
              f"(range {vals.min():+.4f} to {vals.max():+.4f})")
    print("Interpret each metric separately: only a consistently positive delta for")
    print("that metric supports improvement from the transductive topic block.")
    print("A mixed-sign or zero-crossing range does not. None is a causal decomposition.")

    # topic-only permutation check
    obs_macro = f1_score(y, preds[("grouped (cv_group)", "topic only")], average="macro")
    if args.n_perm > 0:
        print(f"\n--- Permutation test against the label-shuffled null "
              f"({args.n_perm} permutations) ---")
        print("  Each permutation refits the whole cross-validated pipeline on shuffled")
        print("  labels, so this is a test of the entire procedure, not just the model.")
        rng = np.random.default_rng(RANDOM_SEED)
        null_scores = []
        y_shuf = y.copy()
        for _ in range(args.n_perm):
            rng.shuffle(y_shuf)
            yp = run_cv(Xtop, y_shuf, groups, make_lr(), grouped=True)
            null_scores.append(f1_score(y_shuf, yp, average="macro"))
        null_scores = np.asarray(null_scores)
        p_emp = (np.sum(null_scores >= obs_macro) + 1) / (args.n_perm + 1)
        print(f"  observed macro F1 {obs_macro:.4f}")
        print(f"  null mean {null_scores.mean():.4f} "
              f"(SD {null_scores.std(ddof=1):.4f}), null max {null_scores.max():.4f}")
        print(f"  empirical p = {p_emp:.4f}"
              + (f"  (floor of this test is 1/{args.n_perm + 1} = "
                 f"{1 / (args.n_perm + 1):.4f})" if p_emp <= 2 / (args.n_perm + 1) else ""))
    else:
        null_scores = np.array([np.nan])
        p_emp = np.nan
        print("\n--- Permutation test skipped (--n-perm 0) ---")

    # per-level breakdown
    print("\n--- Per-level performance, topic only, grouped CV ---")
    rep = classification_report(y, preds[("grouped (cv_group)", "topic only")],
                                labels=levels, output_dict=True, zero_division=0)
    per_level = pd.DataFrame({l: rep[l] for l in levels}).T
    per_level["predicted_n"] = [
        int(np.sum(preds[("grouped (cv_group)", "topic only")] == l)) for l in levels
    ]
    print(per_level.to_string(float_format=lambda x: f"{x:.3f}"))
    never = [l for l in levels if per_level.loc[l, "predicted_n"] == 0]
    if never:
        print(f"\nLevels the topic-only model NEVER predicts: {never}")
    per_level.to_csv(RESULTS_DIR / f"per_level_report_{tag}.csv")

    # reported-model calibration
    out = {"tag": tag, "n": int(len(df)),
           "class_support": support.to_dict(),
           "spearman_length_level": {"rho": float(lr_rho), "p": float(lr_p)},
           "results": res.to_dict(orient="records"),
           "leakage_weighted_f1": leak.round(4).to_dict(),
           "bootstrap_ci": ci_df.to_dict(orient="records"),
           "incremental_topic_over_surface": inc,
           "permutation_null": {"observed_macro_f1": float(obs_macro),
                                "null_mean": float(null_scores.mean()),
                                "null_max": float(null_scores.max()),
                                "empirical_p": float(p_emp)},
           "per_level": per_level.to_dict(orient="index"),
           "never_predicted_levels": never}

    floor = MOST_FREQUENT_WEIGHTED_F1.get(lang)
    ceiling = XLMR_WEIGHTED_F1.get(lang)
    if args.dataset is None and floor is not None and ceiling is not None:
        lang_name = {"de": "German", "en": "English"}.get(lang, lang)
        print(f"\n--- Calibration against Imperial et al. (2025), {lang_name}, "
              "weighted F1 ---")
        span = ceiling - floor
        our_floor = 100 * res[(res.cv == "grouped (cv_group)") &
                              (res.features == "majority class (floor)")
                              ]["weighted_f1"].iloc[0]
        print(f"  NOTE: their most-frequent-class floor is {floor:.1f} while the "
              f"same baseline on our split is {our_floor:.1f}.")
        print("  The two are not interchangeable, because their test split caps "
              "instances per level and ours does not, so the level distributions")
        print("  differ. Their floor is used below because the ceiling is theirs "
              "too, and a share of a span must be measured on one scale.")
        print(f"  their most-frequent-class floor : {floor:.1f}")
        print(f"  their fine-tuned XLM-R          : {ceiling:.1f}")
        print(f"  above-floor span                : {span:.1f} points")
        calib = {}
        for name in ["word count only", "surface features", "topic only",
                     "surface + topic"]:
            wf1 = 100 * res[(res.cv == "grouped (cv_group)") &
                            (res.features == name)]["weighted_f1"].iloc[0]
            share = (wf1 - floor) / span
            calib[name] = {"weighted_f1": round(float(wf1), 1),
                           "share_of_above_floor_span": round(float(share), 3)}
            print(f"  {name:<18}: weighted F1 {wf1:5.1f}  "
                  f"-> {share:6.1%} of the above-floor span")
        print("\n  Cross-study caveats: their test split is capped at")
        print("  200 instances per language per granularity, they evaluate on a held-out")
        print("  split rather than cross-validation, and their portion of this language")
        print("  may differ after their own cleaning. This is an indication of magnitude,")
        print("  not a like-for-like benchmark entry.")
        out["calibration_vs_reported"] = calib
        out["reference_numbers"] = REFERENCE_LADDER.get(lang, {})
        out["our_majority_floor_weighted_f1"] = round(float(our_floor), 1)
    elif args.dataset is None:
        print(f"\n--- Calibration skipped: no reference numbers for '{lang}' in "
              "config.REFERENCE_LADDER ---")

    # regenerated summary for the manifest
    summary_lines = [
        f"LENGTH VS TOPIC BENCHMARK ({tag}), n={len(df)}",
        "Grouped 5-fold CV on cv_group; topic features are transductive.",
    ]
    for name in ("majority class (floor)", "word count only", "surface features",
                 "topic only", "surface + topic"):
        row = g.loc[name]
        summary_lines.append(
            f"{name}: weighted_f1={row['weighted_f1']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, qwk={row['qwk']:.4f}"
        )
    summary_lines.append("Repeated-CV topic increment over surface features:")
    for metric in ("weighted_f1", "macro_f1", "qwk"):
        item = inc[metric]
        summary_lines.append(
            f"{metric}: mean={item['mean_delta']:+.4f}, "
            f"range=[{item['min']:+.4f}, {item['max']:+.4f}]"
        )
    summary_lines.extend([
        "Interpret metrics separately; a zero-crossing range does not support improvement.",
        "These comparisons are predictive diagnostics, not a causal decomposition.",
    ])
    (RESULTS_DIR / f"length_benchmark_{tag}.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    def _safe(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))

    (RESULTS_DIR / f"length_benchmark_{tag}.json").write_text(
        json.dumps(out, indent=2, default=_safe), encoding="utf-8")
    keep_feats = {"majority class (floor)", "word count only", "surface features",
                  "topic only", "surface + topic"}
    missing = keep_feats - set(res["features"])
    assert not missing, (
        f"feature names changed but keep_feats did not: {sorted(missing)}. "
        "This silently emptied the out-of-fold file once already."
    )
    pred_rows = []
    for (cvlabel, name), yp in preds.items():
        if cvlabel != "grouped (cv_group)" or name not in keep_feats:
            continue
        for yt, yhat in zip(y, yp):
            pred_rows.append({"cv": cvlabel, "features": name, "y_true": yt, "y_pred": yhat})
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / f"oof_length_benchmark_{tag}.csv", index=False)
    print(f"\nSaved out/length_benchmark_{tag}.csv / .json")
    print(f"Saved out/per_level_report_{tag}.csv")
    print(f"Saved out/length_vs_level_{tag}.csv")
    print(f"Saved out/oof_length_benchmark_{tag}.csv")


if __name__ == "__main__":
    main()
