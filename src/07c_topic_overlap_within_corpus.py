"""step 7c: within-corpus topic-overlap stress test"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
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
    LEVEL_ORDER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths,
    topic_outputs_problem,
)
from utils_dedup import topic_linked_components

N_SPLITS = 5
RANK = {l: i for i, l in enumerate(LEVEL_ORDER)}


def qwk(a, b) -> float:
    return float(cohen_kappa_score([RANK[x] for x in a], [RANK[x] for x in b],
                                   weights="quadratic"))


def lr_dense():
    return make_pipeline(
        StandardScaler(),
        CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
    )


def topic_pair_components(df: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """merge topics linked by a parallel or near-duplicate family"""
    return topic_linked_components(df["topic"], validation_groups(df))


def pooled_skew_diagnostic(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """quantify how far topic-grouped folds distort the corpus mix"""
    base = df["dataset"].value_counts(normalize=True)
    rows = []
    for i, (_, te) in enumerate(
        GroupKFold(n_splits=N_SPLITS).split(df, df["cefr_level"], groups=df["topic"])
    ):
        mix = df.iloc[te]["dataset"].value_counts(normalize=True).reindex(base.index).fillna(0)
        rows.append({
            "fold": i,
            "total_variation_from_overall": round(float(np.abs(mix - base).sum() / 2), 3),
            **{f"share_{k}": round(float(v), 3) for k, v in mix.items()},
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"topic_overlap_diagnostic_{lang}.csv", index=False)
    print("\n--- Diagnostic: corpus skew induced by topic-grouped folds (pooled data) ---")
    print("overall corpus mix: " + ", ".join(f"{k} {v:.2f}" for k, v in base.items()))
    print(out.to_string(index=False))
    print("A total variation distance far from 0 means the pooled topic-stratified")
    print("estimate in script 07 conflates topic overlap with corpus shift, which is")
    print("why the within-corpus analysis below is the one to report.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    ap.add_argument("--dataset", default="merlin_de")
    ap.add_argument("--skip-diagnostic", action="store_true")
    args = ap.parse_args()
    lang, p = args.lang, paths(args.lang)

    if problem := topic_outputs_problem(lang):
        sys.exit(problem)
    full = pd.read_csv(p["with_topics_csv"])
    doc_topic = np.load(p["doc_topic_matrix"])
    assert len(full) == doc_topic.shape[0], "table and topic matrix out of sync"

    counts_all = full["cefr_level"].value_counts()
    keep_all = (~full["cefr_level"].isin(counts_all[counts_all < N_SPLITS].index)).to_numpy()
    if not args.skip_diagnostic:
        pooled_skew_diagnostic(full[keep_all].reset_index(drop=True), lang)

    sel = (full["dataset"] == args.dataset).to_numpy()
    if not sel.any():
        sys.exit(f"No rows for '{args.dataset}'. Have: {sorted(full['dataset'].unique())}")
    df, X = full[sel].reset_index(drop=True), doc_topic[sel]
    counts = df["cefr_level"].value_counts()
    keep = (~df["cefr_level"].isin(counts[counts < N_SPLITS].index)).to_numpy()
    dropped = sorted(counts[counts < N_SPLITS].index.tolist())
    df, X = df[keep].reset_index(drop=True), normalise_topic_weights(X[keep])
    y = df["cefr_level"].to_numpy()
    n_topics_used = int(df["topic"].nunique())
    topic_groups, n_topic_components, n_cross_topic_pairs = topic_pair_components(df)

    tag = f"{lang}_{args.dataset}"
    print("\n" + "=" * 78)
    print(f"TOPIC OVERLAP WITHIN A SINGLE CORPUS: {args.dataset} (n = {len(df):,})")
    print("=" * 78)
    print(f"Levels dropped for having fewer than {N_SPLITS} texts: "
          f"{dropped if dropped else 'none'}")
    print(f"Distinct topics present in this corpus: {n_topics_used} "
          f"(of {int(full['topic'].nunique())} overall)")
    print(f"Leakage-safe topic components after linking validation families: "
          f"{n_topic_components}; cross-topic pairs: {n_cross_topic_pairs}")
    if n_topic_components < N_SPLITS:
        status = (
            f"not estimable: leakage-safe topic grouping leaves "
            f"{n_topic_components} component(s), fewer than {N_SPLITS} folds"
        )
        print(f"  {status}.")
        print("  No scores are produced: allowing a parallel or near-duplicate text")
        print("  to cross folds would introduce direct document-family leakage.")
        pd.DataFrame([{
            "corpus": args.dataset,
            "n": int(len(df)),
            "n_topics_used": n_topics_used,
            "n_topic_components": n_topic_components,
            "n_cross_topic_pairs": n_cross_topic_pairs,
            "status": status,
        }]).to_csv(RESULTS_DIR / f"topic_overlap_within_{tag}.csv", index=False)
        return
    if n_topics_used < N_SPLITS + 2:
        print("  CAUTION: with this few topics, each grouped fold holds out a large")
        print("  share of them, so the split is coarse. Report the number of topics")
        print("  alongside the result rather than the drop alone.")
    print("Corpus membership is constant here, so a drop cannot be caused by a")
    print("shift in corpus composition the way it can in the pooled analysis.")
    print("That is NOT the same as saying the drop measures topic overlap alone.")
    print("Random and topic-grouped folds also differ in class balance and fold")
    print("difficulty, and the non-topic controls can shift in the same direction")
    print("as the lexical model. Read this as the difference between two split regimes,")
    print("a stress test rather than a clean causal estimate.\n")

    # fold-specific vocabulary and idf
    Xtf = df["text"].astype(str).tolist()
    # separate word-count control
    Xwc = word_count_feature(df["word_count"])
    Xsurf = surface_features(df["text"]).to_numpy()

    specs = [
        ("topic mixture", X, lr_dense(),
         "reference representation directly defined by the topic model"),
        ("full text (TF-IDF)", Xtf,
         make_pipeline(
             TfidfVectorizer(max_features=50_000, min_df=3, sublinear_tf=True),
             CheckedLogisticRegression(max_iter=3000, random_state=RANDOM_SEED)),
         "lexical classifier stress test"),
        ("word count only", Xwc, lr_dense(),
         "minimal non-topic control with one feature"),
        ("surface features", Xsurf, lr_dense(),
         "control matching script 06's baseline: length plus complexity"),
    ]

    rows = []
    pair_groups = validation_groups(df)
    for name, Xm, clf, note in specs:
        rnd = cross_val_predict(
            clf, Xm, y,
            cv=StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
            ),
            groups=pair_groups,
        )
        grp = cross_val_predict(clf, Xm, y, cv=GroupKFold(n_splits=N_SPLITS),
                                groups=topic_groups)
        r = {
            "corpus": args.dataset, "features": name, "n": int(len(df)),
            "n_topics_used": n_topics_used,
            "n_topic_components": n_topic_components,
            "n_cross_topic_pairs": n_cross_topic_pairs,
            "status": "ok",
            "random_weighted_f1": f1_score(y, rnd, average="weighted", zero_division=0),
            "topic_grouped_weighted_f1": f1_score(y, grp, average="weighted", zero_division=0),
            "random_qwk": qwk(y, rnd),
            "topic_grouped_qwk": qwk(y, grp),
        }
        r["drop_weighted_f1"] = r["random_weighted_f1"] - r["topic_grouped_weighted_f1"]
        r["drop_qwk"] = r["random_qwk"] - r["topic_grouped_qwk"]
        rows.append(r)
        print(f"{name}   ({note})")
        print(f"   random folds  : weighted F1 {r['random_weighted_f1']:.3f}   "
              f"QWK {r['random_qwk']:+.3f}")
        print(f"   topic-grouped : weighted F1 {r['topic_grouped_weighted_f1']:.3f}   "
              f"QWK {r['topic_grouped_qwk']:+.3f}")
        print(f"   drop          : {r['drop_weighted_f1']:+.3f} weighted F1   "
              f"{r['drop_qwk']:+.3f} QWK\n")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"topic_overlap_within_{tag}.csv", index=False)
    print(f"Saved out/topic_overlap_within_{tag}.csv")
    print("\nHow to read this. Compare TF-IDF with the word-count and surface")
    print("controls. If controls move by a similar amount, the split difference")
    print("cannot be attributed specifically to topic overlap. Report all signs and")
    print("all corpora; do not call the difference a cost or reliance estimate.")


if __name__ == "__main__":
    main()
