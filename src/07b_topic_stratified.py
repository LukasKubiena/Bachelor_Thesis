"""Step 7b: independent replication of the pooled split stress test.

Recomputes the two rows produced by step 7 with the same estimators and folds,
then checks them against step 7's authoritative table. Keeping an independent
implementation catches accidental drift, while writing to a separate file
prevents this diagnostic from overwriting the primary result. The gap remains
descriptive: grouping also changes class and corpus composition.

Run:
    python src/07b_topic_stratified.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import (
    LEVEL_ORDER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths,
    topic_outputs_problem,
)
from features import CheckedLogisticRegression, normalise_topic_weights, validation_groups
from stats_utils import classification_metrics
from utils_dedup import topic_linked_components

N_SPLITS = 5


def cv_eval(name, X, y, groups=None, pair_groups=None):
    if groups is None:
        if pair_groups is None:
            raise ValueError("pair_groups are required for the random-fold replication")
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
        )
        splits = list(cv.split(X, y, pair_groups))
    else:
        cv = GroupKFold(n_splits=N_SPLITS)
        splits = list(cv.split(X, y, groups))
    y_pred = np.empty(len(y), dtype=object)
    clf = make_pipeline(
        StandardScaler(),
        CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
    )
    for train, test in splits:
        clf.fit(X[train], y[train])
        y_pred[test] = clf.predict(X[test])
    m = classification_metrics(y, y_pred)
    m["model"] = name
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = parser.parse_args()
    p = paths(args.lang)
    lang = args.lang

    if problem := topic_outputs_problem(lang):
        sys.exit(problem)

    df = pd.read_csv(p["with_topics_csv"])
    doc_topic = np.load(p["doc_topic_matrix"])
    counts = df["cefr_level"].value_counts()
    keep = ~df["cefr_level"].isin(counts[counts < N_SPLITS].index)
    df = df[keep].reset_index(drop=True)
    doc_topic = normalise_topic_weights(doc_topic[keep.to_numpy()])
    y = df["cefr_level"].to_numpy()
    pairs = validation_groups(df)
    topics, n_topic_components, n_cross_topic_groups = topic_linked_components(
        df["topic"].to_numpy(), pairs
    )
    replication_path = RESULTS_DIR / f"topic_stratified_replication_{lang}.csv"
    authoritative_path = RESULTS_DIR / f"topic_stratified_{lang}.csv"
    if n_topic_components < N_SPLITS:
        status = (
            "not estimable: leakage-safe topic grouping leaves "
            f"{n_topic_components} component(s), fewer than {N_SPLITS} folds"
        )
        out = pd.DataFrame([{
            "model": "all",
            "status": status,
            "n_topic_components": n_topic_components,
            "n_cross_topic_groups": n_cross_topic_groups,
        }])
        out.to_csv(replication_path, index=False)
        if authoritative_path.exists():
            authoritative = pd.read_csv(authoritative_path)
            if set(authoritative.get("status", pd.Series(dtype=str)).astype(str)) != {status}:
                raise AssertionError("step 7/7b non-estimability status drift")
        (RESULTS_DIR / f"topic_stratified_replication_{lang}.txt").write_text(
            status + "\n", encoding="utf-8"
        )
        print(status)
        print("Independent replication agrees with step 7.")
        return

    rows = []
    print("Topic mixture, random stratified CV ...")
    rows.append(cv_eval(
        "topic_mixture_random", doc_topic, y, groups=None, pair_groups=pairs
    ))
    print("Topic mixture, topic-grouped CV ...")
    rows.append(cv_eval("topic_mixture_topic_grouped", doc_topic, y, groups=topics))

    print("TF-IDF + LR, random stratified CV ...")
    # refit the vectorizer inside each training fold
    texts = df["text"].astype(str).to_numpy()
    def _tfidf_lr():
        return make_pipeline(
            TfidfVectorizer(max_features=50_000, min_df=3, sublinear_tf=True),
            CheckedLogisticRegression(max_iter=3000, random_state=RANDOM_SEED))
    y_pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    for train, test in cv.split(texts, y, pairs):
        clf = _tfidf_lr()
        clf.fit(texts[train], y[train])
        y_pred[test] = clf.predict(texts[test])
    m = classification_metrics(y, y_pred)
    m["model"] = "tfidf_random"
    rows.append(m)
    print(f"  weighted F1={m['weighted_f1']:.3f}")

    print("TF-IDF + LR, topic-grouped CV ...")
    y_pred = np.empty(len(y), dtype=object)
    cv = GroupKFold(n_splits=N_SPLITS)
    for train, test in cv.split(texts, y, topics):
        clf = _tfidf_lr()
        clf.fit(texts[train], y[train])
        y_pred[test] = clf.predict(texts[test])
    m = classification_metrics(y, y_pred)
    m["model"] = "tfidf_topic_grouped"
    rows.append(m)
    print(f"  weighted F1={m['weighted_f1']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(replication_path, index=False)

    # check the independent result against step 7
    if authoritative_path.exists():
        authoritative = pd.read_csv(authoritative_path).set_index("features")
        comparisons = {
            "topic_mixture_random": ("topic mixture", "random_splits"),
            "topic_mixture_topic_grouped": ("topic mixture", "topic_grouped"),
            "tfidf_random": ("full text (TF-IDF)", "random_splits"),
            "tfidf_topic_grouped": ("full text (TF-IDF)", "topic_grouped"),
        }
        for model, (feature, regime) in comparisons.items():
            got = out.set_index("model").loc[model]
            expected = authoritative.loc[feature]
            for metric in ("weighted_f1", "macro_f1", "qwk"):
                column = f"{regime}_{metric}"
                if not np.isclose(got[metric], expected[column], atol=1e-12):
                    raise AssertionError(
                        f"step 7/7b drift for {model} {metric}: "
                        f"{got[metric]} != {expected[column]}"
                    )
        print("Independent replication agrees with step 7 (tolerance 1e-12).")
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    gap = (
        out.loc[out["model"] == "tfidf_random", "weighted_f1"].iloc[0]
        - out.loc[out["model"] == "tfidf_topic_grouped", "weighted_f1"].iloc[0]
    )
    print(f"\nTF-IDF split-regime difference (random - topic-grouped): "
          f"{gap:+.3f} weighted F1")
    (RESULTS_DIR / f"topic_stratified_replication_{lang}.txt").write_text(
        out.to_string(index=False) + f"\ntfidf_split_difference_weighted_f1={gap:.4f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
