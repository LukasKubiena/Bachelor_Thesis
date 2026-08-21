"""Step 8c: Alternative topic models as sensitivity checks (TF-IDF NMF, S3).

Run:
    python src/08c_model_sensitivity.py
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*",
                        category=UserWarning)
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from turftopic import KeyNMF

from config import ENCODER, N_TOPICS, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths
from features import checked_numeric_matrix, vectorizer_stopwords
from stats_utils import cramers_v


def within_source_v(df, topics) -> dict:
    out = {}
    for name, idx in df.groupby("dataset").groups.items():
        sub_levels = df.loc[idx, "cefr_level"]
        sub_topics = pd.Series(topics[df.index.get_indexer(idx)], index=idx)
        if sub_levels.nunique() < 2 or sub_topics.nunique() < 2:
            continue
        v, _ = cramers_v(pd.crosstab(sub_levels, sub_topics))
        out[name] = round(float(v), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = parser.parse_args()
    p = paths(args.lang)
    lang = args.lang

    if not p["combined_csv"].exists():
        sys.exit(f"Run src/01_load_data.py --lang {lang} first.")
    df = pd.read_csv(p["combined_csv"])
    corpus = df["text"].astype(str).tolist()
    stop = vectorizer_stopwords(lang)
    rows = []

    # keynmf (reference)
    print("KeyNMF ...")
    model = KeyNMF(
        N_TOPICS,
        encoder=ENCODER,
        vectorizer=CountVectorizer(stop_words=stop, min_df=5, lowercase=True),
        random_state=RANDOM_SEED,
    )
    doc_topic = checked_numeric_matrix(
        lambda: model.fit_transform(corpus),
        "KeyNMF document-topic matrix",
        nonnegative=True,
    )
    row = {"model": "KeyNMF", **{f"v_{k}": v for k, v in within_source_v(df, doc_topic.argmax(1)).items()}}
    rows.append(row)
    print(row)

    # plain tf-idf nmf
    print("TF-IDF NMF ...")
    X = TfidfVectorizer(stop_words=stop, min_df=5, max_features=20_000).fit_transform(corpus)
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=RANDOM_SEED, max_iter=400)
    W = checked_numeric_matrix(
        lambda: nmf.fit_transform(X),
        "TF-IDF NMF document-topic matrix",
        nonnegative=True,
    )
    row = {"model": "tfidf_NMF", **{f"v_{k}": v for k, v in within_source_v(df, W.argmax(1)).items()}}
    rows.append(row)
    print(row)

    # s3 from turftopic if available
    try:
        from turftopic import SemanticSignalSeparation as S3

        print("S3 (SemanticSignalSeparation) ...")
        s3 = S3(
            N_TOPICS,
            encoder=ENCODER,
            vectorizer=CountVectorizer(stop_words=stop, min_df=5, lowercase=True),
            random_state=RANDOM_SEED,
        )
        doc_topic_s3 = checked_numeric_matrix(
            lambda: s3.fit_transform(corpus),
            "S3 document-axis matrix",
        )
        # s3/fastica produces signed semantic axes rather than non-negative
        # topic memberships. the dominant axis is therefore the loading with
        # greatest magnitude; signed argmax systematically ignores strong
        # negative projections and is not a meaningful hard assignment.
        dominant_axis = np.abs(doc_topic_s3).argmax(1)
        row = {
            "model": "S3 (dominant absolute axis)",
            **{f"v_{k}": v for k, v in within_source_v(df, dominant_axis).items()},
        }
        rows.append(row)
        print(row)
    except Exception as exc:
        print(f"S3 skipped: {exc}")
        rows.append({"model": "S3", "error": str(exc)})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"model_sensitivity_{lang}.csv", index=False)
    print(f"Saved out/model_sensitivity_{lang}.csv")


if __name__ == "__main__":
    main()
