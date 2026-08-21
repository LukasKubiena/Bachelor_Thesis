"""step 5: topic-count robustness"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*",
                        category=UserWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from turftopic import KeyNMF

from config import ENCODER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths
from features import (
    CheckedLogisticRegression,
    checked_numeric_matrix,
    normalise_topic_weights,
    validation_groups,
    vectorizer_stopwords,
)
from stats_utils import cramers_v

N_SPLITS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    parser.add_argument("--n-topics", type=int, nargs="+", default=[10, 15, 20])
    args = parser.parse_args()
    p = paths(args.lang)

    if not p["combined_csv"].exists():
        sys.exit(f"Run src/01_load_data.py --lang {args.lang} first.")
    df = pd.read_csv(p["combined_csv"])
    corpus = df["text"].astype(str).tolist()

    print(f"Embedding {len(corpus)} texts once (reused for all settings) ...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER)
    embeddings = encoder.encode(corpus, show_progress_bar=True)

    stop = vectorizer_stopwords(args.lang)
    rows = []
    for n in args.n_topics:
        print(f"\n=== {n} topics ===")
        model = KeyNMF(
            n,
            encoder=encoder,
            vectorizer=CountVectorizer(stop_words=stop, min_df=5, lowercase=True),
            random_state=RANDOM_SEED,
        )
        try:
            doc_topic = checked_numeric_matrix(
                lambda: model.fit_transform(corpus, embeddings=embeddings),
                f"KeyNMF document-topic matrix (k={n})",
                nonnegative=True,
            )
        except TypeError:  # older turftopic without embeddings kwarg
            doc_topic = checked_numeric_matrix(
                lambda: model.fit_transform(corpus),
                f"KeyNMF document-topic matrix (k={n})",
                nonnegative=True,
            )
        topics = doc_topic.argmax(axis=1)

        # half 1: within-source cramér's v
        for name, idx in df.groupby("dataset").groups.items():
            sub_levels = df.loc[idx, "cefr_level"]
            sub_topics = pd.Series(topics[df.index.get_indexer(idx)], index=idx)
            if sub_levels.nunique() < 2 or sub_topics.nunique() < 2:
                continue
            v, pval = cramers_v(pd.crosstab(sub_levels, sub_topics))
            rows.append({"n_topics": n, "measure": f"cramers_v_{name}", "value": round(v, 3), "p": f"{pval:.1e}"})
            print(f"  V ({name}): {v:.3f}")

        # half 2: script 06 estimator and grouped folds
        counts = df["cefr_level"].value_counts()
        keep = ~df["cefr_level"].isin(counts[counts < N_SPLITS].index)
        y = df.loc[keep, "cefr_level"].to_numpy()
        X = normalise_topic_weights(doc_topic[keep.to_numpy()])
        groups = validation_groups(df.loc[keep])
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        y_pred = cross_val_predict(
            make_pipeline(StandardScaler(),
                          CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
            X, y, cv=cv, groups=groups,
        )
        f1 = f1_score(y, y_pred, average="macro")
        wf1 = f1_score(y, y_pred, average="weighted")
        acc = float(np.mean(y_pred == y))
        rows.append({"n_topics": n, "measure": "topic_mixture_macro_f1", "value": round(f1, 3), "p": ""})
        rows.append({"n_topics": n, "measure": "topic_mixture_weighted_f1", "value": round(wf1, 3), "p": ""})
        rows.append({"n_topics": n, "measure": "topic_mixture_accuracy", "value": round(acc, 3), "p": ""})
        print(f"  topic-mixture macro F1: {f1:.3f}, weighted F1: {wf1:.3f}, accuracy: {acc:.3f}")

    out = RESULTS_DIR / f"robustness_{args.lang}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("Interpret stability from the full range across topic counts; do not rely")
    print("on the k=15 row alone.")


if __name__ == "__main__":
    main()
