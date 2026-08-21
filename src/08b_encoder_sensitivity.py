"""step 8b: encoder sensitivity"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*",
                        category=UserWarning)
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

from config import N_TOPICS, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths
from features import checked_numeric_matrix, vectorizer_stopwords
from stats_utils import cramers_v

ENCODERS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "distiluse-base-multilingual-cased-v2",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    parser.add_argument("--encoders", nargs="+", default=ENCODERS)
    args = parser.parse_args()
    p = paths(args.lang)

    if not p["combined_csv"].exists():
        sys.exit(f"Run src/01_load_data.py --lang {args.lang} first.")
    df = pd.read_csv(p["combined_csv"])
    corpus = df["text"].astype(str).tolist()
    stop = vectorizer_stopwords(args.lang)

    rows = []
    for enc in args.encoders:
        print(f"\n=== encoder: {enc} ===")
        try:
            model = KeyNMF(
                N_TOPICS,
                encoder=enc,
                vectorizer=CountVectorizer(stop_words=stop, min_df=5, lowercase=True),
                random_state=RANDOM_SEED,
            )
            doc_topic = checked_numeric_matrix(
                lambda: model.fit_transform(corpus),
                f"KeyNMF document-topic matrix ({enc})",
                nonnegative=True,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"encoder": enc, "error": str(exc)})
            continue
        topics = doc_topic.argmax(axis=1)
        row = {"encoder": enc}
        for name, idx in df.groupby("dataset").groups.items():
            sub_levels = df.loc[idx, "cefr_level"]
            sub_topics = pd.Series(topics[df.index.get_indexer(idx)], index=idx)
            if sub_levels.nunique() < 2 or sub_topics.nunique() < 2:
                continue
            v, pval = cramers_v(pd.crosstab(sub_levels, sub_topics))
            row[f"v_{name}"] = round(v, 4)
            row[f"p_{name}"] = f"{pval:.2e}"
            print(f"  V({name})={v:.3f}")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"encoder_sensitivity_{args.lang}.csv", index=False)
    print(f"\nSaved out/encoder_sensitivity_{args.lang}.csv")


if __name__ == "__main__":
    main()
