"""step 2: keynmf topic model"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*",
                        category=UserWarning)
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

from config import (
    ENCODER, N_TOPICS, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths,
    sha256_file,
)
from features import checked_numeric_matrix, vectorizer_stopwords


def get_top_words(model, top_k: int = 15) -> pd.DataFrame:
    """extract the top_k highest-weighted words for each topic"""
    vocab = np.asarray(model.get_vocab())
    rows = []
    for topic_id, weights in enumerate(model.components_):
        top = vocab[np.argsort(weights)[::-1][:top_k]]
        rows.append({"topic": topic_id, "top_words": ", ".join(top)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    parser.add_argument("--n-topics", type=int, default=N_TOPICS)
    parser.add_argument("--encoder", type=str, default=ENCODER)
    args = parser.parse_args()
    p = paths(args.lang)

    RESULTS_DIR.mkdir(exist_ok=True)

    if not p["combined_csv"].exists():
        sys.exit(f"{p['combined_csv']} not found. Run: python src/01_load_data.py --lang {args.lang}")

    df = pd.read_csv(p["combined_csv"])
    corpus = df["text"].astype(str).tolist()
    print(f"Fitting KeyNMF with {args.n_topics} topics on {len(corpus)} {args.lang} texts ...")

    # keyword stopwords only; embeddings unchanged
    # remaining style and complexity signal
    stop = vectorizer_stopwords(args.lang)
    vectorizer = CountVectorizer(
        stop_words=stop,
        min_df=5,           # minimum document frequency
        lowercase=True,
    )

    model = KeyNMF(
        args.n_topics,
        encoder=args.encoder,
        vectorizer=vectorizer,
        random_state=RANDOM_SEED,
    )
    doc_topic = checked_numeric_matrix(
        lambda: model.fit_transform(corpus),
        "KeyNMF document-topic matrix",
        nonnegative=True,
    )
    if doc_topic.shape != (len(corpus), args.n_topics):
        raise ValueError(
            f"unexpected KeyNMF shape {doc_topic.shape}; "
            f"expected {(len(corpus), args.n_topics)}"
        )

    # topic preview
    model.print_topics()

    # downstream files
    np.save(p["doc_topic_matrix"], doc_topic)
    print(f"Saved {p['doc_topic_matrix']}  shape={doc_topic.shape}")

    top_words = get_top_words(model)
    top_words.to_csv(p["topic_words_csv"], index=False)
    print(f"Saved {p['topic_words_csv']}")

    # keep all-zero rows out of topic 0
    empty = int((doc_topic.max(axis=1) == 0).sum())
    if empty:
        print(f"WARNING: {empty} document(s) have an all-zero topic vector "
              "(no keyword passed the min_df filter). argmax assigns them to "
              "topic 0; check topic_strength == 0 to find them.")

    df["topic"] = doc_topic.argmax(axis=1)
    df["topic_strength"] = doc_topic.max(axis=1)
    df.to_csv(p["with_topics_csv"], index=False)
    print(f"Saved {p['with_topics_csv']}")
    p["topic_state_json"].write_text(json.dumps({
        "source_sha256": sha256_file(p["combined_csv"]),
        "with_topics_sha256": sha256_file(p["with_topics_csv"]),
        "doc_topic_sha256": sha256_file(p["doc_topic_matrix"]),
        "topic_words_sha256": sha256_file(p["topic_words_csv"]),
        "n_documents": int(len(df)),
        "n_topics": int(doc_topic.shape[1]),
        "encoder": args.encoder,
    }, indent=2), encoding="utf-8")
    print(f"Saved {p['topic_state_json']}")

    print(
        f"\nNext: read {p['topic_words_csv'].name} and sanity check the topics. "
        f"Then run: python src/03_confound_analysis.py --lang {args.lang}"
    )


if __name__ == "__main__":
    main()
