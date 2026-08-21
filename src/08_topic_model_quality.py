"""Step 8: Justify k empirically (coherence, diversity, reconstruction) + seed stability.

Run:
    python src/08_topic_model_quality.py
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*",
                        category=UserWarning)
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

from config import ENCODER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths
from features import checked_numeric_matrix, vectorizer_stopwords
from plotting import savefig, set_style
from stats_utils import cramers_v

K_GRID = [5, 8, 10, 12, 15, 20, 25, 30]
SEED_GRID = [0, 1, 2, 42, 123]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def top_words_for_topics(model, top_k: int = 15) -> list[list]:
    """Return top words for each topic in descending weight order.

    The order ensures that NPMI uses the highest-weighted words. Code that needs
    sets, such as the Jaccard check, converts them locally.
    """
    vocab = np.asarray(model.get_vocab())
    return [
        vocab[np.argsort(weights)[::-1][:top_k]].tolist()
        for weights in model.components_
    ]


def topic_diversity(word_lists: list[list], top_k: int = 25) -> float:
    """Share of distinct words across all topics' top_k words.

    Interpretation note: diversity can fall with k mechanically, because
    more topics means more opportunity for overlap. It is therefore a weak
    model-selection criterion on its own and should be read as a description,
    not as evidence that a smaller k is better.
    """
    all_words = []
    for words in word_lists:
        all_words.extend(list(words)[:top_k])
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def npmi_coherence(texts: list[str], word_lists: list[list], top_k: int = 10) -> float:
    """Mean NPMI over topics via gensim if available; else a simple PMI proxy.

    word_lists must be ORDERED by topic weight; see top_words_for_topics.

    Interpreting the sign: NPMI runs from -1 to +1, and negative values mean the
    top words co-occur in the same document less often than chance. On a corpus
    of short documents that is common and not by itself evidence of a bad model,
    because a learner letter or a simplified news item contains few words and
    most topic keywords simply cannot co-occur within it. The values are used
    here only to compare k against k on identical data, never as an absolute
    quality judgement.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel

        # use the same analyser as the countvectorizer that produced the topic
        # vocabulary. splitting on whitespace leaves punctuation attached
        # ("oesterreich," vs "oesterreich"), so topic words silently fail to
        # match the reference texts and coherence comes out too low.
        from sklearn.feature_extraction.text import CountVectorizer
        _analyzer = CountVectorizer(lowercase=True).build_analyzer()
        tokenized = [_analyzer(t) for t in texts]
        topics = [list(w)[:top_k] for w in word_lists]
        dictionary = Dictionary(tokenized)
        cm = CoherenceModel(
            topics=topics,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
            topn=top_k,
            processes=1,
        )
        return float(cm.get_coherence())
    except Exception as exc:
        print(f"  gensim NPMI unavailable ({exc}); using co-occurrence proxy")
        return _proxy_coherence(texts, word_lists, top_k)


def _proxy_coherence(texts, word_lists, top_k: int = 10) -> float:
    from collections import Counter

    docs = [set(t.lower().split()) for t in texts]
    n = len(docs)
    df_count = Counter()
    for d in docs:
        for w in d:
            df_count[w] += 1
    scores = []
    for s in word_lists:
        words = list(s)[:top_k]
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                df1, df2 = df_count[w1], df_count[w2]
                if df1 == 0 or df2 == 0:
                    continue
                joint = sum(1 for d in docs if w1 in d and w2 in d)
                pmi = np.log((joint * n + 1e-12) / (df1 * df2))
                npmi = pmi / (-np.log((joint + 1e-12) / n))
                pair_scores.append(npmi)
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores)) if scores else float("nan")


def matched_jaccard(lists_a: list[list], lists_b: list[list]) -> float:
    """Mean Jaccard overlap of optimally matched topics between two runs.

    Order is irrelevant here (overlap is a set operation), so the ordered lists
    are converted to sets locally. Topics are matched with the Hungarian
    algorithm because topic indices are arbitrary across runs.
    """
    sets_a = [set(x) for x in lists_a]
    sets_b = [set(x) for x in lists_b]
    n = min(len(sets_a), len(sets_b))
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(sets_a[i] & sets_b[j])
            union = len(sets_a[i] | sets_b[j]) or 1
            cost[i, j] = 1 - inter / union
    r, c = linear_sum_assignment(cost)
    return float(np.mean(1 - cost[r, c]))


def fit_keynmf(corpus, embeddings, encoder, n_topics, seed, stop, vectorizer=None):
    vectorizer = vectorizer or CountVectorizer(stop_words=stop, min_df=5, lowercase=True)
    model = KeyNMF(n_topics, encoder=encoder, vectorizer=vectorizer, random_state=seed)
    try:
        doc_topic = checked_numeric_matrix(
            lambda: model.fit_transform(corpus, embeddings=embeddings),
            f"KeyNMF document-topic matrix (k={n_topics}, seed={seed})",
            nonnegative=True,
        )
    except TypeError:
        doc_topic = checked_numeric_matrix(
            lambda: model.fit_transform(corpus),
            f"KeyNMF document-topic matrix (k={n_topics}, seed={seed})",
            nonnegative=True,
        )
    return model, doc_topic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    parser.add_argument("--skip-seed", action="store_true")
    args = parser.parse_args()
    p = paths(args.lang)
    lang = args.lang
    set_style()

    if not p["combined_csv"].exists():
        sys.exit(f"Run src/01_load_data.py --lang {lang} first.")
    df = pd.read_csv(p["combined_csv"])
    corpus = df["text"].astype(str).tolist()
    stop = vectorizer_stopwords(lang)

    print(f"Embedding {len(corpus)} texts once ...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER)
    embeddings = encoder.encode(corpus, show_progress_bar=True)

    quality_rows = []
    for k in K_GRID:
        print(f"\n=== quality k={k} ===")
        model, doc_topic = fit_keynmf(corpus, embeddings, encoder, k, RANDOM_SEED, stop)
        sets = top_words_for_topics(model, top_k=25)
        coh = npmi_coherence(corpus, sets, top_k=10)
        div = topic_diversity(sets, top_k=25)
        try:
            rec_err = float(model.nmf_.reconstruction_err_)
        except Exception:
            rec_err = float("nan")
        quality_rows.append(
            {"k": k, "npmi": coh, "diversity": div, "reconstruction_err": rec_err}
        )
        print(f"  NPMI={coh:.3f}  diversity={div:.3f}  recon={rec_err}")

    qdf = pd.DataFrame(quality_rows)
    qdf.to_csv(RESULTS_DIR / f"topic_quality_{lang}.csv", index=False)

    # only plot reconstruction error if the installed turftopic actually exposes
    # it. plotting an all-nan series produces a blank third panel, which looks
    # like a broken figure rather than an unavailable metric.
    has_recon = qdf["reconstruction_err"].notna().any()
    if not has_recon:
        print("\n  NOTE: this Turftopic build does not expose the NMF "
              "reconstruction error, so that panel is omitted from the figure "
              "and the column is left empty in topic_quality_*.csv.")
    n_panels = 3 if has_recon else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.7 * n_panels, 3.5))
    axes[0].plot(qdf["k"], qdf["npmi"], marker="o")
    axes[0].set_title("NPMI coherence")
    axes[0].set_ylabel("mean NPMI (higher is better)")
    axes[1].plot(qdf["k"], qdf["diversity"], marker="o", color="C1")
    axes[1].set_title("Topic diversity")
    axes[1].set_ylabel("share of distinct top words")
    if has_recon:
        axes[2].plot(qdf["k"], qdf["reconstruction_err"], marker="o", color="C2")
        axes[2].set_title("NMF reconstruction error")
    for ax in axes:
        ax.set_xlabel("number of topics k")
        ax.axvline(15, color="0.5", ls="--")
        ax.annotate("k = 15 used\nin main analysis", xy=(15, ax.get_ylim()[0]),
                    xytext=(2, 4), textcoords="offset points",
                    fontsize=7, color="0.4")
    fig.suptitle(f"Topic model quality vs k ({lang})")
    savefig(fig, RESULTS_DIR / f"fig_topic_quality_{lang}")

    if args.skip_seed:
        return

    print("\n=== Seed stability at k=15 ===")
    runs = {}
    seed_rows = []
    for seed in SEED_GRID:
        model, doc_topic = fit_keynmf(corpus, embeddings, encoder, 15, seed, stop)
        topics = doc_topic.argmax(axis=1)
        runs[seed] = top_words_for_topics(model, top_k=15)
        row = {"seed": seed}
        for name, idx in df.groupby("dataset").groups.items():
            sub_levels = df.loc[idx, "cefr_level"]
            sub_topics = pd.Series(topics[df.index.get_indexer(idx)], index=idx)
            if sub_levels.nunique() < 2 or sub_topics.nunique() < 2:
                continue
            v, _ = cramers_v(pd.crosstab(sub_levels, sub_topics))
            row[f"v_{name}"] = round(v, 4)
        seed_rows.append(row)
        print(f"  seed={seed}: {row}")

    jaccards = []
    for a, b in combinations(SEED_GRID, 2):
        jaccards.append(matched_jaccard(runs[a], runs[b]))
    mean_j = float(np.mean(jaccards)) if jaccards else float("nan")
    sdf = pd.DataFrame(seed_rows)
    sdf["mean_pairwise_topic_jaccard"] = mean_j
    sdf.to_csv(RESULTS_DIR / f"seed_stability_{lang}.csv", index=False)
    print(f"Mean pairwise matched-topic Jaccard: {mean_j:.3f}")
    v_cols = [c for c in sdf.columns if c.startswith("v_")]
    for c in v_cols:
        print(f"  {c}: mean={sdf[c].mean():.3f} sd={sdf[c].std():.3f}")


if __name__ == "__main__":
    main()
