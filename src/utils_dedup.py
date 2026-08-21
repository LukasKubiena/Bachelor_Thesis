"""near-duplicate and grouping utilities"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def merged_validation_groups(
    df: pd.DataFrame,
    near_pairs: pd.DataFrame | None = None,
) -> pd.Series:
    """return leakage-safe groups spanning parallel and near-duplicate texts"""
    n = len(df)
    parent = np.arange(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if "pair_id" in df.columns:
        grouped = df.groupby(df["pair_id"].astype(str), sort=False).indices.values()
        for indices in grouped:
            indices = list(map(int, indices))
            for idx in indices[1:]:
                union(indices[0], idx)

    if near_pairs is not None and len(near_pairs):
        if not {"i", "j"} <= set(near_pairs.columns):
            raise ValueError("near-duplicate table must contain i and j columns")
        for i, j in near_pairs[["i", "j"]].itertuples(index=False, name=None):
            i, j = int(i), int(j)
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"near-duplicate row index out of range: {(i, j)} for n={n}")
            union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_id = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    return pd.Series(
        [f"cv_{root_to_id[root]}" for root in roots],
        index=df.index,
        name="cv_group",
        dtype="string",
    )


def topic_linked_components(
    topics,
    validation_groups,
) -> tuple[np.ndarray, int, int]:
    """merge dominant-topic groups linked by a validation family"""
    frame = pd.DataFrame({
        "topic": np.asarray(topics, dtype=int),
        "validation_group": np.asarray(validation_groups).astype(str),
    })
    unique_topics = sorted(frame["topic"].unique().tolist())
    parent = {topic: topic for topic in unique_topics}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cross_topic_groups = 0
    for _, group in frame.groupby("validation_group", sort=False):
        linked = group["topic"].unique().tolist()
        if len(linked) > 1:
            cross_topic_groups += 1
            for topic in linked[1:]:
                union(linked[0], topic)

    roots = {topic: find(topic) for topic in unique_topics}
    root_ids = {root: i for i, root in enumerate(sorted(set(roots.values())))}
    groups = frame["topic"].map(lambda t: root_ids[roots[int(t)]]).to_numpy()
    return groups, len(root_ids), cross_topic_groups


def near_duplicate_report(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.85,
    top_k: int = 20,
) -> pd.DataFrame:
    """cosine similarity over char 3-5 gram tf-idf"""
    texts = df[text_col].astype(str).tolist()
    X = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
    ).fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(df)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)

    rows = []
    seen = set()
    for i in range(len(df)):
        for rank in range(1, idx.shape[1]):  # skip self at rank 0
            j = int(idx[i, rank])
            sim = 1.0 - float(dist[i, rank])
            if sim < threshold:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append(
                {
                    "i": a,
                    "j": b,
                    "similarity": round(sim, 4),
                    "dataset_i": df.iloc[a]["dataset"],
                    "dataset_j": df.iloc[b]["dataset"],
                    "level_i": df.iloc[a]["cefr_level"],
                    "level_j": df.iloc[b]["cefr_level"],
                    "same_corpus": df.iloc[a]["dataset"] == df.iloc[b]["dataset"],
                    "pair_id_i": df.iloc[a].get("pair_id", None),
                    "pair_id_j": df.iloc[b].get("pair_id", None),
                }
            )
    return pd.DataFrame(rows)


def bipartite_max_similarity(
    df: pd.DataFrame,
    dataset_a: str,
    dataset_b: str,
    text_col: str = "text",
    chunk: int = 200,
) -> pd.DataFrame:
    """for every document in dataset_a, the nearest neighbour in dataset_b"""
    from sklearn.metrics.pairwise import cosine_similarity

    a_mask = df["dataset"] == dataset_a
    b_mask = df["dataset"] == dataset_b
    if not a_mask.any() or not b_mask.any():
        return pd.DataFrame()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000
    )
    X = vec.fit_transform(df[text_col].astype(str))
    Xa, Xb = X[a_mask.to_numpy()], X[b_mask.to_numpy()]
    a_idx = np.flatnonzero(a_mask.to_numpy())
    b_idx = np.flatnonzero(b_mask.to_numpy())
    rows = []
    for start in range(0, Xa.shape[0], chunk):
        S = cosine_similarity(Xa[start:start + chunk], Xb)
        am = S.argmax(axis=1)
        mx = S.max(axis=1)
        for k, (j_local, sim) in enumerate(zip(am, mx)):
            i = int(a_idx[start + k])
            j = int(b_idx[int(j_local)])
            rows.append({
                "i": i, "j": j, "similarity": round(float(sim), 4),
                "dataset_i": dataset_a, "dataset_j": dataset_b,
                "level_i": df.iloc[i]["cefr_level"],
                "level_j": df.iloc[j]["cefr_level"],
            })
    return pd.DataFrame(rows)
