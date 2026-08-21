"""Step 1b: Descriptive statistics table for the thesis (Table 1).

Per corpus and per CEFR level: n, token length stats, sentences, TTR on a
fixed 200-token window, mean word length. Also Spearman rho of length vs
ordinal CEFR rank with a bootstrap CI — the evidence for the length claim
from the data itself, not a citation.

Run:
    python src/01b_descriptives.py
    python src/01b_descriptives.py --lang en
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import LEVEL_ORDER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths
from plotting import savefig, set_style

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SENT_SPLIT = re.compile(r"[.!?]+[\s\"]+")


def n_sentences(text: str) -> int:
    parts = [p for p in SENT_SPLIT.split(str(text).strip()) if p.strip()]
    return max(1, len(parts)) if str(text).strip() else 0


def mean_word_len(text: str) -> float:
    words = str(text).split()
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def windowed_ttr(text: str, window: int = 200) -> float:
    """Type-token ratio on the first `window` tokens (length-unconfounded)."""
    tokens = str(text).lower().split()[:window]
    if len(tokens) < 20:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def describe_slice(sub: pd.DataFrame) -> dict:
    wc = sub["word_count"]
    return {
        "n": int(len(sub)),
        "tokens_mean": float(wc.mean()),
        "tokens_median": float(wc.median()),
        "tokens_iqr": float(wc.quantile(0.75) - wc.quantile(0.25)),
        "tokens_min": int(wc.min()),
        "tokens_max": int(wc.max()),
        "sentences_median": float(sub["n_sentences"].median()),
        "ttr_200_mean": float(sub["ttr_200"].mean(skipna=True)),
        "mean_word_len": float(sub["mean_word_len"].mean()),
    }


def spearman_length_level(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42):
    idx = {l: i for i, l in enumerate(LEVEL_ORDER)}
    ranks = df["cefr_level"].map(idx).to_numpy()
    length = df["word_count"].to_numpy()
    rho, p = spearmanr(length, ranks)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(df)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        r, _ = spearmanr(length[i], ranks[i])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(p), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = parser.parse_args()
    p = paths(args.lang)
    lang = args.lang

    # step 1b runs before the topic model. always read the table just produced
    # by step 1; an older with-topics file may still exist from a prior run.
    src = p["combined_csv"]
    if not src.exists():
        sys.exit(f"Run src/01_load_data.py --lang {lang} first.")
    df = pd.read_csv(src)
    df["n_sentences"] = df["text"].map(n_sentences)
    df["mean_word_len"] = df["text"].map(mean_word_len)
    df["ttr_200"] = df["text"].map(windowed_ttr)

    rows = []
    for ds, sub in df.groupby("dataset"):
        for level in LEVEL_ORDER:
            sl = sub[sub["cefr_level"] == level]
            if len(sl) == 0:
                continue
            row = {"dataset": ds, "cefr_level": level, **describe_slice(sl)}
            rows.append(row)
        rows.append({"dataset": ds, "cefr_level": "ALL", **describe_slice(sub)})
    rows.append({"dataset": "ALL", "cefr_level": "ALL", **describe_slice(df)})

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"descriptives_{lang}.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"Saved {out_path}")

    # corpus-by-level count matrix
    counts = (
        df.pivot_table(index="dataset", columns="cefr_level", values="text", aggfunc="count", fill_value=0)
        .reindex(columns=[l for l in LEVEL_ORDER if l in set(df["cefr_level"])])
    )
    counts.to_csv(RESULTS_DIR / f"corpus_level_counts_{lang}.csv")
    print("\nCorpus x level counts:")
    print(counts.to_string())

    rho, pval, lo, hi = spearman_length_level(df, seed=RANDOM_SEED)
    print(f"\nSpearman(word_count, CEFR rank): rho = {rho:.3f} [{lo:.3f}, {hi:.3f}], p = {pval:.2e}")
    (RESULTS_DIR / f"length_level_spearman_{lang}.txt").write_text(
        f"rho={rho:.4f}\nci=[{lo:.4f}, {hi:.4f}]\np={pval:.6e}\nn={len(df)}\n",
        encoding="utf-8",
    )

    # figure 5: length by level, per corpus
    set_style()
    corpora = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.2 * len(corpora), 4), sharey=True)
    if len(corpora) == 1:
        axes = [axes]
    levels = [l for l in LEVEL_ORDER if l in set(df["cefr_level"])]
    for ax, ds in zip(axes, corpora):
        sub = df[df["dataset"] == ds]
        data = [sub.loc[sub["cefr_level"] == l, "word_count"].to_numpy() for l in levels]
        ax.boxplot(data, tick_labels=levels, showfliers=False)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("CEFR")
    axes[0].set_ylabel("Tokens per document")
    fig.suptitle(f"Document length by CEFR level ({lang})")
    savefig(fig, RESULTS_DIR / f"fig_length_by_level_{lang}")

    # markdown table for quick paste
    md = out[out["cefr_level"] == "ALL"][
        ["dataset", "n", "tokens_median", "tokens_iqr", "sentences_median", "ttr_200_mean", "mean_word_len"]
    ].to_markdown(index=False, floatfmt=".2f")
    (RESULTS_DIR / f"descriptives_{lang}.md").write_text(md + "\n", encoding="utf-8")
    print(f"Saved out/descriptives_{lang}.md")


if __name__ == "__main__":
    main()
