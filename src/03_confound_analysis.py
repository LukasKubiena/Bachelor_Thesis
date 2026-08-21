"""Step 3 thin pass: topic summary table + labelled heatmaps.

Extended association statistics (bootstrap CIs, CMH, residuals, power) live in
src/03b_association_extended.py and write association_extended_*.{txt,json}.
This script produces the topic interpretability table used in the results.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LEVEL_ORDER, OPEN_HF_DATASETS, RESULTS_DIR, paths, topic_outputs_problem
from plotting import savefig, set_style, topic_axis_labels
from stats_utils import cramers_v

# checked against the highest-strength documents, not just the keywords.
# keeping the labels here makes the tables and figures use the same wording.
PROVISIONAL_LABELS_DE = {
    0: "Austrian regional Covid measures",
    1: "Informal greeting and invitation letters",
    2: "Formal letters seeking housing",
    3: "Austrian domestic politics",
    4: "Au-pair stays in Germany / learning German",
    5: "US politics",
    6: "Covid pandemic and health",
    7: "Informal notes about everyday arrangements",
    8: "Essays on home vs host-country culture",
    9: "Brexit",
    10: "Work, unemployment and housing",
    11: "Austrian survey and opinion reporting",
    12: "IT internship applications",
    13: "Austrian/European news (travel, economy, sport)",
    14: "Congratulations on a birth",
}


def plot_share_heatmap(crosstab: pd.DataFrame, title: str, path) -> None:
    shares = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * shares.shape[1]), 0.55 * shares.shape[0] + 2.2))
    im = ax.imshow(shares.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(shares.shape[1]), shares.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(shares.shape[0]), shares.index)
    ax.set_xlabel("Topic")
    ax.set_title(title)
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            val = shares.iat[i, j]
            if val >= 0.005:
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=6,
                )
    fig.colorbar(im, ax=ax, label="Share of row")
    savefig(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = parser.parse_args()
    p = paths(args.lang)
    lang = args.lang
    set_style()
    RESULTS_DIR.mkdir(exist_ok=True)

    if problem := topic_outputs_problem(lang):
        sys.exit(problem)
    df = pd.read_csv(p["with_topics_csv"])
    levels = [l for l in LEVEL_ORDER if l in set(df["cefr_level"])]
    top_words = pd.read_csv(p["topic_words_csv"]).set_index("topic")["top_words"]
    labels_map = PROVISIONAL_LABELS_DE if lang == "de" else {}
    short_label = topic_axis_labels(top_words.to_dict(), labels_map)

    ct_level = pd.crosstab(df["cefr_level"], df["topic"]).reindex(levels)
    ct_level.to_csv(RESULTS_DIR / f"crosstab_topic_level_{lang}.csv")
    plot_share_heatmap(
        ct_level.rename(columns=short_label),
        f"Where each CEFR level's texts go, by topic ({lang})",
        RESULTS_DIR / f"heatmap_level_topic_{lang}",
    )
    ct_source = pd.crosstab(df["dataset"], df["topic"])
    plot_share_heatmap(
        ct_source.rename(columns=short_label),
        f"Where each source dataset's texts go, by topic ({lang})",
        RESULTS_DIR / f"heatmap_source_topic_{lang}",
    )

    v, pval = cramers_v(ct_level)
    v_src, p_src = cramers_v(ct_source)
    lines = [
        f"=== Topic vs CEFR level (point estimates; see 03b for CIs) ===",
        f"Cramér's V: {v:.3f}  chi2 p: {pval:.2e}",
        f"Topic vs source V: {v_src:.3f}  p: {p_src:.2e}",
        "",
        "=== Within-source point estimates ===",
    ]
    for name, sub in df.groupby("dataset"):
        if sub["cefr_level"].nunique() < 2 or sub["topic"].nunique() < 2:
            continue
        vv, pp = cramers_v(pd.crosstab(sub["cefr_level"], sub["topic"]))
        lines.append(f"{name} (n={len(sub)}): V={vv:.3f}, p={pp:.2e}")

    shares = ct_level.div(ct_level.sum(axis=0), axis=1)
    dataset_share = ct_source.div(ct_source.sum(axis=0), axis=1)
    summary = pd.DataFrame({
        "topic": list(ct_level.columns),
        "top_words": [top_words.get(t, "") for t in ct_level.columns],
        "interpreted_label": [labels_map.get(int(t), "") for t in ct_level.columns],
        "n_texts": [int(ct_level[t].sum()) for t in ct_level.columns],
        "dominant_level": [shares[t].idxmax() for t in ct_level.columns],
        "dominant_level_share": [round(float(shares[t].max()), 2) for t in ct_level.columns],
        "dominant_source": [dataset_share[t].idxmax() for t in ct_level.columns],
        "dominant_source_share": [round(float(dataset_share[t].max()), 2) for t in ct_level.columns],
    })
    summary.to_csv(RESULTS_DIR / f"topic_summary_{lang}.csv", index=False)
    lines.append(f"\nSaved topic_summary_{lang}.csv")
    for _, row in summary.iterrows():
        lines.append(
            f"Topic {int(row['topic']):>2} [{row['interpreted_label']}] "
            f"(n={row['n_texts']}, {row['dominant_level']} "
            f"{row['dominant_level_share']:.0%}): {str(row['top_words'])[:70]}"
        )

    report = "\n".join(lines)
    (RESULTS_DIR / f"association_stats_{lang}.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
