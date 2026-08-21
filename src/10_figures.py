"""Step 10: regenerate thesis figures from files in out/.

No model is refitted here. If an input file is missing, that figure is skipped
with a warning so a partial run is still useful.

Run:
    python src/10_figures.py
    python src/10_figures.py --lang en
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from config import (
    LEVEL_ORDER,
    MOST_FREQUENT_WEIGHTED_F1,
    OPEN_HF_DATASETS,
    RESULTS_DIR,
    XLMR_WEIGHTED_F1,
    paths,
)
from plotting import OKABE_ITO, savefig, set_style, topic_axis_labels

CONTROL_DATASETS = {"deplain_apa_doc", "apa_lha"}
CI_RE = re.compile(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]")


def _exists(path: Path, label: str) -> bool:
    if path.exists():
        return True
    print(f"  SKIP {label}: missing {path.name}")
    return False


def _topic_labels(lang: str) -> dict:
    p = paths(lang)
    interpreted = {}
    summary = RESULTS_DIR / f"topic_summary_{lang}.csv"
    if summary.exists():
        s = pd.read_csv(summary)
        if "interpreted_label" in s.columns:
            interpreted = {
                int(r.topic): str(r.interpreted_label).strip()
                for r in s.itertuples()
                if isinstance(r.interpreted_label, str) and r.interpreted_label.strip()
            }
    top_words = {}
    if p["topic_words_csv"].exists():
        w = pd.read_csv(p["topic_words_csv"])
        top_words = {int(r.topic): r.top_words for r in w.itertuples()}
    return topic_axis_labels(top_words, interpreted or None)


def _parse_ci(text) -> tuple[float, float] | None:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    m = CI_RE.search(str(text))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def fig_cramers_v_forest(lang: str) -> None:
    src = RESULTS_DIR / f"cramers_v_by_corpus_{lang}.csv"
    if not _exists(src, "Cramér's V forest"):
        return
    df = pd.read_csv(src).sort_values("v", ascending=True)
    pooled = None
    assoc = RESULTS_DIR / f"association_extended_{lang}.json"
    if assoc.exists():
        pooled = json.loads(assoc.read_text(encoding="utf-8")).get("overall", {}).get("v")

    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(df) + 1.6))
    y = np.arange(len(df))
    for i, r in enumerate(df.itertuples()):
        is_control = r.corpus in CONTROL_DATASETS
        color = OKABE_ITO[3] if is_control else OKABE_ITO[0]
        marker = "s" if is_control else "o"
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=color, lw=1.6, solid_capstyle="round")
        ax.plot(r.v, i, marker=marker, color=color, ms=7, zorder=3)
        tag = "  (control)" if is_control else ""
        ax.text(1.02, i, f"n = {int(r.n):,}{tag}", va="center", ha="left",
                transform=ax.get_yaxis_transform(), fontsize=8, color="0.3")
    if pooled is not None:
        ax.axvline(pooled, color="0.45", ls="--", lw=1, zorder=0)
        ax.text(pooled, len(df) - 0.35, f" pooled V = {pooled:.3f}",
                fontsize=8, color="0.35", rotation=90, va="top", ha="right")
    ax.set_yticks(y, df["corpus"].tolist())
    ax.set_xlabel("Cramér's V (bias-corrected) with 95% CI")
    ax.set_xlim(left=0)
    ax.set_title(f"Topic–level association by corpus ({lang})")
    savefig(fig, RESULTS_DIR / f"fig_cramers_v_by_corpus_{lang}")


def fig_residual_heatmap(lang: str, merlin: bool = False) -> None:
    tag = f"merlin_{lang}" if merlin else lang
    src = RESULTS_DIR / f"residuals_level_topic_{tag}.csv"
    label = "MERLIN residual heatmap" if merlin else "residual heatmap"
    if not _exists(src, label):
        return
    resid = pd.read_csv(src, index_col=0)
    resid = resid.reindex([l for l in LEVEL_ORDER if l in resid.index])
    topics = [int(c) for c in resid.columns]
    labels_map = _topic_labels(lang)

    # interpreted labels can run to 80 characters. rotated, those consume so
    # much vertical space that tight_layout collapses the axes into a thin
    # strip and the figure becomes unreadable. truncate on a word boundary and
    # size the canvas from the matrix shape plus a fixed allowance for labels.
    def _short(s: str, n: int = 30) -> str:
        s = str(s)
        if len(s) <= n:
            return s
        cut = s[:n].rsplit(" ", 1)[0]
        return (cut if len(cut) > n * 0.6 else s[:n]).rstrip(" ,;/") + "…"

    xticklabels = [_short(labels_map.get(t, str(t))) for t in topics]

    n_rows, n_cols = resid.shape
    width = min(13.0, max(8.5, 0.8 * n_cols + 3.0))
    height = 0.42 * n_rows + 3.4          # 3.4in reserved for rotated labels
    vmax = max(3.0, float(np.nanmax(np.abs(resid.to_numpy()))))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(resid.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(topics)), xticklabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(resid.index)), resid.index.tolist(), fontsize=9)
    ax.set_xlabel("Topic")
    ax.set_ylabel("CEFR level")
    title = f"Standardised residuals, topic × level ({'MERLIN only' if merlin else lang})"
    ax.set_title(title)
    for i in range(resid.shape[0]):
        for j in range(resid.shape[1]):
            val = float(resid.iloc[i, j])
            if abs(val) > 3:
                ax.plot(j, i, marker="*", color="0.1", ms=7)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec="0.15", lw=0.8))
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("standardised residual", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    # the "star marks |residual| > 3" legend belongs in the apa figure note
    # below the figure, not inside the axes, where it collides with the first
    # rotated tick label.
    out = RESULTS_DIR / (f"fig_residuals_merlin_{lang}" if merlin else f"fig_residuals_{lang}")
    savefig(fig, out)


def _grouped_row(js: dict, features: str) -> dict | None:
    for r in js.get("results", []):
        if r.get("cv", "").startswith("grouped") and r.get("features") == features:
            return r
    return None


def _ci_for(js: dict, features: str) -> tuple[float, float] | None:
    for r in js.get("bootstrap_ci", []):
        if r.get("features") == features:
            return _parse_ci(r.get("weighted_f1_ci"))
    return None


def fig_length_vs_topic_bars() -> None:
    pooled_p = RESULTS_DIR / "length_benchmark_de.json"
    merlin_p = RESULTS_DIR / "length_benchmark_de_merlin_de.json"
    if not _exists(pooled_p, "length vs topic bars") or not _exists(merlin_p, "length vs topic bars"):
        return
    pooled = json.loads(pooled_p.read_text(encoding="utf-8"))
    merlin = json.loads(merlin_p.read_text(encoding="utf-8"))
    specs = [
        ("majority class (floor)", "majority floor"),
        ("word count only", "word count only"),
        ("surface features", "surface features"),
        ("topic only", "topic only"),
        ("surface + topic", "surface + topic"),
    ]
    colors = [OKABE_ITO[7], OKABE_ITO[3], OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2]]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, js, title in (
        (axes[0], pooled, "Pooled German"),
        (axes[1], merlin, "MERLIN only"),
    ):
        xs = np.arange(len(specs))
        vals, yerr = [], []
        for key, _ in specs:
            row = _grouped_row(js, key)
            v = 100.0 * float(row["weighted_f1"]) if row else np.nan
            vals.append(v)
            ci = _ci_for(js, key)
            if ci is None or not np.isfinite(v):
                yerr.append((0.0, 0.0))
            else:
                lo, hi = 100.0 * ci[0], 100.0 * ci[1]
                yerr.append((max(0.0, v - lo), max(0.0, hi - v)))
        err = np.array(yerr).T
        ax.bar(xs, vals, color=colors, width=0.72, yerr=err, capsize=3,
               error_kw={"ecolor": "0.25", "elinewidth": 0.9})
        floor = MOST_FREQUENT_WEIGHTED_F1.get("de")
        ceiling = XLMR_WEIGHTED_F1.get("de")
        if floor is not None:
            ax.axhline(floor, color="0.45", ls=":", lw=1)
            ax.text(len(specs) - 0.05, floor + 0.8, f"Imperial most-freq. {floor:.1f}",
                    ha="right", va="bottom", fontsize=7, color="0.35")
        if ceiling is not None:
            ax.axhline(ceiling, color="0.45", ls="--", lw=1)
            ax.text(len(specs) - 0.05, ceiling - 1.2, f"Imperial XLM-R {ceiling:.1f}",
                    ha="right", va="top", fontsize=7, color="0.35")
        ax.set_xticks(xs, [lab for _, lab in specs], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 85)
    axes[0].set_ylabel("weighted F1")
    fig.suptitle("What predicts CEFR level: length, topic, or both")
    fig.text(0.5, 0.01,
             "Dashed/dotted lines are Imperial et al. (2025) German numbers from a different evaluation split.",
             ha="center", fontsize=7.5, color="0.4")
    savefig(fig, RESULTS_DIR / "fig_length_vs_topic_pooled_merlin_de")


def fig_transfer_matrix(lang: str) -> None:
    src = RESULTS_DIR / f"cross_corpus_transfer_{lang}.csv"
    if not _exists(src, "transfer matrix"):
        return
    df = pd.read_csv(src)
    levels = str(df["evaluation_levels"].iloc[0]).replace(",", ", ") \
        if "evaluation_levels" in df else "not recorded"
    corpora = list(dict.fromkeys(list(df["train"]) + list(df["test"])))
    mat = pd.DataFrame(np.nan, index=corpora, columns=corpora)
    setting = pd.DataFrame("", index=corpora, columns=corpora)
    for r in df.itertuples():
        mat.loc[r.train, r.test] = r.weighted_f1
        setting.loc[r.train, r.test] = r.setting
    fig, ax = plt.subplots(figsize=(0.9 * len(corpora) + 4.2, 0.7 * len(corpora) + 3.4))
    im = ax.imshow(mat.to_numpy(), cmap="viridis", vmin=0, vmax=max(0.4, float(np.nanmax(mat.to_numpy()))))
    if "test_coverage" in df:
        coverage = df.groupby("test")["test_coverage"].first().to_dict()
        test_labels = [f"{c}\n({coverage[c]:.0%} retained)" for c in corpora]
    else:
        test_labels = corpora
    ax.set_xticks(range(len(corpora)), test_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(corpora)), corpora)
    ax.set_xlabel("Test corpus")
    ax.set_ylabel("Train corpus")
    ax.set_title(
        f"Transductive topic-mixture transfer ({lang}; common levels: {levels})")
    for i, tr in enumerate(corpora):
        for j, te in enumerate(corpora):
            val = mat.loc[tr, te]
            if not np.isfinite(val):
                continue
            txt = f"{val:.3f}"
            if setting.loc[tr, te] == "in-corpus CV":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec="white", lw=1.6))
                txt += "\nCV"
            color = "white" if val > 0.45 * np.nanmax(mat.to_numpy()) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("weighted F1")
    savefig(fig, RESULTS_DIR / f"fig_transfer_matrix_{lang}")


def fig_confusion(lang: str) -> None:
    oof_06 = RESULTS_DIR / f"oof_length_benchmark_{lang}.csv"
    oof_04 = RESULTS_DIR / f"oof_topic_mixture_{lang}.csv"
    support_p = RESULTS_DIR / f"class_support_{lang}.csv"
    if oof_06.exists():
        raw = pd.read_csv(oof_06)
        if "features" in raw.columns:
            raw = raw[raw["features"] == "topic only"]
        y_true, y_pred = raw["y_true"].to_numpy(), raw["y_pred"].to_numpy()
        source = "06 grouped topic-only"
    elif oof_04.exists():
        raw = pd.read_csv(oof_04)
        y_true, y_pred = raw["y_true"].to_numpy(), raw["y_pred"].to_numpy()
        source = "04 grouped topic-mixture"
    else:
        print("  SKIP confusion matrix: no OOF predictions saved yet")
        return
    levels = [l for l in LEVEL_ORDER if l in set(y_true) or l in set(y_pred)]
    if support_p.exists():
        sup = pd.read_csv(support_p, index_col=0).squeeze("columns")
        if isinstance(sup, pd.DataFrame):
            sup = sup.iloc[:, 0]
    else:
        sup = pd.Series(y_true).value_counts()
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=levels)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(
        range(len(levels)),
        [f"{l} (n={int(sup.get(l, (np.asarray(y_true) == l).sum()))})" for l in levels],
    )
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("True level")
    ax.set_title(f"Topic mixture only ({lang}, {source})")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(levels)):
        for j in range(len(levels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, RESULTS_DIR / f"confusion_topic_mixture_{lang}")


def fig_topic_overlap(lang: str) -> None:
    files = sorted(RESULTS_DIR.glob(f"topic_overlap_within_{lang}_*.csv"))
    if not files:
        print(f"  SKIP topic-overlap figure: no topic_overlap_within_{lang}_*.csv")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    feature_order = ["topic mixture", "full text (TF-IDF)", "word count only",
                     "surface features"]
    features = [f for f in feature_order if f in set(df["features"])]
    corpora = list(dict.fromkeys(df["corpus"]))
    # put learner / reference next to each other when present.
    preferred = [c for c in ["merlin_de", "elg_cefr_de", "icle500_en", "elg_cefr_en"]
                 if c in corpora]
    corpora = preferred + [c for c in corpora if c not in preferred]

    fig, axes = plt.subplots(1, len(features), figsize=(4.1 * len(features), 4.2), sharey=True)
    if len(features) == 1:
        axes = [axes]
    x = np.arange(len(corpora))
    width = 0.36
    for ax, feat in zip(axes, features):
        sub = df[df["features"] == feat].set_index("corpus")
        rnd = [100 * float(sub.loc[c, "random_weighted_f1"]) if c in sub.index else np.nan
               for c in corpora]
        grp = [100 * float(sub.loc[c, "topic_grouped_weighted_f1"]) if c in sub.index else np.nan
               for c in corpora]
        ax.bar(x - width / 2, rnd, width, label="random folds", color=OKABE_ITO[0])
        ax.bar(x + width / 2, grp, width, label="topic-grouped", color=OKABE_ITO[1])
        ax.set_xticks(x, corpora, rotation=30, ha="right")
        ax.set_title(feat)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("weighted F1")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Within-corpus topic-grouped stress test ({lang})")
    savefig(fig, RESULTS_DIR / f"fig_topic_overlap_within_{lang}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = ap.parse_args()
    set_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    lang = args.lang
    print(f"=== figures ({lang}) ===")
    fig_cramers_v_forest(lang)
    fig_residual_heatmap(lang, merlin=False)
    if lang == "de":
        fig_residual_heatmap(lang, merlin=True)
        fig_length_vs_topic_bars()
        fig_confusion(lang)
    fig_transfer_matrix(lang)
    fig_topic_overlap(lang)
    print("Done.")


if __name__ == "__main__":
    main()
