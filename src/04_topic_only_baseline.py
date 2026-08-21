"""step 4: topic-only cefr prediction"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    LEVEL_ORDER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths,
    topic_outputs_problem,
)
from features import CheckedLogisticRegression, normalise_topic_weights, validation_groups
from plotting import savefig, set_style
from stats_utils import classification_metrics

N_SPLITS = 5


def ensure_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pair_id" not in df.columns or df["pair_id"].isna().all():
        df["pair_id"] = [f"{ds}__{i}" for i, ds in enumerate(df["dataset"])]
    df["pair_id"] = df["pair_id"].astype(str)
    return df


def evaluate(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf,
    grouped: bool = True,
) -> tuple[dict, np.ndarray]:
    if grouped:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = cv.split(X, y, groups)
        # older sklearn compatibility
        y_pred = np.empty_like(y)
        fold_metrics = []
        for train, test in split_iter:
            clf.fit(X[train], y[train])
            pred = clf.predict(X[test])
            y_pred[test] = pred
            m = classification_metrics(y[test], pred)
            fold_metrics.append(m)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        fold_metrics = []
        for train, test in cv.split(X, y):
            clf.fit(X[train], y[train])
            fold_metrics.append(classification_metrics(y[test], clf.predict(X[test])))

    row = classification_metrics(y, y_pred)
    row["model"] = name
    row["cv"] = "grouped" if grouped else "ungrouped"
    row["macro_f1_fold_sd"] = float(np.std([m["macro_f1"] for m in fold_metrics]))
    row["weighted_f1_fold_sd"] = float(np.std([m["weighted_f1"] for m in fold_metrics]))
    return row, y_pred


def bootstrap_metric_ci(y, y_pred, metric_key: str = "weighted_f1", n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(classification_metrics(y[i], y_pred[i])[metric_key])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_f1(X, y, groups, clf, observed_f1: float, n_perm: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        y_shuf = rng.permutation(y)
        row, _ = evaluate("perm", X, y_shuf, groups, clf, grouped=True)
        if row["macro_f1"] >= observed_f1 - 1e-15:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--skip-perm", action="store_true")
    parser.add_argument("--with-hgb", action="store_true",
                        help="also run HistGradientBoosting ceiling check")
    args = parser.parse_args()
    p = paths(args.lang)
    tag = args.lang if args.dataset is None else f"{args.lang}_{args.dataset}"
    set_style()
    RESULTS_DIR.mkdir(exist_ok=True)

    if problem := topic_outputs_problem(args.lang):
        sys.exit(problem)

    df = pd.read_csv(p["with_topics_csv"])
    doc_topic = np.load(p["doc_topic_matrix"])
    assert len(df) == doc_topic.shape[0], "table and topic matrix are out of sync; rerun step 2"
    df = ensure_pair_id(df)

    if args.dataset is not None:
        keep = df["dataset"] == args.dataset
        if not keep.any():
            sys.exit(f"No rows for dataset '{args.dataset}'. Available: {sorted(df['dataset'].unique())}")
        df, doc_topic = df[keep].reset_index(drop=True), doc_topic[keep.to_numpy()]
        print(f"Restricted to {args.dataset}: {len(df)} texts")

    # relative loadings remove loading-magnitude effects
    doc_topic = normalise_topic_weights(doc_topic)

    counts = df["cefr_level"].value_counts()
    rare = counts[counts < N_SPLITS].index.tolist()
    if rare:
        print(f"Dropping rare levels with fewer than {N_SPLITS} texts: {rare}")
        keep = ~df["cefr_level"].isin(rare)
        df, doc_topic = df[keep].reset_index(drop=True), doc_topic[keep.to_numpy()]

    y = df["cefr_level"].to_numpy()
    groups = validation_groups(df)
    support = df["cefr_level"].value_counts().reindex([l for l in LEVEL_ORDER if l in set(y)])
    print(f"{len(y)} texts. Class support:")
    print(support.to_string())
    support.to_csv(RESULTS_DIR / f"class_support_{tag}.csv")

    results = []

    # floor
    row, _ = evaluate(
        "majority class",
        np.zeros((len(y), 1)),
        y,
        groups,
        DummyClassifier(strategy="most_frequent"),
        grouped=True,
    )
    results.append(row)

    topic_onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(df[["topic"]])
    row, _ = evaluate(
        "topic id (one-hot)",
        topic_onehot,
        y,
        groups,
        make_pipeline(StandardScaler(), CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
        grouped=True,
    )
    results.append(row)

    # ungrouped (legacy) for leakage transparency
    lr = make_pipeline(StandardScaler(), CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED))
    row_ungrouped, _ = evaluate(
        "topic mixture",
        doc_topic,
        y,
        groups,
        lr,
        grouped=False,
    )
    results.append(row_ungrouped)

    row, y_pred_mix = evaluate(
        "topic mixture",
        doc_topic,
        y,
        groups,
        make_pipeline(StandardScaler(), CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
        grouped=True,
    )
    lo, hi = bootstrap_metric_ci(y, y_pred_mix, "weighted_f1", seed=RANDOM_SEED)
    row["weighted_f1_ci_low"] = lo
    row["weighted_f1_ci_high"] = hi
    results.append(row)

    row_cw, _ = evaluate(
        "topic mixture (class-weighted)",
        doc_topic,
        y,
        groups,
        make_pipeline(
            StandardScaler(),
            CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED, class_weight="balanced"),
        ),
        grouped=True,
    )
    results.append(row_cw)

    if args.with_hgb:
        from sklearn.ensemble import HistGradientBoostingClassifier

        row_hgb, _ = evaluate(
            "topic mixture (HistGB)",
            doc_topic,
            y,
            groups,
            HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=4),
            grouped=True,
        )
        results.append(row_hgb)

    if not args.skip_perm:
        print("Permutation test on topic-mixture macro F1 (200 shuffles) ...")
        p_emp = permutation_f1(
            doc_topic,
            y,
            groups,
            make_pipeline(StandardScaler(), CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
            row["macro_f1"],
            n_perm=200,
            seed=RANDOM_SEED,
        )
        mix_idx = next(
            i for i, r in enumerate(results)
            if r["model"] == "topic mixture" and r["cv"] == "grouped"
        )
        results[mix_idx]["macro_f1_perm_p"] = p_emp
        print(f"  empirical p = {p_emp:.4g}")

    res = pd.DataFrame(results)
    out_csv = RESULTS_DIR / f"baseline_results_{tag}.csv"
    res.to_csv(out_csv, index=False)
    print("\n" + res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved {out_csv}")

    pd.DataFrame({"y_true": y, "y_pred": y_pred_mix}).to_csv(
        RESULTS_DIR / f"oof_topic_mixture_{tag}.csv", index=False
    )

    # per-level report
    report = classification_report(y, y_pred_mix, output_dict=True, zero_division=0)
    per_level = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    per_level.to_csv(RESULTS_DIR / f"per_level_report_{tag}.csv", index=False)

    levels = [l for l in LEVEL_ORDER if l in set(y)]
    cm = confusion_matrix(y, y_pred_mix, labels=levels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(levels)), [f"{l} (n={int(support[l])})" for l in levels])
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("True level")
    ax.set_title(f"Topic mixture only ({tag}, grouped 5-fold CV)")
    for i in range(len(levels)):
        for j in range(len(levels)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    savefig(fig, RESULTS_DIR / f"confusion_topic_mixture_{tag}")

    print(
        "\nPrimary metric is weighted F1 (Imperial et al. 2025). "
        "Compare against the fine-tuned XLM-R weighted F1 for this language."
    )


if __name__ == "__main__":
    main()
