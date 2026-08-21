"""step 7: cross-corpus transfer and split stress test"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    LEVEL_ORDER, OPEN_HF_DATASETS, RANDOM_SEED, RESULTS_DIR, paths,
    topic_outputs_problem,
)
from features import CheckedLogisticRegression, normalise_topic_weights, validation_groups
from utils_dedup import topic_linked_components

N_SPLITS = 5
RANK = {l: i for i, l in enumerate(LEVEL_ORDER)}


def qwk(y_true, y_pred) -> float:
    t = [RANK[l] for l in y_true]
    p = [RANK[l] for l in y_pred]
    return float(cohen_kappa_score(t, p, weights="quadratic"))


def adjacent_accuracy(y_true, y_pred) -> float:
    t = np.array([RANK[l] for l in y_true])
    p = np.array([RANK[l] for l in y_pred])
    return float(np.mean(np.abs(t - p) <= 1))


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(np.mean(np.asarray(y_pred) == np.asarray(y_true))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
        "qwk": qwk(y_true, y_pred),
    }


def make_lr():
    return make_pipeline(
        StandardScaler(), CheckedLogisticRegression(max_iter=5000, random_state=RANDOM_SEED)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    ap.add_argument("--skip-tfidf", action="store_true",
                    help="skip the full-text TF-IDF comparison in experiment B")
    args = ap.parse_args()
    lang = args.lang
    p = paths(lang)

    if problem := topic_outputs_problem(lang):
        sys.exit(problem)
    df = pd.read_csv(p["with_topics_csv"]).reset_index(drop=True)
    doc_topic = normalise_topic_weights(np.load(p["doc_topic_matrix"]))
    assert len(df) == doc_topic.shape[0], "table and topic matrix out of sync"

    y = df["cefr_level"].to_numpy()
    corpora = sorted(df["dataset"].unique())
    out: dict = {"lang": lang, "n": int(len(df)), "corpora": corpora}

    # common label space across transfer cells
    support = pd.crosstab(df["dataset"], df["cefr_level"]).reindex(
        index=corpora, columns=LEVEL_ORDER, fill_value=0)
    common_levels = [level for level in LEVEL_ORDER
                     if level in support and (support[level] >= N_SPLITS).all()]
    if len(common_levels) < 2:
        sys.exit(f"Fewer than two common evaluation levels across corpora: {common_levels}")
    transfer_keep = df["cefr_level"].isin(common_levels).to_numpy()
    df_transfer = df[transfer_keep].reset_index(drop=True)
    topic_transfer = doc_topic[transfer_keep]
    y_transfer = df_transfer["cefr_level"].to_numpy()
    coverage_by_corpus = {
        c: float(np.mean(df.loc[df["dataset"] == c, "cefr_level"].isin(common_levels)))
        for c in corpora
    }
    out["evaluation_levels"] = common_levels
    out["evaluation_coverage_by_corpus"] = coverage_by_corpus

    print("=" * 78)
    print(f"CROSS-CORPUS TRANSFER AND TOPIC-STRATIFIED SPLITS ({lang})")
    print("=" * 78)

    # a. leave-one-corpus-out and the full transfer matrix
    print("\n" + "=" * 78)
    print("A. TRANSFER: does a topic-only model trained on one corpus work on another?")
    print("=" * 78)
    print("\nEach cell is weighted F1 (QWK in brackets) for a topic-only logistic")
    print("regression trained on the row corpus and tested on the column corpus.")
    print("The diagonal is 5-fold cross-validation inside that corpus.")
    print(f"All cells use the same levels: {', '.join(common_levels)}")
    print("Coverage of the original corpus: " + ", ".join(
        f"{c}={coverage_by_corpus[c]:.1%}" for c in corpora))

    rows = []
    for train_c in corpora:
        tr = (df_transfer["dataset"] == train_c).to_numpy()
        for test_c in corpora:
            te = (df_transfer["dataset"] == test_c).to_numpy()
            if train_c == test_c:
                # linked families kept together
                sub_y = y_transfer[tr]
                if len(sub_y) < N_SPLITS or len(np.unique(sub_y)) < 2:
                    continue
                groups = validation_groups(df_transfer.loc[tr])
                cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                          random_state=RANDOM_SEED)
                y_pred = cross_val_predict(make_lr(), topic_transfer[tr],
                                           sub_y, cv=cv, groups=groups)
                m = metrics(sub_y, y_pred)
                m.update({"train": train_c, "test": test_c, "setting": "in-corpus CV",
                          "n_train": int(len(sub_y)), "n_test": int(len(sub_y)),
                          "evaluation_levels": ",".join(common_levels),
                          "test_coverage": coverage_by_corpus[test_c]})
            else:
                clf = make_lr()
                clf.fit(topic_transfer[tr], y_transfer[tr])
                y_pred = clf.predict(topic_transfer[te])
                m = metrics(y_transfer[te], y_pred)
                m.update({"train": train_c, "test": test_c, "setting": "transfer",
                          "n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "evaluation_levels": ",".join(common_levels),
                          "test_coverage": coverage_by_corpus[test_c]})
            rows.append(m)

    transfer = pd.DataFrame(rows)
    transfer.to_csv(RESULTS_DIR / f"cross_corpus_transfer_{lang}.csv", index=False)

    print()
    corner = "train \\ test"
    print(f"{corner:<18}" + "".join(f"{c:>22}" for c in corpora))
    for train_c in corpora:
        line = f"{train_c:<18}"
        for test_c in corpora:
            sel = transfer[(transfer.train == train_c) & (transfer.test == test_c)]
            if sel.empty or pd.isna(sel["weighted_f1"].iloc[0]):
                line += f"{'--':>22}"
            else:
                wf = sel["weighted_f1"].iloc[0]
                k = sel["qwk"].iloc[0]
                mark = "*" if train_c == test_c else " "
                line += f"{f'{wf:.3f} ({k:+.2f}){mark}':>22}"
        print(line)
    print("\n* = in-corpus cross-validation. All other cells are transfer.")

    # leave-one-corpus-out transfer
    print("\n--- Leave-one-corpus-out (train on all other corpora) ---")
    loco_rows = []
    for c in corpora:
        te = (df_transfer["dataset"] == c).to_numpy()
        tr = ~te
        clf = make_lr()
        clf.fit(topic_transfer[tr], y_transfer[tr])
        m_out = metrics(y_transfer[te], clf.predict(topic_transfer[te]))

        # matched in-corpus reference
        sub_y = y_transfer[te]
        m_in = None
        if len(sub_y) >= N_SPLITS and len(np.unique(sub_y)) >= 2:
            groups = validation_groups(df_transfer.loc[te])
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=RANDOM_SEED)
            m_in = metrics(sub_y,
                           cross_val_predict(make_lr(), topic_transfer[te],
                                             sub_y, cv=cv, groups=groups))
        # test-corpus majority floor
        floor = metrics(sub_y, np.full(len(sub_y),
                                       pd.Series(sub_y).value_counts().idxmax()))
        loco_rows.append({
            "held_out_corpus": c, "n_test": int(te.sum()),
            "evaluation_levels": ",".join(common_levels),
            "test_coverage": coverage_by_corpus[c],
            "in_corpus_weighted_f1": m_in["weighted_f1"] if m_in else np.nan,
            "in_corpus_qwk": m_in["qwk"] if m_in else np.nan,
            "transfer_weighted_f1": m_out["weighted_f1"],
            "transfer_qwk": m_out["qwk"],
            "floor_weighted_f1": floor["weighted_f1"],
            "drop_weighted_f1": (m_in["weighted_f1"] - m_out["weighted_f1"]) if m_in else np.nan,
        })
        print(f"  {c:<18} n={te.sum():>5}  "
              f"in-corpus {m_in['weighted_f1']:.3f} (QWK {m_in['qwk']:+.2f})  ->  "
              f"transfer {m_out['weighted_f1']:.3f} (QWK {m_out['qwk']:+.2f})  "
              f"[floor {floor['weighted_f1']:.3f}]"
              if m_in else f"  {c}: in-corpus reference not computable")
    loco = pd.DataFrame(loco_rows)
    out["leave_one_corpus_out"] = loco.to_dict(orient="records")
    out["transfer_matrix"] = transfer.to_dict(orient="records")

    if not loco.empty:
        mean_in = loco["in_corpus_weighted_f1"].mean()
        mean_out = loco["transfer_weighted_f1"].mean()
        print(f"\n  mean in-corpus weighted F1 : {mean_in:.3f}")
        print(f"  mean transfer weighted F1  : {mean_out:.3f}")
        print(f"  mean drop                  : {mean_in - mean_out:+.3f}")
        print("\n  Interpretation. These transductive diagnostics use a common CEFR label")
        print("  space, so cells are comparable with respect to labels. The topic basis")
        print("  still saw every corpus during unsupervised fitting, so the transfer")
        print("  values are optimistic diagnostics rather than unseen-corpus estimates.")
        out["mean_in_corpus_weighted_f1"] = float(mean_in)
        out["mean_transfer_weighted_f1"] = float(mean_out)

    # b. topic-stratified splits
    print("\n" + "=" * 78)
    print("B. TOPIC-GROUPED STRESS TEST")
    print("=" * 78)
    print("\nRandom folds let the same topic appear in train and test; topic-grouped")
    print("folds forbid it. The split regimes also differ in class balance, corpus")
    print("mix and difficulty, so their difference is not an estimate of a pure")
    print("topic-overlap effect.")

    counts = df["cefr_level"].value_counts()
    keep = (~df["cefr_level"].isin(counts[counts < N_SPLITS].index)).to_numpy()
    dfk = df[keep].reset_index(drop=True)
    yk = y[keep]
    Xtopic = doc_topic[keep]
    validation = validation_groups(dfk)
    topic_components, n_topic_components, n_cross_topic_groups = (
        topic_linked_components(dfk["topic"].to_numpy(), validation)
    )
    stress_estimable = n_topic_components >= N_SPLITS
    print(f"Leakage-safe topic components: {n_topic_components}; "
          f"cross-topic validation families: {n_cross_topic_groups}")

    strat_rows = []

    def evaluate_splits(name, X, y_arr, groups_topic, groups_pair):
        rnd = cross_val_predict(
            make_lr(), X, y_arr,
            cv=StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=RANDOM_SEED),
            groups=groups_pair)
        grp = cross_val_predict(
            make_lr(), X, y_arr,
            cv=GroupKFold(n_splits=N_SPLITS), groups=groups_topic)
        m_r, m_g = metrics(y_arr, rnd), metrics(y_arr, grp)
        floor = metrics(y_arr, np.full(len(y_arr),
                                       pd.Series(y_arr).value_counts().idxmax()))
        strat_rows.append({
            "features": name,
            "status": "ok",
            "n_topic_components": n_topic_components,
            "n_cross_topic_groups": n_cross_topic_groups,
            "random_splits_weighted_f1": m_r["weighted_f1"],
            "topic_grouped_weighted_f1": m_g["weighted_f1"],
            "drop_weighted_f1": m_r["weighted_f1"] - m_g["weighted_f1"],
            "random_splits_macro_f1": m_r["macro_f1"],
            "topic_grouped_macro_f1": m_g["macro_f1"],
            "random_splits_qwk": m_r["qwk"],
            "topic_grouped_qwk": m_g["qwk"],
            "floor_weighted_f1": floor["weighted_f1"],
        })
        print(f"\n  {name}")
        print(f"    random  folds : weighted F1 {m_r['weighted_f1']:.3f}  "
              f"macro F1 {m_r['macro_f1']:.3f}  QWK {m_r['qwk']:+.3f}")
        print(f"    topic-grouped : weighted F1 {m_g['weighted_f1']:.3f}  "
              f"macro F1 {m_g['macro_f1']:.3f}  QWK {m_g['qwk']:+.3f}")
        print(f"    drop          : {m_r['weighted_f1'] - m_g['weighted_f1']:+.3f} "
              f"weighted F1   (floor {floor['weighted_f1']:.3f})")

    if stress_estimable:
        evaluate_splits("topic mixture", Xtopic, yk, topic_components, validation)
    else:
        status = (
            "not estimable: leakage-safe topic grouping leaves "
            f"{n_topic_components} component(s), fewer than {N_SPLITS} folds"
        )
        print(f"\n  {status}.")
        print("  No pooled stress-test scores are produced because a linked")
        print("  document family must never cross training and test folds.")
        strat_rows.append({
            "features": "all",
            "status": status,
            "n_topic_components": n_topic_components,
            "n_cross_topic_groups": n_cross_topic_groups,
        })

    if not args.skip_tfidf and stress_estimable:
        print("\n  Full-text TF-IDF stress test under the same two split regimes.")
        # fold-specific vocabulary and idf
        Xtf = dfk["text"].astype(str).tolist()
        clf_tf = make_pipeline(
            TfidfVectorizer(max_features=50_000, min_df=3, sublinear_tf=True),
            CheckedLogisticRegression(max_iter=3000, random_state=RANDOM_SEED))
        rnd = cross_val_predict(
            clf_tf, Xtf, yk,
            cv=StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=RANDOM_SEED),
            groups=validation)
        grp = cross_val_predict(clf_tf, Xtf, yk, cv=GroupKFold(n_splits=N_SPLITS),
                                groups=topic_components)
        m_r, m_g = metrics(yk, rnd), metrics(yk, grp)
        floor = metrics(yk, np.full(len(yk), pd.Series(yk).value_counts().idxmax()))
        strat_rows.append({
            "features": "full text (TF-IDF)",
            "status": "ok",
            "n_topic_components": n_topic_components,
            "n_cross_topic_groups": n_cross_topic_groups,
            "random_splits_weighted_f1": m_r["weighted_f1"],
            "topic_grouped_weighted_f1": m_g["weighted_f1"],
            "drop_weighted_f1": m_r["weighted_f1"] - m_g["weighted_f1"],
            "random_splits_macro_f1": m_r["macro_f1"],
            "topic_grouped_macro_f1": m_g["macro_f1"],
            "random_splits_qwk": m_r["qwk"],
            "topic_grouped_qwk": m_g["qwk"],
            "floor_weighted_f1": floor["weighted_f1"],
        })
        # full-corpus vocabulary reference
        _vocab = TfidfVectorizer(max_features=50_000, min_df=3,
                                 sublinear_tf=True).fit(Xtf)
        print(f"\n  full text (TF-IDF), vocabulary {len(_vocab.vocabulary_):,} "
              "on the full corpus; each fold refits its own")
        print(f"    random  folds : weighted F1 {m_r['weighted_f1']:.3f}  "
              f"macro F1 {m_r['macro_f1']:.3f}  QWK {m_r['qwk']:+.3f}")
        print(f"    topic-grouped : weighted F1 {m_g['weighted_f1']:.3f}  "
              f"macro F1 {m_g['macro_f1']:.3f}  QWK {m_g['qwk']:+.3f}")
        print(f"    drop          : {m_r['weighted_f1'] - m_g['weighted_f1']:+.3f} "
              f"weighted F1   (floor {floor['weighted_f1']:.3f})")
        print("\n  Interpretation caveat: topic-grouped folds also shift the level and")
        print("  corpus distributions. Positive or negative differences can therefore")
        print("  reflect split difficulty; report this only as a stress test.")

    strat = pd.DataFrame(strat_rows)
    strat.to_csv(RESULTS_DIR / f"topic_stratified_{lang}.csv", index=False)
    out["topic_stratified"] = strat.to_dict(orient="records")

    def _safe(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))

    (RESULTS_DIR / f"cross_corpus_{lang}.json").write_text(
        json.dumps(out, indent=2, default=_safe), encoding="utf-8")
    print(f"\nSaved out/cross_corpus_transfer_{lang}.csv")
    print(f"Saved out/topic_stratified_{lang}.csv")
    print(f"Saved out/cross_corpus_{lang}.json")


if __name__ == "__main__":
    main()
