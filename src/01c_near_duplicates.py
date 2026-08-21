"""step 1c: near-duplicate audit"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import OPEN_HF_DATASETS, RESULTS_DIR, paths
from utils_dedup import (
    bipartite_max_similarity,
    merged_validation_groups,
    near_duplicate_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    p = paths(args.lang)

    # clean step 1 input
    src = p["combined_csv"]
    if not src.exists():
        sys.exit(f"Run src/01_load_data.py --lang {args.lang} first.")
    df = pd.read_csv(src)

    print(f"Scanning {len(df)} texts for near-duplicates (threshold={args.threshold}) ...")
    report = near_duplicate_report(df, threshold=args.threshold, top_k=20)
    cols = ["i", "j", "similarity", "dataset_i", "dataset_j", "level_i", "level_j"]
    extra = [c for c in report.columns if c not in cols]
    ordered = [c for c in cols if c in report.columns] + extra
    out = RESULTS_DIR / f"near_duplicates_{args.lang}.csv"
    report[ordered].to_csv(out, index=False) if len(report) else pd.DataFrame(columns=cols).to_csv(out, index=False)

    # merged validation families
    df["cv_group"] = merged_validation_groups(df, report)
    p["combined_csv"].write_text(df.to_csv(index=False), encoding="utf-8")
    n_cv_groups = int(df["cv_group"].nunique())
    n_multirow_groups = int((df.groupby("cv_group").size() > 1).sum())
    print(f"Validation groups: {n_cv_groups:,}; "
          f"multi-row families: {n_multirow_groups:,}")

    n_within = int(report["same_corpus"].sum()) if len(report) else 0
    n_cross = int((~report["same_corpus"]).sum()) if len(report) else 0
    print(f"Near-duplicate pairs: {len(report)} total")
    print(f"  within-corpus: {n_within} (expected: parallel pairs)")
    print(f"  cross-corpus:  {n_cross} (problem if non-trivial)")

    n_deplain = int((df["dataset"] == "deplain_apa_doc").sum())
    n_apa = int((df["dataset"] == "apa_lha").sum())
    n_deplain_apa = 0
    n_label_mismatch = 0
    share: float | str = (0.0 if n_deplain else "NA")
    if len(report):
        print("\nWithin-corpus pairs by dataset:")
        print(report[report["same_corpus"]].groupby("dataset_i").size().to_string()
              if n_within else "  none")
    if n_cross:
        print("\nCross-corpus pairs by dataset pair:")
        cross = report[~report["same_corpus"]].copy()
        cross["pair"] = cross.apply(
            lambda r: " | ".join(sorted([str(r["dataset_i"]), str(r["dataset_j"])])), axis=1
        )
        print(cross["pair"].value_counts().to_string())
        mask = (
            ((cross["dataset_i"] == "deplain_apa_doc") & (cross["dataset_j"] == "apa_lha"))
            | ((cross["dataset_i"] == "apa_lha") & (cross["dataset_j"] == "deplain_apa_doc"))
        )
        overlap = cross[mask]
        n_deplain_apa = int(len(overlap))
        n_label_mismatch = int((overlap["level_i"] != overlap["level_j"]).sum())
        share = (n_deplain_apa / n_deplain) if n_deplain else "NA"
        print(f"\nDEplain-APA vs APA-LHA pairs ≥ {args.threshold}: {n_deplain_apa}")
        print(f"  share of DEplain's {n_deplain} documents: {share:.3f}"
              if n_deplain else "  DEplain not in this language")
        print(f"  of which different CEFR labels: {n_label_mismatch}")
        if n_label_mismatch:
            print("  WARNING: same (or near-same) article labelled inconsistently "
                  "across the two corpora.")
        print("Do not delete anything yet. These numbers decide whether to deduplicate.")
    elif n_deplain and n_apa:
        print("\nDEplain-APA vs APA-LHA pairs ≥ "
              f"{args.threshold}: 0  (share of DEplain's {n_deplain} documents: 0.000)")
        print("No cross-corpus near-duplicates at this threshold.")

    # nearest deplain–apa-lha matches
    bip_max = ""
    bip_n85 = 0
    bip_mismatch = 0
    if n_deplain and n_apa:
        print("\nFull bipartite cosine, every DEplain doc vs nearest APA-LHA ...")
        bip = bipartite_max_similarity(df, "deplain_apa_doc", "apa_lha")
        if len(bip):
            bip.to_csv(RESULTS_DIR / f"near_duplicates_deplain_vs_apa_lha_{args.lang}.csv",
                       index=False)
            sims = bip["similarity"]
            bip_max = f"{float(sims.max()):.4f}"
            for t in (0.5, 0.7, 0.8, 0.85, 0.9):
                n = int((sims >= t).sum())
                print(f"  ≥ {t:.2f}: {n:4d}  ({n / n_deplain:.3f} of DEplain)")
            bip_n85 = int((sims >= args.threshold).sum())
            hi = bip[sims >= args.threshold]
            bip_mismatch = int((hi["level_i"] != hi["level_j"]).sum()) if len(hi) else 0
            print(f"  max similarity = {bip_max}")
            print(f"  pairs ≥ {args.threshold}: {bip_n85}; label mismatches: {bip_mismatch}")
            n_deplain_apa = bip_n85
            n_label_mismatch = bip_mismatch
            share = bip_n85 / n_deplain
    print(f"Saved {out}")

    summary = (
        f"near_duplicate_pairs={len(report)}\n"
        f"within_corpus={n_within}\n"
        f"cross_corpus={n_cross}\n"
        f"threshold={args.threshold}\n"
        f"n_deplain_apa_doc={n_deplain}\n"
        f"n_apa_lha={n_apa}\n"
        f"cross_corpus_deplain_apa_vs_apa_lha={n_deplain_apa}\n"
        f"share_of_deplain={share if n_deplain else 'NA'}\n"
        f"deplain_apa_label_mismatch={n_label_mismatch}\n"
        f"bipartite_max_similarity={bip_max or 'NA'}\n"
        f"bipartite_n_ge_threshold={bip_n85}\n"
        f"bipartite_label_mismatch={bip_mismatch}\n"
        f"cv_groups={n_cv_groups}\n"
        f"multirow_cv_groups={n_multirow_groups}\n"
    )
    (RESULTS_DIR / f"near_duplicates_summary_{args.lang}.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
