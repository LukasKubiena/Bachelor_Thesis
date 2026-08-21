"""Step 1: Build the combined CEFR table for one language.

Downloads the openly available UniversalCEFR datasets for the chosen language
from HuggingFace, merges any gated corpora you have placed in
in/external/<lang>/ (UniversalCEFR JSON schema), normalizes the labels, and
writes one combined CSV.

Run:
    python src/01_load_data.py             # German (default)
    python src/01_load_data.py --lang en   # English

Output:
    in/texts_<lang>.csv   one row per text with columns:
        text, cefr_level, source_name, dataset, format, category, word_count
"""

import argparse
import hashlib
import json
import sys

import pandas as pd
from datasets import load_dataset

from config import (
    DATA_DIR,
    LEVEL_ORDER,
    MIN_TOKENS,
    OPEN_HF_DATASETS,
    RESULTS_DIR,
    external_dir,
    normalize_level,
    paths,
)

# columns of the universalcefr schema we keep.
KEEP_COLS = ["text", "cefr_level", "source_name", "format", "category", "pair_id", "title"]


def load_open_datasets(lang: str) -> list:
    """Pull the open datasets for this language from the HuggingFace hub."""
    frames = []
    for name in OPEN_HF_DATASETS[lang]:
        print(f"Downloading {name} ...")
        ds = load_dataset(name)
        # datasets may have one or more splits; concatenate them all.
        parts = [ds[split].to_pandas() for split in ds.keys()]
        df = pd.concat(parts, ignore_index=True)
        df["dataset"] = name.split("/")[-1]
        frames.append(df)
        print(f"  -> {len(df)} rows")
    return frames


def load_external_files(lang: str) -> list:
    """Merge gated corpora if you have added them.

    Any .json or .jsonl file in in/external/<lang>/ that follows the
    UniversalCEFR schema is picked up automatically.
    """
    frames = []
    ext = external_dir(lang)
    if not ext.exists():
        return frames
    for path in sorted(ext.glob("*.json*")):
        print(f"Loading external file {path.name} ...")
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            records = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(records)
        df["dataset"] = path.stem
        frames.append(df)
        print(f"  -> {len(df)} rows")
    return frames


def assign_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every row has a pair_id.

    Parallel corpora (DEplain, APA-LHA) already carry article-level pair_ids
    from the converter. Learner/reference corpora get a unique per-document id
    so StratifiedGroupKFold degrades gracefully to ordinary stratified CV.
    """
    df = df.copy()
    if "pair_id" not in df.columns:
        df["pair_id"] = pd.NA
    missing = df["pair_id"].isna() | (df["pair_id"].astype(str).str.strip() == "") | (
        df["pair_id"].astype(str).str.lower() == "nan"
    )
    # unique ids for unpaired rows (merlin, elg, and any leftover).
    for i in df.index[missing]:
        df.at[i, "pair_id"] = f"{df.at[i, 'dataset']}__{i}"
    df["pair_id"] = df["pair_id"].astype(str)
    return df


def exact_duplicate_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise exact duplicates before pooled de-duplication.

    Text itself is not written to the audit because some input corpora cannot
    be redistributed. A SHA-256 digest is sufficient to identify and trace a
    duplicate locally without copying licensed text into ``out/``.
    """
    duplicated = df[df.duplicated(subset=["text"], keep=False)]
    rows = []
    for text, group in duplicated.groupby("text", sort=False):
        rows.append({
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "n_rows": int(len(group)),
            "n_corpora": int(group["dataset"].nunique()),
            "n_levels": int(group["cefr_level"].nunique()),
            "datasets": "|".join(sorted(group["dataset"].astype(str).unique())),
            "levels": "|".join(sorted(group["cefr_level"].astype(str).unique())),
            "action": (
                "exclude_all_conflicting_rows"
                if group["dataset"].nunique() > 1 or group["cefr_level"].nunique() > 1
                else "keep_first_redundant_copy"
            ),
        })
    return pd.DataFrame(rows, columns=[
        "text_sha256", "n_rows", "n_corpora", "n_levels", "datasets", "levels",
        "action",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = parser.parse_args()
    p = paths(args.lang)

    DATA_DIR.mkdir(exist_ok=True)
    external_dir(args.lang).mkdir(parents=True, exist_ok=True)

    frames = load_open_datasets(args.lang) + load_external_files(args.lang)
    df = pd.concat(frames, ignore_index=True)

    # keep only the schema columns that exist plus our dataset tag.
    cols = [c for c in KEEP_COLS if c in df.columns] + ["dataset"]
    df = df[cols].copy()

    n_start = len(df)

    # i log every filter separately. the paper reports a final n, and a reader
    # should be able to see how it was arrived at rather than take it on trust.
    raw_labels = df["cefr_level"].copy()
    df["cefr_level"] = raw_labels.map(normalize_level)
    n_sublevel = int(
        raw_labels.astype(str).str.strip().str.upper()
        .str.contains(r"(?:\+|\.\d)$", regex=True, na=False).sum()
    )
    n_bad_label = int(df["cefr_level"].isna().sum())
    df = df.dropna(subset=["cefr_level", "text"])

    df["text"] = df["text"].astype(str).str.strip()
    df["word_count"] = df["text"].str.split().str.len()
    n_short = int((df["word_count"] < MIN_TOKENS).sum())
    df = df[df["word_count"] >= MIN_TOKENS]

    n_before_dedup = len(df)
    duplicate_audit = exact_duplicate_audit(df)
    RESULTS_DIR.mkdir(exist_ok=True)
    duplicate_audit.to_csv(
        RESULTS_DIR / f"exact_duplicates_preclean_{args.lang}.csv", index=False)
    conflicting_texts = set(
        df.groupby("text").filter(
            lambda group: group["dataset"].nunique() > 1
            or group["cefr_level"].nunique() > 1
        )["text"]
    )
    conflict_mask = df["text"].isin(conflicting_texts)
    n_conflicting_rows = int(conflict_mask.sum())
    df = df[~conflict_mask].drop_duplicates(
        subset=["dataset", "cefr_level", "text"])
    n_redundant = n_before_dedup - n_conflicting_rows - len(df)
    n_dupes = n_before_dedup - len(df)

    df = df.reset_index(drop=True)
    df = assign_pair_ids(df)

    print("\nCleaning, step by step:")
    print(f"  rows loaded                                  {n_start:>7,}")
    print(f"  plus/sub-level labels mapped to base level   {n_sublevel:>7,}")
    print(f"  dropped, label not one of the six levels     {n_bad_label:>7,}")
    print(f"  dropped, fewer than {MIN_TOKENS} tokens              {n_short:>7,}")
    print(f"  dropped, exact duplicate text                {n_dupes:>7,}")
    print(f"    conflicting corpus/level rows (all copies) {n_conflicting_rows:>7,}")
    print(f"    redundant same-label copies (keep first)   {n_redundant:>7,}")
    print(f"  kept                                         {len(df):>7,}")
    n_cross = int((duplicate_audit["n_corpora"] > 1).sum())
    print(f"  duplicate text groups spanning corpora          {n_cross:>7,}")
    print("  NOTE: all copies of a text spanning corpora or labels are excluded;")
    print("  only redundant copies within one corpus/label keep their first row.")
    print("  01c separately checks near-duplicate articles.")
    print(f"  audit: out/exact_duplicates_preclean_{args.lang}.csv")

    # acceptance check for parallel structure.
    sizes = df.groupby("pair_id").size()
    print("\npair_id group-size counts (expect ~483 size-2 for deplain):")
    print(sizes.value_counts().sort_index().to_string())
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        gs = sub.groupby("pair_id").size().value_counts().sort_index()
        print(f"  {ds}: {dict(gs)}")

    df.to_csv(p["combined_csv"], index=False)
    print(f"Saved {p['combined_csv']}")

    # ------------------------------------------------------------------
    # summary: this table is worth eyeballing before any modelling.
    # ------------------------------------------------------------------
    print("\nTexts per dataset and CEFR level:")
    summary = (
        df.pivot_table(index="dataset", columns="cefr_level", values="text", aggfunc="count", fill_value=0)
        .reindex(columns=[l for l in LEVEL_ORDER if l in df["cefr_level"].unique()])
    )
    print(summary.to_string())

    print("\nMedian word count per level (the known length signal):")
    print(df.groupby("cefr_level")["word_count"].median().reindex(LEVEL_ORDER).dropna().to_string())

    if not any(external_dir(args.lang).glob("*.json*")):
        print(
            f"\nNOTE: no gated corpora found in in/external/{args.lang}/. "
            "See README for which gated corpora exist for this language."
        )


if __name__ == "__main__":
    sys.exit(main())
