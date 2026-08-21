"""Step 0 (optional): Convert the gated raw corpora to UniversalCEFR schema.

Reads the raw DEPlain and APA-LHA releases from in/raw/ and writes
UniversalCEFR-style JSONL files into in/external/, where 01_load_data.py
picks them up automatically. Run this once, then rerun the pipeline from
step 1.

Expected layout (as downloaded from Zenodo):
    in/raw/DEPlain/B__Document-level_Corpus/DEplain-APA-doc/plain-text/all.csv
    in/raw/DEPlain/E__Sentence-level_Corpus/DEplain-APA-sent/all.csv
    in/raw/APA_sentence-aligned_LHA/A2-OR/*.de, *_A2.simpde
    in/raw/APA_sentence-aligned_LHA/B1-OR/*.de, *_B1.simpde

Label sources (nothing is guessed):
    DEplain-APA carries its levels in the data itself: originals are B1,
    simplifications are A2 (columns complex_level / simple_level in the doc
    CSV and language_level_original / language_level_simple in the sentence
    CSV). APA-LHA carries the level in the filename (_A2 / _B1). The APA-LHA
    originals (.de) have no CEFR label and are therefore excluded.

Run:
    python src/00_convert_raw.py
    python src/00_convert_raw.py --include-sentences   # also emit DEplain-APA-sent

Sentence-level data is skipped by default: single sentences are a poor unit
for topic modelling and would swamp the ~5k documents with ~25k fragments.
Include them only for a deliberate granularity comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from config import DATA_DIR, external_dir

EXTERNAL_DIR = external_dir("de")
RAW_DIR = DATA_DIR / "raw"
DEPLAIN_DOC = RAW_DIR / "DEPlain" / "B__Document-level_Corpus" / "DEplain-APA-doc" / "plain-text" / "all.csv"
DEPLAIN_SENT = RAW_DIR / "DEPlain" / "E__Sentence-level_Corpus" / "DEplain-APA-sent" / "all.csv"
APA_LHA = RAW_DIR / "APA_sentence-aligned_LHA"


def detokenize(text: str) -> str:
    """Undo the pre-tokenized spacing in APA-LHA files ('Wort , Wort .')."""
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def record(
    text: str,
    level: str,
    source_name: str,
    fmt: str,
    title: str = "NA",
    pair_id: str | None = None,
) -> dict:
    return {
        "title": title,
        "lang": "de",
        "source_name": source_name,
        "format": fmt,
        "category": "reference",  # professionally written or simplified, not learner-produced
        "cefr_level": level.upper(),
        "license": "restricted, academic use only, do not redistribute",
        "text": text,
        "pair_id": pair_id,
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):>6} records -> {path}")


def convert_deplain_doc() -> None:
    if not DEPLAIN_DOC.exists():
        print(f"SKIP DEplain-APA-doc, not found at {DEPLAIN_DOC}")
        return
    print("Converting DEplain-APA-doc ...")
    df = pd.read_csv(DEPLAIN_DOC)
    records = []
    # each row is one original/simplification document pair. deduplicate by
    # document id in case a document participates in several pairs. the acl
    # release already ships pair_id linking original <-> simplification.
    for id_col, text_col, level_col in [
        ("complex_document_id", "original", "complex_level"),
        ("simple_document_id", "simplification", "simple_level"),
    ]:
        sub = df.drop_duplicates(subset=[id_col])
        for _, row in sub.iterrows():
            text = str(row[text_col]).strip()
            level = str(row[level_col]).strip()
            if text and level.lower() != "nan":
                pid = f"deplain_{row['pair_id']}"
                records.append(
                    record(text, level, "deplain-apa-doc", "document-level", pair_id=pid)
                )
    write_jsonl(records, EXTERNAL_DIR / "deplain_apa_doc.jsonl")


def convert_deplain_sent() -> None:
    if not DEPLAIN_SENT.exists():
        print(f"SKIP DEplain-APA-sent, not found at {DEPLAIN_SENT}")
        return
    print("Converting DEplain-APA-sent ...")
    df = pd.read_csv(DEPLAIN_SENT)
    records = []
    for id_col, text_col, level_col in [
        ("original_id", "original", "language_level_original"),
        ("simplification_id", "simplification", "language_level_simple"),
    ]:
        sub = df.drop_duplicates(subset=[id_col])
        for _, row in sub.iterrows():
            text = str(row[text_col]).strip()
            level = str(row[level_col]).strip()
            if text and level.lower() != "nan":
                records.append(record(text, level, "deplain-apa-sent", "sentence-level"))
    write_jsonl(records, EXTERNAL_DIR / "deplain_apa_sent.jsonl")


def convert_apa_lha() -> None:
    if not APA_LHA.exists():
        print(f"SKIP APA-LHA, not found at {APA_LHA}")
        return
    print("Converting APA-LHA (simplified texts only, originals have no CEFR label) ...")
    records = []
    n_originals = 0
    for folder in sorted(APA_LHA.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.simpde")):
            # filename pattern: <id>_<year>_<level>.simpde
            # shared article id is the leading <id>_<year> prefix.
            match = re.search(r"_(A2|B1)\.simpde$", path.name)
            prefix = re.match(r"(\d+_\d+)", path.name)
            if not match:
                continue
            lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                continue
            text = detokenize(" ".join(lines))
            pid = f"apa_lha_{prefix.group(1)}" if prefix else f"apa_lha_{path.stem}"
            records.append(
                record(
                    text,
                    match.group(1),
                    "apa-lha",
                    "document-level",
                    title=path.stem,
                    pair_id=pid,
                )
            )
        n_originals += len(list(folder.glob("*.de")))
    print(f"  ({n_originals} unlabeled originals excluded)")
    write_jsonl(records, EXTERNAL_DIR / "apa_lha.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-sentences", action="store_true",
                        help="also convert DEplain-APA-sent (~25k single sentences)")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        sys.exit(f"{RAW_DIR} not found. Move the downloaded corpora there first (see README).")

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    convert_deplain_doc()
    convert_apa_lha()
    if args.include_sentences:
        convert_deplain_sent()
    else:
        print("Skipping DEplain-APA-sent (rerun with --include-sentences to add it).")

    print("\nDone. Now rerun the pipeline from step 1:")
    print("  python src/01_load_data.py")


if __name__ == "__main__":
    main()
