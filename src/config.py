"""Shared settings for the whole pipeline.

Every script imports its paths and constants from here, so if I want to change
the number of topics or the encoder I only have to change it in one place.

All scripts take --lang (default "de"). Outputs are suffixed with the language
so the German and English runs don't overwrite each other.
"""

from __future__ import annotations

import re
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "in"       # input data (not committed, see README)
OUT_DIR = PROJECT_ROOT / "out"     # everything the scripts produce

# older names, kept so i don't have to touch every script at once.
DATA_DIR = IN_DIR
RESULTS_DIR = OUT_DIR


def external_dir(lang: str) -> Path:
    """Where the gated corpora go after I convert them (see README)."""
    return IN_DIR / "external" / lang


def paths(lang: str) -> dict:
    """The files that get passed between scripts, per language."""
    return {
        "combined_csv": IN_DIR / f"texts_{lang}.csv",
        "with_topics_csv": IN_DIR / f"texts_{lang}_with_topics.csv",
        "doc_topic_matrix": OUT_DIR / f"doc_topic_matrix_{lang}.npy",
        "topic_words_csv": OUT_DIR / f"topics_top_words_{lang}.csv",
        "topic_state_json": OUT_DIR / f"topic_state_{lang}.json",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topic_outputs_problem(lang: str) -> str | None:
    """Return an explanation if topic outputs do not match the cleaned input."""
    p = paths(lang)
    needed = (
        p["combined_csv"], p["with_topics_csv"], p["doc_topic_matrix"],
        p["topic_words_csv"], p["topic_state_json"],
    )
    missing = [path.name for path in needed if not path.exists()]
    if missing:
        return (f"Missing topic inputs/outputs {missing}. Run "
                f"src/02_topic_model.py --lang {lang}.")
    try:
        state = json.loads(p["topic_state_json"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"Invalid {p['topic_state_json'].name}: {exc}. Rerun step 2."
    current = sha256_file(p["combined_csv"])
    if state.get("source_sha256") != current:
        return (f"Topic outputs are stale relative to {p['combined_csv'].name}. "
                f"Rerun src/02_topic_model.py --lang {lang}.")
    if state.get("with_topics_sha256") != sha256_file(p["with_topics_csv"]):
        return f"{p['with_topics_csv'].name} has changed since topic fitting; rerun step 2."
    if state.get("doc_topic_sha256") != sha256_file(p["doc_topic_matrix"]):
        return f"{p['doc_topic_matrix'].name} has changed since topic fitting; rerun step 2."
    if state.get("topic_words_sha256") != sha256_file(p["topic_words_csv"]):
        return f"{p['topic_words_csv'].name} has changed since topic fitting; rerun step 2."
    try:
        matrix = np.load(p["doc_topic_matrix"], mmap_mode="r")
    except (OSError, ValueError) as exc:
        return f"Invalid {p['doc_topic_matrix'].name}: {exc}. Rerun step 2."
    expected_shape = (state.get("n_documents"), state.get("n_topics"))
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        return (f"{p['doc_topic_matrix'].name} has shape {matrix.shape}, expected "
                f"{expected_shape}; rerun step 2.")
    if not np.isfinite(matrix).all():
        return f"{p['doc_topic_matrix'].name} contains non-finite values; rerun step 2."
    return None


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------
# these download automatically from huggingface. the gated corpora (deplain-apa
# and apa-lha) have to be requested separately and converted with 00_convert_raw.py.
OPEN_HF_DATASETS = {
    "de": [
        "UniversalCEFR/merlin_de",     # learner exam essays
        "UniversalCEFR/elg_cefr_de",   # manually CEFR-annotated reference texts
    ],
    "en": [
        "UniversalCEFR/icle500_en",    # learner essays
        "UniversalCEFR/cefr_asag_en",  # learner short answers
        "UniversalCEFR/elg_cefr_en",    # manually CEFR-annotated reference texts
    ],
}

# ---------------------------------------------------------------------------
# cefr levels
# ---------------------------------------------------------------------------
LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


# sub-level notations that the cefr treats as the same band. the companion
# volume (council of europe, 2020, section 2.6) distinguishes "criterion levels"
# (a2, or a2.1) from "plus levels" (a2+, or a2.2), so a2+ and a2.2 mean the same
# thing and both sit inside a2. my analysis works at the six-level granularity,
# so i map both notations down to the base level. that is a real simplification
# and i count how often it fires (see 01_load_data.py) rather than let it happen
# silently.
_SUBLEVEL = re.compile(r"^(PRE-A1|A1|A2|B1|B2|C1|C2)\s*(\+|\.\d+)?$")


def normalize_level(raw) -> str | None:
    """Map a raw label onto one of the six Common Reference Levels.

    'a1', 'A1+', 'A1.2' and ' A1 ' all become 'A1'. Pre-A1, which the Companion
    Volume added in 2020, has no place in the six-level scheme I use, so it is
    returned as its own string and dropped downstream by the caller, but at
    least it is distinguishable from a parse failure. Anything else (a bare 'A',
    a range like 'B1-B2', an empty cell) returns None.
    """
    if raw is None:
        return None
    level = str(raw).strip().upper()
    m = _SUBLEVEL.match(level)
    if not m:
        return None
    base = m.group(1)
    return base if base in LEVEL_ORDER else None


# ---------------------------------------------------------------------------
# model settings
# ---------------------------------------------------------------------------
# multilingual so german and english go through the same encoder, and small
# enough to run on a laptop cpu. alternatives are tested in 08b.
ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"

N_TOPICS = 15      # override with --n-topics
MIN_TOKENS = 10    # drop very short fragments, they carry no topic signal
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# numbers copied from imperial et al. (2025), table 3
# ---------------------------------------------------------------------------
# these are weighted f1, which is the paper's primary metric, not macro f1.
# column order in their table is en es de nl cs it fr et pt ar hi ru cy.
XLMR_WEIGHTED_F1 = {"de": 73.2, "en": 75.5}
MOST_FREQUENT_WEIGHTED_F1 = {"de": 26.8, "en": 7.39}

# the rest of the same column, used for the comparison in section 4.5.
REFERENCE_LADDER = {
    "de": {"most_frequent_class": 26.8, "logregr_topfeats": 52.5,
           "randforest_allfeats": 65.4, "eurobert": 70.6,
           "modernbert": 72.1, "xlmr": 73.2},
    "en": {"most_frequent_class": 7.39, "logregr_allfeats": 32.1,
           "randforest_allfeats": 63.4, "eurobert": 74.6,
           "modernbert": 75.8, "xlmr": 75.5},
}
# careful with english: their floor is 7.39 but mine is 14.7 on a different
# split, so the two aren't interchangeable. script 06 prints both.


def log_environment() -> dict:
    """Package versions, saved into the manifest so results are traceable."""
    import platform

    import numpy
    import pandas
    import scipy
    import sklearn

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "encoder": ENCODER,
        "random_seed": RANDOM_SEED,
        "n_topics": N_TOPICS,
    }
    for name in ("turftopic", "sentence_transformers"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                )
                env[name] = __import__(name).__version__
        except Exception:
            pass
    return env
