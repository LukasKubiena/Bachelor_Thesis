from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_duplicate_audit_preserves_cross_corpus_evidence():
    loader = load_script("load_data", "01_load_data.py")
    df = pd.DataFrame({
        "text": ["same", "same", "within", "within", "unique"],
        "dataset": ["a", "b", "a", "a", "b"],
        "cefr_level": ["A2", "B1", "A2", "A2", "B1"],
    })
    audit = loader.exact_duplicate_audit(df)
    assert len(audit) == 2
    cross = audit[audit["n_corpora"] == 2].iloc[0]
    assert cross["datasets"] == "a|b"
    assert cross["levels"] == "A2|B1"
    assert cross["action"] == "exclude_all_conflicting_rows"
    assert len(cross["text_sha256"]) == 64


def test_validation_groups_merge_pair_and_near_duplicate_links():
    from utils_dedup import merged_validation_groups

    df = pd.DataFrame({
        "pair_id": ["a", "a", "b", "c", "d"],
        "dataset": ["x"] * 5,
    })
    near = pd.DataFrame({"i": [1, 3], "j": [2, 4]})
    groups = merged_validation_groups(df, near)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.iloc[3] == groups.iloc[4]
    assert groups.iloc[0] != groups.iloc[3]


def test_topic_pair_components_merge_cross_topic_parallel_versions():
    overlap = load_script("topic_overlap", "07c_topic_overlap_within_corpus.py")
    df = pd.DataFrame({
        "pair_id": ["a", "a", "b", "c"],
        "topic": [0, 1, 2, 2],
    })
    groups, n_components, n_cross_topic_pairs = overlap.topic_pair_components(df)
    assert n_components == 2
    assert n_cross_topic_pairs == 1
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


def test_manifest_integrity_rejects_partial_oof(tmp_path):
    manifest = load_script("build_manifest_integrity", "09_build_manifest.py")
    manifest.RESULTS_DIR = tmp_path
    df = pd.DataFrame({
        "dataset": ["only"] * 10,
        "cefr_level": ["A2"] * 5 + ["B1"] * 5,
    })
    pd.DataFrame({
        "features": ["topic only"] * 10,
        "y_true": df["cefr_level"],
        "y_pred": df["cefr_level"],
    }).to_csv(tmp_path / "oof_length_benchmark_en.csv", index=False)
    errors = manifest.collect_output_integrity(df, "en")
    assert any("feature sets" in error for error in errors)
    assert any("missing required output" in error for error in errors)


def test_topic_state_detects_stale_cleaned_input(tmp_path, monkeypatch):
    import config

    p = {
        "combined_csv": tmp_path / "texts.csv",
        "with_topics_csv": tmp_path / "texts_with_topics.csv",
        "doc_topic_matrix": tmp_path / "topics.npy",
        "topic_words_csv": tmp_path / "words.csv",
        "topic_state_json": tmp_path / "topic_state.json",
    }
    p["combined_csv"].write_text("current", encoding="utf-8")
    p["with_topics_csv"].write_text("augmented", encoding="utf-8")
    np.save(p["doc_topic_matrix"], np.ones((1, 2)))
    p["topic_words_csv"].write_text("topic,top_words\n0,test\n", encoding="utf-8")
    p["topic_state_json"].write_text(json.dumps({
        "source_sha256": "old",
        "with_topics_sha256": config.sha256_file(p["with_topics_csv"]),
        "doc_topic_sha256": config.sha256_file(p["doc_topic_matrix"]),
        "topic_words_sha256": config.sha256_file(p["topic_words_csv"]),
        "n_documents": 1,
        "n_topics": 2,
    }), encoding="utf-8")
    monkeypatch.setattr(config, "paths", lambda _lang: p)
    assert "stale" in config.topic_outputs_problem("de")

    p["topic_state_json"].write_text(json.dumps({
        "source_sha256": config.sha256_file(p["combined_csv"]),
        "with_topics_sha256": config.sha256_file(p["with_topics_csv"]),
        "doc_topic_sha256": config.sha256_file(p["doc_topic_matrix"]),
        "topic_words_sha256": config.sha256_file(p["topic_words_csv"]),
        "n_documents": 1,
        "n_topics": 2,
    }), encoding="utf-8")
    assert config.topic_outputs_problem("de") is None
