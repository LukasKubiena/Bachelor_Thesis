"""step 9: collect headline results in out/manifest_<lang>.json.

the manifest keeps the reported results in one place and checks for missing or
inconsistent output files.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    MOST_FREQUENT_WEIGHTED_F1,
    OPEN_HF_DATASETS,
    RESULTS_DIR,
    XLMR_WEIGHTED_F1,
    log_environment,
    paths,
    topic_outputs_problem,
)


def _json_safe(obj):
    """Replace NaN/Inf so the manifest is RFC-compliant JSON (not Python/JS NaN)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "item") and not isinstance(obj, (bytes, str)):
        try:
            return _json_safe(obj.item())
        except (ValueError, AttributeError):
            pass
    return obj


def collect_macro_f1_drift(manifest: dict, tol: float = 0.001) -> list[str]:
    """check that topic-mixture macro f1 agrees across steps 04, 05 and 06.
    """
    pairs: list[tuple[str, float]] = []
    for row in manifest.get("baselines", {}).get("all") or []:
        if row.get("model") != "topic mixture":
            continue
        cv = str(row.get("cv", "grouped"))
        if cv in ("grouped", "grouped (pair_id)", "grouped (cv_group)") or "cv" not in row:
            try:
                pairs.append((f"baselines[{cv}].macro_f1", float(row["macro_f1"])))
            except (TypeError, ValueError, KeyError):
                pass
            break
    for row in manifest.get("length_benchmark", {}).get("models") or []:
        if row.get("features") != "topic only":
            continue
        cv = str(row.get("cv", ""))
        if not cv.startswith("grouped"):
            continue
        try:
            pairs.append(("length_benchmark[grouped topic only].macro_f1",
                          float(row["macro_f1"])))
        except (TypeError, ValueError, KeyError):
            pass
        break
    for row in manifest.get("robustness", {}).get("k_sweep") or []:
        if int(row.get("n_topics", -1)) != 15:
            continue
        if row.get("measure") != "topic_mixture_macro_f1":
            continue
        try:
            pairs.append(("robustness[k=15].topic_mixture_macro_f1", float(row["value"])))
        except (TypeError, ValueError, KeyError):
            pass
        break
    if len(pairs) < 2:
        return []
    vals = [v for _, v in pairs]
    if max(vals) - min(vals) <= tol:
        return []
    listing = ", ".join(f"{src}={val:.4f}" for src, val in pairs)
    return [
        f"topic-mixture macro F1 differs by more than {tol}: {listing}"
    ]


def collect_output_integrity(df: pd.DataFrame, lang: str) -> list[str]:
    """Detect incomplete or incomparable outputs before they reach the thesis."""
    errors: list[str] = []
    if "cv_group" not in df.columns or df["cv_group"].isna().any():
        errors.append(
            "cleaned data lack cv_group; rerun 01c_near_duplicates.py and all "
            "predictive analyses"
        )
    else:
        if "pair_id" in df.columns:
            split_pairs = df.groupby("pair_id")["cv_group"].nunique()
            if (split_pairs > 1).any():
                errors.append("a pair_id family is split across cv_group values")
        near_path = RESULTS_DIR / f"near_duplicates_{lang}.csv"
        if not near_path.exists():
            errors.append(f"missing required output: {near_path.name}")
        else:
            near = pd.read_csv(near_path)
            for i, j in near[["i", "j"]].itertuples(index=False, name=None):
                i, j = int(i), int(j)
                if not (0 <= i < len(df) and 0 <= j < len(df)):
                    errors.append(f"{near_path.name} contains out-of-range row indices")
                    break
                if str(df.iloc[i]["cv_group"]) != str(df.iloc[j]["cv_group"]):
                    errors.append(
                        f"near-duplicate rows {(i, j)} are split across cv_group values"
                    )
                    break
    expected_features = {
        "majority class (floor)", "word count only", "surface features",
        "topic only", "surface + topic",
    }
    subsets: list[tuple[str | None, pd.DataFrame]] = [(None, df)]
    if lang == "de":
        subsets.extend((name, df[df["dataset"] == name])
                       for name in sorted(df["dataset"].unique()))
    for dataset, sub in subsets:
        suffix = lang if dataset is None else f"{lang}_{dataset}"
        path = RESULTS_DIR / f"oof_length_benchmark_{suffix}.csv"
        if not path.exists():
            errors.append(f"missing required OOF file: {path.name}")
            continue
        eligible = sub["cefr_level"].value_counts()
        eligible_levels = set(eligible[eligible >= 5].index)
        expected_n = int(sub["cefr_level"].isin(eligible_levels).sum())
        oof = pd.read_csv(path)
        if "features" not in oof or not {"y_true", "y_pred"} <= set(oof.columns):
            errors.append(f"{path.name} lacks required prediction columns")
            continue
        seen = set(oof["features"].astype(str))
        if seen != expected_features:
            errors.append(
                f"{path.name} feature sets are {sorted(seen)}, expected "
                f"{sorted(expected_features)}"
            )
        counts = oof["features"].value_counts().to_dict()
        bad = {feature: int(counts.get(feature, 0)) for feature in expected_features
               if int(counts.get(feature, 0)) != expected_n}
        if bad:
            errors.append(
                f"{path.name} has stale/incomplete row counts {bad}; "
                f"expected {expected_n} per feature"
            )

    xfer_path = RESULTS_DIR / f"cross_corpus_transfer_{lang}.csv"
    if xfer_path.exists():
        xfer = pd.read_csv(xfer_path)
        if "evaluation_levels" not in xfer:
            errors.append(f"{xfer_path.name} predates the common-label-space fix")
        elif xfer["evaluation_levels"].nunique(dropna=False) != 1:
            errors.append(f"{xfer_path.name} uses more than one evaluation label space")

    stress_path = RESULTS_DIR / f"topic_stratified_{lang}.csv"
    replication_path = RESULTS_DIR / f"topic_stratified_replication_{lang}.csv"
    if stress_path.exists() and replication_path.exists():
        stress = pd.read_csv(stress_path)
        replication = pd.read_csv(replication_path)
        required_stress = {
            "features", "random_splits_weighted_f1",
            "topic_grouped_weighted_f1", "random_splits_macro_f1",
            "topic_grouped_macro_f1", "random_splits_qwk",
            "topic_grouped_qwk",
        }
        required_rep = {"model", "weighted_f1", "macro_f1", "qwk"}
        stress_status = set(stress.get("status", pd.Series(dtype=str)).astype(str))
        not_estimable = bool(stress_status) and all(
            status.startswith("not estimable:") for status in stress_status
        )
        if not_estimable:
            rep_status = set(replication.get("status", pd.Series(dtype=str)).astype(str))
            if rep_status != stress_status:
                errors.append("step 7/7b non-estimability status drift")
        elif not required_stress <= set(stress.columns):
            errors.append(f"{stress_path.name} has the wrong schema or was overwritten")
        elif not required_rep <= set(replication.columns):
            errors.append(f"{replication_path.name} has the wrong schema")
        else:
            stress = stress.set_index("features")
            replication = replication.set_index("model")
            comparisons = {
                "topic_mixture_random": ("topic mixture", "random_splits"),
                "topic_mixture_topic_grouped": ("topic mixture", "topic_grouped"),
                "tfidf_random": ("full text (TF-IDF)", "random_splits"),
                "tfidf_topic_grouped": ("full text (TF-IDF)", "topic_grouped"),
            }
            for model, (feature, regime) in comparisons.items():
                if model not in replication.index or feature not in stress.index:
                    errors.append(f"missing step 7/7b stress-test row: {model}")
                    continue
                for metric in ("weighted_f1", "macro_f1", "qwk"):
                    a = float(replication.loc[model, metric])
                    b = float(stress.loc[feature, f"{regime}_{metric}"])
                    if not math.isclose(a, b, abs_tol=1e-12):
                        errors.append(
                            f"step 7/7b drift for {model} {metric}: {a} != {b}"
                        )
    elif not stress_path.exists():
        errors.append(f"missing required output: {stress_path.name}")
    else:
        errors.append(f"missing required output: {replication_path.name}")

    if lang == "de":
        for dataset in ("merlin_de", "elg_cefr_de", "deplain_apa_doc", "apa_lha"):
            overlap_path = RESULTS_DIR / f"topic_overlap_within_de_{dataset}.csv"
            if not overlap_path.exists():
                errors.append(f"missing required output: {overlap_path.name}")
                continue
            overlap = pd.read_csv(overlap_path)
            if "status" not in overlap:
                errors.append(f"{overlap_path.name} predates the pair-safe split fix")
                continue
            statuses = set(overlap["status"].astype(str))
            if dataset in {"merlin_de", "elg_cefr_de"}:
                if statuses != {"ok"} or len(overlap) != 4:
                    errors.append(f"{overlap_path.name} lacks four valid feature rows")
            elif not all(status.startswith("not estimable:") for status in statuses):
                errors.append(
                    f"{overlap_path.name} should record the pair-safe non-estimability"
                )

    required = [
        RESULTS_DIR / f"ci_coverage_{lang}.csv",
        RESULTS_DIR / f"exact_duplicates_preclean_{lang}.csv",
        RESULTS_DIR / f"encoder_sensitivity_{lang}.csv",
        RESULTS_DIR / f"model_sensitivity_{lang}.csv",
    ]
    errors.extend(f"missing required output: {path.name}" for path in required
                  if not path.exists())
    return errors


def _read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _read_txt(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _read_json(path: Path):
    """Read the JSON that 03b, 06 and 07 write.

    Reading their JSON rather than rebuilding numbers from the CSVs means the
    manifest stays in step with the analysis on its own. If a file is missing
    the script says so at the end instead of leaving a silently empty section.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  WARNING: {path.name} is not valid JSON ({e}); skipped")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    args = parser.parse_args()
    lang = args.lang
    p = paths(lang)
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": lang,
        "environment": log_environment(),
        "data": {},
        "association": {},
        "baselines": {},
        "length_benchmark": {},
        "cross_corpus": {},
        "robustness": {},
        "reference": {
            "xlmr_weighted_f1": XLMR_WEIGHTED_F1.get(lang),
            "most_frequent_class_weighted_f1": MOST_FREQUENT_WEIGHTED_F1.get(lang),
        },
    }

    src = p["with_topics_csv"] if p["with_topics_csv"].exists() else p["combined_csv"]
    if src.exists():
        df = pd.read_csv(src)
        manifest["data"] = {
            "n_documents": int(len(df)),
            "per_corpus": df["dataset"].value_counts().to_dict(),
            "per_level": df["cefr_level"].value_counts().to_dict(),
        }
        if "pair_id" in df.columns:
            sizes = df.groupby("pair_id").size().value_counts().to_dict()
            manifest["data"]["pair_id_group_sizes"] = {str(k): int(v) for k, v in sizes.items()}
        if "cv_group" in df.columns:
            sizes = df.groupby("cv_group").size().value_counts().to_dict()
            manifest["data"]["cv_group_size_counts"] = {
                str(k): int(v) for k, v in sizes.items()
            }
        duplicate_audit = _read_csv(
            RESULTS_DIR / f"exact_duplicates_preclean_{lang}.csv")
        if duplicate_audit is not None:
            manifest["data"]["exact_duplicate_groups_preclean"] = int(len(duplicate_audit))
            manifest["data"]["cross_corpus_exact_duplicate_groups_preclean"] = int(
                (duplicate_audit["n_corpora"] > 1).sum())

    # ------------------------------------------------------------------
    # structured json from scripts 03b / 06 / 07. these are the authoritative
    # sources; the csv readers below remain as a fallback for older filenames.
    # ------------------------------------------------------------------
    assoc = _read_json(RESULTS_DIR / f"association_extended_{lang}.json")
    if assoc:
        manifest["association"].update({
            "overall": assoc.get("overall"),
            "topic_vs_source": assoc.get("source"),
            "by_corpus": {r["corpus"]: r for r in assoc.get("by_corpus", [])},
            "cmh_per_level": assoc.get("cmh"),
            "ranking_raw": assoc.get("ranking_raw"),
            "ranking_matched_n": assoc.get("ranking_matched_n"),
            "grouping_holds_at_matched_n": assoc.get("grouping_holds_at_matched_n"),
            "matched_n": assoc.get("matched_n"),
            "n_cells_resid_gt3": assoc.get("n_cells_resid_gt3"),
            "top_associations": assoc.get("top_associations"),
            "top_associations_merlin": assoc.get("top_associations_merlin"),
            "confident_half": assoc.get("confident_half"),
        })

    lb = _read_json(RESULTS_DIR / f"length_benchmark_{lang}.json")
    if lb:
        manifest["length_benchmark"].update({
            "results": lb.get("results"),
            "bootstrap_ci": lb.get("bootstrap_ci"),
            "leakage_weighted_f1": lb.get("leakage_weighted_f1"),
            "incremental_topic_over_surface": lb.get("incremental_topic_over_surface"),
            "permutation_null": lb.get("permutation_null"),
            "per_level": lb.get("per_level"),
            "never_predicted_levels": lb.get("never_predicted_levels"),
            "calibration_vs_reported": lb.get("calibration_vs_reported"),
            "spearman_length_level": lb.get("spearman_length_level"),
        })

    cc = _read_json(RESULTS_DIR / f"cross_corpus_{lang}.json")
    if cc:
        manifest["cross_corpus"].update({
            "transfer_matrix": cc.get("transfer_matrix"),
            "leave_one_corpus_out": cc.get("leave_one_corpus_out"),
            "mean_in_corpus_weighted_f1": cc.get("mean_in_corpus_weighted_f1"),
            "mean_transfer_weighted_f1": cc.get("mean_transfer_weighted_f1"),
            "topic_stratified": cc.get("topic_stratified"),
        })

    by_corp = _read_csv(RESULTS_DIR / f"association_by_corpus_{lang}.csv")
    if by_corp is not None:
        overall_txt = _read_txt(RESULTS_DIR / f"association_stats_{lang}.txt") or ""
        # prefer structured csv
        by = {}
        for _, r in by_corp.iterrows():
            by[r["dataset"]] = {
                "cramers_v": float(r["cramers_v"]),
                "ci": [float(r["ci_low"]), float(r["ci_high"])],
                "perm_p": float(r["perm_p"]),
                "perm_p_holm": float(r["perm_p_holm"]) if "perm_p_holm" in r and pd.notna(r["perm_p_holm"]) else None,
                "ami": float(r["ami"]) if "ami" in r else None,
                "theils_u_level_given_topic": float(r["theils_u_level_given_topic"])
                if "theils_u_level_given_topic" in r
                else None,
                "n": int(r["n"]),
            }
        manifest["association"]["by_corpus"] = by
        manifest["association"]["stats_text"] = overall_txt[:2000]

    power = _read_csv(RESULTS_DIR / f"power_analysis_{lang}.csv")
    if power is not None:
        manifest["association"]["power"] = power.to_dict(orient="records")

    cmh = _read_csv(RESULTS_DIR / f"cmh_per_level_{lang}.csv")
    if cmh is not None:
        manifest["association"]["cmh_per_level"] = cmh.to_dict(orient="records")

    base = _read_csv(RESULTS_DIR / f"baseline_results_{lang}.csv")
    if base is not None:
        manifest["baselines"]["all"] = base.to_dict(orient="records")
    for tag in [f"{lang}_merlin_de", f"{lang}_elg_cefr_de"]:
        b = _read_csv(RESULTS_DIR / f"baseline_results_{tag}.csv")
        if b is not None:
            manifest["baselines"][tag] = b.to_dict(orient="records")

    length = _read_csv(RESULTS_DIR / f"length_benchmark_{lang}.csv")
    if length is not None:
        manifest["length_benchmark"]["models"] = length.to_dict(orient="records")
        txt = _read_txt(RESULTS_DIR / f"length_benchmark_{lang}.txt")
        if txt:
            manifest["length_benchmark"]["summary"] = txt

    xfer = _read_csv(RESULTS_DIR / f"cross_corpus_transfer_{lang}.csv")
    if xfer is not None:
        manifest["cross_corpus"]["rows"] = xfer.to_dict(orient="records")
    strat = _read_csv(RESULTS_DIR / f"topic_stratified_{lang}.csv")
    if strat is not None:
        manifest["cross_corpus"]["topic_stratified_table"] = strat.to_dict(orient="records")

    seed = _read_csv(RESULTS_DIR / f"seed_stability_{lang}.csv")
    if seed is not None:
        manifest["robustness"]["seed_stability"] = seed.to_dict(orient="records")
    enc = _read_csv(RESULTS_DIR / f"encoder_sensitivity_{lang}.csv")
    if enc is not None:
        manifest["robustness"]["encoder_sensitivity"] = enc.to_dict(orient="records")
    model_s = _read_csv(RESULTS_DIR / f"model_sensitivity_{lang}.csv")
    if model_s is not None:
        manifest["robustness"]["model_sensitivity"] = model_s.to_dict(orient="records")
    quality = _read_csv(RESULTS_DIR / f"topic_quality_{lang}.csv")
    if quality is not None:
        manifest["robustness"]["topic_quality"] = quality.to_dict(orient="records")
    rob = _read_csv(RESULTS_DIR / f"robustness_{lang}.csv")
    if rob is not None:
        manifest["robustness"]["k_sweep"] = rob.to_dict(orient="records")
    ci_coverage = _read_csv(RESULTS_DIR / f"ci_coverage_{lang}.csv")
    if ci_coverage is not None:
        manifest["association"]["ci_coverage_simulation"] = ci_coverage.to_dict(
            orient="records")

    nd = _read_txt(RESULTS_DIR / f"near_duplicates_summary_{lang}.txt")
    if nd:
        manifest["data"]["near_duplicates"] = nd

    ls = _read_json(RESULTS_DIR / f"length_stratified_{lang}.json")
    if ls:
        manifest["length_stratified"] = {
            "by_corpus": ls.get("by_corpus"),
            "cautions": ls.get("cautions"),
        }

    # ------------------------------------------------------------------
    # drift guard. the same quantity must not silently disagree across
    # sections. do not reconcile: warn, and record the values as they are.
    # ------------------------------------------------------------------
    warnings = collect_macro_f1_drift(manifest)
    integrity_errors = collect_output_integrity(df, lang) if src.exists() else [
        f"missing analysis table: {src}"]
    if topic_problem := topic_outputs_problem(lang):
        integrity_errors.append(topic_problem)
    manifest["consistency_warnings"] = warnings
    manifest["integrity_errors"] = integrity_errors
    all_errors = warnings + integrity_errors
    if all_errors:
        print("\n" + "!" * 72)
        print("OUTPUT INTEGRITY FAILURE")
        print("Do not use the manifest until the producer scripts have been rerun.")
        for w in all_errors:
            print(f"  {w}")
        print("!" * 72)
    else:
        print("\nDrift guard: topic-mixture macro F1 agrees across sections (≤ 0.001).")

    # ------------------------------------------------------------------
    # say out loud what is missing. an empty section in the manifest means the
    # script that fills it has not been run, and a silently empty manifest is
    # exactly how the thesis text ends up disagreeing with the code.
    # ------------------------------------------------------------------
    expected = {
        "association": "src/03b_association_extended.py",
        "baselines": "src/04_topic_only_baseline.py",
        "length_benchmark": "src/06_length_benchmark.py",
        "cross_corpus": "src/07_cross_corpus.py",
        "robustness": "src/05_robustness.py, 08*, needs the sentence encoder",
        "length_stratified": "src/03c_length_stratified.py",
    }
    missing = [(k, v) for k, v in expected.items() if not manifest.get(k)]
    if missing:
        print("\nSections still EMPTY in the manifest:")
        for k, script in missing:
            print(f"  {k:<18} run: {script}")
    else:
        print("\nAll manifest sections populated.")

    out = RESULTS_DIR / f"manifest_{lang}.json"
    out.write_text(
        json.dumps(_json_safe(manifest), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")
    if all_errors or missing:
        raise SystemExit("Manifest is incomplete or inconsistent; see messages above.")


if __name__ == "__main__":
    main()
