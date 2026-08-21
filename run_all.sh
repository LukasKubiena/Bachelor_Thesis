#!/usr/bin/env bash
# runs the full analysis for one language
# usage: bash run_all.sh [de|en]
#
# run setup.sh first; the topic and sensitivity models take the most time
set -euo pipefail
cd "$(dirname "$0")"
LANG_ARG="${1:-de}"

PYTHON="${PYTHON:-python}"
for candidate in env/bin/python .venv/bin/python; do
  if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thesis-matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON" src/00_check_environment.py

echo "=== Pipeline lang=${LANG_ARG} ==="
if [[ -d in/raw ]] && find in/raw -type f -print -quit | grep -q .; then
  "$PYTHON" src/00_convert_raw.py
elif [[ -d in/external/de ]] && find in/external/de -type f -name '*.json*' -print -quit | grep -q .; then
  echo "No in/raw files; using the already converted gated corpora in in/external/de/."
else
  echo "No gated-corpus files found; the pipeline will use open corpora only."
fi
"$PYTHON" src/01_load_data.py            --lang "$LANG_ARG"
"$PYTHON" src/01b_descriptives.py        --lang "$LANG_ARG"
"$PYTHON" src/01c_near_duplicates.py     --lang "$LANG_ARG"
"$PYTHON" src/02_topic_model.py          --lang "$LANG_ARG"
"$PYTHON" src/03_confound_analysis.py    --lang "$LANG_ARG"
"$PYTHON" src/03b_association_extended.py --lang "$LANG_ARG"
"$PYTHON" src/03c_length_stratified.py   --lang "$LANG_ARG"
"$PYTHON" src/03d_ci_coverage.py         --lang "$LANG_ARG" --n-rep 1000 --n-bootstrap-rep 120 --n-boot 120
"$PYTHON" src/04_topic_only_baseline.py  --lang "$LANG_ARG"
"$PYTHON" src/05_robustness.py           --lang "$LANG_ARG"
"$PYTHON" src/06_length_benchmark.py     --lang "$LANG_ARG"
"$PYTHON" src/07_cross_corpus.py         --lang "$LANG_ARG"
"$PYTHON" src/07b_topic_stratified.py    --lang "$LANG_ARG"
"$PYTHON" src/08_topic_model_quality.py  --lang "$LANG_ARG"
"$PYTHON" src/08b_encoder_sensitivity.py --lang "$LANG_ARG"
"$PYTHON" src/08c_model_sensitivity.py   --lang "$LANG_ARG"

if [[ "$LANG_ARG" == "de" ]]; then
  "$PYTHON" src/04_topic_only_baseline.py --lang de --dataset merlin_de
  "$PYTHON" src/04_topic_only_baseline.py --lang de --dataset elg_cefr_de
  "$PYTHON" src/06_length_benchmark.py --lang de --dataset merlin_de
  # use the same surface baseline for every german corpus
  "$PYTHON" src/06_length_benchmark.py --lang de --dataset elg_cefr_de --n-perm 0
  "$PYTHON" src/06_length_benchmark.py --lang de --dataset deplain_apa_doc --n-perm 0
  "$PYTHON" src/06_length_benchmark.py --lang de --dataset apa_lha --n-perm 0
  "$PYTHON" src/07c_topic_overlap_within_corpus.py --lang de --dataset merlin_de
  "$PYTHON" src/07c_topic_overlap_within_corpus.py --lang de --dataset elg_cefr_de --skip-diagnostic
  "$PYTHON" src/07c_topic_overlap_within_corpus.py --lang de --dataset deplain_apa_doc --skip-diagnostic
  "$PYTHON" src/07c_topic_overlap_within_corpus.py --lang de --dataset apa_lha --skip-diagnostic
else
  "$PYTHON" src/04_topic_only_baseline.py --lang en --dataset icle500_en
  "$PYTHON" src/07c_topic_overlap_within_corpus.py --lang en --dataset icle500_en
  "$PYTHON" src/07c_topic_overlap_within_corpus.py --lang en --dataset elg_cefr_en --skip-diagnostic
fi

# repeat a few scripts to check that seeded results stay the same
echo "=== Determinism check (03, 04, 06, 07) ==="
CHECK_DIR="$(mktemp -d)"
trap 'rm -rf "$CHECK_DIR"' EXIT
cp "out/association_stats_${LANG_ARG}.txt" "$CHECK_DIR/assoc_a.txt"
cp "out/baseline_results_${LANG_ARG}.csv" "$CHECK_DIR/base_a.csv"
cp "out/length_benchmark_${LANG_ARG}.csv" "$CHECK_DIR/len_a.csv"
cp "out/length_benchmark_${LANG_ARG}.json" "$CHECK_DIR/len_a.json"
cp "out/cross_corpus_transfer_${LANG_ARG}.csv" "$CHECK_DIR/xfer_a.csv"
"$PYTHON" src/03_confound_analysis.py --lang "$LANG_ARG" > /dev/null
# skip slow permutation tests during this check and restore their json result
"$PYTHON" src/04_topic_only_baseline.py --lang "$LANG_ARG" --skip-perm > /dev/null
"$PYTHON" src/06_length_benchmark.py --lang "$LANG_ARG" --n-perm 0 > /dev/null
"$PYTHON" src/07_cross_corpus.py --lang "$LANG_ARG" > /dev/null
cmp "$CHECK_DIR/assoc_a.txt" "out/association_stats_${LANG_ARG}.txt"
"$PYTHON" - "$LANG_ARG" "$CHECK_DIR" <<'PY'
import pandas as pd, sys
lang = sys.argv[1]
check_dir = sys.argv[2]
a = pd.read_csv(f"{check_dir}/base_a.csv")
b = pd.read_csv(f"out/baseline_results_{lang}.csv")
cols = [c for c in ["model", "cv", "accuracy", "macro_f1", "weighted_f1"]
        if c in a.columns and c in b.columns]
pd.testing.assert_frame_equal(
    a[cols].reset_index(drop=True), b[cols].reset_index(drop=True), atol=1e-12
)
print("04 CSV metrics identical")
PY
cmp "$CHECK_DIR/len_a.csv" "out/length_benchmark_${LANG_ARG}.csv"
cmp "$CHECK_DIR/xfer_a.csv" "out/cross_corpus_transfer_${LANG_ARG}.csv"
cp "$CHECK_DIR/base_a.csv" "out/baseline_results_${LANG_ARG}.csv"
cp "$CHECK_DIR/len_a.json" "out/length_benchmark_${LANG_ARG}.json"
echo "Determinism OK"

echo "=== Unit tests ==="
# stop before figures and manifests if a test fails
"$PYTHON" -m pytest -q tests/

"$PYTHON" src/10_figures.py --lang "$LANG_ARG"
"$PYTHON" src/09_build_manifest.py --lang "$LANG_ARG"

echo "Done. Headline numbers are in out/manifest_${LANG_ARG}.json"
