#!/usr/bin/env bash
# environment setup
# usage: bash setup.sh
# activation: source env/bin/activate

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is too old for the pinned analysis "
        "environment. Set PYTHON to Python 3.10 or newer."
    )
PY

echo "[1/3] creating virtual environment in ./env"
"$PYTHON" -m venv env

echo "[2/3] installing requirements"
./env/bin/python -m pip install --upgrade pip --quiet
./env/bin/python -m pip install -r requirements.txt
./env/bin/python src/00_check_environment.py

echo "[3/3] creating folders"
mkdir -p in out

echo
echo "Done. Activate it with:"
echo "    source env/bin/activate"
echo
echo "Then run the pipeline with:"
echo "    bash run_all.sh de"
echo
echo "Note: the two gated corpora are not downloaded automatically."
echo "See the Data section of README.md."
