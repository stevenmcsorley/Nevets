#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import sys
assert (3,11) <= sys.version_info[:2] < (3,15), sys.version
PY
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
else
  pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
fi
pip install -e '.[dev]'
python scripts/doctor.py
