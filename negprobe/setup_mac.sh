#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon): virtual env, packages, data.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt ipykernel
bash get_data.sh
echo
echo "Setup done. Activate with: source .venv/bin/activate"
