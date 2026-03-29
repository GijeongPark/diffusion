#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "ERROR: no Python interpreter found."
  exit 1
fi

exec "$PYTHON_BIN" -m peh_inverse_design.pipeline_runner "$@"
