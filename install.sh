#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
if [ -d wheelhouse ]; then
  .venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python run.py --check
