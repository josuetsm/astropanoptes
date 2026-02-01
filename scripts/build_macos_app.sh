#!/usr/bin/env bash
set -euo pipefail

APP_NAME="AstroPanoptes"
ENTRYPOINT="ui/pyqt6_app.py"

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "pyinstaller not found. Install with: python -m pip install pyinstaller" >&2
  exit 1
fi

python -m PyInstaller \
  --name "${APP_NAME}" \
  --windowed \
  --onefile \
  --clean \
  "${ENTRYPOINT}"

echo "Built dist/${APP_NAME}.app (or dist/${APP_NAME})"
