#!/usr/bin/env bash
# Chạy ứng dụng HR Analytics dạng Desktop (cửa sổ native, không mở web).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-hr-desktop}"
mkdir -p "$MPLCONFIGDIR"

cd "$SCRIPT_DIR"
exec python app.py "$@"
