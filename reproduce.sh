#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- table2
fi
exec python3 "$(dirname "$0")/reproduction/reproduce.py" "$@"
