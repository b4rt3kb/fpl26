#!/usr/bin/env bash
# Official current-season data plus validated historical archives.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 fetch_fpl.py "$@"
