#!/usr/bin/env bash
# Regenerate every reference file. Run from anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RSCRIPT="${RSCRIPT:-$HOME/miniconda3/envs/latest-r/bin/Rscript}"
mkdir -p "$HERE/data"
cd "$REPO"
for f in "$HERE"/make_*.R; do
  echo "== $(basename "$f")"
  "$RSCRIPT" "$f"
done
# the bundled example dataset is built from the R fixture by the Python loader
"${PYTHON:-/scratch/user/uqdbhuva/envs/spidepy/bin/python}" "$REPO/tests/_ref.py"
