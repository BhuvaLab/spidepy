#!/usr/bin/env bash
# Install the R reference packages into tests/ref/Rlib only (never the user library):
# SpaNorm f233b7b (1.7.16) first, then spiDE e31e744 (0.99.37). Override the sources with
# SPANORM_REPO / SPIDE_REPO and the refs with SPANORM_REF / SPIDE_REF.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SPANORM_REF="${SPANORM_REF:-f233b7b}"
SPANORM_REPO="${SPANORM_REPO:-$REPO/../SpaNorm}"
SPIDE_REF="${SPIDE_REF:-e31e744}"
SPIDE_REPO="${SPIDE_REPO:-$REPO/../spiDE}"
RBIN="${RBIN:-$HOME/miniconda3/envs/latest-r/bin}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$HERE/Rlib" "$TMP/SpaNorm" "$TMP/spiDE"

git -C "$SPANORM_REPO" archive "$SPANORM_REF" | tar -x -C "$TMP/SpaNorm"
"$RBIN/R" CMD INSTALL --library="$HERE/Rlib" "$TMP/SpaNorm"

# spiDE has a `research` submodule; git archive emits it as an empty directory or omits it.
git -C "$SPIDE_REPO" archive "$SPIDE_REF" | tar -x -C "$TMP/spiDE"
rm -rf "$TMP/spiDE/research"
R_LIBS="$HERE/Rlib" "$RBIN/R" CMD INSTALL --no-docs --library="$HERE/Rlib" "$TMP/spiDE"

# clubSandwich is only a Suggests (sandwich engine); keep going if there is no network.
"$RBIN/Rscript" -e "if (!requireNamespace('clubSandwich', lib.loc='$HERE/Rlib', quietly=TRUE)) tryCatch(install.packages('clubSandwich', lib='$HERE/Rlib', repos='https://cloud.r-project.org'), error=function(e) message('NOTE: clubSandwich not installed: ', conditionMessage(e)), warning=function(w) message('NOTE: clubSandwich not installed: ', conditionMessage(w)))"

"$RBIN/Rscript" -e "library(SpaNorm, lib.loc='$HERE/Rlib'); library(spiDE, lib.loc='$HERE/Rlib'); print(packageVersion('SpaNorm')); print(packageVersion('spiDE')); cat('clubSandwich:', requireNamespace('clubSandwich', lib.loc='$HERE/Rlib', quietly=TRUE), '\n')"
