"""Input glue: counts, coordinates, cell metadata and niche matrices from / to AnnData.

This is the only module that knows the AnnData layout. Counts leave it genes x cells (R's
orientation); everything else leaves it as plain numpy / pandas objects. Niches live in
``obsm[name + sigma_str]`` as DataFrames (R ``reducedDim``) and the merge groups of
``mergeNiches()`` in ``uns["spiDE_niche_groups"][key]`` (R ``metadata(spe)``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ._rcompat import r_num_str

GROUPS_KEY = "spiDE_niche_groups"


def as_actual(adata: AnnData) -> None:
    """Turn an AnnData view into an independent object in place (copy-on-write for setters)."""
    if adata.is_view:
        adata._init_as_actual(adata.copy())


def get_counts(adata: AnnData, assay: str = "counts"):
    """Counts as genes x cells, from ``layers[assay]`` if present, else ``X``.

    Mirrors R ``assay(spe, assay)``: a sparse input stays sparse (CSC, as R's ``dgCMatrix``)
    and is never densified; a dense input is returned as a (transposed) view of ``X``.
    """
    M = adata.layers[assay] if assay in adata.layers else adata.X
    if sp.issparse(M):
        return sp.csc_matrix(M.T)  # CSR (cells x genes) transposes to CSC for free
    return np.asarray(M).T


def get_coords(adata: AnnData) -> np.ndarray:
    """First two columns of ``obsm["spatial"]`` as a float (n, 2) array (R ``spatialCoords``)."""
    if "spatial" not in adata.obsm:
        raise ValueError("spatial coordinates not found in adata.obsm['spatial']")
    xy = adata.obsm["spatial"]
    xy = xy.to_numpy() if isinstance(xy, pd.DataFrame) else np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have at least 2 columns (x and y coordinates)")
    try:
        xy = xy[:, :2].astype(float)
    except (TypeError, ValueError):
        raise ValueError("adata.obsm['spatial'] must be numeric") from None
    if not np.all(np.isfinite(xy)):
        raise ValueError("spatial coordinates contain non-finite values")
    return xy


def _as_character(v) -> object:
    """One value as R ``as.character()`` prints it (missing stays ``NaN``)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (float, np.floating)):
        return r_num_str(float(v))
    return str(v)


def as_character_array(x) -> np.ndarray:
    """A column as an object array of R ``as.character()`` strings, missing values ``NaN``.

    Each distinct value is formatted once (``pd.factorize``), so a million-cell column costs a
    hash pass, not a Python call per cell.
    """
    x = pd.Series(x) if not isinstance(x, pd.Series) else x
    codes, uniq = pd.factorize(x)
    lut = np.empty(len(uniq) + 1, dtype=object)
    lut[:-1] = [_as_character(v) for v in uniq]
    lut[-1] = np.nan  # code -1 (missing) picks the last entry
    return lut[codes]


def get_obs(adata: AnnData, column: str) -> np.ndarray:
    """An ``obs`` column as an object array of strings, missing values kept as ``NaN``.

    The port of ``as.character(colData(spe)[[column]])``: numbers print as R prints them
    (``1.0`` -> ``"1"``), logicals as ``TRUE`` / ``FALSE``.
    """
    if column not in adata.obs.columns:
        raise ValueError(f"column '{column}' not found in adata.obs")
    return as_character_array(adata.obs[column])


def niche_key(name: str, sigma: float) -> str:
    """``obsm`` key of a niche matrix, R ``paste0(name, sigma)`` (``"Niche20"``, ``"Niche2.5"``)."""
    return name + r_num_str(sigma)


def niche_keys(adata: AnnData, name: str = "Niche") -> list[str]:
    """``obsm`` keys that hold niche matrices named ``name`` + bandwidth.

    Matches R ``grep("^name[0-9.]+$", reducedDimNames(spe))`` (so an exponent form such as
    ``"Niche1e+05"`` is not recognised, as in R; see ``docs/r-discrepancies.md``).
    """
    pat = re.compile(f"^{re.escape(name)}[0-9.]+$")
    return [k for k in adata.obsm.keys() if pat.match(k)]


def get_niche(adata: AnnData, key: str) -> pd.DataFrame:
    """The niche matrix at ``obsm[key]`` as a DataFrame (cells x cell types)."""
    if key not in adata.obsm:
        raise ValueError(f"niche '{key}' not found in adata.obsm")
    m = adata.obsm[key]
    if isinstance(m, pd.DataFrame):
        return m
    return pd.DataFrame(np.asarray(m), index=adata.obs_names)


def set_niche(adata: AnnData, key: str, df: pd.DataFrame) -> None:
    """Store a niche DataFrame (indexed by ``obs_names``) at ``obsm[key]``, in place."""
    if len(df) != adata.n_obs or not df.index.equals(adata.obs_names):
        raise ValueError("the niche matrix must have one row per cell, indexed by adata.obs_names")
    as_actual(adata)
    adata.obsm[key] = df.copy()


def get_group_map(adata: AnnData, key: str) -> dict[str, list[str]] | None:
    """Merge groups recorded for niche ``key`` (``{merged: [members]}``), or ``None``."""
    groups = adata.uns.get(GROUPS_KEY)
    if groups is None or key not in groups:
        return None
    return {str(k): [str(m) for m in np.atleast_1d(v)] for k, v in groups[key].items()}


def set_group_map(adata: AnnData, key: str, group_map: dict | None) -> None:
    """Record the merge groups of niche ``key`` (``None`` removes them), in place."""
    as_actual(adata)
    groups = dict(adata.uns.get(GROUPS_KEY, {}))
    if group_map is None:
        groups.pop(key, None)
    else:
        groups[key] = {str(k): [str(m) for m in np.atleast_1d(v)] for k, v in group_map.items()}
    adata.uns[GROUPS_KEY] = groups
