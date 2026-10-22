"""Niche covariates: per-sample Gaussian cell-type densities and their merging.

Ports `R/buildNiches.R` (`buildNiches`, `.effectiveNiche`, `.fillMissingDims`,
`.speApply`), `R/mergeNiches.R` (`mergeNiches`, `.resolveMembers`) and `.isSelfNiche`
of `R/utils.R`. Niche matrices are stored as DataFrames in `adata.obsm[name + sigma]`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ._checks import check_adata
from ._io import (
    get_coords,
    get_group_map,
    get_niche,
    get_obs,
    niche_key,
    set_group_map,
    set_niche,
)
from ._kde import density_at
from ._rcompat import r_sort_str, sanitise


def _unique(xs) -> list:
    """Distinct values in order of first appearance (R `unique`)."""
    return list(dict.fromkeys(xs))


def _is_missing(v) -> bool:
    return v is None or (not isinstance(v, str) and bool(pd.isna(v)))


def _effective_niche(xy: np.ndarray, cts: np.ndarray, sigma: float, edge: bool, diggle: bool):
    """Density of every cell type at every cell of one sample (R `.effectiveNiche`).

    Columns follow the first appearance of each cell type; values are scaled by
    `pi * sigma^2` so they are on the scale of expected neighbour counts. The observation
    window is the bounding box of the sample's coordinates.
    """
    window = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    uct = _unique(cts)
    cols = [
        density_at(xy[cts == ct], xy, sigma, window, edge=edge, diggle=diggle) * np.pi * sigma**2
        for ct in uct
    ]
    return pd.DataFrame(np.column_stack(cols), columns=uct)


def _sample_niches(xy, cts, sigmas, all_cell_types, edge, diggle) -> list[np.ndarray]:
    """One sample's niche matrices, one per sigma, padded to `all_cell_types` with zeros.

    The padding is R `.fillMissingDims(m, all_cell_types, "column")` followed by
    `m[is.na(m)] <- 0`.
    """
    return [
        _effective_niche(xy, cts, sg, edge, diggle)
        .reindex(columns=all_cell_types)
        .fillna(0.0)
        .to_numpy()
        for sg in sigmas
    ]


def build_niches(
    adata: AnnData,
    sigma: float | Sequence[float] = (10, 30, 50, 70),
    cell_type: str = "cell_type",
    sample_id: str = "sample_id",
    edge: bool = True,
    diggle: bool = True,
    name: str = "Niche",
    n_jobs: int = 1,
    copy: bool = False,
) -> AnnData:
    """Build niche covariate matrices (R `buildNiches`).

    For each sample and bandwidth, the Gaussian kernel density of every cell type is evaluated
    at each cell's location (edge-corrected as in `spatstat.explore::densityfun`) and scaled
    by `pi * sigma**2`. The cells x cell types matrix of each bandwidth is stored as the
    DataFrame `adata.obsm[name + sigma]` with the same, sorted, set of cell-type columns for
    every sample (cell types absent from a sample are 0) and the original cell order.

    Parameters
    ----------
    adata : AnnData
        Cells x genes with `obsm["spatial"]`, a cell type and a sample id column.
    sigma : float or sequence of float
        Kernel bandwidths in the units of the spatial coordinates.
    cell_type, sample_id : str
        `adata.obs` columns holding cell type labels and sample identifiers.
    edge, diggle : bool
        Edge correction passed to the density (see `spidepy._kde`).
    name : str
        Prefix of the `obsm` keys.
    n_jobs : int
        Number of joblib workers over samples; the result does not depend on it.
    copy : bool
        Return a modified copy instead of modifying `adata` in place.

    Returns
    -------
    AnnData
        `adata` (or its copy) with one niche matrix per bandwidth.
    """
    check_adata(adata, cell_type=cell_type, sample_id=sample_id)
    if copy:
        adata = adata.copy()
    sigmas = [float(s) for s in np.atleast_1d(sigma)]

    xy = get_coords(adata)
    cts = get_obs(adata, cell_type)
    sids = get_obs(adata, sample_id)
    # sort() drops missing labels
    all_cell_types = r_sort_str(c for c in _unique(cts) if not _is_missing(c))

    # samples in order of first appearance; each subset keeps cell order (R .speApply)
    uniq_sids = _unique(sids)
    idx = [np.flatnonzero(sids == s) for s in uniq_sids]
    args = [(xy[i], cts[i], sigmas, all_cell_types, edge, diggle) for i in idx]
    if n_jobs == 1:
        per_sample = [_sample_niches(*a) for a in args]
    else:
        from joblib import Parallel, delayed

        per_sample = Parallel(n_jobs=n_jobs)(delayed(_sample_niches)(*a) for a in args)

    # writing each sample's rows at its cells' positions equals R's rbind followed by the
    # inverse of the sample-grouping permutation
    for k, sg in enumerate(sigmas):
        mat = np.zeros((adata.n_obs, len(all_cell_types)))
        for i, res in zip(idx, per_sample):
            mat[i] = res[k]
        set_niche(
            adata,
            niche_key(name, sg),
            pd.DataFrame(mat, index=adata.obs_names, columns=all_cell_types),
        )
    return adata


def _resolve_members(members: Sequence[str], prior: dict | None) -> list[str]:
    """Expand merged names back to their fine members (R `.resolveMembers`).

    A member that is a name in the `prior` mapping (a column from an earlier merge) is
    replaced by that column's fine members. An empty prior entry is R's `NULL` and leaves
    the member as it is.
    """
    if prior is None:
        return list(members)
    out: list[str] = []
    for m in members:
        out.extend(prior[m] if prior.get(m) else [m])
    return _unique(out)


def merge_niches(
    adata: AnnData,
    groups: dict[str, Sequence[str]],
    sigma: float | Sequence[float] | None = None,
    name: str = "Niche",
    copy: bool = False,
) -> AnnData:
    """Merge niche cell-type columns into groups by summing them (R `mergeNiches`).

    Cell types not named in `groups` are kept as their own (singleton) niche. The group
    membership, composed through earlier merges so that it always resolves to the original fine
    cell types, is recorded in `adata.uns["spiDE_niche_groups"][key]` for each niche matrix.

    Parameters
    ----------
    adata : AnnData
        With niche matrices (see `build_niches`).
    groups : dict
        Merged niche name -> cell-type columns to sum.
    sigma : float or sequence of float, optional
        Bandwidths to update; `None` updates every `obsm` key starting with `name`
        (R greps `^name`, so this also catches keys such as `"NicheFoo"`).
    name : str
        Key prefix of the niche matrices.
    copy : bool
        Return a modified copy instead of modifying `adata` in place.

    Returns
    -------
    AnnData
    """
    if not isinstance(groups, dict) or len(groups) == 0:
        raise ValueError("'groups' should be a named (non-empty) dict of cell-type lists")
    if copy:
        adata = adata.copy()

    if sigma is None:
        targets = [k for k in adata.obsm.keys() if re.search(f"^{name}", k)]
    else:
        targets = [niche_key(name, s) for s in np.atleast_1d(sigma)]
    missing = [k for k in _unique(targets) if k not in adata.obsm]
    if missing:
        raise ValueError(f"niche reducedDim(s) not found: {', '.join(missing)}")

    for rd in targets:
        mat = get_niche(adata, rd)
        cols = [str(c) for c in mat.columns]
        named = {m for g in groups.values() for m in g}
        leftover = _unique(c for c in cols if c not in named)
        # (name, members) pairs: a leftover that shares a name with a group stays a second
        # column, as in R
        full = [(str(g), [str(m) for m in ms]) for g, ms in groups.items()]
        full += [(c, [c]) for c in leftover]

        prior = get_group_map(adata, rd)
        members = [_unique(m for m in ms if m in cols) for _, ms in full]
        group_map: dict[str, list[str]] = {}
        for (g, _), ms in zip(full, members):
            group_map.setdefault(g, _resolve_members(ms, prior))  # R's [[ ]] finds the first

        merged = np.column_stack(
            [
                mat.iloc[:, [cols.index(m) for m in ms]].sum(axis=1).to_numpy()
                if ms
                else np.zeros(len(mat))
                for ms in members
            ]
        )
        set_niche(adata, rd, pd.DataFrame(merged, index=mat.index, columns=[g for g, _ in full]))
        set_group_map(adata, rd, group_map)
    return adata


def _is_self_niche(index: str, niches, group_map: dict | None = None) -> np.ndarray:
    """Which niche columns are the index type's own niche (R `.isSelfNiche`).

    Parameters
    ----------
    index : str
        A single index cell type.
    niches : sequence of str
        Niche column names.
    group_map : dict, optional
        Merged niche column -> fine members (raw names). Names and members are sanitised here
        before matching. A niche absent from it has the member set `{itself}`; the niche's own
        name is always a member.

    Returns
    -------
    numpy.ndarray of bool
        `True` where the niche column is the index type's own.
    """
    san_map: dict[str, list[str]] = {}
    for k, v in (group_map or {}).items():
        san_map.setdefault(sanitise(str(k)), sanitise([str(m) for m in np.atleast_1d(v)]))
    index = sanitise(str(index))
    nm = sanitise([str(n) for n in niches])
    return np.array([index == n or index in san_map.get(n, ()) for n in nm], dtype=bool)
