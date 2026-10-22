"""The slopes engine's per-patient niche slopes and their variances (``R/patient-slopes.R``).

From the shared, condition-free fit of an index type, every patient gets a one-step negative
binomial estimate of its own niche slopes (one Fisher-scoring step from the pooled slope, with
the other coefficients and the dispersion held at the shared fit and the patient's intercept
profiled out). Two variances accompany each slope: the model-based ``diag(I_s^-1)`` and a
within-patient spatial tile sandwich. Arrays are genes x patients x tested niches in R's
dimension order, NaN where R has NA.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._design import _partial_block, _patient_blocks
from ._glm import _bp_genes, _row, _working_wr


def _median_cols(x: np.ndarray) -> np.ndarray:
    """Column medians ignoring NaN; NaN for an all-NaN column (R ``median(na.rm = TRUE)``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=0)


def _patient_slopes(
    fit: dict,
    des: dict,
    Yk,
    L,
    tiles,
    min_cells: int = 10,
    min_tiles: int = 5,
    offset=None,
    n_jobs: int = 1,
) -> dict:
    """Per-patient niche slopes and variances for every gene (``.patientSlopes``).

    Parameters
    ----------
    fit : dict
        Shared fit with ``alpha`` (genes x columns of the design) and ``psi`` (genes).
    des : dict
        The per-index design from ``_index_design``.
    Yk : array or sparse matrix
        Counts of the index cells, genes x cells.
    L : array
        Niche log-densities of the index cells, cells x columns, tested niches first.
    tiles : array
        Tile label of each index cell (the spatial variance clusters).
    min_cells, min_tiles : int
        Minimum index cells a patient needs; minimum tiles for its own spatial variance.
    offset : array, optional
        Per-gene, per-cell log offset, genes x cells.
    n_jobs : int
        Gene blocks run in parallel; the result does not depend on it.

    Returns
    -------
    dict
        ``beta``, ``v_tile``, ``v_model`` (genes x patients x tested niches), ``pooled``
        (genes x tested niches), plus the ``patients`` and ``niches``
        labels.
    """
    alpha = np.asarray(fit["alpha"], dtype=float)
    psi = np.asarray(fit["psi"], dtype=float)
    G = alpha.shape[0]
    S = des["npat"]
    nt = len(des["tested_niche"])
    jn = np.asarray(des["niche_cols"])
    L = np.asarray(L, dtype=float)
    patient = np.asarray(des["patient"])
    rows_of = [np.flatnonzero(patient == s) for s in range(S)]
    # each patient's absorbed block (its intercept, and its library-size spline under
    # depth = "spatial_spline") over its own cells
    Z_of = _patient_blocks(des, rows_of)
    tiles = np.asarray(tiles)
    tile_codes = [np.unique(tiles[i], return_inverse=True)[1] for i in rows_of]

    def one_gene(g: int):
        bb = np.full((S, nt), np.nan)
        vv = np.full((S, nt), np.nan)
        vt = np.full((S, nt), np.nan)
        al, ps = alpha[g], psi[g]
        if not (np.all(np.isfinite(al)) and np.isfinite(ps)):
            return bb, vv, vt
        wr = _working_wr(_row(Yk, g), des["W"], al, ps, offset=0.0 if offset is None else offset[g])
        w, r = wr["w"], wr["r"]
        beta = al[jn]
        for s in range(S):
            i = rows_of[s]
            if len(i) < min_cells:
                continue
            ws = w[i]
            Ls = L[i]
            Lt = _partial_block(Ls, ws, Z_of[s])
            if Lt is None:
                continue
            Is = (Lt * np.sqrt(ws)[:, None]).T @ (Lt * np.sqrt(ws)[:, None])
            us = (Lt * r[i][:, None]).sum(axis=0)
            # a non-finite weight or residual (an overflowing mean): this patient drops
            if not (np.all(np.isfinite(Is)) and np.all(np.isfinite(us))):
                continue
            # a niche absent from (or constant across) this patient's cells leaves nothing
            # after centring: only that slope drops
            dg = np.diag(Is)
            keep = np.flatnonzero(dg > 1e-12 * (Ls**2 * ws[:, None]).sum(axis=0))
            if keep.size == 0:
                continue
            # the rank check runs on the correlation scale: collinearity alone, not a
            # column's spread or units
            sc = 1.0 / np.sqrt(dg[keep])
            ssc = np.outer(sc, sc)
            ev, vec = np.linalg.eigh(Is[np.ix_(keep, keep)] * ssc)  # ascending (R: descending)
            if not np.all(np.isfinite(ev)) or ev[0] <= ev[-1] * 1e-8:
                continue
            Iinv = ((vec / ev) @ vec.T) * ssc
            d = np.full(Is.shape[0], np.nan)
            vd = np.full(Is.shape[0], np.nan)
            d[keep] = Iinv @ us[keep]
            vd[keep] = np.diag(Iinv)
            bb[s] = (beta + d)[:nt]
            vv[s] = vd[:nt]
            nT = int(tile_codes[s].max()) + 1
            if nT >= min_tiles:
                Ut = np.zeros((nT, keep.size))
                np.add.at(Ut, tile_codes[s], Lt[:, keep] * r[i][:, None])
                Ut -= Ut.mean(axis=0)
                vd[keep] = np.diag(Iinv @ (Ut.T @ Ut) @ Iinv) * nT / (nT - 1)
                vt[s] = vd[:nt]
        # patients without their own tile variance take the model variance times the gene's
        # median tile / model ratio (at least 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = _median_cols(vt / vv)
        rat[~np.isfinite(rat)] = 1.0
        miss = np.isnan(vt) & np.isfinite(vv)
        vt[miss] = (vv * np.maximum(rat, 1.0)[None, :])[miss]
        return bb, vv, vt

    res = [x for b in _bp_genes(G, lambda idx: [one_gene(int(g)) for g in idx], n_jobs) for x in b]
    beta_a = np.full((G, S, nt), np.nan)
    vt_a = np.full((G, S, nt), np.nan)
    vm_a = np.full((G, S, nt), np.nan)
    for g, (b, vm, vt) in enumerate(res):
        beta_a[g], vm_a[g], vt_a[g] = b, vm, vt
    return {
        "beta": beta_a,
        "v_tile": vt_a,
        "v_model": vm_a,
        "pooled": alpha[:, jn[:nt]],
        "patients": list(des["patients"]),
        "niches": list(des["tested_niche"]),
    }


def _patient_factor(v_tile: np.ndarray, v_model: np.ndarray) -> np.ndarray:
    """The per-(patient, niche) variance factor of the pooled test (``.patientFactor``).

    ``f_sn = max(1, median over genes of v_tile / v_model)`` over the whole gene family, with
    non-finite ratios ignored and 1 where no ratio is left. Returns patients x niches.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(v_tile, dtype=float) / np.asarray(v_model, dtype=float)
    r[~np.isfinite(r)] = np.nan
    S, nt = r.shape[1], r.shape[2]
    med = _median_cols(r.reshape(r.shape[0], S * nt)).reshape(S, nt)
    f = np.maximum(med, 1.0)  # NaN stays NaN
    f[~np.isfinite(f)] = 1.0
    return f
