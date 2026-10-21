"""Per-index-type design matrices and the covariate helpers of ``fitSpiDE``.

Ports ``R/fit-design.R`` (``.indexCells``, ``.nicheColumns``, ``.indexDesign``, ``.linPred``,
``.denseX``, ``.patientBlocks``, ``.partialBlock``, ``.depthBlocks``) and, from
``R/fitSpiDE.R``, ``.pickSigma``, ``.cellCovariates``, ``.indexCovariates``, ``.indexOffset``
and ``.patientTable``.

Every index cell type is fitted on its own, with the design
``[patient intercepts | covariates | niche log-densities | (condition x niche)]``. The design
dict of :func:`_index_design` keeps R's list names (``W, patient, patients, npat, absorb, start,
pen, dense, intercept_cols, Zs, tested, niche_cols, tested_niche``) with these differences:
every index (``patient``, ``dense``, ``intercept_cols``, ``tested``, ``niche_cols``) is 0-based,
and a ``colnames`` key holds R's ``colnames(W)`` (a dense ``W`` is a plain array here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from spanormpy._block_design import NBBlockDesign, nb_block_design
from spanormpy._tps import _ns_eval, _ns_knots, tps_basis

from ._io import as_character_array, niche_keys
from ._rcompat import r_num_str, r_sort_str, sanitise
from .niches import _is_self_niche

_EPS = np.finfo(float).eps


def _is_missing(v) -> bool:
    return v is None or (not isinstance(v, str) and bool(pd.isna(v)))


def _factor_codes(x) -> tuple[list[str], np.ndarray]:
    """Levels and 0-based codes (-1 for missing) of R ``factor(x)``.

    A ``Categorical`` keeps its category order (unused ones dropped); a numeric vector sorts
    numerically (``factor(c(10, 9))`` has levels ``9, 10``) and is labelled as R prints it;
    anything else sorts as strings (:func:`r_sort_str`). Vectorised: no per-element Python.
    """
    if isinstance(x, pd.Categorical):
        codes = np.asarray(x.codes, dtype=int)
        used = sorted(set(codes.tolist()) - {-1})
        remap = np.full(len(x.categories) + 1, -1, dtype=int)
        remap[used] = np.arange(len(used))
        return [str(x.categories[i]) for i in used], np.where(codes < 0, -1, remap[codes])
    arr = np.asarray(x)
    if arr.dtype.kind in "iuf":
        v = arr.astype(float)
        ok = ~np.isnan(v)
        uniq = np.unique(v[ok])
        codes = np.full(len(v), -1, dtype=int)
        codes[ok] = np.searchsorted(uniq, v[ok])
        return [r_num_str(u) for u in uniq], codes
    raw, uniq = pd.factorize(pd.Series(arr, dtype=object))
    labs = [str(u) for u in uniq]
    levels = r_sort_str(set(labs))
    pos = {s: i for i, s in enumerate(levels)}
    lut = np.array([pos[s] for s in labs] + [-1], dtype=int)
    return levels, lut[raw]  # raw == -1 (missing) picks the trailing -1


def _levels(x) -> list[str]:
    """Levels of R ``factor(x)``: sorted unique non-missing labels (numbers sort numerically)."""
    return _factor_codes(x)[0]


# --------------------------------------------------------------------------------------
# index cells and niche columns
# --------------------------------------------------------------------------------------
def _index_cells(ct, smp, k: str, min_cells: int) -> np.ndarray:
    """0-based cells of index type ``k`` whose patient has >= ``min_cells`` of them (``.indexCells``).

    ``ct`` may hold missing values (cells left out of the fit); cell order is kept.
    """
    ct = pd.Series(np.asarray(ct, dtype=object))
    ik = np.flatnonzero((ct.notna() & (ct == k)).to_numpy())
    if ik.size == 0:
        return ik
    s = pd.Series(np.asarray(smp, dtype=object)[ik]).astype(str)
    # a missing patient is never kept (R: NA is not in names(table))
    s[pd.isna(np.asarray(smp, dtype=object)[ik])] = None
    n = s.map(s.value_counts())
    return ik[(n >= min_cells).to_numpy()]


def _niche_columns(NM: pd.DataFrame, k: str, ik, niche=None, group_map=None) -> dict:
    """Niche columns of index type ``k`` (``.nicheColumns``).

    Every column that is not ``k``'s own niche (merge groups honoured) and has positive,
    finite variance of ``log1p`` over the index cells; ``tested`` is ``cols`` or its
    intersection with ``niche`` (in the order of ``cols``).
    """
    nm = [str(c) for c in NM.columns]
    if len(nm):
        self_ = _is_self_niche([sanitise(k)] * len(nm), sanitise(nm), group_map)
    else:
        self_ = np.zeros(0, dtype=bool)
    cols = [c for c, s in zip(nm, self_) if not s]
    if cols:
        with np.errstate(invalid="ignore"):
            M = np.log1p(NM.iloc[np.asarray(ik, dtype=int)][cols].to_numpy(dtype=float))
        v = _colvar(M)
        cols = [c for c, vi in zip(cols, v) if np.isfinite(vi) and vi > 0]
    tested = cols if niche is None else [c for c in cols if c in set(np.atleast_1d(niche))]
    return {"cols": cols, "tested": list(tested)}


def _colvar(M: np.ndarray) -> np.ndarray:
    """Column variances as R ``var``: a column holding NaN, or a single row, gives NaN.

    Accumulated in long double with a second pass for the mean, as R's ``cov`` does, so a column
    of identical values has variance exactly 0 (a plain double mean can leave ~1e-33 and let a
    constant niche column through the ``v > 0`` filter).
    """
    n = M.shape[0]
    if n < 2:
        return np.full(M.shape[1], np.nan)
    Ml = M.astype(np.longdouble)
    m = Ml.mean(axis=0)
    m = m + (Ml - m).mean(axis=0)
    v = ((Ml - m) ** 2).sum(axis=0) / (n - 1)
    return np.where(np.isnan(M).any(axis=0), np.nan, v.astype(float))


# --------------------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------------------
def _index_design(
    L: pd.DataFrame,
    cov: pd.DataFrame | None,
    patient_codes,
    patient_levels,
    trt=None,
    strata=None,
    tested=None,
    blocks: dict | None = None,
) -> dict:
    """The design for one index type (``.indexDesign``).

    Parameters
    ----------
    L
        index cells x niche columns (``log1p`` densities, tested columns first).
    cov
        index cells x covariates (centred), or ``None``.
    patient_codes, patient_levels
        0-based patient code of each index cell and the patient labels (R ``factor``).
    trt
        ``None`` (the pooled model) or 0/1 per cell (the condition model, adding
        ``condition:<niche>`` columns).
    strata
        ``None`` or a patient-level factor per cell, entered as ``strata:<level>:<niche>``
        nuisance columns (treatment coding, the first level dropped).
    tested
        niche columns to test (default all of ``L``).
    blocks
        output of :func:`_depth_blocks` (``Zc``, ``Zs``): the within-patient library-size
        spline, held as a compact block design.

    Returns
    -------
    dict
        R's list (see the module docstring for the 0-based indices). ``dense`` and
        ``intercept_cols`` index the columns of ``W``; for a block design ``W`` is an
        ``NBBlockDesign`` and ``absorb`` is ``None``.
    """
    codes = np.asarray(patient_codes, dtype=int)
    levels = [str(x) for x in patient_levels]
    np_ = len(levels)
    n = len(L)
    lnames = [str(c) for c in L.columns]
    tested = lnames if tested is None else list(tested)
    Lm = L.to_numpy(dtype=float)
    parts: list[np.ndarray] = []
    names: list[str] = []
    if cov is not None and cov.shape[1]:
        parts.append(cov.to_numpy(dtype=float))
        names += [str(c) for c in cov.columns]
    if strata is not None:
        st_levels, st_codes = _factor_codes(strata)
        if len(st_levels) > 1:
            # model.matrix(~ st)[, -1]: one 0/1 column per level but the first
            Sd = np.column_stack([(st_codes == i).astype(float) for i in range(1, len(st_levels))])
            SN = np.column_stack([Sd * Lm[:, [j]] for j in range(Lm.shape[1])])
            sn_names = [
                f"strata:{lv}:{ln}" for ln in lnames for lv in st_levels[1:]
            ]  # rep(levels[-1], ncol(L)) pasted with rep(colnames(L), each = ncol(Sd))
            nz = (SN != 0).sum(axis=0) > 0
            parts.append(SN[:, nz])
            names += [s for s, k in zip(sn_names, nz) if k]
    parts.append(Lm)
    names += lnames
    if trt is None:
        tcols = tested
    else:
        trt = np.asarray(trt, dtype=float)
        parts.append(Lm * trt[:, None])
        names += [f"condition:{c}" for c in lnames]
        tcols = [f"condition:{c}" for c in tested]
    X = np.hstack(parts)
    px = X.shape[1]
    base = {"patient": codes, "patients": levels, "npat": np_}
    if blocks is not None:
        q = blocks["Zc"].shape[1]
        cat = pd.Categorical.from_codes(codes, categories=levels)
        W = nb_block_design(pd.DataFrame(X, columns=names), blocks["Zc"], cat)
        cn = list(W.colnames)
        return {
            "W": W,
            **base,
            "absorb": None,
            "start": np.array([False] * px + [True] + [False] * (q - 1)),
            "pen": np.concatenate([np.zeros(px), np.full(q, 1e-3)]),
            "dense": np.arange(px),
            "intercept_cols": px + np.arange(np_) * q,
            "Zs": blocks["Zs"],
            "tested": _match(tcols, cn),
            "niche_cols": _match(lnames, cn),
            "tested_niche": tested,
            "colnames": cn,
        }
    P = np.zeros((n, np_))
    P[np.arange(n), codes] = 1.0
    W = np.hstack([P, X])
    cn = [f"patient:{lv}" for lv in levels] + names
    return {
        "W": W,
        **base,
        "absorb": np.array([True] * np_ + [False] * px),
        "start": np.array([True] * np_ + [False] * px),
        "pen": np.concatenate([np.full(np_, 1e-3), np.zeros(px)]),
        "dense": np_ + np.arange(px) if px else np.zeros(0, dtype=int),
        "intercept_cols": np.arange(np_),
        "Zs": None,
        "tested": _match(tcols, cn),
        "niche_cols": _match(lnames, cn),
        "tested_niche": tested,
        "colnames": cn,
    }


def _match(x, table) -> np.ndarray:
    """0-based first position of each ``x`` in ``table`` (R ``match``, an absent name is an error)."""
    pos = {}
    for i, t in enumerate(table):
        pos.setdefault(t, i)
    try:
        return np.array([pos[v] for v in x], dtype=int)
    except KeyError as e:
        raise ValueError(f"column {e.args[0]!r} not found in the design") from None


def _lin_pred(W, alpha) -> np.ndarray:
    """Linear predictor ``W alpha`` of one gene, for a dense or compact block design (``.linPred``)."""
    alpha = np.asarray(alpha, dtype=float)
    if isinstance(W, NBBlockDesign):
        X = W.X.cpu().numpy()
        Z = W.Z.cpu().numpy()
        bid = W.bid.cpu().numpy()
        A = alpha[W.px :].reshape(W.G, W.q)
        return X @ alpha[: W.px] + (Z * A[bid]).sum(axis=1)
    return np.asarray(W) @ alpha


def _dense_x(des: dict) -> np.ndarray:
    """The non-absorbed (dense) columns of the design (``.denseX``)."""
    if isinstance(des["W"], NBBlockDesign):
        return des["W"].X.cpu().numpy()
    return des["W"][:, des["dense"]]


def _patient_blocks(des: dict, rows_of) -> list[np.ndarray]:
    """Each patient's absorbed block over its own cells (its intercept alone without depth blocks)."""
    if des["Zs"] is not None:
        return des["Zs"]
    return [np.ones((len(r), 1)) for r in rows_of]


def _partial_block(X, w, Z):
    """``X`` residualised on a patient's block ``Z`` with weights ``w`` (``.partialBlock``).

    Frisch-Waugh-Lovell: ``X - Z (Z'WZ)^-1 Z'WX``; with ``Z`` the intercept alone this is
    weighted centring within the patient. ``None`` if the block is singular (R ``solve`` fails
    when the reciprocal condition number is below machine epsilon).
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.shape[1] == 1 and np.all(Z == 1):
        return X - ((X * w[:, None]).sum(axis=0) / w.sum())[None, :]
    Zs = Z * np.sqrt(w)[:, None]
    G = Zs.T @ Zs
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return None
    rcond = 1.0 / (np.linalg.norm(G, 1) * np.linalg.norm(Ginv, 1))
    if not np.isfinite(rcond) or rcond < _EPS:
        return None
    cf = np.linalg.solve(G, Z.T @ (X * w[:, None]))
    return X - Z @ cf


def _depth_blocks(ell, xy_all, sec_all, ik, patient_codes, patient_levels, L: pd.DataFrame) -> dict:
    """The within-patient library-size spline of ``depth = "spatial_spline"`` (``.depthBlocks``).

    For each section of each patient, the columns ``[l, l * B]`` over that section's index
    cells: ``l`` the log library size centred within (section, index type) and ``B`` the
    section's centred natural-spline tensor basis of position (spanormpy ``tps_basis`` on the
    whole section's coordinates). The basis has ``df x df`` columns, ``df = 3`` for a section
    with >= 200 index cells, 2 with >= 100 and none (``l`` alone) below.

    Returns ``Zs`` (per patient: ``[1 | its sections' columns]`` over its cells), ``Zc`` (a
    DataFrame, every cell's row of its patient's ``Zs`` zero-padded to the widest patient) and
    ``r2`` (patients x niche columns: R^2 of each niche column on the patient's block).
    """
    ell = np.asarray(ell, dtype=float)
    if not np.all(np.isfinite(ell)):
        raise ValueError('depth = "spatial_spline" needs a finite log library size for every cell')
    ik = np.asarray(ik, dtype=int)
    sec_all = np.asarray(sec_all, dtype=object).astype(str)
    sec = sec_all[ik]
    pid = np.asarray(patient_codes, dtype=int)
    levels = [str(x) for x in patient_levels]
    S = len(levels)
    Lm = L.to_numpy(dtype=float)
    Zs: list[np.ndarray] = []
    r2 = np.full((S, Lm.shape[1]), np.nan)
    for s in range(S):
        i = np.flatnonzero(pid == s)
        cols = []
        for sg in pd.unique(sec[i]):  # unique, in order of appearance
            j = np.flatnonzero(sec[i] == sg)
            lc = ell[i[j]] - ell[i[j]].mean()
            df = 3 if len(j) >= 200 else 2 if len(j) >= 100 else 0
            z = np.zeros((len(i), 1 + (df**2 if df else 0)))
            z[j, 0] = lc
            if df:
                ref = np.flatnonzero(sec_all == sg)
                B = tps_basis(
                    xy_all[ik[i[j]], 0],
                    xy_all[ik[i[j]], 1],
                    df=(df, df),
                    ref_x=xy_all[ref, 0],
                    ref_y=xy_all[ref, 1],
                )
                z[j, 1:] = lc[:, None] * B
            cols.append(z)
        Zs.append(np.hstack([np.ones((len(i), 1)), *cols]))
        for k in range(Lm.shape[1]):
            y = Lm[i, k]
            tss = float(((y - y.mean()) ** 2).sum())
            if tss > 0:
                res = y - Zs[s] @ np.linalg.lstsq(Zs[s], y, rcond=None)[0]
                r2[s, k] = 1 - float((res**2).sum()) / tss
    q = max(z.shape[1] for z in Zs)
    Zc = np.zeros((len(ell), q))
    for s in range(S):
        Zc[pid == s, : Zs[s].shape[1]] = Zs[s]
    names = ["intercept"] + [f"depth{i}" for i in range(1, q)]
    return {
        "Zc": pd.DataFrame(Zc, columns=names),
        "Zs": Zs,
        "r2": pd.DataFrame(r2, index=levels, columns=[str(c) for c in L.columns]),
    }


# --------------------------------------------------------------------------------------
# helpers of fitSpiDE
# --------------------------------------------------------------------------------------
def _as_numeric(x: str) -> float:
    """R ``suppressWarnings(as.numeric(x))`` of a digits-and-dots string: NaN when unparseable."""
    try:
        return float(x)
    except ValueError:
        return float("nan")


def _pick_sigma(adata, sigma, name: str) -> float:
    """The niche bandwidth: the only one present, or the one asked for (``.pickSigma``)."""
    keys = niche_keys(adata, name)
    have = [_as_numeric(k[len(name) :]) for k in keys]
    if not have:
        raise ValueError("no niche reducedDims found; run buildNiches() first")
    if sigma is None:
        if len(have) > 1:
            # sort() drops the NA of an unparseable key such as "Niche1.2.3"
            shown = ", ".join(r_num_str(h) for h in sorted(h for h in have if not np.isnan(h)))
            raise ValueError(
                f"spe carries several niche bandwidths ({shown}): choose one with 'sigma'"
            )
        return have[0]
    if np.size(sigma) != 1:
        raise ValueError("'sigma' must be a single bandwidth")
    return float(np.ravel(sigma)[0])


def _numeric_column(x) -> np.ndarray:
    """R ``as.numeric`` of one obs column (a factor gives its 1-based codes)."""
    if isinstance(x.dtype, pd.CategoricalDtype):
        c = x.cat.codes.to_numpy(dtype=float) + 1
        c[c == 0] = np.nan
        return c
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def _cell_covariates(cd: pd.DataFrame, covariates, depth: str, Y) -> pd.DataFrame:
    """Cell-level covariates over all cells: ``loglib`` (unless depth is "none") and user ones.

    Port of ``.cellCovariates``; ``Y`` is genes x cells. A cell with no counts has a missing
    (NaN) ``loglib``. A table with no columns has ``len(cd)`` rows.
    """
    cols: dict[str, np.ndarray] = {}
    if depth != "none":
        lib = np.asarray(Y.sum(axis=0), dtype=float).ravel()
        with np.errstate(divide="ignore"):
            cols["loglib"] = np.where(lib > 0, np.log(np.where(lib > 0, lib, 1.0)), np.nan)
    for cv in covariates:
        cols[cv] = _numeric_column(cd[cv])
    if not cols:
        return pd.DataFrame(index=cd.index)
    return pd.DataFrame(cols, index=cd.index)


def _index_covariates(covk: pd.DataFrame, depth: str) -> pd.DataFrame | None:
    """The index type's covariates, centred over its cells (``.indexCovariates``).

    Under ``depth = "nonlinear"`` the log library size becomes a 3-df natural cubic spline
    (``splines::ns(x, df = 3)``, knots at this index type's quantiles); under
    ``"spatial_spline"`` it is dropped (depth enters through the blocks). ``None`` if nothing
    is left.
    """
    if covk.shape[1] == 0:
        return None
    if depth == "spatial_spline":
        covk = covk.drop(columns="loglib")
        if covk.shape[1] == 0:
            return None
    elif depth == "nonlinear":
        x = covk["loglib"].to_numpy(dtype=float)
        knots, bk = _ns_knots(x, 3)
        B = _ns_eval(x, knots, bk)
        Bdf = pd.DataFrame(
            B, index=covk.index, columns=[f"loglib_ns{i + 1}" for i in range(B.shape[1])]
        )
        covk = pd.concat([Bdf, covk.drop(columns="loglib")], axis=1)
    return covk - np.nanmean(covk.to_numpy(dtype=float), axis=0)


def _index_offset(adata, offset: str, gk, ik) -> np.ndarray:
    """The offset of one index type: genes x index cells, dense (``.indexOffset``).

    ``offset`` names a layer of ``adata`` (cells x genes); ``gk`` are gene names and ``ik``
    0-based cell positions.
    """
    M = adata.layers[offset]
    gi = adata.var_names.get_indexer(list(gk))
    if (gi < 0).any():  # R: assay(spe, offset)[gk, ik] on an unknown gene name
        raise ValueError("subscript out of bounds")
    ik = np.asarray(ik, dtype=int)
    if sp.issparse(M):  # columns first (CSC-friendly), then rows; never cells x all genes dense
        sub = sp.csc_matrix(M)[:, gi][ik].toarray()
    else:  # index cells and genes together
        sub = np.asarray(M, dtype=float)[np.ix_(ik, gi)]
    O = sub.T
    if not np.all(np.isfinite(O)):
        raise ValueError(f"offset assay '{offset}' has non-finite values")
    return O


def _patient_table(cd: pd.DataFrame, smp) -> pd.DataFrame:
    """One row per patient: id, cell count and every obs column constant within patients.

    Port of ``.patientTable``. A column counts as constant when each patient has at most one
    distinct non-missing value (compared as R ``as.character``); the patient's value is its
    first cell's (so a missing first cell gives a missing value, as in R).
    """
    raw, uniq = pd.factorize(
        pd.Series(np.asarray(smp, dtype=object))
        .astype(str)
        .where(pd.notna(np.asarray(smp, dtype=object)))
    )
    pats = r_sort_str(uniq)
    order = {p: i for i, p in enumerate(pats)}
    lut = np.array([order[u] for u in uniq] + [-1], dtype=int)
    pid = lut[raw]  # patient position of each cell, -1 for a missing sample id
    ok = pid >= 0
    ncells = np.bincount(pid[ok], minlength=len(pats)).astype(int)
    first = np.full(len(pats), -1, dtype=int)
    first[pid[ok][::-1]] = np.flatnonzero(ok)[::-1]  # first occurrence wins
    out = {"patient": pats, "ncells": ncells}
    for cn in cd.columns:
        x = cd[cn]
        ccodes, _ = pd.factorize(as_character_array(x))
        # distinct non-missing values per patient (cells with a missing sample are ignored)
        use = ok & (ccodes >= 0)
        pairs = pd.DataFrame({"p": pid[use], "c": ccodes[use]}).drop_duplicates()
        if not (pairs.groupby("p").size() > 1).any():
            if isinstance(x.dtype, pd.CategoricalDtype):
                out[cn] = x.iloc[first].cat.remove_unused_categories().reset_index(drop=True)
            else:
                out[cn] = x.iloc[first].reset_index(drop=True).to_numpy()
    return pd.DataFrame(out)
