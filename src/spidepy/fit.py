"""``fit_spide``: the per-patient niche models of every index cell type (``R/fitSpiDE.R``).

For each index cell type, one negative binomial GLM per gene of its expression on the local
density of every other (niche) cell type, with an intercept per patient so that each niche
effect is a within-patient slope. The slopes engine then gives every patient a one-step
estimate of its own slopes; the sandwich engine tests the fit's coefficients with a
patient-clustered CR2 sandwich.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from . import __version__
from ._checks import (
    check_adata,
    check_column,
    check_condition,
    check_counts,
    check_covariates,
    check_niche,
    check_sample,
)
from ._classes import SpiDEFit
from ._design import (
    _cell_covariates,
    _depth_blocks,
    _factor_codes,
    _index_cells,
    _index_covariates,
    _index_design,
    _index_offset,
    _niche_columns,
    _patient_table,
    _pick_sigma,
)
from ._glm import _fit_index_glm
from ._io import (
    as_character_array,
    get_coords,
    get_counts,
    get_group_map,
    get_niche,
    get_obs,
    niche_key,
)
from ._rcompat import r_sort_str
from ._sandwich import _sandwich_cr2
from ._slopes import _patient_factor, _patient_slopes

_ENGINES = ("slopes", "sandwich")
_DEPTHS = ("loglib", "nonlinear", "spatial_spline", "none")


def _match_arg(x, choices, what: str) -> str:
    """R ``match.arg`` (exact matches only)."""
    if not isinstance(x, str) or x not in choices:
        raise ValueError(f"'{what}' should be one of {', '.join(repr(c) for c in choices)}")
    return x


def fit_spide(
    adata: AnnData,
    condition: str | None = None,
    engine: str = "slopes",
    index: str | Sequence[str] | None = None,
    niche: str | Sequence[str] | None = None,
    sigma: float | None = None,
    cell_type: str = "cell_type",
    sample_id: str = "sample_id",
    section: str | None = None,
    covariates: str | Sequence[str] = (),
    depth: str = "loglib",
    offset: str | None = None,
    strata: str | None = None,
    assay: str = "counts",
    name: str = "Niche",
    min_cells: int = 10,
    min_patients: int = 6,
    min_detect: float = 0.1,
    genes: Sequence[str] | None = None,
    tile: float | None = None,
    min_tiles: int = 5,
    n_jobs: int = 1,
    verbose: bool = True,
) -> SpiDEFit:
    """Fit spiDE's per-patient niche models (R ``fitSpiDE``).

    Fits, for every index cell type, one negative binomial GLM per gene of its expression on
    the local density of every other (niche) cell type, with an intercept per patient so that
    each niche effect is a within-patient slope. The slopes are then tested across patients,
    pooled or between conditions.

    Two engines estimate the between-patient error the tests need. ``"slopes"`` (default)
    fits the shared model once, without the condition, then gives every patient a one-step NB
    estimate of its own niche slopes with a within-patient spatial sandwich variance; its fit
    never sees the condition. ``"sandwich"`` fits the condition-specific model directly and
    tests its coefficients with a patient-clustered CR2 sandwich and Bell-McCaffrey degrees of
    freedom; it needs the condition at fit time.

    Parameters
    ----------
    adata : AnnData
        Cells x genes with raw integer counts, ``obsm["spatial"]`` and the niche matrices of
        :func:`~spidepy.niches.build_niches`.
    condition : str, optional
        The ``obs`` column holding a two-level, patient-level condition. The slopes engine
        only records it; the sandwich engine fits the condition-specific model only when it
        is given.
    engine : {"slopes", "sandwich"}
        The between-patient error engine.
    index : str or sequence of str, optional
        Index cell types (default: every type with at least ``min_cells`` cells in at least
        ``min_patients`` patients).
    niche : str or sequence of str, optional
        Niche columns to test (default all); the others stay in the model as adjustments. An
        index type is never tested against its own niche (see ``merge_niches``).
    sigma : float, optional
        The niche bandwidth (needed only if ``adata`` carries several).
    cell_type, sample_id : str
        The ``obs`` columns of the cell type and of the patient (the replicated unit).
    section : str, optional
        The ``obs`` column of the tissue section (default ``sample_id``), used to lay the
        spatial tiles of the variance estimate.
    covariates : sequence of str
        Cell-level ``obs`` columns to adjust for (centred; patient-level ones are refused, the
        patient intercepts absorb them).
    depth : {"loglib", "nonlinear", "spatial_spline", "none"}
        How sequencing depth enters: the centred log library size as a covariate; a 3-df
        natural spline of it; a library-size spline within each patient that may vary over the
        section; or not at all (e.g. when ``offset`` carries depth).
    offset : str, optional
        The name of a layer holding a log-scale offset per cell and gene, added to every
        linear predictor with its coefficient fixed at 1.
    strata : str, optional
        A patient-level ``obs`` column (e.g. slide or batch) whose niche slopes are adjusted
        for in the condition test. Used by the sandwich engine's condition model here and by
        the slopes engine at test time; the pooled model never uses it.
    assay : str
        The layer of raw integer counts (``X`` for ``"counts"`` when no such layer exists).
    name : str
        The niche ``obsm`` prefix.
    min_cells, min_patients : int
        Minimum cells of an index type a patient must contribute; minimum patients a type needs.
    min_detect : float
        Minimum fraction of an index type's cells in which a gene is detected.
    genes : sequence of str, optional
        Genes to test (default every gene). Library size is always computed from every gene.
    tile : float, optional
        Side of the square tiles of the spatial variance estimate (default 3 x ``sigma``).
    min_tiles : int
        Minimum tiles a patient needs for its own spatial variance.
    n_jobs : int
        Genes are fitted in blocks over this many workers; the result does not depend on it
        beyond floating-point summation order.
    verbose : bool
        Report progress on stderr.

    Returns
    -------
    SpiDEFit
    """
    engine = _match_arg(engine, _ENGINES, "engine")
    depth = _match_arg(depth, _DEPTHS, "depth")
    if isinstance(covariates, str):
        covariates = [covariates]
    covariates = list(covariates)
    check_adata(adata, assay=assay, cell_type=cell_type, sample_id=sample_id)
    if condition is not None:
        check_condition(adata, condition)
    check_sample(adata, condition=condition, sample_id=sample_id, covariates=covariates)
    check_covariates(adata, covariates, finite_only=True)
    sigma = _pick_sigma(adata, sigma, name)
    check_niche(adata, sigma, name)
    check_column(adata, strata, "strata")
    check_column(adata, section, "section")
    Y = get_counts(adata, assay)  # genes x cells
    check_counts(Y, integer_only=True)
    if offset is not None and (not isinstance(offset, str) or offset not in adata.layers):
        raise ValueError(
            "'offset' must name an assay of adata (a log-scale offset per gene and cell)"
        )

    def say(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    cd = adata.obs
    ct = get_obs(adata, cell_type)
    smp = get_obs(adata, sample_id)
    sec = smp if section is None else get_obs(adata, section)
    key = niche_key(name, sigma)
    NM = get_niche(adata, key)
    xy = get_coords(adata)
    tile = 3 * sigma if tile is None else tile
    cov = _cell_covariates(cd, covariates, depth, Y)
    # a cell with no counts has no library size: it cannot enter a model with a log-depth
    # term, so it is left out (reported, not silently)
    if depth != "none":
        usable = np.isfinite(cov["loglib"].to_numpy())
    else:
        usable = np.ones(Y.shape[1], dtype=bool)
    if not usable.all():
        say(f"fit_spide: {int((~usable).sum())} cell(s) with no counts left out")
    patients = _patient_table(cd, smp)
    group_map = get_group_map(adata, key)

    if index is None:
        index = _default_index(ct, smp, min_cells, min_patients)
    else:
        index = [index] if isinstance(index, str) else list(index)
    seen = set(pd.unique(ct[pd.notna(ct)]))
    index = [k for k in dict.fromkeys(index) if k in seen]  # R intersect(index, unique(ct))
    if not index:
        raise ValueError("no index cell type has enough cells in enough patients")

    trt_of = _condition_coding(patients, condition) if engine == "sandwich" and condition else None
    str_of = _strata_by_patient(patients, strata) if strata is not None else None
    # every cell's log library size, for the patients' mean depth that the plots read
    if "loglib" in cov.columns:
        ll = cov["loglib"].to_numpy()
    else:
        with np.errstate(divide="ignore"):
            ll = np.log(np.asarray(Y.sum(axis=0), dtype=float).ravel())
    ct_use = np.where(usable, ct, np.nan).astype(object)
    var_names = adata.var_names
    gene_keep = None if genes is None else set(genes)

    fits: dict[str, dict] = {}
    for k in index:
        ik = _index_cells(ct_use, smp, k, min_cells)
        pat_levels, pat_codes = _factor_codes(smp[ik])
        if len(pat_levels) < min_patients:
            say(f"fit_spide: skipping {k} ({len(pat_levels)} patients)")
            continue
        nc = _niche_columns(NM, k, ik, niche, group_map)
        if not nc["tested"]:
            continue
        Ysub = Y[:, ik]
        if sp.issparse(Ysub):
            Ysub = Ysub.tocsr()
        det = np.asarray((Ysub > 0).sum(axis=1)).ravel() / len(ik)
        gi = np.flatnonzero(det >= min_detect)
        if gene_keep is not None:
            gi = np.array([i for i in gi if var_names[i] in gene_keep], dtype=int)
        if gi.size == 0:
            continue
        gk = [str(g) for g in var_names[gi]]
        cols = nc["tested"] + [c for c in nc["cols"] if c not in set(nc["tested"])]
        L = pd.DataFrame(np.log1p(NM.iloc[ik][cols].to_numpy(dtype=float)), columns=cols)
        Yk = Ysub[gi]
        covk = _index_covariates(cov.iloc[ik], depth)
        blk = (
            _depth_blocks(cov["loglib"].to_numpy()[ik], xy, sec, ik, pat_codes, pat_levels, L)
            if depth == "spatial_spline"
            else None
        )
        Ok = None if offset is None else _index_offset(adata, offset, gk, ik)
        say(
            f"fit_spide: {engine} engine, index {k}: {len(gk)} genes, {len(ik)} cells, "
            f"{len(pat_levels)} patients, {len(nc['tested'])} niches"
        )
        # the pooled model has no strata columns: with them (treatment-coded) its niche
        # coefficient would be the slope of the first stratum's patients, not the average
        # over all patients the pooled test is about
        des0 = _index_design(L, covk, pat_codes, pat_levels, tested=nc["tested"], blocks=blk)
        fit0 = _fit_index_glm(Yk, des0, offset=Ok, n_jobs=n_jobs)
        head = {
            "genes": gk,
            "niches": list(nc["tested"]),
            "adjusted": [c for c in nc["cols"] if c not in set(nc["tested"])],
            "patients": list(des0["patients"]),
        }
        depth_r2 = None if blk is None else blk["r2"][nc["tested"]]
        mean_expr = pd.Series(np.asarray(Yk.mean(axis=1)).ravel(), index=gk)
        levels = _patient_levels(fit0, des0, gk, L[nc["tested"]], pat_codes, ll[ik])
        if engine == "slopes":
            tiles = _tile_codes(sec, xy, tile, ik)
            ps = _patient_slopes(
                fit0,
                des0,
                Yk,
                L.to_numpy(),
                tiles,
                min_cells=min_cells,
                min_tiles=min_tiles,
                offset=Ok,
                n_jobs=n_jobs,
            )
            pats, nich = head["patients"], head["niches"]
            fits[k] = {
                **head,
                "beta": ps["beta"],
                "v_tile": ps["v_tile"],
                "v_model": ps["v_model"],
                "pooled": pd.DataFrame(ps["pooled"], index=gk, columns=nich),
                "ncells": pd.Series(ps["ncells"], index=pats),
                "factor": pd.DataFrame(
                    _patient_factor(ps["v_tile"], ps["v_model"]), index=pats, columns=nich
                ),
                "mean_expr": mean_expr,
                "psi": pd.Series(fit0["psi"], index=gk),
                "status": fit0["status"],
                "depth_r2": depth_r2,
                **levels,
            }
        else:
            coef = _sandwich_cr2(fit0, des0, Yk, offset=Ok, n_jobs=n_jobs, genes=gk)
            coef.insert(0, "test", "pooled")
            if trt_of is not None:
                trt = trt_of.reindex(pat_levels).to_numpy()[pat_codes]  # by patient name, as R
                stk = None if str_of is None else str_of.reindex(pat_levels).to_numpy()[pat_codes]
                des1 = _index_design(
                    L,
                    covk,
                    pat_codes,
                    pat_levels,
                    trt=trt,
                    strata=stk,
                    tested=nc["tested"],
                    blocks=blk,
                )
                fit1 = _fit_index_glm(Yk, des1, offset=Ok, n_jobs=n_jobs)
                c1 = _sandwich_cr2(fit1, des1, Yk, offset=Ok, n_jobs=n_jobs, genes=gk)
                c1.insert(0, "test", "condition")
                coef = pd.concat([coef, c1], ignore_index=True)
            fits[k] = {
                **head,
                "coef": coef,
                "mean_expr": mean_expr,
                "status": fit0["status"],
                "depth_r2": depth_r2,
                "ncells": pd.Series(
                    np.bincount(pat_codes, minlength=len(pat_levels)), index=pat_levels
                ),
                **levels,
            }
    if not fits:
        raise ValueError("no index cell type could be fitted")
    return SpiDEFit(
        engine=engine,
        sigma=sigma,
        condition=condition,
        patients=patients,
        index=fits,
        params={
            "cell_type": cell_type,
            "sample_id": sample_id,
            "section": section,
            "covariates": covariates,
            "depth": depth,
            "offset": offset,
            "strata": strata,
            "min_cells": min_cells,
            "min_patients": min_patients,
            "min_detect": min_detect,
            "tile": tile,
            "min_tiles": min_tiles,
            "version": __version__,
        },
    )


def _default_index(ct, smp, min_cells: int, min_patients: int) -> list[str]:
    """Cell types with >= ``min_cells`` cells in >= ``min_patients`` patients, sorted."""
    df = pd.DataFrame({"ct": ct, "smp": smp}).dropna()
    n = df.groupby(["ct", "smp"], sort=False).size()
    npat = (n >= min_cells).groupby(level="ct").sum()
    return r_sort_str(npat.index[npat.to_numpy() >= min_patients])


def _tile_codes(sec, xy, tile: float, ik) -> np.ndarray:
    """Integer label of each index cell's spatial tile: (section, floor(x / tile), floor(y / tile))."""
    sc = pd.factorize(pd.Series(np.asarray(sec, dtype=object)[ik]))[0]
    fx = np.floor(xy[ik, 0] / tile).astype(np.int64)
    fy = np.floor(xy[ik, 1] / tile).astype(np.int64)
    return np.unique(np.column_stack([sc, fx, fy]), axis=0, return_inverse=True)[1].ravel()


def _condition_coding(patients: pd.DataFrame, condition: str) -> pd.Series:
    """0/1 per patient for a two-level condition: 1 = the second level (``.conditionCoding``).

    Levels are the factor levels (a ``Categorical``), else the sorted values, so the contrast
    is "second - first".
    """
    if condition not in patients.columns:
        raise ValueError(f"condition '{condition}' is not a patient-level column")
    x = patients[condition]
    chars = as_character_array(x)
    if isinstance(x.dtype, pd.CategoricalDtype):
        lv = [str(c) for c in x.cat.remove_unused_categories().cat.categories]
    else:
        lv = r_sort_str(set(chars[pd.notna(chars)]))
    if len(lv) != 2:
        raise ValueError(f"condition '{condition}' must have exactly two levels")
    return pd.Series(
        (chars == lv[1]).astype(float), index=patients["patient"].astype(str).to_numpy()
    )


def _strata_by_patient(patients: pd.DataFrame, strata: str) -> pd.Series:
    """The strata label (R ``as.character``) of each patient, in patient order."""
    if strata not in patients.columns:
        raise ValueError(f"strata column '{strata}' is not a patient-level column")
    return pd.Series(
        as_character_array(patients[strata]), index=patients["patient"].astype(str).to_numpy()
    )


def _patient_levels(fit0: dict, des: dict, genes, L: pd.DataFrame, codes, ll) -> dict:
    """What the plots read of each patient (``.patientLevels``).

    Its intercept per gene in the shared, condition-free fit (the level every between-patient
    difference is absorbed into; under ``depth = "spatial_spline"`` the intercept column of the
    patient's block), the mean ``log1p`` density of each tested niche over its index cells, and
    their mean log library size.
    """
    pats = list(des["patients"])
    a = np.asarray(fit0["alpha"])[:, des["intercept_cols"]]
    S = len(pats)
    n = np.bincount(codes, minlength=S).astype(float)
    Lm = L.to_numpy(dtype=float)
    nm = np.column_stack(
        [np.bincount(codes, weights=Lm[:, j], minlength=S) for j in range(Lm.shape[1])]
    )
    fin = np.isfinite(ll)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = np.bincount(codes[fin], weights=ll[fin], minlength=S) / np.bincount(
            codes[fin], minlength=S
        )
    return {
        "intercept": pd.DataFrame(a, index=genes, columns=pats),
        "niche_mean": pd.DataFrame(nm / n[:, None], index=pats, columns=list(L.columns)),
        "loglib_mean": pd.Series(dm, index=pats),
    }
