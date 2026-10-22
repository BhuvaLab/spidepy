"""Data builders shared by the plots (`R/plot-helpers.R`).

Anything shared across genes is computed over the whole gene family, as the tests compute it;
counts are read only as the plotted genes' rows or through a sparse product. Every builder
returns what R's returns (data frames as `pandas.DataFrame` with R's column names and row
order), so the plots and their tests read the same numbers. Per-patient values are always joined
by patient name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._checks import check_fit_adata, check_niche
from .._io import get_counts, get_group_map, get_niche, get_obs, niche_key
from .._rcompat import r_sort_str
from .._test_slopes import _dl_tau2
from ..fit import _condition_coding, _match_arg


def _slope_weights(xi: dict) -> tuple[np.ndarray, np.ndarray]:
    """The slopes engine's pooled-test variance of each patient's slope and the DerSimonian-Laird
    `tau2` of each (gene, niche) over the whole gene family (R `.slopeWeights`).

    The variances are `v_model` times the per-patient factor (genes x patients x niches) and
    `tau2` is genes x niches: the weights are `1 / (v + tau2)`.
    """
    # the factor is patients x niches; align it to the arrays' patients and niches by name
    fac = xi["factor"].reindex(index=list(xi["patients"]), columns=list(xi["niches"]))
    v = np.asarray(xi["v_model"], dtype=float) * fac.to_numpy(dtype=float)[None, :, :]
    return v, _dl_tau2(np.asarray(xi["beta"], dtype=float), v)


def _resolve_test(x, test: str | None) -> str:
    """The condition-specific test if the results have one, else the pooled test (R
    `.resolveTest`, as `results()` does)."""
    have = set(x.table["test"])
    if test is None:
        test = "condition" if "condition" in have else "pooled"
    test = _match_arg(test, ("pooled", "condition"), "test")
    if test not in have:
        raise ValueError("these results have no condition-specific test")
    return test


def _group_levels(fit, condition) -> pd.Series:
    """Each patient's group as a categorical indexed by patient name (R `.groupLevels`).

    Its condition level (categories in contrast order), or one group, `"all patients"`, when
    there is no condition.
    """
    pt = fit.patients
    names = pt["patient"].astype(str).to_numpy()
    if not condition:
        return pd.Series(pd.Categorical(["all patients"] * len(pt)), index=names)
    lv = _condition_coding(pt, condition).attrs["levels"]
    chars = pt[condition].map(lambda v: v if pd.isna(v) else str(v)).to_numpy(dtype=object)
    return pd.Series(pd.Categorical(chars, categories=lv), index=names)


def _direction_labels(x, test: str) -> dict[str, str]:
    """The legend labels of an `up` and a `down` call (R `.directionLabels`)."""
    if test == "pooled":
        return {"up": "rises with density", "down": "falls with density"}
    lv = _condition_coding(x.fit.patients, x.condition).attrs["levels"]
    return {"up": f"higher slope in {lv[1]}", "down": f"higher slope in {lv[0]}"}


def _triplet_rows(tab: pd.DataFrame, test: str, gene, index: str, niche: str) -> pd.DataFrame:
    """The table rows of one test for these genes, in the genes' order (R `.tripletRows`).

    A gene without a row gives an all-missing row, as R's `match` indexing does.
    """
    r = tab[(tab["test"] == test) & (tab["index"] == index) & (tab["niche"] == niche)]
    r = r.drop_duplicates("gene").set_index("gene", drop=False)
    out = r.reindex(list(np.atleast_1d(gene)))
    out.index = pd.RangeIndex(len(out))
    return out


def _expression_bands(fit, k: int = 5) -> pd.DataFrame:
    """Each index type's genes in `k` bands by mean expression (R `.expressionBands`).

    Columns `gene, index, band` with `band` a categorical `Q1 (dimmest) ... Qk
    (brightest)`.
    """
    labs = [f"Q{i}" for i in range(1, k + 1)]
    labs[0] += " (dimmest)"
    labs[-1] += " (brightest)"
    brk = np.linspace(0.0, 1.0, k + 1)
    out = []
    for name, xi in fit.index.items():
        me = np.asarray(xi["mean_expr"], dtype=float)
        # rank(ties.method = "first")
        r = np.empty(len(me))
        r[np.argsort(me, kind="stable")] = np.arange(1, len(me) + 1)
        r = r / len(xi["genes"])
        # cut(r, brk, include.lowest = TRUE): bins (b[i-1], b[i]], the first closed
        code = np.maximum(np.searchsorted(brk, r, side="left"), 1) - 1
        out.append(
            pd.DataFrame(
                {
                    "gene": [str(g) for g in xi["genes"]],
                    "index": name,
                    "band": pd.Categorical.from_codes(code, categories=labs),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def _gene_rows(Y_index: pd.Index, genes) -> np.ndarray:
    """Row positions of `genes` in the counts, failing as R does if one is missing."""
    pos = Y_index.get_indexer(list(genes))
    if (pos < 0).any():
        miss = [g for g, p in zip(genes, pos, strict=True) if p < 0]
        raise ValueError(f"gene(s) not in 'adata': {', '.join(miss)}")
    return pos


def _cell_type_means(
    adata, genes, assay: str = "counts", cell_type: str = "cell_type"
) -> tuple[pd.DataFrame, pd.Series]:
    """Mean expression per 10,000 counts of these genes in every cell type (R `.cellTypeMeans`).

    A product with a sparse, library-scaled indicator, so only genes x cell types is dense.
    Returns the genes x cell types means (cell types sorted, cells without counts or a type
    left out) and the number of cells of each type.
    """
    Y = get_counts(adata, assay)
    genes = [str(g) for g in genes]
    rows = _gene_rows(adata.var_names, genes)
    lib = np.asarray(Y.sum(axis=0)).ravel().astype(float)
    ct = get_obs(adata, cell_type)
    ok = np.flatnonzero((lib > 0) & pd.notna(ct))
    types = r_sort_str(set(ct[ok]))
    j = pd.Index(types).get_indexer(ct[ok])
    ind = sp.csr_matrix((1e4 / lib[ok], (ok, j)), shape=(Y.shape[1], len(types)))
    n = np.bincount(j, minlength=len(types))
    Ysub = Y[rows, :] if sp.issparse(Y) else np.asarray(Y)[rows, :]
    prod = Ysub @ ind
    M = (prod.toarray() if sp.issparse(prod) else np.asarray(prod)) / n[None, :]
    return pd.DataFrame(M, index=genes, columns=types), pd.Series(n, index=types)


def _niche_fold(
    tab: pd.DataFrame, adata, fit, assay: str = "counts", name: str = "Niche", pseudo: float = 0.01
) -> np.ndarray:
    """For each row of a results table: the gene's expression in the niche type over its
    expression in the index type (CP10k, plus `pseudo` on both; R `.nicheFold`).

    A merged niche (`merge_niches`) pools its member cell types' cells.
    """
    check_fit_adata(
        adata,
        fit,
        list(dict.fromkeys(tab["index"])),
        list(dict.fromkeys(tab["niche"])),
        assay,
        name,
    )
    genes = list(dict.fromkeys(tab["gene"]))
    M, n = _cell_type_means(adata, genes, assay, fit.params["cell_type"])
    groups = get_group_map(adata, niche_key(name, fit.sigma)) or {}

    def pooled(members) -> np.ndarray:
        m = [x for x in dict.fromkeys(members) if x in M.columns]
        if not m:
            return np.full(len(M), np.nan)
        return M[m].to_numpy() @ n[m].to_numpy(dtype=float) / n[m].sum()

    # a niche column is its group's members (a group may be named after one of them); an index
    # type is its own cells
    def side(types, members):
        return pd.DataFrame({t: pooled(members(t)) for t in types}, index=M.index)

    En = side(list(dict.fromkeys(tab["niche"])), lambda t: groups.get(t, [t]))
    Ei = side(list(dict.fromkeys(tab["index"])), lambda t: [t])
    gi = M.index.get_indexer(tab["gene"])
    en = En.to_numpy()[gi, En.columns.get_indexer(tab["niche"])]
    ei = Ei.to_numpy()[gi, Ei.columns.get_indexer(tab["index"])]
    return (en + pseudo) / (ei + pseudo)


def _index_cell_data(
    adata, fit, index: str, niche: str, genes, assay: str = "counts", name: str = "Niche"
) -> dict:
    """One index type's cells as the fit used them (R `.indexCellData`).

    The fit's patients and cells with counts: these genes' counts (a dense genes x cells array),
    library sizes, patients and the log1p density of one niche column.
    """
    ct = get_obs(adata, fit.params["cell_type"])
    smp = get_obs(adata, fit.params["sample_id"])
    xi = fit.index[index]
    genes = [str(g) for g in genes]
    Y = get_counts(adata, assay)
    rows = _gene_rows(adata.var_names, genes)
    check_niche(adata, fit.sigma, name)
    check_fit_adata(adata, fit, index, niche, assay, name)
    lib = np.asarray(Y.sum(axis=0)).ravel().astype(float)
    ik = np.flatnonzero(
        pd.notna(ct)
        & (ct == index)
        & pd.Series(smp).isin(list(xi["patients"])).to_numpy()
        & (lib > 0)
    )
    NM = get_niche(adata, niche_key(name, fit.sigma))
    if niche not in NM.columns:
        raise ValueError(f"niche '{niche}' is not a column of {niche_key(name, fit.sigma)}")
    # the plotted genes' rows first, so no genes x cells copy is made (sparse input stays sparse)
    counts = Y[rows, :][:, ik]
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    return {
        "counts": counts.astype(float),
        "lib": lib[ik],
        "patient": smp[ik],
        "density": np.log1p(NM[niche].to_numpy(dtype=float)[ik]),
    }
