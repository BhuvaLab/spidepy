"""``test_niche_abundance``: the between-patient niche-abundance association (``R/niche-abundance.R``).

The per-patient intercepts of :func:`~spidepy.fit_spide` absorb the association between a
patient's mean niche density around an index cell type and the patient's mean expression in
that type. Here it is tested on its own terms: per-(sample, index type) pseudobulk log2-CPM
regressed on the sample's mean log1p niche density, across samples, with a limma moderated t.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ._checks import (
    check_adata,
    check_condition,
    check_covariates,
    check_genesets,
    check_niche,
)
from ._design import _factor_codes
from ._io import get_counts, get_group_map, get_niche, get_obs, niche_key
from ._limma import _dqrdc2_rank, e_bayes, lm_fit
from ._rcompat import p_adjust_bh, r_sort_str
from .fit import _match_arg
from .niches import _is_self_niche

_TYPES = ("competitive", "self-contained")


def _sanitise_array(x) -> np.ndarray:
    """R ``.sanitise`` on a character vector that may hold missing values (kept missing)."""
    return pd.Series(x, dtype=object).str.replace(r" |-", ".", regex=True).to_numpy(dtype=object)


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(s) for s in x]


def _intersect(a, b) -> list[str]:
    """R ``intersect(a, b)``: unique elements of ``a`` (in its order) that are in ``b``."""
    sb = set(b)
    return [s for s in dict.fromkeys(a) if s in sb]


# --------------------------------------------------------------------------------------
# pseudobulk
# --------------------------------------------------------------------------------------
def _pseudobulk_by_index(Y, NM: pd.DataFrame, ct, smp, index, min_cells, prior_count) -> dict:
    """Per-(sample, index type) pseudobulk log2-CPM and mean log1p niche density.

    Port of ``.pseudobulkByIndex``. ``Y`` is counts (genes x cells, dense or sparse), ``NM``
    the niche matrix (cells x niche types). Returns ``{k: {"Y": genes x samples log2-CPM,
    "niche": samples x niches DataFrame, "ncells", "samples"}}`` for each index type ``k``
    with at least three samples of ``min_cells`` or more type-``k`` cells.
    """
    ct = np.asarray(ct, dtype=object)
    smp = np.asarray(smp, dtype=object)
    all_ct = r_sort_str(pd.unique(ct[pd.notna(ct)]))
    cts = all_ct if index is None else _intersect(_sanitise_array(_as_list(index)), all_ct)
    out = {}
    nm_log = np.log1p(NM.to_numpy(dtype=float))
    for k in cts:
        ik = np.flatnonzero(ct == k)
        s_k = smp[ik]
        # table(smp[ik]): missing sample ids are not counted; names sorted
        counts = pd.Series(s_k[pd.notna(s_k)]).value_counts()
        keep = [s for s in r_sort_str(counts.index) if counts[s] >= min_cells]
        if len(keep) < 3:
            continue
        pos = {s: i for i, s in enumerate(keep)}
        g = np.array([pos.get(s, -1) if isinstance(s, str) else -1 for s in s_k])
        ok = g >= 0
        ik, g = ik[ok], g[ok]
        ind = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)), shape=(len(g), len(keep)))
        # the counts are never densified whole: only genes x samples is
        sums = Y[:, ik] @ ind
        sums = sums.toarray() if sp.issparse(sums) else np.asarray(sums)
        lib = sums.sum(axis=0)
        lcpm = np.log2(((sums + prior_count) / (lib + 2 * prior_count)) * 1e6)
        n = np.bincount(g, minlength=len(keep)).astype(float)
        nm = (ind.T @ nm_log[ik]) / n[:, None]
        out[k] = {
            "Y": lcpm,
            "niche": pd.DataFrame(nm, index=keep, columns=list(NM.columns)),
            "ncells": n,
            "samples": keep,
        }
    return out


def _patient_value(x, pat, pats) -> pd.Series:
    """First non-missing value of a cell-level vector per patient (``.patientValue``).

    A patient with no non-missing value takes its first cell's (missing) value; a patient
    with no cell takes a missing value. The original dtype (a factor stays a factor) is kept
    and the result is indexed by patient name, so callers look values up by name.
    """
    x = x if isinstance(x, pd.Series) else pd.Series(x)
    x = x.reset_index(drop=True)
    pat = np.asarray(pat, dtype=object)
    ok = x.notna().to_numpy()
    cell = pd.DataFrame({"p": pat, "i": np.arange(len(pat))})
    cell = cell[pd.notna(cell["p"])]
    first_any = cell.drop_duplicates("p").set_index("p")["i"]
    first_ok = cell[ok[cell["i"].to_numpy()]].drop_duplicates("p").set_index("p")["i"]
    idx = [
        int(first_ok[p])
        if p in first_ok.index
        else (int(first_any[p]) if p in first_any.index else -1)
        for p in pats
    ]
    take = x.iloc[[max(i, 0) for i in idx]].reset_index(drop=True)
    miss = np.array(idx) < 0
    if miss.any():
        take = take.where(~miss)
    take.index = list(pats)
    return take


# --------------------------------------------------------------------------------------
# gene-set scores
# --------------------------------------------------------------------------------------
def _abundance_set_scores(Y, genes, genesets, type, min_size, max_size):
    """Per-sample gene-set scores for one index type (``.abundanceSetScores``).

    Each gene (row of ``Y``, genes x samples) is standardised across samples; a set's score in
    a sample is the mean of its genes' standardised values there, less the mean over all the
    other genes when ``type`` is ``"competitive"``. Returns ``None`` when no set has between
    ``min_size`` and ``max_size`` genes among the index type's, else a dict with ``S`` (sets x
    samples), ``sets`` (their names) and ``size``.
    """
    Y = np.asarray(Y, dtype=float)
    genes = np.asarray(genes, dtype=object)
    sdv = Y.std(axis=1, ddof=1) if Y.shape[1] > 1 else np.full(Y.shape[0], np.nan)
    use = np.isfinite(sdv) & (sdv > 0)
    Y, genes, sdv = Y[use], genes[use], sdv[use]
    if not Y.shape[0]:
        return None
    Z = (Y - Y.mean(axis=1, keepdims=True)) / sdv[:, None]
    nz = Z.shape[0]
    member = {name: np.flatnonzero(np.isin(genes, _as_list(g))) for name, g in genesets.items()}
    size = np.array([len(m) for m in member.values()])
    keep = (size >= min_size) & (size <= max_size) & ((type == "self-contained") | (size < nz))
    if not keep.any():
        return None
    names = [n for n, kp in zip(member, keep) if kp]
    size = size[keep]
    S = np.vstack([Z[member[n]].sum(axis=0) / s for n, s in zip(names, size)])
    if type == "competitive":
        tot = Z.sum(axis=0)
        S = S - (tot[None, :] - S * size[:, None]) / (nz - size)[:, None]
    return {"S": S, "sets": names, "size": size}


# --------------------------------------------------------------------------------------
# leverage cap and the per-pair fit
# --------------------------------------------------------------------------------------
def _hat(X: np.ndarray) -> np.ndarray:
    """``stats::hat(X, intercept = FALSE)``: row sums of squares of the first ``rank`` columns of Q.

    R takes Q from LINPACK ``dqrdc2``, whose pivoting moves negligible columns last; the
    leading ``rank`` columns span the same space as ``X[:, pivot[:rank]]``.
    """
    rank, pivot = _dqrdc2_rank(X)
    if rank == 0:
        return np.zeros(X.shape[0])
    q, _ = np.linalg.qr(X[:, pivot[:rank]])
    return (q**2).sum(axis=1)


def _cap_leverage(X, max_leverage, floor: float = 0.05) -> dict:
    """Mallows-type weights capping sample leverage (``.capLeverage``).

    Samples whose leverage exceeds ``max_leverage`` times the average ``ncol(X) / nrow(X)`` are
    down-weighted until none does or until they reach ``floor``. Returns ``w``, ``leverage``
    (the largest leverage before the cap, as a multiple of the average) and ``downweighted``.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    avg = X.shape[1] / n
    h = _hat(X)
    w = np.ones(n)
    cap = max_leverage * avg
    if np.isfinite(cap) and cap < 1 and h.max() > cap * (1 + 1e-6):
        for _ in range(100):
            hw = _hat(X * np.sqrt(w)[:, None])
            if hw.max() <= cap * (1 + 1e-3):
                break
            # never below the floor: a sample weighted towards zero would leave its condition,
            # and so the design, without support
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.maximum(w * np.minimum(1.0, cap / hw), floor)
    return {"w": w, "leverage": float(h.max() / avg), "downweighted": int((w < 1).sum())}


def _covariate_columns(x: pd.Series):
    """Design columns of one covariate as ``model.matrix`` makes them, or ``None`` if missing.

    Numbers enter as they are; anything else is a factor with treatment contrasts (a
    ``Categorical`` keeps all its categories, so an unused one gives an all-zero column).
    """
    if x.dtype.kind in "iuf":
        v = x.to_numpy(dtype=float)
        return None if np.isnan(v).any() else [v]
    if isinstance(x.dtype, pd.CategoricalDtype):
        levels, codes = list(map(str, x.cat.categories)), np.asarray(x.cat.codes)
    else:
        levels, codes = _factor_codes(x.to_numpy(dtype=object))
    if (codes < 0).any():
        return None
    return [(codes == i).astype(float) for i in range(1, len(levels))]


def _model_matrix(df: pd.DataFrame, covariates, interaction: bool):
    """``model.matrix(~ niche [* condition] + covariates, df)``, or ``None`` if rows would drop.

    Columns in R's term order: intercept, niche, condition, covariates, niche:condition.
    Returns ``(X, names)``.
    """
    cols = [np.ones(len(df)), df["niche"].to_numpy(dtype=float)]
    names = ["(Intercept)", "niche"]
    if interaction:
        cols.append(df["condition"].to_numpy(dtype=float))
        names.append("condition")
    for cv in covariates:
        cc = _covariate_columns(df[cv])
        if cc is None:
            return None
        cols += cc
        names += [cv] * len(cc)
    if interaction:
        cols.append(cols[1] * cols[2])
        names.append("niche:condition")
    X = np.column_stack(cols)
    if np.isnan(X).any():  # model.matrix drops such rows, so nrow(X) != nrow(df)
        return None
    return X, names


def _abundance_pair(Y, df: pd.DataFrame, covariates=(), max_leverage=3):
    """One (index, niche) pair: the design, the leverage cap and the limma fit (``.abundancePair``).

    ``df`` holds the samples' niche abundance, optionally ``condition`` (a categorical with the
    two levels of the whole data) and the covariates, one row per column of ``Y`` (genes x
    samples). The condition is coded -1/2, +1/2, so ``"niche"`` is the mean of the two
    conditions' associations and ``"condition:niche"`` their difference (second level minus
    first); where the conditions leave fewer than two samples each the design is the pooled one
    and only ``"niche"`` is reported. Returns ``None`` when no design is estimable, else
    ``terms`` (per term ``estimate``, ``t``, ``p``), ``n``, ``leverage`` and ``downweighted``.
    """
    df = df.copy()
    # a covariate constant over these samples is carried by the intercept
    covariates = [cv for cv in _as_list(covariates) if df[cv].dropna().nunique() > 1]
    interactions = [False]
    if "condition" in df.columns:
        cond = df["condition"]
        lev = list(cond.cat.categories)
        counts = cond.value_counts().reindex(lev, fill_value=0)
        if len(lev) == 2 and (counts >= 2).all():
            df["condition"] = (cond.astype(object) == lev[1]).astype(float) - 0.5
            df.loc[cond.isna(), "condition"] = np.nan
            interactions = [True, False]
    X = names = None
    for inter in interactions:
        mm = _model_matrix(df, covariates, inter)
        if mm is not None:
            X, names = mm
            if X.shape[0] >= X.shape[1] + 2 and _dqrdc2_rank(X)[0] == X.shape[1]:
                break
        X = None
    if X is None:
        return None
    cap = _cap_leverage(X, max_leverage)
    # the weighted design must still estimate every column (a pair that cannot drops out)
    if cap["downweighted"] and _dqrdc2_rank(X * np.sqrt(cap["w"])[:, None])[0] < X.shape[1]:
        return None
    w = None
    if cap["downweighted"]:
        # a weight vector as long as the genes would be read as gene weights
        w = np.tile(cap["w"], (Y.shape[0], 1)) if Y.shape[0] == Y.shape[1] else cap["w"]
    try:
        fit = e_bayes(lm_fit(Y, X, weights=w), robust=X.shape[0] >= 6)
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        return None
    terms = {}
    for tm, label in (("niche", "niche"), ("niche:condition", "condition:niche")):
        if tm in names:
            j = names.index(tm)
            terms[label] = {
                "estimate": fit["coefficients"][:, j],
                "t": fit["t"][:, j],
                "p": fit["p_value"][:, j],
            }
    return {
        "terms": terms,
        "n": int(X.shape[0]),
        "leverage": cap["leverage"],
        "downweighted": cap["downweighted"],
    }


# --------------------------------------------------------------------------------------
# public function
# --------------------------------------------------------------------------------------
def test_niche_abundance(
    adata: AnnData,
    condition: str | None = None,
    sigma: float | None = None,
    index: str | Sequence[str] | None = None,
    niche: str | Sequence[str] | None = None,
    covariates: str | Sequence[str] = (),
    assay: str = "counts",
    cell_type: str = "cell_type",
    sample_id: str = "sample_id",
    name: str = "Niche",
    min_cells: int = 10,
    prior_count: float = 1,
    max_leverage: float = 3,
    genesets: dict | None = None,
    type: str = "competitive",
    min_size: int = 5,
    max_size: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """Test whether patients with more of a niche cell type express genes differently.

    A between-patient question that the within-patient engines cannot see: does a gene's
    expression in index cell type k differ between patients whose type-k cells sit, on average,
    among more type-n cells? (R ``testNicheAbundance``.)

    For each index type k and niche type n, the per-sample pseudobulk log2-CPM of every gene in
    the type-k cells is regressed on the per-sample mean log1p niche density of type n around
    those cells, across samples, with a limma moderated t-test. Without a ``condition`` the
    design is ``~ niche + covariates`` and one term, ``"niche"``, is reported. With a
    ``condition`` the design is ``~ niche * condition + covariates`` with the condition coded
    -1/2, +1/2, and two terms are reported: ``"niche"`` (the association averaged over the two
    conditions) and ``"condition:niche"`` (its difference, second level minus first). Where the
    conditions leave fewer than two samples each, only ``"niche"`` is reported.

    Samples whose leverage in the design exceeds ``max_leverage`` times the average are
    down-weighted until none does (weights from the design alone, never from expression). With
    ``genesets``, sets are tested instead of genes: genes are standardised across samples and a
    set's score is the mean of its genes' values, less the other genes' mean when
    ``type="competitive"``. ``q_global`` of R (column ``q.global``) is one Benjamini-Hochberg
    family per term.

    Parameters
    ----------
    adata : AnnData
        With niche matrices in ``obsm`` (see :func:`~spidepy.build_niches`).
    condition : str, optional
        The ``obs`` column of a two-level condition (constant within sample).
    sigma : float
        The niche bandwidth (one value).
    index, niche : str or sequence of str, optional
        Restrict the index / niche cell types. An index type is never tested against its own
        niche: itself, or the merged niche it is a member of (see :func:`~spidepy.merge_niches`).
    covariates : str or sequence of str
        Sample-level ``obs`` columns to adjust for (first non-missing value per sample).
    assay, cell_type, sample_id, name : str
        The counts layer, the ``obs`` columns of cell type and sample, the niche key prefix.
    min_cells : int
        Minimum cells of the index type a sample must contribute (else dropped for that type).
    prior_count : float
        The pseudocount of the log2-CPM.
    max_leverage : float
        A number above 1; ``inf`` turns the cap off.
    genesets : dict, optional
        ``{set name: gene identifiers}``; tests sets instead of genes.
    type : {"competitive", "self-contained"}
        With ``genesets``: the set against the other genes, or the set alone.
    min_size, max_size : int
        With ``genesets``: size limits after intersecting with each index type's genes.
    verbose : bool
        Report progress on stderr.

    Returns
    -------
    pandas.DataFrame
        One row per (gene, index type, niche type, term) with columns ``gene``, ``index``,
        ``niche``, ``term``, ``estimate``, ``t``, ``p``, ``n_patients``, ``leverage``,
        ``downweighted``, ``q`` (BH within each index, niche, term) and ``q.global``. With
        ``genesets`` the column ``gene`` becomes ``set``, followed by ``size``, and a
        ``direction`` column is added.
    """
    check_adata(adata, assay=assay, cell_type=cell_type, sample_id=sample_id)
    if (
        isinstance(max_leverage, (bool, np.bool_))
        or not np.isscalar(max_leverage)
        or not isinstance(max_leverage, (int, float, np.number))
        or np.isnan(max_leverage)
        or max_leverage <= 1
    ):
        raise ValueError(
            "'max_leverage' must be a single number above 1 (inf switches the cap off)"
        )
    if condition is not None:
        check_condition(adata, condition)
    covariates = _as_list(covariates)
    check_covariates(adata, covariates)
    if genesets is not None:
        check_genesets(genesets)
    type = _match_arg(type, _TYPES, "type")
    if sigma is None or np.size(sigma) != 1:
        raise ValueError("'sigma' must be a single bandwidth")
    check_niche(adata, sigma, name=name)

    ct_raw = get_obs(adata, cell_type)
    ct = _sanitise_array(ct_raw)
    # sanitised -> original labels (the first of each), so the output reads as fit_spide's does
    ct_lab: dict = {}
    for r, c in zip(np.asarray(ct_raw, dtype=object), ct):
        if pd.notna(c):
            ct_lab.setdefault(c, r)
    smp = get_obs(adata, sample_id)
    Y = get_counts(adata, assay)
    genes = np.asarray(adata.var_names, dtype=object)
    nkey = niche_key(name, float(sigma))
    NM = get_niche(adata, nkey).copy()
    nm_lab: dict = {}
    for r, c in zip(list(NM.columns), _sanitise_array(NM.columns)):
        nm_lab.setdefault(c, r)
    NM.columns = _sanitise_array(NM.columns)
    group_map = get_group_map(adata, nkey)
    NM_cols = list(NM.columns)
    niches = NM_cols if niche is None else _intersect(_sanitise_array(_as_list(niche)), NM_cols)
    if not niches:
        raise ValueError("no requested niche cell types found in the niche reducedDim")
    NM = NM[niches]

    pb = _pseudobulk_by_index(Y, NM, ct, smp, index, min_cells, prior_count)
    if not pb:
        raise ValueError("no index cell type has at least three samples with >= min_cells cells")
    # sample-level condition and covariates: first non-missing value per sample, by name
    pats = list(pd.unique(smp[pd.notna(smp)]))
    cond_s = cond_levels = None
    if condition is not None:
        cond_s = _patient_value(get_obs(adata, condition), smp, pats)
        cond_levels, _ = _factor_codes(cond_s.to_numpy(dtype=object))
    cov_s = {cv: _patient_value(adata.obs[cv], smp, pats) for cv in covariates}

    rows = []
    for k, b in pb.items():
        Yk, names_k = b["Y"], list(genes)
        sc = None
        if genesets is not None:
            # with gene sets, each patient's set score takes the place of a gene
            sc = _abundance_set_scores(Yk, genes, genesets, type, min_size, max_size)
            if sc is None:
                continue
            Yk, names_k = sc["S"], sc["sets"]
        # an index type is never tested against its own niche, merged or not
        own = _is_self_niche(ct_lab[k], [nm_lab[x] for x in niches], group_map)
        for n in [x for x, s in zip(niches, own) if not s]:
            if verbose:
                print(
                    f"test_niche_abundance: {ct_lab[k]} x {nm_lab[n]} "
                    f"({len(b['samples'])} samples)",
                    file=sys.stderr,
                )
            df = pd.DataFrame({"niche": b["niche"].loc[b["samples"], n].to_numpy(dtype=float)})
            if cond_s is not None:
                df["condition"] = pd.Categorical(
                    cond_s.loc[b["samples"]].to_numpy(dtype=object), categories=cond_levels
                )
            for cv in covariates:
                df[cv] = cov_s[cv].loc[b["samples"]].reset_index(drop=True)
            r = _abundance_pair(Yk, df, covariates, max_leverage)
            if r is None:
                continue
            for tm, v in r["terms"].items():
                d = {"set" if sc is not None else "gene": names_k}
                if sc is not None:
                    d["size"] = sc["size"]
                d.update(index=ct_lab[k], niche=nm_lab[n], term=tm, estimate=v["estimate"], t=v["t"], p=v["p"],
                         n_patients=r["n"], leverage=r["leverage"],
                         downweighted=r["downweighted"])  # fmt: skip
                rows.append(pd.DataFrame(d))
    if not rows:
        raise ValueError(
            "no (index, niche) pair had enough samples to fit"
            if genesets is None
            else "no gene set within min_size and max_size, or no (index, niche) pair had "
            "enough samples to fit"
        )
    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(out["p"].to_numpy())].reset_index(drop=True)
    key = out["index"] + " " + out["niche"] + " " + out["term"]
    out["q"] = out.groupby(key, sort=False)["p"].transform(lambda p: p_adjust_bh(p.to_numpy()))
    # one family per term: pooling them let a strong "niche" signal loosen the threshold for
    # the condition:niche rows
    out["q.global"] = out.groupby("term", sort=False)["p"].transform(
        lambda p: p_adjust_bh(p.to_numpy())
    )
    if genesets is not None:
        out["direction"] = np.where(out["t"] > 0, "up", "down")
    return out


# pytest would otherwise collect the public function as a test
test_niche_abundance.__test__ = False
