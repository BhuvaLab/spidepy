"""Gene-set tests on the slopes engine's per-patient slopes (``R/spiGSEA.R``)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._checks import check_genesets
from ._classes import SpiDEResults
from ._rcompat import p_adjust_bh
from ._test_slopes import _dl_tau2, _pooled_column_test, _robust_condition_test
from .fit import _condition_coding, _match_arg, _strata_by_patient

_TYPES = ("competitive", "self-contained")
_POOLED_DF = ("proportional", "capped")
_TESTS = ("pooled", "condition")
_COLUMNS = [
    "set",
    "index",
    "niche",
    "test",
    "size",
    "estimate",
    "se",
    "t",
    "df",
    "p",
    "q",
    "q.global",
    "n_patients",
    "direction",
]


def _set_slopes(x: dict, member: list, type: str) -> dict:
    """Each set's per-patient slope for one index type, for every niche column (``.setSlopes``).

    A gene's slopes are put on a common scale by ``sigma_g = sqrt(tau2_g + median_s v_gs)``, its
    typical spread of slopes between patients (DerSimonian-Laird tau2 plus a typical
    within-patient variance, both from the pooled weights and computed without the condition).
    A set's slope in a patient is the mean of its genes' scaled slopes there (genes missing in
    that patient left out; the patient drops when under half the set is present), minus, when
    competitive, the mean over the column's other genes. Its variance treats the genes' errors as
    independent within the patient.

    Parameters
    ----------
    x : dict
        One index type of a slopes fit (``genes``, ``beta``, ``v_model``, ``factor``,
        ``mean_expr``; slope arrays genes x patients x niches).
    member : list of array-like of int
        Each set's positions in ``x["genes"]``.
    type : {"competitive", "self-contained"}

    Returns
    -------
    dict
        ``b`` and ``v`` (sets x patients x niches) and ``mean_expr`` (per set, the geometric
        mean of ``mean_expr + 1e-3`` over its genes).
    """
    beta = np.asarray(x["beta"], dtype=float)
    G, S, nt = beta.shape
    K = len(member)
    vpool = np.asarray(x["v_model"], dtype=float) * x["factor"].to_numpy(dtype=float)[None, :, :]
    tau = _dl_tau2(beta, vpool)
    B = np.full((K, S, nt), np.nan)
    V = np.full((K, S, nt), np.nan)
    for j in range(nt):
        bj, vj = beta[:, :, j].copy(), vpool[:, :, j].copy()
        bad = ~(np.isfinite(bj) & np.isfinite(vj))
        bj[bad] = np.nan
        vj[bad] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN rows: R's median(na.rm) gives NA too
            sig = np.sqrt(tau[:, j] + np.nanmedian(vj, axis=1))
            sig[~np.isfinite(sig) | (sig <= 0)] = np.nan
            u = bj / sig[:, None]
            vu = vj / sig[:, None] ** 2
            fin = np.isfinite(u) & np.isfinite(vu)
            u[~fin] = 0.0
            vu[~fin] = 0.0
            tot_n, tot_b, tot_v = fin.sum(axis=0), u.sum(axis=0), vu.sum(axis=0)
            for s, g in enumerate(member):
                g = np.asarray(g, dtype=int)
                n_in = fin[g].sum(axis=0)
                sb, sv = u[g].sum(axis=0), vu[g].sum(axis=0)
                mb = sb / n_in
                mv = sv / n_in.astype(float) ** 2
                drop = n_in < len(g) / 2
                if type == "competitive":
                    n_out = tot_n - n_in
                    mb = mb - (tot_b - sb) / n_out
                    mv = mv + (tot_v - sv) / n_out.astype(float) ** 2
                    drop = drop | (n_out < 1)
                mb[drop] = np.nan
                mv[drop] = np.nan
                B[s, :, j], V[s, :, j] = mb, mv
    me = np.asarray(x["mean_expr"], dtype=float)
    mean_expr = np.array(
        [np.exp(np.mean(np.log(me[np.asarray(g, dtype=int)] + 1e-3))) for g in member]
    )
    return {"b": B, "v": V, "mean_expr": mean_expr}


def spi_gsea(
    res: SpiDEResults,
    genesets: dict,
    test: str | list[str] | None = None,
    type: str = "competitive",
    min_size: int = 5,
    max_size: int = 500,
    pooled_df: str | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Gene-set tests on spiDE's niche results (R ``spiGSEA``).

    Asks, for every gene set and every (index cell type, niche cell type) column, whether the
    set's genes as a group change with the niche (the pooled test) or change with it differently
    between the two conditions (the condition-specific test).

    Every patient gets a slope for the set: the average of its genes' niche slopes in that
    patient, each gene's slope first divided by the gene's typical spread of slopes between
    patients. By default the set's slope is *competitive*: the same average over the column's
    other tested genes is subtracted, so the test asks whether the set moves more than the
    typical gene. With ``type="self-contained"`` nothing is subtracted. The patients' set slopes
    then go through the same pooled and condition tests as the genes' slopes
    (:func:`~spidepy.test_spide`); the patient is the unit of replication, so the correlation
    between a set's genes is accounted for without being estimated.

    Needs the slopes engine, whose fit holds each patient's slopes.

    Parameters
    ----------
    res : SpiDEResults
        Results of the slopes engine.
    genesets : dict of str to list of str
        Gene sets by name.
    test : {"pooled", "condition"} or list of them, optional
        The tests to run; default every test ``res`` holds.
    type : {"competitive", "self-contained"}
    min_size, max_size : int
        Set size limits after intersecting with the genes tested in each index type.
    pooled_df : {"proportional", "capped"}, optional
        The pooled test's df rule; default the one ``res`` used (``"capped"`` when it records
        none).
    n_jobs : int
        Workers for the condition test (results do not depend on it).

    Returns
    -------
    pandas.DataFrame
        One row per (set, index, niche, test): ``set, index, niche, test, size, estimate, se, t,
        df, p, q`` (Benjamini-Hochberg within the (index, niche, test) column), ``q.global``
        (over every row of that test), ``n_patients`` and ``direction``. The estimate is in units
        of the genes' between-patient spread. Empty when no set is usable.
    """
    if not isinstance(res, SpiDEResults):
        raise TypeError("'res' should be a SpiDEResults from test_spide() or spide()")
    type = _match_arg(type, _TYPES, "type")
    if pooled_df is None:
        pooled_df = res.pooled_df if res.pooled_df else "capped"  # R .pooledDfOf
    else:
        pooled_df = _match_arg(pooled_df, _POOLED_DF, "pooled_df")
    check_genesets(genesets)
    fit = res.fit
    if fit.engine != "slopes":
        raise ValueError(
            'spiGSEA() needs the slopes engine\'s per-patient slopes: fit with engine = "slopes"'
        )
    have = ["pooled", "condition"] if res.condition else ["pooled"]
    if test is None:
        test = have
    else:
        test = [test] if isinstance(test, str) else list(test)
        test = [_match_arg(t, _TESTS, "test") for t in test]
    if not all(t in have for t in test):
        raise ValueError("the condition-specific set test needs results tested with a condition")
    pt = fit.patients
    trt_of = _condition_coding(pt, res.condition) if "condition" in test else None
    strata = fit.params.get("strata")
    str_of = _strata_by_patient(pt, strata) if trt_of is not None and strata is not None else None
    sets = {k: ({v} if isinstance(v, str) else set(v)) for k, v in genesets.items()}
    out = []
    for k, x in fit.index.items():
        pats = list(x["patients"])
        trt = None if trt_of is None else trt_of.reindex(pats).to_numpy()
        st = None if str_of is None else str_of.reindex(pats).to_numpy()
        genes = list(x["genes"])
        member = {n: np.flatnonzero([g in s for g in genes]) for n, s in sets.items()}
        member = {
            n: m
            for n, m in member.items()
            if min_size <= len(m) <= max_size and len(m) < len(genes)
        }
        if not member:
            continue
        names, mem = list(member), list(member.values())
        ss = _set_slopes(x, mem, type)
        size = np.array([len(m) for m in mem])
        for j, niche in enumerate(x["niches"]):
            b, v = ss["b"][:, :, j], ss["v"][:, :, j]
            tau = _dl_tau2(b[:, :, None], v[:, :, None])[:, 0]
            if "pooled" in test:
                r = _pooled_column_test(
                    b, v, tau, ss["mean_expr"], trend=len(mem) >= 20, df_rule=pooled_df
                )
                out.append(_rows(names, k, niche, "pooled", size, r))
            if "condition" in test:
                r = _robust_condition_test(b, v, tau, trt=trt, strata=st, n_jobs=n_jobs)
                out.append(_rows(names, k, niche, "condition", size, r))
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    p = d["p"].to_numpy(dtype=float)
    ok = np.isfinite(p)
    q, qg = np.full(len(d), np.nan), np.full(len(d), np.nan)
    col = (d["index"] + "\r" + d["niche"] + "\r" + d["test"]).to_numpy()
    for key in pd.unique(col[ok]):
        i = np.flatnonzero(ok & (col == key))
        q[i] = p_adjust_bh(p[i])
    for key in pd.unique(d["test"].to_numpy()[ok]):
        i = np.flatnonzero(ok & (d["test"].to_numpy() == key))
        qg[i] = p_adjust_bh(p[i])
    d["q"], d["q.global"] = q, qg
    t = d["t"].to_numpy(dtype=float)
    d["direction"] = np.where(np.isfinite(t), np.where(t > 0, "up", "down"), None)
    # R order(test != "pooled", p): pooled first, p ascending, NA last, ties in row order
    pk = np.where(np.isnan(p), np.inf, p)
    o = np.lexsort((pk, (d["test"] != "pooled").to_numpy()))
    return d.iloc[o][_COLUMNS].reset_index(drop=True)


def _rows(names, index, niche, test, size, r: pd.DataFrame) -> pd.DataFrame:
    head = pd.DataFrame({"set": names, "index": index, "niche": niche, "test": test, "size": size})
    return pd.concat([head, r.reset_index(drop=True)], axis=1)
