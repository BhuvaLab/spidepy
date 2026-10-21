"""The slopes engine's tests across patients (``R/test-slopes.R``).

For every (index, tested niche) the per-patient slopes ``beta_s`` of each gene are combined
across patients, weighted by ``1 / (v_model_s * f_s + tau2_g)``: the pooled test (``~ 1``) and
the condition test (``~ condition [+ strata]``). The pooled test is a moderated limma test
with an expression-trended, robust prior and the t statistic referred to limma's df scaled by
the Kish effective share of the patients. The condition test is the same weighted least
squares with an HC2 standard error across patients and Bell-McCaffrey df. ``tau2_g`` is the
gene's DerSimonian-Laird between-patient variance of the slope; Cochran's Q behind it gives the
label-free heterogeneity p that the default condition family filters on.

Slope arrays are genes x patients x niches (R ``x$beta[, , j]``); a column of one niche is
genes x patients.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ._glm import _bp_genes
from ._limma import _dqrdc2_rank, e_bayes, lm_fit
from ._rcompat import p_adjust_bh, r_sort_str

_EPS = np.finfo(float).eps


def _usable(b: np.ndarray, v: np.ndarray):
    """Weights ``1 / v``, the mask of usable (finite slope, finite positive weight) entries,
    and the slopes / weights with the unusable ones set to 0 (the head of ``.dlTau2``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / v
    ok = np.isfinite(b) & np.isfinite(w) & (w > 0)
    return np.where(ok, w, 0.0), np.where(ok, b, 0.0), ok


def _dl_tau2(b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Label-free DerSimonian-Laird between-patient variance per (gene, niche) (``.dlTau2``).

    ``b`` and ``v`` are genes x patients x niches; the result is genes x niches.
    """
    w, b, ok = _usable(b, v)
    with np.errstate(all="ignore"):
        sw = w.sum(axis=1)
        sw2 = (w**2).sum(axis=1)
        bbar = (w * b).sum(axis=1) / sw
        m = ok.sum(axis=1)
        dev = b - bbar[:, None, :]
        q = (w * dev**2).sum(axis=1)
        # np.maximum propagates NaN as R's pmax does; non-finite tau2 then becomes 0
        tau2 = np.maximum((q - (m - 1)) / (sw - sw2 / sw), 0.0)
    tau2[~np.isfinite(tau2)] = 0.0
    return tau2


def _heterogeneity_p(b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cochran's Q of the patients' slopes, referred to chi-square on ``m - 1`` df
    (``.heterogeneityP``); NA under two usable patients or for an overflowing Q."""
    w, b, ok = _usable(b, v)
    with np.errstate(all="ignore"):
        sw = w.sum(axis=1)
        bbar = (w * b).sum(axis=1) / sw
        m = ok.sum(axis=1)
        q = (w * (b - bbar[:, None, :]) ** 2).sum(axis=1)
        p = stats.chi2.sf(q, np.maximum(m - 1, 1))
    p[(m < 2) | ~np.isfinite(q) | ~np.isfinite(p)] = np.nan
    return p


def _kish_df(wts: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """The Kish effective number of patients behind each gene's weights, less the design's
    columns, floored at 1 (``.kishDf``)."""
    ww = np.where(np.isfinite(wts) & np.isfinite(b), wts, np.nan)
    with np.errstate(all="ignore"):
        neff = np.nansum(ww, axis=1) ** 2 / np.nansum(ww**2, axis=1)
        return np.maximum(neff - k, 1.0)


def _pooled_column_test(
    b: np.ndarray,
    v: np.ndarray,
    tau2: np.ndarray,
    mean_expr: np.ndarray,
    min_pooled: int = 6,
    trend: bool = True,
    df_rule: str = "proportional",
) -> pd.DataFrame:
    """The pooled test of one (index, niche) column: moderated limma on an intercept
    (``.pooledColumnTest``).

    ``b`` and ``v`` are genes x patients slopes and variances, ``tau2`` the genes' between-patient
    variance. The prior is robust and trended on ``log(mean_expr + 1e-3)`` over the column's
    gene family (``trend=False``: a flat prior); the df are limma's ``df.total`` scaled by the
    Kish effective share of the patients (``"proportional"``) or capped by the Kish effective
    patients less one (``"capped"``).

    R wraps ``eBayes`` in ``tryCatch(suppressWarnings(...))`` and retries with
    ``robust = FALSE`` when the robust call fails; the retry reads the ``fit`` variable the
    failed call has just overwritten with ``NULL`` and so always fails too (see
    ``docs/r-discrepancies.md``): a column whose robust prior fails is entirely NA here, as in R.
    """
    if df_rule not in ("proportional", "capped"):
        raise ValueError("'df_rule' should be one of 'proportional', 'capped'")
    b = np.asarray(b, dtype=float)
    G = b.shape[0]
    with np.errstate(all="ignore"):
        wts = 1.0 / (np.asarray(v, dtype=float) + np.asarray(tau2, dtype=float)[:, None])
    wts[~np.isfinite(wts) | (wts <= 0) | ~np.isfinite(b)] = np.nan
    n_ok = np.isfinite(wts).sum(axis=1)
    est = np.full(G, np.nan)
    tt = np.full(G, np.nan)
    df = np.full(G, np.nan)
    # rows with fewer than two usable patients carry no residual variance
    fitted = np.flatnonzero(n_ok >= 2)
    if fitted.size:
        w = wts[fitted]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = w / np.nanmean(w, axis=1)[:, None]
            fit = lm_fit(b[fitted], np.ones((b.shape[1], 1)), weights=w)
            fit["Amean"] = np.log(np.asarray(mean_expr, dtype=float)[fitted] + 1e-3)
            try:
                eb = e_bayes(fit, robust=True, trend=trend)
            except Exception:  # R: tryCatch(..., error = function(e) NULL)
                eb = None
        if eb is not None:
            est[fitted] = eb["coefficients"][:, 0]
            tt[fitted] = eb["t"][:, 0]
            bf = b[fitted]
            if df_rule == "capped":
                df[fitted] = np.minimum(eb["df_total"], _kish_df(w, bf, 1))
            else:
                ww = np.where(np.isfinite(w) & np.isfinite(bf), w, np.nan)
                with np.errstate(all="ignore"):
                    neff = np.nansum(ww, axis=1) ** 2 / np.nansum(ww**2, axis=1)
                    df[fitted] = eb["df_total"] * neff / np.isfinite(ww).sum(axis=1)
    tt[(n_ok < min_pooled) | ~np.isfinite(tt)] = np.nan
    with np.errstate(all="ignore"):
        p = 2 * stats.t.cdf(-np.abs(tt), df)
        se = est / tt
    return pd.DataFrame(
        {
            "estimate": est,
            "se": se,
            "t": tt,
            "df": df,
            "p": p,
            "n_patients": np.isfinite(b).sum(axis=1).astype(float),
        }
    )


def _strata_dummies(strata, trt: np.ndarray) -> np.ndarray | None:
    """Treatment-coded strata columns (``model.matrix(~ sf)[, -1]``), None for one level.

    Refuses a condition that lies in the strata's span, since pivoting would otherwise drop a
    strata column and test the condition unadjusted.
    """
    st = np.asarray(strata, dtype=object)
    levels = r_sort_str({str(s) for s in st if pd.notna(s)})  # droplevels(factor(strata))
    if len(levels) <= 1:
        return None
    sd = np.column_stack([(st == lv).astype(float) for lv in levels[1:]])
    s1 = np.column_stack([np.ones(len(st)), sd])
    if _dqrdc2_rank(np.column_stack([s1, trt]))[0] == _dqrdc2_rank(s1)[0]:
        raise ValueError("the condition is fully confounded with 'strata'")
    return sd


def _solve_inv(m: np.ndarray):
    """``solve(m)`` as R: None when LAPACK would call the matrix computationally singular."""
    try:
        inv = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return None
    with np.errstate(all="ignore"):
        rcond = 1.0 / (np.linalg.norm(m, 1) * np.linalg.norm(inv, 1))
    if not np.isfinite(rcond) or rcond < _EPS:
        return None
    return inv


def _kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum()) if len(w) else 0.0


def _condition_gene(b, v, tau2, trt, x0, min_group):
    """One gene's condition test: ``(estimate, se, df)`` or None (the body of ``one()`` in
    ``.robustConditionTest``)."""
    if not np.isfinite(tau2):
        return None
    with np.errstate(all="ignore"):
        w = 1.0 / (v + tau2)
    ok = np.isfinite(b) & np.isfinite(w) & (w > 0)
    if _kish(w[ok & (trt == 1)]) < min_group or _kish(w[ok & (trt == 0)]) < min_group:
        return None
    X = x0[ok]
    rank, pivot = _dqrdc2_rank(X)
    keep = np.sort(pivot[:rank])
    if not (0 in keep and 1 in keep) or len(keep) >= X.shape[0]:
        return None
    X = X[:, keep]
    w = w[ok] / w[ok].mean()
    y = b[ok]
    A = _solve_inv((X * np.sqrt(w)[:, None]).T @ (X * np.sqrt(w)[:, None]))
    if A is None or not np.isfinite(A).all():
        return None
    XA = X @ A
    h = (XA * X).sum(axis=1) * w  # leverages
    if (h >= 1 - 1e-8).any():
        return None
    coef = A @ (X.T @ (w * y))
    e = y - X @ coef
    d = (XA[:, 1] * w) ** 2 / (1 - h)
    V = float((d * e**2).sum())
    sig = -(XA @ X.T)  # W^-1 - K
    sig[np.diag_indices_from(sig)] += 1.0 / w
    nu = float((d * np.diag(sig)).sum() ** 2 / ((np.outer(d, d) * sig**2).sum()))
    if not np.isfinite(V) or V <= 0 or not np.isfinite(nu) or nu <= 0:
        return None
    return coef[1], np.sqrt(V), nu


def _robust_condition_test(
    b: np.ndarray,
    v: np.ndarray,
    tau2: np.ndarray,
    trt,
    strata=None,
    min_group: int = 3,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """The condition test of one (index, niche) column (``.robustConditionTest``).

    Per gene, weighted least squares of the patients' slopes on ``[1, condition, strata]`` with
    weights ``1 / (v + tau2)``, the condition coefficient's HC2 standard error across patients
    and Bell-McCaffrey df under the working model ``var(b_s) ~ 1 / w_s``. A patient with an
    unusable slope or weight drops out; a gene needs ``min_group`` Kish-effective patients in
    each condition. Every per-gene inversion is guarded and a gene that cannot be tested is NA.
    Genes are blocked over ``n_jobs``.
    """
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    trt = np.asarray(trt, dtype=float)
    G = b.shape[0]
    x0 = np.column_stack([np.ones(len(trt)), trt])
    if strata is not None:
        sd = _strata_dummies(strata, trt)
        if sd is not None:
            x0 = np.column_stack([x0, sd])
    M = np.full((G, 3), np.nan)

    def one(gi):
        out = np.full((len(gi), 3), np.nan)
        for a, g in enumerate(gi):
            r = _condition_gene(b[g], v[g], tau2[g], trt, x0, min_group)
            if r is not None:
                out[a] = r
        return gi, out

    for gi, out in _bp_genes(G, one, n_jobs):
        M[gi] = out
    with np.errstate(all="ignore"):
        tt = M[:, 0] / M[:, 1]
        p = 2 * stats.t.cdf(-np.abs(tt), M[:, 2])
    return pd.DataFrame(
        {
            "estimate": M[:, 0],
            "se": M[:, 1],
            "t": tt,
            "df": M[:, 2],
            "p": p,
            "n_patients": np.isfinite(b).sum(axis=1).astype(float),
        }
    )


def _slopes_tests(
    x: dict, trt=None, strata=None, n_jobs: int = 1, pooled_df: str = "proportional"
) -> pd.DataFrame:
    """The slopes engine's tests for one index type's stored slopes (``.slopesTests``).

    Every row carries its triplet's label-free heterogeneity p. Rows are ordered by niche,
    pooled test then condition test.
    """
    genes = list(x["genes"])
    beta = np.asarray(x["beta"], dtype=float)
    vpool = np.asarray(x["v_model"], dtype=float) * x["factor"].to_numpy(dtype=float)[None, :, :]
    tau_pool = _dl_tau2(beta, vpool)
    het = _heterogeneity_p(beta, vpool)
    mean_expr = x["mean_expr"].to_numpy(dtype=float)
    out = []
    for j, niche in enumerate(x["niches"]):
        bj, vj = beta[:, :, j], vpool[:, :, j]
        tests = [
            ("pooled", _pooled_column_test(bj, vj, tau_pool[:, j], mean_expr, df_rule=pooled_df))
        ]
        if trt is not None:
            tests.append(
                (
                    "condition",
                    _robust_condition_test(
                        bj, vj, tau_pool[:, j], trt=trt, strata=strata, n_jobs=n_jobs
                    ),
                )
            )
        for name, d in tests:
            head = pd.DataFrame({"gene": genes, "niche": niche, "test": name})
            out.append(pd.concat([head, d, pd.DataFrame({"p.heterogeneity": het[:, j]})], axis=1))
    return pd.concat(out, ignore_index=True)


def _bh_families(tab: pd.DataFrame, procedure: str, fdr: float) -> pd.DataFrame:
    """Benjamini-Hochberg families (``.bhFamilies``), adding ``q`` and ``in_family``.

    The pooled test: every (gene, index, niche). The condition test: the triplets that pass
    the pooled test or the label-free heterogeneity test at ``fdr`` (``"pooled_or_heterogeneity"``),
    whose heterogeneity passes BH over every triplet (``"heterogeneity"``), whose pooled test
    passes (``"filtered"``), or every triplet (``"all"``). No filter looks at the condition.
    """
    tab = tab.copy()
    n = len(tab)
    q = np.full(n, np.nan)
    in_family = np.zeros(n, dtype=bool)
    is_pooled = (tab["test"] == "pooled").to_numpy()
    is_cond = (tab["test"] == "condition").to_numpy()
    p = tab["p"].to_numpy(dtype=float)
    pl = np.flatnonzero(is_pooled & np.isfinite(p))
    q[pl] = p_adjust_bh(p[pl])
    in_family[pl] = True
    cl = np.flatnonzero(is_cond & np.isfinite(p))
    if len(cl) and procedure != "all":
        key = list(zip(tab["gene"], tab["index"], tab["niche"]))
        pooled_pass = {key[i] for i in pl[q[pl] < fdr]}
        het_pass: set = set()
        if procedure != "filtered":
            ph = tab["p.heterogeneity"].to_numpy(dtype=float)
            hl = np.flatnonzero(is_pooled & np.isfinite(ph))
            het_pass = {key[i] for i in hl[p_adjust_bh(ph[hl]) < fdr]}
        if procedure == "filtered":
            keep = pooled_pass
        elif procedure == "heterogeneity":
            keep = het_pass
        else:
            keep = pooled_pass | het_pass
        cl = np.array([i for i in cl if key[i] in keep], dtype=int)
    if len(cl):
        q[cl] = p_adjust_bh(p[cl])
        in_family[cl] = True
    tab["q"] = q
    tab["in_family"] = in_family
    return tab
