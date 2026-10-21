"""The slice of limma that spiDE's pooled and niche-abundance tests use (limma 3.66.0).

``lm_fit`` is ``lmFit`` (least squares, per-gene weights, NA observations; ``lm.series``),
``fit_f_dist`` is ``fitFDist`` (with the natural-spline trend), ``squeeze_var`` is
``squeezeVar`` (which picks ``fitFDist`` when every gene has the same residual df and
``fitFDistUnequalDF1`` with a weighted lowess trend otherwise) and ``e_bayes`` is
``eBayes`` for the quantities spiDE reads: ``t``, ``p.value``, ``df.total``, ``s2.prior``,
``df.prior`` and ``s2.post``. ``var.prior`` and ``lods`` (``tmixture.matrix``) are not read
anywhere in spiDE and are not computed.

``fit_f_dist_robustly`` is ``fitFDistRobustly`` (Winsorised moments of the log variances,
used by ``squeezeVar(robust = TRUE)`` when every gene has the same residual df), and the robust
half of ``fitFDistUnequalDF1`` (FDR prior weights, then df shrinkage for right outliers) covers
unequal residual df.

Matrices are genes x observations, as in R. R ``NA`` is ``nan``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import special, stats
from spanormpy._edger import (
    _loess_fit_unweighted,
    _pf_upper_log,
    _winsorized_moments,
    _zeroin2,
)
from spanormpy._polish_engine import brent_fmin
from spanormpy._tps import _design, _ns_knots

__all__ = ["lm_fit", "fit_f_dist", "fit_f_dist_robustly", "squeeze_var", "e_bayes"]

_QR_TOL = 1e-7  # lm.fit / lm.wfit default rank tolerance


# ---------------------------------------------------------------------------------------------
# special functions
# ---------------------------------------------------------------------------------------------
def _logmdigamma(z) -> np.ndarray:
    """``log(z) - digamma(z)`` as statmod's ``logmdigamma`` computes it (real ``z``).

    For ``z < 5`` the recurrence moves to ``z + 5``; from there an asymptotic series is used,
    which stays accurate where ``log(z) - digamma(z)`` would cancel. ``z <= 0`` and ``nan``
    give ``nan``.
    """
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    good = np.isfinite(z) & (z > 0) | (z == np.inf)
    small = good & (z < 5)
    big = good & ~small
    if small.any():
        x = z[small]
        out[small] = (
            np.log(x / (x + 5)) + _logmdigamma_tail(x + 5) + sum(1 / (x + k) for k in range(5))
        )
    if big.any():
        out[big] = _logmdigamma_tail(z[big])
    return out


def _logmdigamma_tail(z: np.ndarray) -> np.ndarray:
    """The ``Mod(z) >= 5`` branch of statmod's ``logmdigamma``."""
    x = 1 / z**2
    tail = x * (
        -1 / 12
        + x
        * (
            1 / 120
            + x
            * (
                -1 / 252
                + x
                * (1 / 240 + x * (-1 / 132 + x * (691 / 32760 + x * (-1 / 12 + 3617 * x / 8160))))
            )
        )
    )
    return 1 / (2 * z) - tail


def _trigamma_inverse(x) -> np.ndarray:
    """Inverse of ``trigamma`` (limma ``trigammaInverse``).

    Newton's method on the elements in the iterative range, all stopped together when
    ``max(-dif / y) < 1e-8`` (R's rule), so the result is accurate to about that tolerance and
    agrees with R's iterates exactly up to rounding.
    """
    x = np.array(x, dtype=float, copy=True)
    y = x.copy()
    ok = ~np.isnan(x)
    neg = ok & (x < 0)
    if neg.any():
        y[neg] = np.nan
        warnings.warn("NaNs produced", UserWarning, stacklevel=2)
    ok &= ~neg
    hi = ok & (x > 1e7)
    y[hi] = 1 / np.sqrt(x[hi])
    ok &= ~hi
    lo = ok & (x < 1e-6)
    y[lo] = 1 / x[lo]
    ok &= ~lo
    if ok.any():
        xs = x[ok]
        ys = 0.5 + 1 / xs
        it = 0
        while True:
            it += 1
            tri = special.polygamma(1, ys)
            dif = tri * (1 - tri / xs) / special.polygamma(2, ys)
            ys = ys + dif
            if np.max(-dif / ys) < 1e-8:
                break
            if it > 50:
                warnings.warn("Iteration limit exceeded", UserWarning, stacklevel=2)
                break
        y[ok] = ys
    return y


# ---------------------------------------------------------------------------------------------
# least squares with R's rank rule
# ---------------------------------------------------------------------------------------------
def _dqrdc2_rank(x: np.ndarray, tol: float = _QR_TOL) -> tuple[int, np.ndarray]:
    """Rank and column pivot of ``x`` as LINPACK ``dqrdc2`` (R's ``lm.fit``) decides them.

    Householder QR in which a column whose norm, after the reflections of the columns before
    it, falls below ``tol`` times its original norm is moved to the end. Returns the rank (the
    columns ``pivot[:rank]`` are estimable) and the 0-based pivot.
    """
    x = np.array(x, dtype=float, copy=True)
    n, p = x.shape
    jpvt = np.arange(p)
    qraux = np.linalg.norm(x, axis=0)
    work1 = qraux.copy()
    work2 = qraux.copy()
    work2[work2 == 0] = 1.0
    k = p + 1  # Fortran's 1-based counter of the first negligible column
    for l0 in range(min(n, p)):
        # cycle negligible columns to the end (the l < k test stops infinite cycling)
        while l0 + 1 < k and qraux[l0] < work2[l0] * tol:
            for a in (x, jpvt, qraux, work1, work2):
                moved = a[..., l0].copy()
                a[..., l0 : p - 1] = a[..., l0 + 1 : p]
                a[..., p - 1] = moved
            k -= 1
        if l0 + 1 == n:
            break
        nrmxl = np.linalg.norm(x[l0:, l0])
        if nrmxl == 0.0:
            continue
        if x[l0, l0] != 0.0:
            nrmxl = np.copysign(nrmxl, x[l0, l0])
        x[l0:, l0] /= nrmxl
        x[l0, l0] += 1.0
        for j in range(l0 + 1, p):
            t = -np.dot(x[l0:, l0], x[l0:, j]) / x[l0, l0]
            x[l0:, j] += t * x[l0:, l0]
            if qraux[j] != 0.0:
                tt = max(1.0 - (abs(x[l0, j]) / qraux[j]) ** 2, 0.0)
                if abs(tt) < 1e-6:  # relative tolerance, modified 9/99 by Doug Bates
                    qraux[j] = np.linalg.norm(x[l0 + 1 :, j])
                    work1[j] = qraux[j]
                else:
                    qraux[j] *= np.sqrt(tt)
        qraux[l0] = x[l0, l0]
        x[l0, l0] = -nrmxl
    return min(k - 1, n), jpvt


def _ls_one(X: np.ndarray, y: np.ndarray):
    """One least-squares fit (``lm.fit`` on pre-weighted data).

    Returns ``(coef, fitted, rss, rank, rinv)``: ``coef`` has ``nan`` for aliased columns,
    ``rinv`` is the inverse of the triangular factor of the estimable columns (so that
    ``(X'X)^-1 = rinv @ rinv.T`` on them). ``rss`` is the residual sum of squares, i.e. the
    sum of squared effects beyond the rank.
    """
    n, p = X.shape
    rank, pivot = _dqrdc2_rank(X)
    est = np.sort(pivot[:rank])
    coef = np.full(p, np.nan)
    if rank == 0:
        return coef, np.zeros(n), float(y @ y), 0, np.empty((0, 0))
    q, r = np.linalg.qr(X[:, est])
    b = np.linalg.solve(r, q.T @ y)
    coef[est] = b
    fitted = X[:, est] @ b
    res = y - fitted
    return coef, fitted, float(res @ res), rank, np.linalg.inv(r)


def _as_matrix_weights(weights, shape: tuple[int, int]) -> np.ndarray:
    """limma ``asMatrixWeights``: a weight matrix, gene weights or array weights as G x n."""
    g, n = shape
    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        w = w.reshape(1, 1)
    elif w.ndim == 1:
        w = w.reshape(-1, 1)  # R's as.matrix() on a vector is a column
    elif w.ndim > 2:
        raise ValueError("weights is of unexpected shape")
    if w.shape == (g, n):
        return w
    if min(w.shape) != 1:
        raise ValueError("weights is of unexpected shape")
    if w.shape[1] > 1 and w.shape[1] == n:  # a row vector: array weights
        return np.tile(w, (g, 1))
    lw = w.size
    if lw == 1 or lw == g:  # gene weights
        return np.tile(w.reshape(-1, 1), (1, n)) if lw == g else np.full((g, n), w.item())
    if lw == n:  # array weights
        return np.tile(w.reshape(1, -1), (g, 1))
    raise ValueError("weights is of unexpected size")


def lm_fit(Y, design, weights=None) -> dict:
    """Least-squares fit of every row of ``Y`` on ``design`` (limma ``lmFit``, method ``"ls"``).

    Each gene is fitted on its own observations: non-finite entries of ``Y`` and weights that
    are not positive and finite are dropped (``lm.series``), so genes can have different residual
    df. Aliased coefficients (rank deficiency after dropping observations, R's ``dqrdc2`` rule)
    are ``nan``.

    Parameters
    ----------
    Y : array (G, n)
        Expression, genes x observations; ``nan`` marks a missing observation.
    design : array (n, p)
        Design matrix without ``nan``.
    weights : None, array (G, n), (n,) or (G,)
        Observation weights; a vector is read as array weights (length ``n``) or gene weights
        (length ``G``) as ``asMatrixWeights`` does (gene weights win when ``G == n``).

    Returns
    -------
    dict
        ``coefficients`` and ``stdev_unscaled`` (G, p), ``sigma`` and ``df_residual`` (G,),
        ``Amean`` (G,; row means of ``Y`` ignoring ``nan``, as ``lmFit`` on a matrix) and
        ``design``. ``sigma`` is ``nan`` for genes with no residual df.
    """
    Y = np.array(Y, dtype=float, copy=True)
    if Y.ndim != 2:
        raise ValueError("Y must be a genes x observations matrix")
    if Y.shape[0] == 0:
        raise ValueError("expression matrix has zero rows")
    g, n = Y.shape
    design = np.asarray(design, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.shape[0] != n:
        raise ValueError("row dimension of design doesn't match column dimension of data object")
    if np.isnan(design).any():
        raise ValueError("NAs not allowed in design matrix")
    p = design.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        amean = np.nanmean(np.where(np.isnan(Y), np.nan, Y), axis=1)
    w = None
    if weights is not None:
        w = _as_matrix_weights(weights, (g, n)).copy()
        w[w <= 0] = np.nan
        Y[~np.isfinite(w)] = np.nan

    beta = np.full((g, p), np.nan)
    sdu = np.full((g, p), np.nan)
    sigma = np.full(g, np.nan)
    dfres = np.zeros(g)
    obs = np.isfinite(Y)
    m_obs = obs.sum(axis=1)
    for m in np.unique(m_obs):
        if m == 0:
            continue
        rows = np.flatnonzero(m_obs == m)
        # the observations of each gene, in order
        cols = np.argsort(~obs[rows], axis=1, kind="stable")[:, :m]
        yg = np.take_along_axis(Y[rows], cols, axis=1)
        a = design[cols]  # (g, m, p)
        if w is not None:
            sw = np.sqrt(np.take_along_axis(w[rows], cols, axis=1))
            a = a * sw[:, :, None]
            yg = yg * sw
        fast = False
        if m >= p:
            # LAPACK's |R_jj| is the norm of column j left after the columns before it, which
            # is what dqrdc2 compares with tol * its original norm: if no column is negligible
            # dqrdc2 pivots nothing and the batched fit is the same fit.
            q, r = np.linalg.qr(a)
            norms = np.linalg.norm(a, axis=1)
            norms[norms == 0] = 1.0
            fast_rows = (np.abs(np.diagonal(r, axis1=1, axis2=2)) >= _QR_TOL * norms).all(axis=1)
            if fast_rows.any():
                qf, rf = q[fast_rows], r[fast_rows]
                b = np.linalg.solve(rf, np.einsum("gmp,gm->gp", qf, yg[fast_rows])[:, :, None])[
                    :, :, 0
                ]
                res = yg[fast_rows] - np.einsum("gmp,gp->gm", a[fast_rows], b)
                rss = (res**2).sum(axis=1)
                rinv = np.linalg.inv(rf)
                idx = rows[fast_rows]
                beta[idx] = b
                sdu[idx] = np.sqrt((rinv**2).sum(axis=2))
                dfres[idx] = m - p
                if m > p:
                    sigma[idx] = np.sqrt(rss / (m - p))
            fast = True
            slow = rows[~fast_rows]
            slow_pos = np.flatnonzero(~fast_rows)
        if not fast:
            slow = rows
            slow_pos = np.arange(len(rows))
        for gi, pos in zip(slow, slow_pos):
            coef, _, rss, rank, rinv = _ls_one(a[pos], yg[pos])
            est = ~np.isnan(coef)
            beta[gi] = coef
            sdu[gi, est] = np.sqrt((rinv**2).sum(axis=1))
            dfres[gi] = m - rank
            if m - rank > 0:
                sigma[gi] = np.sqrt(rss / (m - rank))
    return {
        "coefficients": beta,
        "stdev_unscaled": sdu,
        "sigma": sigma,
        "df_residual": dfres,
        "Amean": amean,
        "design": design,
    }


# ---------------------------------------------------------------------------------------------
# natural spline with intercept (the trend design of fitFDist)
# ---------------------------------------------------------------------------------------------
def _ns_intercept(x: np.ndarray, df: int):
    """``splines::ns(x, df = df, intercept = TRUE)`` and a predictor for new values.

    ``ns`` with an intercept puts ``df - 2`` interior knots at quantiles of ``x`` (spanormpy's
    ``_ns_knots`` places ``df - 1`` for the no-intercept basis, so it is called with
    ``df - 1``), keeps the constant column of the B-spline basis and projects both onto the
    null space of the second-derivative boundary constraints. Returns ``(basis, predict)``;
    ``predict(newx)`` extends linearly beyond the boundary knots as ``predict.ns`` does.
    """
    knots, bk = _ns_knots(x, df - 1)
    aknots = np.sort(np.concatenate([np.repeat(bk, 4), knots]))
    const = _design(aknots, bk, 2)  # 2 x (nIknots + 4); the intercept column is kept
    q, _ = np.linalg.qr(const.T, mode="complete")
    proj = q[:, 2:]

    def predict(newx: np.ndarray) -> np.ndarray:
        newx = np.asarray(newx, dtype=float)
        basis = np.zeros((len(newx), len(knots) + 4))
        lo, hi = newx < bk[0], newx > bk[1]
        inside = ~(lo | hi)
        for mask, pivot in ((lo, bk[:1]), (hi, bk[1:])):
            if mask.any():
                tt = np.vstack([_design(aknots, pivot, 0), _design(aknots, pivot, 1)])
                basis[mask] = np.column_stack([np.ones(mask.sum()), newx[mask] - pivot[0]]) @ tt
        if inside.any():
            basis[inside] = _design(aknots, newx[inside], 0)
        return basis @ proj

    return predict(x), predict


# ---------------------------------------------------------------------------------------------
# fitFDist
# ---------------------------------------------------------------------------------------------
def fit_f_dist(x, df1, covariate=None):
    """Moment estimates of the scaled F prior of the variances (limma ``fitFDist``).

    Models ``x ~ scale * F(df1, df2)``. With a ``covariate`` the log scale is a natural cubic
    spline of it (``ns(covariate, df = splinedf, intercept = TRUE)`` with ``splinedf`` between
    1 and 4 by the number of usable values); without, it is constant.

    Parameters
    ----------
    x : array (n,)
        Sample variances; ``nan``, infinite and negative values are skipped (their trend
        value is predicted from ``covariate``).
    df1 : float or array (n,)
        Residual df of ``x``; values <= 1e-15 are skipped.
    covariate : array (n,) or None
        Trend covariate, no ``nan``; infinite values are moved to just outside the finite range.

    Returns
    -------
    scale, df2
        ``scale`` is a float (no trend) or an array (n,) (trend), ``df2`` a float, possibly
        ``inf``; both ``nan`` when nothing can be estimated.
    """
    x = np.array(x, dtype=float, copy=True).ravel()
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), 0.0
    df1 = np.array(df1, dtype=float, copy=True).ravel()
    scalar_df = len(df1) == 1
    ok = np.isfinite(df1) & (df1 > 1e-15)
    if scalar_df:
        if not ok[0]:
            return np.nan, np.nan
        ok = np.ones(n, dtype=bool)
    elif len(df1) != n:
        raise ValueError("x and df1 have different lengths")
    if covariate is not None:
        covariate = np.array(covariate, dtype=float, copy=True).ravel()
        if len(covariate) != n:
            raise ValueError("x and covariate must be of same length")
        if np.isnan(covariate).any():
            raise ValueError("NA covariate values not allowed")
        isfin = np.isfinite(covariate)
        if not isfin.all():
            if isfin.any():
                lo, hi = covariate[isfin].min(), covariate[isfin].max()
                covariate[covariate == -np.inf] = lo - 1
                covariate[covariate == np.inf] = hi + 1
            else:
                covariate = np.sign(covariate)
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 1:
        return float(x[ok][0]), 0.0
    notallok = nok < n
    if notallok:
        x = x[ok]
        if not scalar_df:
            df1 = df1[ok]
        if covariate is not None:
            covariate_notok = covariate[~ok]
            covariate = covariate[ok]
    if covariate is not None:
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))
        if splinedf < 2:
            scale, df2 = fit_f_dist(x, df1)
            return np.full(n, scale), df2
    x = np.maximum(x, 0.0)
    m = float(np.median(x))
    if m == 0:
        warnings.warn(
            "More than half of residual variances are exactly zero: eBayes unreliable",
            UserWarning,
            stacklevel=2,
        )
        m = 1.0
    elif (x == 0).any():
        warnings.warn(
            "Zero sample variances detected, have been offset away from zero",
            UserWarning,
            stacklevel=2,
        )
    x = np.maximum(x, 1e-5 * m)
    z = np.log(x)
    e = z + _logmdigamma(df1 / 2)
    if covariate is None:
        emean = e.mean()
        evar = ((e - emean) ** 2).sum() / (nok - 1)
    else:
        design, predict = _ns_intercept(covariate, splinedf)
        coef, fitted, rss, rank, _ = _ls_one(design, e)
        if notallok:
            emean = np.zeros(n)
            emean[ok] = fitted
            emean[~ok] = predict(covariate_notok) @ coef
        else:
            emean = fitted
        evar = rss / (nok - rank)  # mean of the squared effects beyond the rank
    evar = evar - np.mean(special.polygamma(1, df1 / 2))
    if evar > 0:
        df2 = 2 * float(_trigamma_inverse(evar))
        s20 = np.exp(emean - _logmdigamma(df2 / 2))
    else:
        df2 = np.inf
        s20 = x.mean() if covariate is None else np.exp(emean)
    return (float(s20) if np.ndim(s20) == 0 else s20), df2


# ---------------------------------------------------------------------------------------------
# weighted lowess and fitFDistUnequalDF1
# ---------------------------------------------------------------------------------------------
_WL_THRESHOLD = 1e-7  # THRESHOLD in weighted_lowess.c


def _wl_seeds(xs: np.ndarray, delta: float) -> np.ndarray:
    """``find_seeds``: first and last point plus every point more than ``delta`` past the last."""
    n = len(xs)
    seeds = [0]
    last = 0
    for pt in range(1, n - 1):
        if xs[pt] - xs[last] > delta:
            seeds.append(pt)
            last = pt
    seeds.append(n - 1)
    return np.array(seeds)


def _wl_limits(xs, ws, pt: int, spanweight: float) -> tuple[int, int, float]:
    """``find_limits`` for one seed: window ends and the largest distance in it.

    The C code grows the window one point at a time, taking the left neighbour only when it is
    strictly nearer than the right one, until the weights add up to ``spanweight``. That is the
    other points taken in order of distance with the right side first on ties, so the window is
    found with one sort and a cumulative sum (same order of additions), then extended over ties.
    """
    n = len(xs)
    others = np.concatenate([np.arange(pt), np.arange(pt + 1, n)])
    d = np.abs(xs[others] - xs[pt])
    order = np.lexsort((others < pt, d))  # by distance, right (index > pt) before left
    others, d = others[order], d[order]
    cum = np.cumsum(np.concatenate([[ws[pt]], ws[others]]))
    k = int(np.searchsorted(cum, spanweight, side="left"))  # points added once cum >= target
    k = min(k, len(others))
    taken = others[:k]
    left = min(pt, int(taken.min())) if k else pt
    right = max(pt, int(taken.max())) if k else pt
    mdist = float(d[k - 1]) if k else 0.0
    while left > 0 and xs[left] == xs[left - 1]:
        left -= 1
    while right < n - 1 and xs[right] == xs[right + 1]:
        right += 1
    return left, right, mdist


def _wl_fit(xs, ys, ws, rw, pt: int, left: int, right: int, dist: float) -> float:
    """``lowess_fit``: local linear fit at ``xs[pt]`` over ``[left, right]``."""
    sl = slice(left, right + 1)
    if dist < _WL_THRESHOLD:
        work = ws[sl] * rw[sl]
        return float((ys[sl] * work).sum() / work.sum())
    work = (1 - (np.abs(xs[pt] - xs[sl]) / dist) ** 3) ** 3 * ws[sl] * rw[sl]
    allw = work.sum()
    xmean = (work * xs[sl]).sum() / allw
    ymean = (work * ys[sl]).sum() / allw
    var = ((xs[sl] - xmean) ** 2 * work).sum()
    covar = ((xs[sl] - xmean) * (ys[sl] - ymean) * work).sum()
    if var < _WL_THRESHOLD:
        return float(ymean)
    slope = covar / var
    return float(slope * xs[pt] + (ymean - slope * xmean))


def _weighted_lowess(x, y, weights, span: float, iterations: int = 1, npts: int = 200):
    """limma ``weightedLowess`` (R wrapper and ``weighted_lowess.c``), fitted values only.

    A line-by-line port. ``delta`` comes from ``npts`` as in the wrapper (0 when
    ``npts >= n``); seeds are the points more than ``delta`` apart, fitted exactly, with the
    points between interpolated linearly. Each window holds ``span`` times the total prior
    weight. With ``iterations > 1`` robustness weights (bisquare on the weighted MAD) are
    iterated as in the C code. Ties in ``x`` are handled as the C code does (windows extend
    over ties; the interpolation slope falls back to the end-point average when the seeds
    coincide).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least two points")
    if iterations <= 0:
        raise ValueError("number of robustness iterations should be positive")
    o = np.argsort(x, kind="stable")
    xs, ys, ws = x[o], y[o], w[o]
    npts = int(npts + 0.5)
    if npts < 1:
        raise ValueError("number of points should be a positive integer")
    if npts >= n:
        delta = 0.0
    else:
        cum = np.cumsum(np.sort(np.diff(xs)))
        nc = np.arange(npts)
        delta = float(np.min(cum[len(cum) - 1 - nc] / (npts - nc)))
    totalweight = ws.sum()
    spanweight = totalweight * span
    subrange = (xs[-1] - xs[0]) / n
    seeds = _wl_seeds(xs, delta)
    limits = [_wl_limits(xs, ws, int(pt), spanweight) for pt in seeds]
    rob = np.ones(n)
    fit = np.empty(n)
    for _ in range(iterations):
        fit[0] = _wl_fit(xs, ys, ws, rob, 0, *limits[0])
        last = 0
        for s in range(1, len(seeds)):
            pt = int(seeds[s])
            fit[pt] = _wl_fit(xs, ys, ws, rob, pt, *limits[s])
            if pt - last > 1:
                cur = xs[pt] - xs[last]
                if cur > _WL_THRESHOLD * subrange:
                    slope = (fit[pt] - fit[last]) / cur
                    fit[last + 1 : pt] = slope * xs[last + 1 : pt] + (fit[pt] - slope * xs[pt])
                else:
                    fit[last + 1 : pt] = 0.5 * (fit[pt] + fit[last])
            last = pt
        rsd = np.abs(ys - fit)
        resid_scale = rsd.sum() / n
        ror = np.argsort(rsd, kind="stable")
        rsd_sorted = rsd[ror]
        cmad = 0.0
        half = totalweight / 2
        cur_w = 0.0
        for j in range(n):
            cur_w += ws[ror[j]]
            if cur_w == half:  # the unlikely exact match
                cmad = 3 * (rsd_sorted[j] + rsd_sorted[j + 1])
                break
            if cur_w > half:
                cmad = 6 * rsd_sorted[j]
                break
        if cmad <= _WL_THRESHOLD * resid_scale:
            break
        rob[ror] = np.where(rsd_sorted < cmad, (1 - (rsd_sorted / cmad) ** 2) ** 2, 0.0)
    out = np.empty(n)
    out[o] = fit
    return out


def _loess_fit_trend(y, x, weights, span, min_weight, max_weight) -> np.ndarray:
    """limma ``loessFit(y, x, weights, span, iterations = 1)$fitted`` (weighted, non-null weights).

    Weights are clamped to ``[min_weight, max_weight]``. Equal weights (limma then calls
    ``lowess``) cannot arise from ``fitFDistUnequalDF1`` (the weights differ with the df) and
    raise ``NotImplementedError``.
    """
    n = len(y)
    fitted = np.full(n, np.nan)
    obs = np.isfinite(y) & np.isfinite(x)
    xobs, yobs = x[obs], y[obs]
    nobs = len(yobs)
    if nobs == 0:
        return fitted
    if span < 1 / nobs:
        fitted[obs] = yobs
        return fitted
    wobs = np.where(np.isnan(weights[obs]), 0.0, weights[obs])
    wobs = np.minimum(np.maximum(wobs, max(min_weight, 0.0)), max_weight)
    if wobs.max() - wobs.min() < 1e-15:
        raise NotImplementedError("loessFit with equal weights (plain lowess) is not ported")
    nwobs = nobs if min_weight > 0 else int((wobs > 0).sum())
    if nwobs < 4 + 1 / span:
        if nwobs == 1:
            fitted[obs] = yobs[wobs > 0][0]
        else:  # lm.wfit(cbind(1, x), y, w)
            sw = np.sqrt(wobs)
            a = np.column_stack([np.ones(nobs), xobs]) * sw[:, None]
            coef = np.linalg.lstsq(a, yobs * sw, rcond=None)[0]
            fitted[obs] = coef[0] + coef[1] * xobs
        return fitted
    fitted[obs] = _weighted_lowess(xobs, yobs, wobs, span, iterations=1, npts=200)
    return fitted


def _p_adjust_bh(p: np.ndarray) -> np.ndarray:
    """``p.adjust(p, method = "BH")`` (no missing values)."""
    n = len(p)
    o = np.argsort(-p, kind="stable")
    ro = np.argsort(o, kind="stable")
    i = np.arange(n, 0, -1)
    return np.minimum(1.0, np.minimum.accumulate(n / i * p[o]))[ro]


def _choose_lowess_span(n: int, small_n: float = 50, min_span: float = 0.3, power=1 / 3) -> float:
    """limma ``chooseLowessSpan``."""
    return min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0)


def _force_monotone(df2_shrunk: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Make ``df2_shrunk`` non-decreasing along the order ``o`` (the monotone block of limma).

    Ports the ``which.min(cumsum(x)/seq_along(x))`` pooling of ``fitFDistUnequalDF1``: values up
    to the minimum running mean are replaced by it, then a running maximum is taken. ``which.min``
    skips NaN and fails (``1:integer(0)``) only when every running mean is NaN.
    """
    n = len(df2_shrunk)
    ordered = df2_shrunk[o]
    m = np.cumsum(ordered) / np.arange(1, n + 1)
    if np.isnan(m).all():
        raise ValueError("argument of length 0")
    imin = int(np.nanargmin(m))
    ordered[: imin + 1] = m[imin]
    out = np.empty(n)
    out[o] = np.maximum.accumulate(ordered)
    return out


def _fit_f_dist_unequal_df1(
    x, df1, covariate=None, span=None, robust=False, prior_weights=None
) -> dict:
    """limma ``fitFDistUnequalDF1``: maximum-likelihood ``df2`` for genes with different df.

    The prior scale (log) is the weighted mean of ``log(x) + log(1/psi(df/2)...)`` (weights
    ``1/trigamma(df1/2)``) or, with a covariate, its weighted lowess trend; ``df2`` maximises the
    F likelihood by Brent's method on ``par = df2/2 / (1 + df2/2)`` in (0.5, 0.9998).
    With ``robust`` the genes are tested two-sided against the fitted prior, Benjamini-Hochberg
    FDRs (capped to 1 above 0.3) become prior weights for a refit, and ``df2_shrunk`` lowers the
    prior df of right-outlying genes.

    Returns a dict with ``scale`` and ``df2`` (``nan`` if fewer than two informative genes);
    robust fits that found outliers add ``df2_outlier`` and the per-gene ``df2_shrunk``.
    """
    x = np.array(x, dtype=float, copy=True).ravel()
    n = len(x)
    df1 = np.array(df1, dtype=float, copy=True).ravel()
    if len(df1) not in (1, n):
        raise ValueError("x and df1 are different lengths")
    if np.isnan(df1).any():
        raise ValueError("NA df1 values")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float).ravel()
        if len(covariate) != n:
            raise ValueError("x and covariate are different lengths")
        if np.isnan(covariate).any():
            raise ValueError("prior.weights contain NA values")
    prior_w = None
    if prior_weights is not None:
        prior_w = np.array(prior_weights, dtype=float, copy=True).ravel()
        if len(prior_w) != n:
            raise ValueError("x and covariate are different lengths")
        if np.isnan(prior_w).any():
            raise ValueError("prior.weights contain NA values")
        if prior_w.min() < 0:
            raise ValueError("prior.weights are negative")
    if np.isnan(x).any():
        i = np.isnan(x)
        prior_w = (~i).astype(float) if prior_w is None else np.where(i, 0.0, prior_w)
        x[i] = 0.0
    if df1.min() < 0.01:
        i = df1 < 0.01
        prior_w = (~i).astype(float) if prior_w is None else np.where(i, 0.0, prior_w)
        df1 = np.where(i, 1.0, df1)
    informative = x > 0
    if prior_w is not None:
        informative &= prior_w != 0
    n_inf = int(informative.sum())
    if n_inf < 2:
        return {"scale": np.nan, "df2": np.nan}
    # R sets prior.weights to NULL here but keeps its PriorWeights flag: with NA / tiny-df genes
    # present the weighted branch then runs on NULL weights, so emean is 0/0 and the objective is
    # sum(NULL * ...) = 0 for every df2. Reproduced below (scale is NaN, df2 is Brent's start).
    r_null_weights_accident = n_inf == 2 and prior_w is not None
    if n_inf == 2:
        covariate, prior_w, robust = None, None, False
    m = float(np.median(x[informative]))
    xpos = np.maximum(x, 1e-12 * m)
    d1 = df1 / 2
    e = np.log(xpos) + _logmdigamma(d1)
    w = 1 / special.polygamma(1, d1)
    if len(w) < n:
        w = np.full(n, w[0])
    if prior_w is not None:
        w = w * prior_w
    if r_null_weights_accident:
        emean = np.nan
    elif covariate is None:
        emean = float((w * e).sum() / w.sum())
    else:
        if span is None:
            span = _choose_lowess_span(n, small_n=500)
        emean = _loess_fit_trend(
            e, covariate, w / np.quantile(w, 0.75), span, min_weight=1e-8, max_weight=100
        )
    d1x = d1 * xpos
    pw = np.ones(n) if prior_w is None else prior_w

    def minus_twice_loglik(par: float) -> float:
        if r_null_weights_accident:
            return 0.0
        d2 = par / (1 - par)
        d2s20 = d2 * np.exp(emean - _logmdigamma(d2))
        return -2 * float(
            (
                pw
                * (
                    -(d1 + d2) * np.log1p(d1x / d2s20)
                    - d1 * np.log(d2s20)
                    + special.gammaln(d1 + d2)
                    - special.gammaln(d2)
                )
            ).sum()
        )

    par = brent_fmin(minus_twice_loglik, 0.5, 0.9998)
    d2 = par / (1 - par)
    s20 = np.exp(emean - _logmdigamma(d2))
    if np.ndim(s20) == 0:
        s20 = float(s20)
    df2 = 2 * float(d2)
    if not robust:
        return {"scale": s20, "df2": df2}

    # two-sided outlier test against the fitted prior; the BH FDR (1 above 0.3) is the
    # approximate probability of not being an outlier
    df1n = np.broadcast_to(df1, (n,))
    fstat = x / s20
    right_p = stats.f.sf(fstat, df1n, df2)
    left_p = 1 - right_p
    if left_p.min() < 0.001:
        i = left_p < 0.001
        left_p[i] = stats.f.cdf(fstat[i], df1n[i], df2)
    fdr = _p_adjust_bh(2 * np.minimum(left_p, right_p))
    fdr[fdr > 0.3] = 1.0
    if fdr.min() == 1:
        return {"scale": s20, "df2": df2}

    # refit with the FDRs as prior weights. R's Recall() drops `span`, so a user span is
    # not used for the refit trend (logged in docs/r-discrepancies.md)
    outpw = _fit_f_dist_unequal_df1(x, df1, covariate=covariate, robust=False, prior_weights=fdr)
    s20, df2 = outpw["scale"], outpw["df2"]

    # qqplot-type method for right outliers
    r = stats.rankdata(fstat)
    uniform_p = (n - r + 0.5) / n
    prob_not_outlier = np.minimum(right_p / uniform_p, 1)
    if prob_not_outlier.min() == 1:
        return outpw

    # df2 of the outlier distribution makes the largest F statistic its median
    i = int(np.argmin(right_p))
    min_right_p = right_p[i]
    if min_right_p == 0:
        df2_outlier = 0.0
        df2_shrunk = prob_not_outlier * df2
    else:
        df2_outlier = math.log(0.5) / math.log(min_right_p) * df2
        new_log_right_p = float(_pf_upper_log(fstat[i : i + 1], df1n[i], df2_outlier)[0])
        df2_outlier = math.log(0.5) / new_log_right_p * df2_outlier
        df2_shrunk = prob_not_outlier * df2 + (1 - prob_not_outlier) * df2_outlier
    if np.isnan(df2_shrunk).any():  # R: 1:integer(0) fails when which.min finds nothing
        raise ValueError("argument of length 0")
    # df2_shrunk monotone in RightP
    df2_shrunk = _force_monotone(df2_shrunk, np.argsort(right_p, kind="stable"))
    return {"scale": s20, "df2": df2, "df2_outlier": df2_outlier, "df2_shrunk": df2_shrunk}


# ---------------------------------------------------------------------------------------------
# squeezeVar and eBayes
# ---------------------------------------------------------------------------------------------
def _squeeze_var_post(var, df, var_prior, df_prior) -> np.ndarray:
    """limma ``.squeezeVar``: the posterior variances, allowing infinite ``df_prior``."""
    var = np.asarray(var, dtype=float)
    df = np.asarray(df, dtype=float)
    df_prior = np.asarray(df_prior, dtype=float)
    var_prior = np.asarray(var_prior, dtype=float)
    if np.isfinite(df_prior.max()):
        return (df * var + df_prior * var_prior) / (df + df_prior)
    n = len(var)
    var_post = np.broadcast_to(var_prior, (n,)).astype(float)  # a copy
    dfp = np.broadcast_to(df_prior, (n,))
    if dfp.min() > 1e100:
        return var_post
    i = np.isfinite(dfp)
    dfi = df[i] if df.size > 1 else df
    var_post[i] = (dfi * var[i] + dfp[i] * var_post[i]) / (dfi + dfp[i])
    return var_post


def squeeze_var(
    var, df, covariate=None, span=None, robust=False, winsor_tail_p=(0.05, 0.1), legacy=None
) -> dict:
    """Empirical-Bayes shrinkage of sample variances (limma ``squeezeVar``).

    Estimates the prior (``df_prior``, ``var_prior``) from ``var`` with ``fit_f_dist`` when
    all genes with residual df share one value (``legacy``), and with the unequal-df
    maximum-likelihood fit (weighted lowess trend, ``span``) otherwise; then returns the
    posterior ``(df * var + df_prior * var_prior) / (df + df_prior)``.

    Returns
    -------
    dict
        ``df_prior`` (float, or an array when the robust fit shrinks per gene), ``var_prior``
        (float, or array (n,) with a covariate) and ``var_post`` (n,). With fewer than 3
        genes the prior is the data: ``df_prior = 0``.
    """
    var = np.array(var, dtype=float, copy=True).ravel()
    n = len(var)
    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return {"df_prior": 0.0, "var_prior": var.copy(), "var_post": var.copy()}
    df = np.array(df, dtype=float, copy=True).ravel()
    if len(df) > 1:
        var[df == 0] = 0.0
    if span is not None:
        legacy = False
    if legacy is None:
        dfp = df[df > 0]
        legacy = len(dfp) > 0 and dfp.min() == dfp.max()
    if legacy:
        if robust:
            fit = fit_f_dist_robustly(var, df, covariate, winsor_tail_p)
            fit_scale, df_prior = fit["scale"], fit["df2_shrunk"]
        else:
            fit_scale, df_prior = fit_f_dist(var, df, covariate)
    else:
        fit = _fit_f_dist_unequal_df1(var, df, covariate=covariate, span=span, robust=robust)
        fit_scale = fit["scale"]
        df_prior = fit.get("df2_shrunk", fit["df2"])
    if np.isnan(df_prior).any():
        raise ValueError("Could not estimate prior df")
    return {
        "df_prior": df_prior,
        "var_prior": fit_scale,
        "var_post": _squeeze_var_post(var, df, fit_scale, df_prior),
    }


def _trimmed_mean(z: np.ndarray, trim: float) -> float:
    """R ``mean(z, trim = trim)`` (``trim >= 0.5`` gives the median)."""
    n = len(z)
    if trim >= 0.5:
        return float(np.median(z))
    lo = math.floor(n * trim)
    return float(np.mean(np.sort(z)[lo : n - lo]))


def _pf_log(f, df1, df2, lower: bool):
    """``pf(f, df1, df2, lower.tail = lower, log.p = TRUE)`` (``df2`` may be ``inf``)."""
    f = np.asarray(f, dtype=float)
    if not lower:
        return _pf_upper_log(f, df1, df2)
    if np.isinf(df2):
        return stats.chi2.logcdf(f * df1, df1)
    return stats.f.logcdf(f, df1, df2)


def _qf_log(logp, df1, df2, lower: bool):
    """``qf(logp, df1, df2, lower.tail = lower, log.p = TRUE)`` (``df2`` may be ``inf``)."""
    p = np.exp(np.asarray(logp, dtype=float))
    if np.isinf(df2):
        q = stats.chi2.ppf(p, df1) if lower else stats.chi2.isf(p, df1)
        return q / df1
    return stats.f.ppf(p, df1, df2) if lower else stats.f.isf(p, df1, df2)


def _approx_ties_mean(x, y, xout) -> np.ndarray:
    """``approx(x, y, xout, rule = 2, ties = mean)$y``: linear, constant outside the range."""
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    return np.interp(xout, ux, uy)


def fit_f_dist_robustly(x, df1, covariate=None, winsor_tail_p=(0.05, 0.1)) -> dict:
    """Robust moment estimates of the scaled F prior (limma ``fitFDistRobustly``).

    Matches the variance of the Winsorised log variances (after removing the mean, or a
    ``loessFit(span = 0.4)`` trend on ``covariate``) to its theoretical value under
    ``scale * F(df1, df2)``, and shrinks ``df2`` per gene for variances in the right tail
    (``df2_shrunk``), which is what ``squeeze_var`` uses as the prior df.

    Parameters
    ----------
    x : array (n,)
        Sample variances; ``nan`` and values with ``df1 <= 1e-6`` are set aside (their
        ``df2_shrunk`` is ``df2`` and their trend scale is interpolated).
    df1 : float or array (n,)
        Residual df of ``x``. Unequal values are first transformed to the largest df by
        matching tail probabilities.
    covariate : array (n,) or None
        Trend covariate (finite).
    winsor_tail_p : pair of float
        Lower and upper Winsorising tail probabilities.

    Returns
    -------
    dict
        ``scale`` (float, or array (n,) with a covariate), ``df2`` (float, possibly ``inf``),
        ``df2_shrunk`` (n,), and, when computed, ``tail_p_value``, ``prob_outlier`` and
        ``df2_outlier``.
    """
    x = np.array(x, dtype=float, copy=True).ravel()
    n = len(x)
    if n < 2:
        return {"scale": np.nan, "df2": np.nan, "df2_shrunk": np.nan}
    if n == 2:
        scale, df2 = fit_f_dist(x, df1, covariate)
        return {"scale": scale, "df2": df2}
    df1 = np.array(df1, dtype=float, copy=True).ravel()
    if len(df1) not in (1, n):
        raise ValueError("x and df1 are different lengths")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float).ravel()
        if len(covariate) != n:
            raise ValueError("x and covariate are different lengths")
        if not np.isfinite(covariate).all():
            raise ValueError("covariate contains NA or infinite values")
    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    ok = np.broadcast_to(ok, (n,))
    if not ok.all():
        covariate2 = None if covariate is None else covariate[~ok]
        fit = fit_f_dist_robustly(
            x[ok],
            df1[ok] if len(df1) > 1 else df1,
            None if covariate is None else covariate[ok],
            winsor_tail_p,
        )
        if "df2_shrunk" not in fit:
            raise ValueError("replacement has length zero")  # R: NULL assigned into a vector
        df2_shrunk = x.copy()  # R starts from x; both subsets are overwritten below
        df2_shrunk[ok] = fit["df2_shrunk"]
        df2_shrunk[~ok] = fit["df2"]
        if covariate is None:
            scale = fit["scale"]
        else:
            scale = np.empty(n)
            scale[ok] = fit["scale"]
            scale[~ok] = np.exp(_approx_ties_mean(covariate[ok], np.log(fit["scale"]), covariate2))
        return {"scale": scale, "df2": fit["df2"], "df2_shrunk": df2_shrunk}

    # avoid zero or negative x values
    m = float(np.median(x))
    if m <= 0:
        raise ValueError("Variances are mostly <= 0")
    x[x < m * 1e-12] = m * 1e-12

    # non-robust estimates, also the lower bound for df2
    nr_scale, nr_df2 = fit_f_dist(x, df1, covariate)

    wtp = np.resize(np.asarray(winsor_tail_p, dtype=float), 2)
    prob = np.array([wtp[0], 1 - wtp[1]])
    if (wtp < 1 / n).all():
        return {"scale": nr_scale, "df2": nr_df2, "df2_shrunk": np.full(n, nr_df2)}

    # transform x to the largest df1
    if len(df1) > 1:
        df1max = df1.max()
        i = df1 < df1max - 1e-14
        if i.any():
            s = np.asarray(nr_scale)
            s = s[i] if s.ndim else float(s)
            f = x[i] / s
            pupper = _pf_log(f, df1[i], nr_df2, lower=False)
            plower = _pf_log(f, df1[i], nr_df2, lower=True)
            up = pupper < plower
            if up.any():
                f[up] = _qf_log(pupper[up], df1max, nr_df2, lower=False)
            if (~up).any():
                f[~up] = _qf_log(plower[~up], df1max, nr_df2, lower=True)
            x[i] = f * s
            df1 = np.array([df1max])
    df1 = float(df1[0])

    z = np.log(x)
    if covariate is None:
        ztrend = _trimmed_mean(z, wtp[1])
        zresid = z - ztrend
    else:
        ztrend, zresid = _loess_fit_unweighted(z, covariate, 0.4)

    # moments of Winsorised residuals
    zrq = np.quantile(zresid, prob)
    zwins = np.minimum(np.maximum(zresid, zrq[0]), zrq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.mean((zwins - zwmean) ** 2)) * n / (n - 1)

    # try df2 == Inf
    mom = _winsorized_moments(df1, np.inf, wtp)
    funval_inf = math.log(zwvar / mom[1])
    if funval_inf <= 0:
        ztc = ztrend + zwmean - mom[0]
        fstat = np.exp(z - ztc)
        tail_p = stats.chi2.sf(fstat * df1, df1)
        r = stats.rankdata(fstat)
        prob_not_outlier = np.minimum(tail_p / ((n - r + 0.5) / n), 1)
        df2_shrunk = np.full(n, np.inf)
        o_mask = prob_not_outlier < 1
        if o_mask.any():
            df2_shrunk[o_mask] = prob_not_outlier[o_mask] * (n * df1)
            o = np.argsort(tail_p, kind="stable")
            df2_shrunk[o] = np.maximum.accumulate(df2_shrunk[o])
        return {"scale": np.exp(ztc), "df2": np.inf, "tail_p_value": tail_p,
                "df2_shrunk": df2_shrunk}  # fmt: skip

    # estimate df2 by matching the variance of the Winsorised z
    def fun(t):
        d2 = t / (1 - t)
        return math.log(zwvar / _winsorized_moments(df1, d2, wtp)[1])

    if nr_df2 == np.inf:
        return {"scale": nr_scale, "df2": nr_df2, "df2_shrunk": np.full(n, np.inf)}
    rbx = nr_df2 / (1 + nr_df2)
    funval_low = fun(rbx)
    if funval_low >= 0:
        df2 = nr_df2
    else:
        root = _zeroin2(fun, rbx, 1.0, funval_low, funval_inf, 1e-8)
        df2 = root / (1 - root)

    # correct ztrend for bias
    mom = _winsorized_moments(df1, df2, wtp)
    ztc = ztrend + zwmean - mom[0]
    fstat = np.exp(z - ztc)
    log_tail_p = _pf_log(fstat, df1, df2, lower=False)
    tail_p = np.exp(log_tail_p)
    r = stats.rankdata(fstat)
    log_emp = np.log(n - r + 0.5) - math.log(n)
    log_pno = np.minimum(log_tail_p - log_emp, 0)
    pno = np.exp(log_pno)
    pout = -np.expm1(log_pno)
    if (log_pno < 0).any():
        # df2.outlier makes the largest F statistic the median of its distribution
        min_log_tail_p = float(log_tail_p.min())
        if min_log_tail_p == -np.inf:
            df2_outlier = 0.0
            df2_shrunk = pno * df2
        else:
            df2_outlier = math.log(0.5) / min_log_tail_p * df2
            new_log = float(_pf_log(np.array([fstat.max()]), df1, df2_outlier, lower=False)[0])
            df2_outlier = math.log(0.5) / new_log * df2_outlier
            df2_shrunk = pno * df2 + pout * df2_outlier
        # force df2.shrunk monotone in TailP
        df2_shrunk = _force_monotone(df2_shrunk, np.argsort(log_tail_p, kind="stable"))
    else:
        df2_outlier = df2
        df2_shrunk = np.full(n, df2)
    return {"scale": np.exp(ztc), "df2": df2, "tail_p_value": tail_p, "prob_outlier": pout,
            "df2_outlier": df2_outlier, "df2_shrunk": df2_shrunk}  # fmt: skip


def e_bayes(
    fit: dict,
    proportion: float = 0.01,
    stdev_coef_lim=(0.1, 4),
    trend=False,
    span=None,
    robust: bool = False,
    winsor_tail_p=(0.05, 0.1),
    legacy=None,
) -> dict:
    """Moderated t-statistics for a ``lm_fit`` result (limma ``eBayes``).

    Parameters
    ----------
    fit : dict
        ``lm_fit`` output (``coefficients``, ``stdev_unscaled``, ``sigma``, ``df_residual``,
        and ``Amean`` when ``trend`` is true).
    proportion, stdev_coef_lim
        Accepted for signature parity with limma; they only enter ``var.prior`` and the log-odds,
        which are not computed here.
    trend : bool or array (G,)
        Fit the prior variance as a trend on ``Amean`` (or on the given covariate).
    robust : bool
        Estimate the prior robustly against outlying variances (``df_prior`` is then per gene).

    Returns
    -------
    dict
        A copy of ``fit`` with ``t`` and ``p_value`` (G, p), ``df_total`` (G,), ``s2_prior``
        (float or (G,)), ``df_prior`` (float or (G,)) and ``s2_post`` (G,). ``df_total`` is
        ``df_residual + df_prior`` capped at the pooled residual df, as limma does.
    """
    for key in ("coefficients", "stdev_unscaled", "sigma", "df_residual"):
        if fit.get(key) is None:
            raise ValueError("No data, or argument is not a valid lmFit object")
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    sdu = np.asarray(fit["stdev_unscaled"], dtype=float)
    sigma = np.asarray(fit["sigma"], dtype=float)
    dfres = np.asarray(fit["df_residual"], dtype=float)
    if np.max(dfres) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    if not np.isfinite(sigma).any():
        raise ValueError("No finite residual standard deviations")
    if isinstance(trend, (bool, np.bool_)):
        if trend:
            if fit.get("Amean") is None:
                raise ValueError("Need Amean component in fit to estimate trend")
            covariate = np.asarray(fit["Amean"], dtype=float)
        else:
            covariate = None
    else:
        covariate = np.asarray(trend, dtype=float)
        if len(covariate) != len(sigma):
            raise ValueError(
                "If trend is numeric then it should have length equal to the number of genes"
            )
    sq = squeeze_var(
        sigma**2, dfres, covariate=covariate, span=span, robust=robust,
        winsor_tail_p=winsor_tail_p, legacy=legacy,
    )  # fmt: skip
    s2_post = sq["var_post"]
    t = coefficients / sdu / np.sqrt(s2_post)[:, None]
    df_total = dfres + sq["df_prior"]
    df_pooled = np.nansum(dfres)
    df_total = np.minimum(df_total, df_pooled)
    p_value = 2 * stats.t.cdf(-np.abs(t), df=df_total[:, None])
    out = dict(fit)
    out.update(
        t=t,
        p_value=p_value,
        df_total=df_total,
        s2_prior=sq["var_prior"],
        df_prior=sq["df_prior"],
        s2_post=s2_post,
    )
    return out
