"""The slice of limma that spiDE's pooled and niche-abundance tests use (limma 3.66.0).

``lm_fit`` is ``lmFit`` (least squares, per-gene weights, NA observations; ``lm.series``),
``fit_f_dist`` is ``fitFDist`` (with the natural-spline trend), ``squeeze_var`` is
``squeezeVar`` (which picks ``fitFDist`` when every gene has the same residual df and
``fitFDistUnequalDF1`` with a weighted lowess trend otherwise) and ``e_bayes`` is
``eBayes`` for the quantities spiDE reads: ``t``, ``p.value``, ``df.total``, ``s2.prior``,
``df.prior`` and ``s2.post``. ``var.prior`` and ``lods`` (``tmixture.matrix``) are not read
anywhere in spiDE and are not computed.

Matrices are genes x observations, as in R. R ``NA`` is ``nan``. The robust branch
(``fitFDistRobustly`` and the robust half of ``fitFDistUnequalDF1``) is the next task: the
``robust`` arguments exist and raise ``NotImplementedError``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import special, stats
from spanormpy._polish_engine import brent_fmin
from spanormpy._tps import _design, _ns_knots

__all__ = ["lm_fit", "fit_f_dist", "squeeze_var", "e_bayes"]

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
def _weighted_lowess(x, y, weights, span: float, npts: int = 200) -> np.ndarray:
    """Single-pass weighted lowess (limma ``weightedLowess(..., iterations = 1)``).

    The window of each fitted point is the smallest set of nearest points whose weights add up
    to ``span`` times the total weight; the local weights are tricube in the distance over the
    window radius times the prior weights, and the fit is a local line. Points are fitted only
    every ``delta`` along ``x`` (``delta`` chosen from ``npts`` as limma does; ``npts >= n``
    gives 0) and the others interpolated linearly, the next fitted point being the first one
    beyond ``x[last] + delta``. Reproduced from limma's C routine by matching R's output (the
    source is not shipped with the installed package); verified against R on continuous
    ``x``. Behaviour on heavily tied ``x`` is not verified.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(x)
    o = np.argsort(x, kind="stable")
    xs, ysrt, ws = x[o], y[o], w[o]
    npts = int(npts + 0.5)
    if npts >= n:
        delta = 0.0
    else:
        cum = np.cumsum(np.sort(np.diff(xs)))
        nc = np.arange(npts)
        delta = float(np.min(cum[len(cum) - 1 - nc] / (npts - nc)))
    thr = span * ws.sum()
    xrange = xs[-1] - xs[0]

    def fit_at(i: int) -> float:
        d = np.abs(xs - xs[i])
        order = np.argsort(d, kind="stable")
        j = min(int(np.searchsorted(np.cumsum(ws[order]), thr, side="left")), n - 1)
        h = d[order[j]]
        t = np.where(d < h, (1 - (d / h) ** 3) ** 3, 0.0) if h > 0 else (d == 0).astype(float)
        wt = t * ws
        a = wt.sum()
        if not a > 0:
            return float(ysrt[i])
        wt = wt / a
        if h > 0:
            a = (wt * xs).sum()
            c = (wt * (xs - a) ** 2).sum()
            if np.sqrt(c) > 0.001 * xrange:
                wt = wt * ((xs[i] - a) / c * (xs - a) + 1)
        return float((wt * ysrt).sum())

    fitted = np.empty(n)
    last, i = -1, 0
    while True:
        fitted[i] = fit_at(i)
        if last < i - 1:
            al = (xs[last + 1 : i] - xs[last]) / (xs[i] - xs[last])
            fitted[last + 1 : i] = al * fitted[i] + (1 - al) * fitted[last]
        last = i
        cut = xs[last] + delta
        i = last + 1
        while i < n and xs[i] <= cut:
            if xs[i] == xs[last]:
                fitted[i] = fitted[last]
                last = i
            i += 1
        if last >= n - 1:
            break
        i = min(max(last + 1, i), n - 1)
    out = np.empty(n)
    out[o] = fitted
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
    fitted[obs] = _weighted_lowess(xobs, yobs, wobs, span, npts=200)
    return fitted


def _choose_lowess_span(n: int, small_n: float = 50, min_span: float = 0.3, power=1 / 3) -> float:
    """limma ``chooseLowessSpan``."""
    return min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0)


def _fit_f_dist_unequal_df1(x, df1, covariate=None, span=None, robust=False):
    """limma ``fitFDistUnequalDF1``: maximum-likelihood ``df2`` for genes with different df.

    The prior scale (log) is the weighted mean of ``log(x) + log(1/psi(df/2)...)`` (weights
    ``1/trigamma(df1/2)``) or, with a covariate, its weighted lowess trend; ``df2`` maximises the
    F likelihood by Brent's method on ``par = df2/2 / (1 + df2/2)`` in (0.5, 0.9998).
    Returns ``(scale, df2)`` (``nan`` if fewer than two informative genes).
    """
    if robust:
        raise NotImplementedError("robust eBayes is not implemented yet")
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
    if np.isnan(x).any():
        i = np.isnan(x)
        prior_w = (~i).astype(float)
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
        return np.nan, np.nan
    if n_inf == 2:
        covariate, prior_w = None, None
    m = float(np.median(x[informative]))
    xpos = np.maximum(x, 1e-12 * m)
    d1 = df1 / 2
    e = np.log(xpos) + _logmdigamma(d1)
    w = 1 / special.polygamma(1, d1)
    if len(w) < n:
        w = np.full(n, w[0])
    if prior_w is not None:
        w = w * prior_w
    if covariate is None:
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
    return (float(s20) if np.ndim(s20) == 0 else s20), 2 * float(d2)


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
            fit_scale, df_prior = _fit_f_dist_robustly(var, df, covariate, winsor_tail_p)
        else:
            fit_scale, df_prior = fit_f_dist(var, df, covariate)
    else:
        fit_scale, df_prior = _fit_f_dist_unequal_df1(
            var, df, covariate=covariate, span=span, robust=robust
        )
    if np.isnan(df_prior).any():
        raise ValueError("Could not estimate prior df")
    return {
        "df_prior": df_prior,
        "var_prior": fit_scale,
        "var_post": _squeeze_var_post(var, df, fit_scale, df_prior),
    }


def _fit_f_dist_robustly(x, df1, covariate=None, winsor_tail_p=(0.05, 0.1)):
    """limma ``fitFDistRobustly`` (the robust branch of ``squeeze_var``; a later task)."""
    raise NotImplementedError("robust eBayes is not implemented yet")


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
        Not implemented yet (raises ``NotImplementedError``).

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
