"""Edge-corrected Gaussian kernel density, a port of ``spatstat.explore::densityfun``.

Reading of spatstat.explore 3.5-3 (``densityfun.ppp`` -> ``densitycrossEngine``) for the call
made by spiDE (Gaussian kernel, scalar ``sigma``, rectangular window):

* The edge correction is **exact**, not pixel based. For a rectangular window the engine takes
  the closed-form Gaussian mass inside the window, ``edgeweight = (pnorm(xmax, x, sigma) -
  pnorm(xmin, x, sigma)) * (pnorm(ymax, y, sigma) - pnorm(ymin, y, sigma))`` (the branch
  ``isgauss && is.null(varcov) && win$type == "rectangle"``). ``second.moment.calc`` (the
  pixel/FFT route) is only used for non-rectangular windows or anisotropic kernels, so no grid,
  ``dimyx``, ``eps`` or interpolation is involved.
* ``edge = TRUE, diggle = TRUE``: each *data* point gets weight ``1 / edgeweight(data point)``
  and the density is ``sum_i w_i K(q - x_i)``.
* ``edge = TRUE, diggle = FALSE`` (uniform correction): the plain sum is divided by the
  edge weight at each *query* point (the query must lie in the window).
* ``edge = FALSE``: the plain sum ``sum_i K(q - x_i)``.
* ``K`` is the isotropic Gaussian ``exp(-d^2 / (2 sigma^2)) / (2 pi sigma^2)`` and the C
  routine drops data points further than ``8 * sigma`` from the query (``cutoff2Dkernel``
  default for the Gaussian); the same cut-off is applied here so the two agree to rounding.
* A query that coincides with a data point includes that point (``K(0)``), as in R.
* ``densityfun`` returns only the queries that lie inside the window (``drop = TRUE``); here
  they come back as ``nan`` so the output keeps the query order.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr

_CUTOFF_SD = 8.0  # spatstat's default Gaussian cut-off, in units of sigma
_BLOCK = 2000  # queries handled per block, bounding the number of stored pairs


def _edge_weight(x, y, sigma, window) -> np.ndarray:
    """Gaussian mass inside the rectangle for kernels centred at ``(x, y)``.

    The ``pnorm`` product of ``densitycrossEngine``.
    """
    xmin, xmax, ymin, ymax = window
    xprob = ndtr((xmax - x) / sigma) - ndtr((xmin - x) / sigma)
    yprob = ndtr((ymax - y) / sigma) - ndtr((ymin - y) / sigma)
    return xprob * yprob


def density_at(
    points_xy,
    query_xy,
    sigma: float,
    window,
    edge: bool = True,
    diggle: bool = True,
) -> np.ndarray:
    """Edge-corrected Gaussian density of ``points_xy`` at ``query_xy``.

    Reproduces ``spatstat.explore::densityfun(ppp, sigma, edge, diggle)(qx, qy)`` for a ppp on
    a rectangular window (see the module docstring for the algorithm).

    Parameters
    ----------
    points_xy : array-like, shape (n, 2)
        Data points.
    query_xy : array-like, shape (m, 2)
        Locations to evaluate; those outside ``window`` give ``nan``.
    sigma : float
        Kernel standard deviation.
    window : tuple
        ``(xmin, xmax, ymin, ymax)`` of the observation window.
    edge, diggle : bool
        As in ``densityfun``.

    Returns
    -------
    numpy.ndarray, shape (m,)
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    qry = np.asarray(query_xy, dtype=float).reshape(-1, 2)
    xmin, xmax, ymin, ymax = (float(v) for v in window)
    win = (xmin, xmax, ymin, ymax)
    inside = (qry[:, 0] >= xmin) & (qry[:, 0] <= xmax) & (qry[:, 1] >= ymin) & (qry[:, 1] <= ymax)
    out = np.full(len(qry), np.nan)
    q = qry[inside]
    res = np.zeros(len(q))
    if len(pts) > 0 and len(q) > 0:
        # Diggle weights live on the data points; the uniform correction divides at queries
        w = 1.0 / _edge_weight(pts[:, 0], pts[:, 1], sigma, win) if edge and diggle else None
        rmax = _CUTOFF_SD * sigma
        tree = cKDTree(pts)
        norm = 1.0 / (2.0 * np.pi * sigma**2)
        for s in range(0, len(q), _BLOCK):
            qb = q[s : s + _BLOCK]
            pairs = tree.sparse_distance_matrix(cKDTree(qb), rmax, output_type="ndarray")
            k = norm * np.exp(-0.5 * (pairs["v"] / sigma) ** 2)
            if w is not None:
                k = k * w[pairs["i"]]
            res[s : s + len(qb)] = np.bincount(pairs["j"], weights=k, minlength=len(qb))
        if edge and not diggle:
            res = res / _edge_weight(q[:, 0], q[:, 1], sigma, win)
    out[inside] = res
    return out
