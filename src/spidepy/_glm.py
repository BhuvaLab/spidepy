"""The per-index negative binomial fit: one GLM per gene through spanormpy's ``polish_nb``.

Ports ``R/fit-glm.R`` (``.glmStart``, ``.fitIndexGLM``, ``.fitStatus``, ``.workingWR``). Damped
Newton on each gene's own penalised likelihood with the patient intercepts absorbed by Schur
complement and the dispersion by profile maximum likelihood. Genes are independent, so
``n_jobs`` only schedules them; results do not depend on it.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed, effective_n_jobs
from scipy.stats import nbinom
from spanormpy import polish_nb

from ._design import _lin_pred


def _glm_start(Yk, des: dict, offset=None) -> np.ndarray:
    """A sane start per gene: each patient's log mean count (net of any offset) (``.glmStart``).

    ``Yk`` is genes x index cells (dense or sparse; per-patient sums are a product with a sparse
    indicator, so a sparse matrix is never densified). Returns genes x ``ncol(W)``.
    """
    codes = np.asarray(des["patient"], dtype=int)
    npat = des["npat"]
    ng = Yk.shape[0]
    npc = np.bincount(codes, minlength=npat).astype(float)
    ind = sp.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), npat)
    )
    lm = Yk @ ind  # genes x patients
    lm = (lm.toarray() if sp.issparse(lm) else np.asarray(lm)) / npc[None, :]
    a = np.zeros((ng, des["W"].shape[1]))
    ic = des["intercept_cols"]
    a[:, ic] = np.log(lm + 1e-3)
    if offset is not None:
        offset = np.asarray(offset, dtype=float)
        if offset.ndim == 2:
            mo = (offset @ ind.toarray()) / npc[None, :]
        else:
            mo = np.tile((ind.T @ offset) / npc, (ng, 1))
        a[:, ic] = a[:, ic] - mo
    return a


def _fit_index_glm(Yk, des: dict, offset=None, n_jobs: int = 1) -> dict:
    """Fit one NB GLM per gene of one index type (``.fitIndexGLM``).

    Returns ``alpha`` (genes x ``ncol(W)``), ``psi``, ``loglik``, ``polish`` (spanormpy's
    per-gene diagnostics) and ``status`` (``genes, not_fitted, psi_at_bound, runaway``). A gene
    that could not be fitted has NaN ``alpha`` and ``psi``.
    """
    a0 = _glm_start(Yk, des, offset)
    fit = polish_nb(
        Yk,
        des["W"],
        a0,
        np.ones(Yk.shape[0]),
        lambda_a=des["pen"],
        offset=offset,
        absorb=des["absorb"],
        start_cols=des["start"],
        psi_method="profile",
        backend="cpu",
        n_jobs=n_jobs,
    )
    return _fit_status(fit, Yk, des, offset, n_jobs)


def _fit_status(
    fit: dict, Yk, des: dict, offset, n_jobs: int = 1, psi_range=(1e-3, 1e3), max_coef: float = 20
) -> dict:
    """Drop what did not fit, settle bounded dispersions and count both (``.fitStatus``).

    A gene polish_nb could not fit (returned at its start with ``polished = False``, or
    non-finite) drops out (NaN). A gene whose dispersion optimum lies on the search bound takes
    the bound its likelihood prefers (``dnbinom(size = 1 / psi, mu)`` at both) and its mean is
    refitted at that dispersion. A fitted gene whose non-absorbed coefficients ran off
    (``|coef| > max_coef``) also drops out. Counts are kept in ``status``.
    """
    pol = fit["polish"]
    alpha = np.array(fit["alpha"], dtype=float)
    psi = np.array(fit["psi"], dtype=float)
    ng = Yk.shape[0]
    bad = ~pol["polished"].to_numpy(dtype=bool) if "polished" in pol else np.zeros(ng, dtype=bool)
    bad = bad | ~np.isfinite(psi) | ~np.isfinite(alpha).all(axis=1)
    dense = np.asarray(des["dense"], dtype=int)
    runaway = ~bad & (np.abs(alpha[:, dense]) > max_coef).any(axis=1)
    bad = bad | runaway
    alpha[bad] = np.nan
    psi[bad] = np.nan
    at_bound = pol["psi_bound"].to_numpy(dtype=bool) if "psi_bound" in pol else np.zeros(ng, bool)
    bnd = np.flatnonzero(~bad & at_bound)
    mat = offset is not None and np.ndim(offset) == 2
    for g in bnd:
        o = 0.0 if offset is None else (offset[g] if mat else offset)
        mu = np.exp(_lin_pred(des["W"], alpha[g]) + o)
        y = _row(Yk, g)
        ll = [nbinom.logpmf(y, 1.0 / p, 1.0 / (1.0 + p * mu)).sum() for p in psi_range]
        psi[g] = psi_range[int(np.argmax(ll))]
    if len(bnd):
        ob = offset[bnd] if mat else offset
        ref = polish_nb(
            Yk[bnd],
            des["W"],
            alpha[bnd],
            psi[bnd],
            lambda_a=des["pen"],
            offset=ob,
            absorb=des["absorb"],
            start_cols=des["start"],
            psi_method="fixed",
            backend="cpu",
            n_jobs=n_jobs,
        )
        ok = ref["polish"]["polished"].to_numpy(dtype=bool) & np.isfinite(ref["alpha"]).all(axis=1)
        alpha[bnd[ok]] = ref["alpha"][ok]
        alpha[bnd[~ok]] = np.nan
        psi[bnd[~ok]] = np.nan
        bad[bnd[~ok]] = True
        bnd = bnd[ok]
    status = {
        "genes": int(ng),
        "not_fitted": int(bad.sum()),
        "psi_at_bound": int(len(bnd)),
        "runaway": int(runaway.sum()),
    }
    return {**fit, "alpha": alpha, "psi": psi, "status": status}


def _row(Y, g: int) -> np.ndarray:
    r = Y[g]
    return np.asarray(r.toarray()).ravel() if sp.issparse(r) else np.asarray(r, dtype=float)


def _working_wr(y, W, alpha, psi, offset=0.0) -> dict:
    """Working weights ``w = mu / (1 + psi mu)`` and residuals ``r = (y - mu) / (1 + psi mu)``.

    Port of ``.workingWR``; ``mu = exp(W alpha + offset)``.
    """
    mu = np.exp(_lin_pred(W, alpha) + offset)
    d = 1.0 + psi * mu
    return {"w": mu / d, "r": (np.asarray(y, dtype=float) - mu) / d, "mu": mu}


def _bp_genes(G: int, fun, n_jobs: int = 1) -> list:
    """Run ``fun`` over blocks of gene indices and return the block results in order (``.bpGenes``).

    The ``G`` genes are cut into ``min(G, 4 * workers)`` contiguous blocks with R's
    ``cut(seq_len(G), nch)`` boundaries (equal-width bins, empty ones dropped), so each worker gets
    a few blocks rather than one task per gene. ``fun`` receives the 0-based indices of a block.
    With ``n_jobs > 1`` blocks run on threads, and BLAS is held to one thread for the whole run
    (R pins one per worker; the limit is process-global, so it wraps the pool, not each block).
    The result never depends on ``n_jobs``.
    """
    nw = max(1, effective_n_jobs(n_jobs))
    chunks = _cut_blocks(G, min(G, nw * 4))
    if nw == 1:
        return [fun(c) for c in chunks]
    with _one_blas_thread():
        return Parallel(n_jobs=nw, backend="threading")(delayed(fun)(c) for c in chunks)


def _cut_blocks(G: int, nch: int) -> list[np.ndarray]:
    """0-based blocks of ``range(G)`` as ``split(seq_len(G), cut(seq_len(G), nch, labels = FALSE))``.

    ``cut`` takes ``nch`` equal-width intervals over ``[1, G]`` (outer breaks widened by 0.1% of
    the range, right-closed); a single block is built directly, as R does.
    """
    if nch <= 1 or G <= 1:
        return [np.arange(G)]
    x = np.arange(1, G + 1, dtype=float)
    br = np.linspace(1.0, float(G), nch + 1)
    dx = br[-1] - br[0]
    br[0] -= dx / 1000
    br[-1] += dx / 1000
    code = np.searchsorted(br, x, side="left")  # bin i holds (br[i-1], br[i]]
    return [np.flatnonzero(code == k) for k in np.unique(code)]


@contextmanager
def _one_blas_thread():
    """Limit BLAS to one thread inside the block when ``threadpoolctl`` is available."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1):
        yield
