"""The sandwich engine's inference: patient-clustered CR2 with Bell-McCaffrey df (`R/sandwich.R`).

A patient-clustered CR2 sandwich (Bell & McCaffrey 2002; Pustejovsky & Tipton 2018) on the
fixed-effects NB fit of one index type. In the working-weights space of the converged IRLS,
`w = mu / (1 + psi mu)` and `r = (y - mu) / (1 + psi mu)`; `X~` is the non-absorbed
columns residualised on each patient's absorbed block, `B = (X~' W X~)^-1`,
`G_s = sum_{i in s} w_i x~_i x~_i'` and `h_s = sum_{i in s} x~_i r_i`. With
`G_s = U S U'` (rank `r_s`), `R_s = S^1/2 U'` factors the patient's weighted block, the
nonzero eigenvalues of its hat block are those of `R_s B R_s' = V L V'`, and the CR2 score is
`u_s = R_s' V D V' S^-1/2 U' h_s` with `D = diag((1 - L)^-1/2)`, so
`V_CR2 = B (sum_s u_s u_s') B` and no `n_s x n_s` matrix is ever formed. For coefficient
`j` the Bell-McCaffrey df is `tr(Gm)^2 / ||Gm||_F^2` with
`Gm = diag(||D V' R_s B e_j||^2) - Q' B Q`, `Q[, s] = R_s' V D V' R_s B e_j`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist

from ._design import _dense_x, _partial_block, _patient_blocks
from ._glm import _bp_genes, _row, _working_wr

_EPS = np.finfo(float).eps
_COLUMNS = ["gene", "niche", "estimate", "se", "df", "se_model", "psi"]


def _solve_info(info: np.ndarray):
    """Inverse of the information matrix, or `None` where R `solve` fails or gives non-finite.

    R's `solve` errors on an exactly singular matrix or one with reciprocal condition number
    below machine epsilon; both (and a non-finite inverse) make the gene drop out.
    """
    if not np.all(np.isfinite(info)):
        return None
    try:
        B = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(B)):
        return None
    rcond = 1.0 / (np.linalg.norm(info, 1) * np.linalg.norm(B, 1))
    return B if np.isfinite(rcond) and rcond >= _EPS else None


def _sandwich_cr2(
    fit: dict, des: dict, Yk, offset=None, n_jobs: int = 1, genes=None
) -> pd.DataFrame:
    """Patient-clustered CR2 standard errors and Bell-McCaffrey df for every gene (`.sandwichCR2`).

    Parameters
    ----------
    fit : dict
        Shared fit with `alpha` (genes x design columns) and `psi` (genes); a gene with a
        non-finite entry is left out.
    des : dict
        The per-index design from `_index_design` (pooled or condition model).
    Yk : array or sparse matrix
        Counts of the index cells, genes x cells; row names are the caller's gene labels.
    offset : array, optional
        Per-gene, per-cell log offset, genes x cells.
    n_jobs : int
        Gene blocks run in parallel; the result does not depend on it.
    genes : sequence of str, optional
        Gene labels (R `rownames(Yk)`); default the row labels of a DataFrame `Yk`, else the
        0-based row numbers as strings.

    Returns
    -------
    pandas.DataFrame
        One row per fitted gene and tested niche, columns `gene, niche, estimate, se, df,
        se_model, psi, t, p`, genes in the order of `Yk`. A gene whose weights,
        residuals or information matrix are non-finite or singular is omitted.
    """
    genes = _gene_labels(genes, Yk)
    alpha = np.asarray(fit["alpha"], dtype=float)
    psi = np.asarray(fit["psi"], dtype=float)
    Yv = Yk.to_numpy() if isinstance(Yk, pd.DataFrame) else Yk
    Xd = _dense_x(des)
    pos = {c: i for i, c in enumerate(des["dense"])}
    jt = np.array([pos[c] for c in des["tested"]], dtype=int)  # tested columns within the dense X
    tested = np.asarray(des["tested"], dtype=int)
    pid = np.asarray(des["patient"], dtype=int)
    S = des["npat"]
    rows_of = [np.flatnonzero(pid == s) for s in range(S)]
    Z_of = _patient_blocks(des, rows_of)
    intercept_only = des["Zs"] is None
    d = Xd.shape[1]

    def one(g: int):
        al, ps = alpha[g], psi[g]
        if not (np.all(np.isfinite(al)) and np.isfinite(ps)):
            return None
        wr = _working_wr(_row(Yv, g), des["W"], al, ps, offset=0.0 if offset is None else offset[g])
        w, r = wr["w"], wr["r"]
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(r))):  # an overflowing mean
            return None
        if intercept_only:
            sw = np.bincount(pid, weights=w, minlength=S)
            xbar = np.zeros((S, d))
            np.add.at(xbar, pid, Xd * w[:, None])
            Xt = Xd - (xbar / sw[:, None])[pid]
        else:
            # residualise on each patient's absorbed block (Frisch-Waugh-Lovell); the blocks nest
            # within patients, so CR2 on the residualised design equals CR2 on the full one
            Xt = Xd.copy()
            for s in range(S):
                i = rows_of[s]
                if not len(i):
                    continue
                xs = _partial_block(Xd[i], w[i], Z_of[s])
                if xs is None:
                    return None
                Xt[i] = xs
        sq = np.sqrt(w)
        B = _solve_info((Xt * sq[:, None]).T @ (Xt * sq[:, None]))
        if B is None:  # singular information for this gene: it drops out
            return None
        Us = np.zeros((d, S))
        fac = [None] * S
        for s in range(S):
            i = rows_of[s]
            if not len(i):
                continue
            Xs = Xt[i]
            Gs = (Xs * sq[i][:, None]).T @ (Xs * sq[i][:, None])
            hs = (Xs * r[i][:, None]).sum(axis=0)
            ev, vec = np.linalg.eigh(Gs)
            keep = ev > ev.max() * 1e-10
            if not keep.any():
                continue
            Uu, sv = vec[:, keep], ev[keep]
            Rs = np.sqrt(sv)[:, None] * Uu.T
            Ms = Rs @ B @ Rs.T
            lam, Vv = np.linalg.eigh((Ms + Ms.T) / 2)
            Dg = 1.0 / np.sqrt(1.0 - np.minimum(lam, 1 - 1e-8))
            qe = (Uu.T @ hs) / np.sqrt(sv)
            Us[:, s] = Rs.T @ (Vv @ (Dg * (Vv.T @ qe)))
            fac[s] = (Rs, Vv, Dg)
        Vcr2 = B @ (Us @ Us.T) @ B
        df = np.empty(len(jt))
        for a, j in enumerate(jt):
            Bc = B[:, j]
            pn = np.zeros(S)
            Q = np.zeros((d, S))
            for s in range(S):
                if fac[s] is None:
                    continue
                Rs, Vv, Dg = fac[s]
                z = Vv.T @ (Rs @ Bc)
                pn[s] = np.sum((Dg * z) ** 2)
                Q[:, s] = Rs.T @ (Vv @ (Dg * z))
            Gm = np.diag(pn) - Q.T @ B @ Q
            df[a] = np.trace(Gm) ** 2 / np.sum(Gm**2)
        return {
            "gene": genes[g],
            "estimate": al[tested],
            "se": np.sqrt(np.maximum(np.diag(Vcr2)[jt], 0.0)),
            "df": df,
            "se_model": np.sqrt(np.diag(B)[jt]),
            "psi": ps,
        }

    def block(idx):
        return [one(int(g)) for g in idx]

    res = [x for b in _bp_genes(len(genes), block, n_jobs) for x in b if x is not None]
    nt = len(des["tested_niche"])
    if not res:
        out = pd.DataFrame(
            {c: pd.Series(dtype=object if c in ("gene", "niche") else float) for c in _COLUMNS}
        )
    else:
        out = pd.DataFrame(
            {
                "gene": np.repeat([x["gene"] for x in res], nt),
                "niche": list(des["tested_niche"]) * len(res),
                "estimate": np.concatenate([x["estimate"] for x in res]),
                "se": np.concatenate([x["se"] for x in res]),
                "df": np.concatenate([x["df"] for x in res]),
                "se_model": np.concatenate([x["se_model"] for x in res]),
                "psi": np.repeat([x["psi"] for x in res], nt),
            }
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["t"] = out["estimate"] / out["se"]
        out["p"] = 2 * _t_dist.cdf(-np.abs(out["t"].to_numpy(dtype=float)), out["df"].to_numpy())
    return out


def _gene_labels(genes, Yk) -> list[str]:
    """Gene labels (R `rownames(Yk)`): as given, else a DataFrame's index, else row numbers."""
    if genes is not None:
        return [str(x) for x in genes]
    if isinstance(Yk, pd.DataFrame):
        return [str(x) for x in Yk.index]
    return [str(i) for i in range(Yk.shape[0])]
