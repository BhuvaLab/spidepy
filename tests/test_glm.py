"""The per-index NB fit against R (``tests/ref/make_design.R``): start values, ``.fitIndexGLM``
alpha / psi / status (including genes engineered to hit the dispersion bound and the runaway
guard), working quantities, and independence from ``n_jobs``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import gammaln

from _design_ref import CONFIGS, a1, r_design, ref
from spidepy._glm import _fit_index_glm, _fit_status, _glm_start, _working_wr

STATUS_KEYS = ["genes", "not_fitted", "psi_at_bound", "runaway"]


@pytest.fixture(scope="module")
def R():
    return ref()


def _offset(R, name):
    return np.asarray(R["fit_inputs"]["Ok"]) if name in ("c5", "c6") else None


@pytest.fixture(scope="module")
def fits(R):
    """Fit every config once (the slowest step)."""
    Yk = np.asarray(R["fit_inputs"]["Yk"])
    return {n: _fit_index_glm(Yk, r_design(R[n]), offset=_offset(R, n)) for n in CONFIGS}


@pytest.mark.parametrize("name", CONFIGS)
def test_glm_start_matches_r(R, name):
    Yk = np.asarray(R["fit_inputs"]["Yk"])
    des = r_design(R[name])
    off = _offset(R, name)
    a0 = _glm_start(Yk, des, off)
    np.testing.assert_allclose(a0, R[name]["a0"], rtol=1e-10, atol=1e-12)
    # a sparse count matrix gives the same start without being densified
    np.testing.assert_allclose(_glm_start(sp.csr_matrix(Yk), des, off), a0, rtol=1e-12, atol=1e-14)


def test_glm_start_per_cell_offset_vector(R):
    des = r_design(R["c5"])
    Yk = np.asarray(R["fit_inputs"]["Yk"])
    ov = np.asarray(R["fit_inputs"]["Ok"])[0]
    a = _glm_start(Yk, des, ov)
    ic = des["intercept_cols"]
    mo = np.array([ov[des["patient"] == s].mean() for s in range(des["npat"])])
    base = _glm_start(Yk, des)
    np.testing.assert_allclose(a[:, ic], base[:, ic] - mo[None, :], rtol=1e-12)


@pytest.mark.parametrize("name", CONFIGS)
def test_fit_index_glm_matches_r(R, fits, name):
    f, cfg = fits[name], R[name]
    assert f["status"] == dict(zip(STATUS_KEYS, [int(v) for v in a1(cfg["status"])]))
    alpha, psi = np.asarray(cfg["alpha"]), np.asarray(cfg["psi"])
    # the NA pattern is exact; rtol 1e-6 downstream of the iterative fit (the Newton / profile
    # dispersion stopping rules end at slightly different iterates, as in the polish_nb tests)
    np.testing.assert_array_equal(np.isnan(f["alpha"]), np.isnan(alpha))
    np.testing.assert_array_equal(np.isnan(f["psi"]), np.isnan(psi))
    np.testing.assert_allclose(f["psi"], psi, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(f["alpha"], alpha, rtol=1e-6, atol=1e-8, equal_nan=True)
    np.testing.assert_array_equal(
        f["polish"]["polished"].to_numpy(), a1(cfg["polished"]).astype(bool)
    )
    np.testing.assert_array_equal(
        f["polish"]["psi_bound"].to_numpy(), a1(cfg["psi_bound"]).astype(bool)
    )


def test_engineered_genes_reach_every_status_branch(R, fits):
    """The fixture holds genes at the dispersion bound, runaway genes and unfitted genes."""
    genes = list(a1(R["fit_inputs"]["genes"]))
    f = fits["c1"]
    s = f["status"]
    assert s["psi_at_bound"] >= 1 and s["runaway"] >= 1 and s["not_fitted"] >= s["runaway"]
    bound = [
        g
        for g, b in zip(genes, f["polish"]["psi_bound"])
        if b and np.isfinite(f["psi"][genes.index(g)])
    ]
    assert bound, "no gene was settled at the dispersion bound"
    for g in bound:  # a bound gene takes exactly one of the two bounds
        assert f["psi"][genes.index(g)] in (1e-3, 1e3)
    gone = [g for g, p in zip(genes, f["psi"]) if np.isnan(p)]
    assert "allzero" in gone  # no counts: runs off
    assert np.isnan(f["alpha"][genes.index("allzero")]).all()
    # a gene polish_nb flagged at the bound but that then drops out is not counted at the bound
    assert s["psi_at_bound"] == int(
        (f["polish"]["psi_bound"].to_numpy() & np.isfinite(f["psi"])).sum()
    )


def test_status_without_bound_or_runaway(R):
    Yk = np.asarray(R["fit_inputs"]["Yk"])[:4]
    # the four fixture genes under the plain design fit cleanly in R (fit_ok)
    import pandas as pd

    from spidepy._design import _index_design

    fi = R["fit_inputs"]
    des = r_design(R["c5"])
    L = pd.DataFrame(fi["L"], columns=["C", "B"])
    d = _index_design(L, None, des["patient"], des["patients"], tested=["C"])
    f = _fit_index_glm(Yk, d)
    ok = R["fit_ok"]
    assert f["status"] == dict(zip(STATUS_KEYS, [int(v) for v in a1(ok["status"])]))
    np.testing.assert_allclose(f["psi"], ok["psi"], rtol=1e-6)
    np.testing.assert_allclose(f["alpha"], ok["alpha"], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ["c1", "c4"])
def test_n_jobs_does_not_change_results(R, fits, name):
    Yk = np.asarray(R["fit_inputs"]["Yk"])
    g = _fit_index_glm(Yk, r_design(R[name]), offset=_offset(R, name), n_jobs=2)
    f = fits[name]
    # n_jobs only changes how genes are blocked (polish_nb sizes blocks by worker), so batched
    # BLAS sums are re-associated: equal to rounding (~1e-13), with identical NA pattern / status
    np.testing.assert_array_equal(np.isnan(g["alpha"]), np.isnan(f["alpha"]))
    np.testing.assert_allclose(g["alpha"], f["alpha"], rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(g["psi"], f["psi"], rtol=1e-10, equal_nan=True)
    assert g["status"] == f["status"]


def test_sparse_counts_give_the_same_fit(R, fits):
    Yk = np.asarray(R["fit_inputs"]["Yk"])
    g = _fit_index_glm(sp.csc_matrix(Yk), r_design(R["c1"]))
    np.testing.assert_allclose(
        g["alpha"], fits["c1"]["alpha"], rtol=1e-9, atol=1e-10, equal_nan=True
    )
    assert g["status"] == fits["c1"]["status"]


@pytest.mark.parametrize("name", CONFIGS)
def test_working_wr_matches_r(R, name):
    cfg, Yk = R[name], np.asarray(R["fit_inputs"]["Yk"])
    des = r_design(cfg)
    alpha, psi = np.asarray(cfg["alpha"]), np.asarray(cfg["psi"])
    for g in (1, 3):
        o = np.asarray(R["fit_inputs"]["Ok"])[g - 1] if name in ("c5", "c6") else 0.0
        got = _working_wr(Yk[g - 1], des["W"], alpha[g - 1], psi[g - 1], o)
        for k in ("w", "r", "mu"):
            np.testing.assert_allclose(got[k], cfg[f"wr{g}"][k], rtol=1e-10, atol=1e-12)


def test_status_bound_choice_uses_the_nb_loglik(R):
    """Force a gene onto the bound branch: it takes the bound with the larger dnbinom loglik."""
    Yk = np.asarray(R["fit_inputs"]["Yk"])
    des = r_design(R["c1"])
    base = _fit_index_glm(Yk, des)
    genes = list(a1(R["fit_inputs"]["genes"]))
    from spidepy._design import _lin_pred

    for gname in ("G2", "over2", "pois"):
        g = genes.index(gname)
        fit = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
        pol = fit["polish"].copy()
        pol["psi_bound"] = False
        pol.loc[pol.index[g], "psi_bound"] = True
        fit["polish"] = pol
        out = _fit_status(fit, Yk, des, None)
        assert out["status"]["psi_at_bound"] >= 1
        assert out["psi"][g] in (1e-3, 1e3)
        # the chosen bound is the likelihood-preferred one at the pre-refit mean
        mu0 = np.exp(_lin_pred(des["W"], base["alpha"][g]))
        l0 = [
            (
                gammaln(Yk[g] + 1 / p)
                - gammaln(1 / p)
                - gammaln(Yk[g] + 1)
                + (1 / p) * np.log((1 / p) / ((1 / p) + mu0))
                + Yk[g] * np.log(mu0 / ((1 / p) + mu0))
            ).sum()
            for p in (1e-3, 1e3)
        ]
        assert out["psi"][g] == (1e-3, 1e3)[int(np.argmax(l0))]
