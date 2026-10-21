"""The sandwich engine's CR2 + Bell-McCaffrey df against R (``tests/ref/make_sandwich.R``).

R's own shared fits (``design.h5``) feed ``_sandwich_cr2`` for the closed-form checks at
``rtol=1e-10``; the clubSandwich values written by R (``vcovCR(type = "CR2", inverse_var = TRUE)``
and ``coef_test(test = "Satterthwaite")`` on the working weighted lm) port the assertions of
R's ``test-sandwich.R`` and ``test-depth.R``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from _design_ref import r_design, strs
from _ref import load_ref
from spidepy._design import _depth_blocks, _index_design
from spidepy._glm import _bp_genes, _fit_index_glm
from spidepy._sandwich import _sandwich_cr2, _solve_info

CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
COLS = ["gene", "niche", "estimate", "se", "df", "se_model", "psi", "t", "p"]


@pytest.fixture(scope="module")
def S():
    return load_ref("sandwich")


@pytest.fixture(scope="module")
def D():
    return load_ref("design")


def _args(D, cfg):
    fit = {"alpha": np.asarray(D[cfg]["alpha"]), "psi": np.asarray(D[cfg]["psi"])}
    off = np.asarray(D["fit_inputs"]["Ok"]) if cfg in ("c5", "c6") else None
    return fit, r_design(D[cfg]), np.asarray(D["fit_inputs"]["Yk"]), off


def _genes(D):
    return strs(D["fit_inputs"]["genes"])


def _assert_table(got, ref, rtol=1e-10, atol=1e-12):
    assert list(got.columns) == COLS
    assert len(got) == int(ref["n"])
    assert got["gene"].tolist() == strs(ref["gene"])
    assert got["niche"].tolist() == strs(ref["niche"])
    for c in COLS[2:8]:
        np.testing.assert_allclose(got[c], np.atleast_1d(ref[c]), rtol=rtol, atol=atol, err_msg=c)
    # p-values below 1e-300 compare as equal-or-both-tiny
    p, rp = got["p"].to_numpy(), np.atleast_1d(ref["p"])
    tiny = (p < 1e-300) & (rp < 1e-300)
    np.testing.assert_allclose(p[~tiny], rp[~tiny], rtol=rtol, atol=1e-300)


@pytest.mark.parametrize("cfg", CFGS)
def test_cr2_matches_r_given_r_fit(S, D, cfg):
    fit, des, Yk, off = _args(D, cfg)
    got = _sandwich_cr2(fit, des, Yk, offset=off, genes=_genes(D))
    _assert_table(got, S[cfg])
    # genes whose fit failed (NaN alpha / psi) are absent, as in R
    ok = np.isfinite(fit["psi"]) & np.isfinite(fit["alpha"]).all(axis=1)
    assert set(got["gene"]) == {g for g, k in zip(_genes(D), ok) if k}


def test_row_order_is_gene_major_tested_niche_minor(S, D):
    fit, des, Yk, off = _args(D, "c6")
    got = _sandwich_cr2(fit, des, Yk, offset=off, genes=_genes(D))
    assert len(des["tested_niche"]) == 2
    assert got["niche"].tolist() == des["tested_niche"] * (len(got) // 2)
    assert got["gene"].tolist()[::2] == got["gene"].tolist()[1::2]


@pytest.mark.parametrize("cfg", ["c1", "c4"])
def test_n_jobs_and_sparse_do_not_change_results(S, D, cfg):
    fit, des, Yk, off = _args(D, cfg)
    base = _sandwich_cr2(fit, des, Yk, offset=off, genes=_genes(D))
    pd.testing.assert_frame_equal(
        _sandwich_cr2(fit, des, Yk, offset=off, genes=_genes(D), n_jobs=3), base, check_exact=True
    )
    pd.testing.assert_frame_equal(
        _sandwich_cr2(fit, des, sp.csr_matrix(Yk), offset=off, genes=_genes(D)), base
    )
    # gene labels may come from a DataFrame index
    df = pd.DataFrame(Yk, index=_genes(D))
    pd.testing.assert_frame_equal(_sandwich_cr2(fit, des, df, offset=off), base)


def test_default_labels_are_row_numbers(S, D):
    fit, des, Yk, _ = _args(D, "c1")
    got = _sandwich_cr2(fit, des, Yk)
    assert got["gene"].iloc[0] == str(np.flatnonzero(np.isfinite(fit["psi"]))[0])


def test_all_genes_unfitted_gives_typed_empty_table(S, D):
    fit, des, Yk, _ = _args(D, "c1")
    bad = {"alpha": np.full_like(fit["alpha"], np.nan), "psi": np.full_like(fit["psi"], np.nan)}
    got = _sandwich_cr2(bad, des, Yk)
    assert int(S["empty"]["n"]) == 0 and len(got) == 0
    assert list(got.columns) == COLS
    assert got[COLS[2:]].dtypes.eq(float).all()


def test_singular_information_drops_every_gene(D):
    """A duplicated design column makes the information singular: R's solve() fails, genes drop."""
    fit, des, Yk, _ = _args(D, "c1")
    des = {**des, "W": des["W"].copy()}
    dn = des["dense"]
    des["W"][:, dn[-1]] = des["W"][:, dn[-2]]
    assert len(_sandwich_cr2(fit, des, Yk)) == 0


def test_solve_info_mimics_r_solve():
    assert _solve_info(np.eye(3)) is not None
    assert _solve_info(np.ones((3, 3))) is None  # exactly singular
    assert _solve_info(np.diag([1.0, 1.0, 1e-17])) is None  # rcond below machine epsilon
    assert _solve_info(np.diag([1.0, np.nan])) is None


def test_bp_genes_blocks_and_order():
    def f(idx):
        return [int(i) for i in idx]

    out = _bp_genes(10, f, n_jobs=1)  # min(G, 4 * 1) blocks, R's cut() boundaries
    assert [len(b) for b in out] == [3, 2, 2, 3]
    assert [i for b in out for i in b] == list(range(10))
    out = _bp_genes(100, f, n_jobs=2)
    assert len(out) == 8  # a few blocks per worker, not a task per gene
    assert [i for b in out for i in b] == list(range(100))
    assert _bp_genes(1, f, n_jobs=4) == [[0]]  # a one-gene index type: a single block
    assert [len(b) for b in _bp_genes(3, f, n_jobs=4)] == [1, 1, 1]  # never more blocks than genes
    assert _bp_genes(0, f) == [[]]


def test_bp_genes_leaves_blas_threads_alone():
    from threadpoolctl import threadpool_info

    before = [d["num_threads"] for d in threadpool_info()]
    _bp_genes(40, lambda idx: list(idx), n_jobs=4)
    assert [d["num_threads"] for d in threadpool_info()] == before


@pytest.mark.parametrize("cfg", ["c1", "c3"])
def test_cr2_end_to_end_on_python_fit(S, D, cfg):
    """Refit in Python, then CR2: iterative-fit tolerance (rtol 1e-6)."""
    _, des, Yk, _ = _args(D, cfg)
    fit = _fit_index_glm(Yk, des)
    got = _sandwich_cr2(fit, des, Yk, genes=_genes(D))
    ref = S[cfg]
    assert got["gene"].tolist() == strs(ref["gene"])
    # rtol 1e-6: alpha / psi are polish_nb outputs; the sandwich amplifies them slightly
    for c in ("estimate", "se", "df", "se_model", "psi"):
        np.testing.assert_allclose(got[c], ref[c], rtol=1e-6, atol=1e-8, err_msg=c)


# ---- clubSandwich oracles (test-sandwich.R, test-depth.R) ------------------------------


def _r_equal(current, target, tol):
    """testthat ``expect_equal`` tolerance: mean absolute difference over mean absolute target."""
    cur, tgt = np.atleast_1d(current).astype(float), np.atleast_1d(target).astype(float)
    return np.mean(np.abs(cur - tgt)) / np.mean(np.abs(tgt)) < tol


def _cs16_design(S, key):
    i = S["cs16"]["inputs"]
    cols = strs(i["cols"])
    L = pd.DataFrame(np.asarray(i["L"]), columns=cols)
    levels = sorted(set(strs(i["patient"])))
    codes = np.array([levels.index(p) for p in strs(i["patient"])])
    trt = np.asarray(i["trt"]) if key == "cond" else None
    return _index_design(L, None, codes, levels, trt=trt, tested=strs(i["tested"]))


@pytest.mark.parametrize("key", ["pooled", "cond"])
def test_cr2_agrees_with_clubsandwich_on_working_model(S, key):
    """test-sandwich.R: estimate 1e-3, SE 0.5%, df 0.5% of clubSandwich's CR2 / Satterthwaite."""
    r = S["cs16"][key]
    des = _cs16_design(S, key)
    fit = {"alpha": np.asarray(r["alpha"]), "psi": np.asarray(r["psi"])}
    genes = strs(S["cs16"]["inputs"]["genes"])
    got = _sandwich_cr2(fit, des, np.asarray(S["cs16"]["inputs"]["Y"]), genes=genes)
    _assert_table(got, r["got"])
    nt = len(des["tested_niche"])
    for g in genes:
        mine = got[got["gene"] == g]
        o = r["cs"][g]
        assert len(mine) == nt
        # the lm takes one working step from the converged alpha: the pooled fit's estimate is
        # 0.3% off on G3 (R's own test only covers the condition design, held to 1e-3)
        assert _r_equal(mine["estimate"], o["beta"], 1e-3 if key == "cond" else 5e-3), g
        assert _r_equal(mine["se"], o["SE"], 0.005), g
        assert _r_equal(mine["df"], o["df"], 0.005), g


def test_spatial_spline_cr2_agrees_with_clubsandwich_full_design(S):
    """test-depth.R: with block designs the SE is held to 3%, the df to 0.5%."""
    r = S["cs_spline"]
    i = r["inputs"]
    cols = strs(i["cols"])
    L = pd.DataFrame(np.asarray(i["L"]), columns=cols)
    levels = sorted(set(strs(i["patient"])))
    codes = np.array([levels.index(p) for p in strs(i["patient"])])
    smp = np.array(strs(i["smp"]), dtype=object)
    blk = _depth_blocks(
        i["ell"], np.asarray(i["xy"]), smp, np.asarray(i["ik"]) - 1, codes, levels, L
    )
    des = _index_design(
        L, None, codes, levels, trt=np.asarray(i["trt"]), tested=strs(i["tested"]), blocks=blk
    )
    genes = strs(i["genes"])
    fit = {"alpha": np.asarray(r["alpha"]), "psi": np.asarray(r["psi"])}
    got = _sandwich_cr2(fit, des, np.asarray(i["Y"]), genes=genes)
    _assert_table(got, r["got"])
    for g in genes:
        mine = got[got["gene"] == g]
        o = r["cs"][g]
        assert _r_equal(mine["se"], o["SE"], 0.03), g
        assert _r_equal(mine["df"], o["df"], 0.005), g
