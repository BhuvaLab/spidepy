"""``fit_spide`` end to end against R (``tests/ref/make_fit.R``) and the testthat cases of
``test-fitSpiDE.R``, ``test-intercepts.R``, the fit-level parts of ``test-depth.R`` /
``test-engine.R`` and the strata case of ``test-sandwich.R``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from _ref import load_ref, load_spe
from spidepy import SpiDEFit, build_niches, fit_spide
from spidepy._design import (
    _cell_covariates,
    _index_cells,
    _index_covariates,
    _index_design,
    _niche_columns,
)
from spidepy._glm import _fit_index_glm, _working_wr

RT = dict(rtol=1e-6, atol=1e-8)  # downstream of the iterative NB fit


def strs(x):
    return [str(s) for s in np.atleast_1d(x)]


# --------------------------------------------------------------------------------------
# the fixtures R used
# --------------------------------------------------------------------------------------
def _off(G, N):
    g = np.arange(1, G + 1)[:, None]
    c = np.arange(1, N + 1)[None, :]
    return 0.3 * np.sin(1.3 * g + c / 37)


@pytest.fixture(scope="module")
def toy():
    a = load_spe("toySpiDE")
    num = a.obs["sample_id"].str.slice(1).astype(int)
    a.obs["slide"] = np.where(num % 4 < 2, "slide1", "slide2")
    a.layers["off"] = _off(a.n_vars, a.n_obs).T.copy()
    return build_niches(a, sigma=30)


@pytest.fixture(scope="module")
def big():
    a = load_spe("toyspe_s8_n650_g4_seed11")
    smp = a.obs["sample_id"].to_numpy()
    sec = smp.copy().astype(object)
    sec[smp == "S1"] = np.where(a.obs["x"].to_numpy()[smp == "S1"] < 250, "S1a", "S1b")
    sec[smp == "S3"] = np.where(a.obs["x"].to_numpy()[smp == "S3"] < 100, "S3a", "S3b")
    a.obs["section"] = sec
    a.obs["slide"] = np.where(np.isin(smp, ["S1", "S2", "S4", "S5"]), "X", "Y")
    return build_niches(a, sigma=30)


def _drop_s1(a):
    """toySpiDE with S1 left holding 3 index-A cells (below min_cells): S1 drops out of A."""
    a = a.copy()
    ct = a.obs["cell_type"].to_numpy().astype(object)
    i = np.flatnonzero((a.obs["sample_id"].to_numpy() == "S1") & (ct == "A"))
    ct[i[3:]] = "B"
    a.obs["cell_type"] = ct
    return build_niches(a, sigma=30)


@pytest.fixture(scope="module")
def dropped(toy):
    return _drop_s1(toy)


@pytest.fixture(scope="module")
def R():
    return load_ref("fit")


# config name -> (dataset, fit_spide kwargs); mirrors the run() calls of make_fit.R
CONFIGS = {
    "slopes_all": ("toy", {}),
    "slopes_cond": ("toy", dict(condition="condition", index="A")),
    "sand_pooled": ("toy", dict(engine="sandwich", index="A")),
    "sand_cond_all": ("toy", dict(engine="sandwich", condition="condition")),
    "sand_cov": (
        "toy",
        dict(engine="sandwich", condition="condition", index="A", covariates="Area"),
    ),
    "slopes_nonlinear": ("toy", dict(index="A", depth="nonlinear")),
    "slopes_none": ("toy", dict(index="A", depth="none")),
    "slopes_spline": ("toy", dict(index="A", depth="spatial_spline")),
    "slopes_offset": ("toy", dict(index="A", offset="off")),
    "sand_none_offset": (
        "toy",
        dict(engine="sandwich", condition="condition", index="A", depth="none", offset="off"),
    ),
    "slopes_genes": ("toy", dict(index="A", genes=["G5", "G1", "G2"])),
    "slopes_subset": ("toy", dict(index=["B", "A"], niche="C")),
    "sand_subset": (
        "toy",
        dict(engine="sandwich", condition="condition", index=["C", "A"], niche="B"),
    ),
    "sand_strata": (
        "toy",
        dict(engine="sandwich", condition="condition", index="A", strata="slide"),
    ),
    "sand_strata_pooled_only": ("toy", dict(engine="sandwich", index="A", strata="slide")),
    "slopes_strata": ("toy", dict(condition="condition", index="A", strata="slide")),
    "slopes_detect": (
        "toy",
        dict(index="A", min_detect=0.5, min_cells=40, min_patients=8, tile=50, min_tiles=3),
    ),
    "sand_dropped": (
        "dropped",
        dict(
            engine="sandwich",
            condition="condition",
            index="A",
            strata="slide",
            min_patients=3,
        ),
    ),
    "big_slopes_spline": ("big", dict(index="A", depth="spatial_spline", section="section")),
    "big_sand_spline": (
        "big",
        dict(
            engine="sandwich",
            condition="condition",
            index="A",
            depth="spatial_spline",
            section="section",
            strata="slide",
        ),
    ),
    "big_slopes_loglib": ("big", dict(section="section", min_patients=6)),
    "big_slopes_nonlinear": ("big", dict(index="A", depth="nonlinear")),
}


@pytest.fixture(scope="module")
def fits(toy, big, dropped):
    data = {"toy": toy, "big": big, "dropped": dropped}
    return {
        k: fit_spide(data[ds], sigma=30, verbose=False, **kw) for k, (ds, kw) in CONFIGS.items()
    }


# --------------------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------------------
def _close(got, ref, what, rtol=RT["rtol"], atol=RT["atol"]):
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    assert got.shape == ref.shape, f"{what}: shape {got.shape} vs {ref.shape}"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref), err_msg=f"{what}: NA pattern")
    ok = ~np.isnan(ref)
    np.testing.assert_allclose(got[ok], ref[ok], rtol=rtol, atol=atol, err_msg=what)


def _mean_rel(got, ref, tol, what):
    """testthat ``expect_equal(tolerance =)``: mean relative difference over the vector."""
    got, ref = np.ravel(got).astype(float), np.ravel(ref).astype(float)
    ok = np.isfinite(got) & np.isfinite(ref)
    assert np.isfinite(got).sum() == np.isfinite(ref).sum(), what
    assert np.abs(got[ok] - ref[ok]).sum() / np.abs(ref[ok]).sum() <= tol, what


def _pvals(got, ref, what):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref), err_msg=what)
    tiny = (got < 1e-300) & (ref < 1e-300)
    ok = ~np.isnan(ref) & ~tiny
    np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-5, atol=1e-300, err_msg=what)


def _frame_equal(df, ref, what, num=()):
    """A table written by ``as.list(data.frame)``: same columns, exact labels, close numbers."""
    ref = dict(ref)
    assert list(df.columns) == strs(ref.pop("colnames__")), what
    assert len(df) == len(np.atleast_1d(next(iter(ref.values())))), what
    for c, v in ref.items():
        v = np.atleast_1d(v)
        if v.dtype.kind in "fiu":
            if c == "p":
                _pvals(df[c], v, f"{what}.{c}")
            else:
                _close(df[c], v, f"{what}.{c}")
        else:
            assert [str(s) for s in df[c]] == strs(v), f"{what}.{c}"


def _compare_index(x, r, what):
    for k in ("genes", "niches", "adjusted", "patients"):
        assert list(x[k]) == strs(r[k]) if len(np.atleast_1d(r[k])) else list(x[k]) == [], (what, k)
    assert x["status"] == {k: int(v) for k, v in r["status"].items()}, what
    assert list(x["mean_expr"].index) == list(x["genes"])
    _close(x["mean_expr"], r["mean_expr"], f"{what}.mean_expr")
    _close(x["ncells"], r["ncells"], f"{what}.ncells", rtol=0, atol=0)
    assert list(x["ncells"].index) == list(x["patients"])
    _close(x["intercept"], r["intercept"], f"{what}.intercept")
    assert list(x["intercept"].index) == list(x["genes"])
    assert list(x["intercept"].columns) == list(x["patients"])
    _close(x["niche_mean"], r["niche_mean"], f"{what}.niche_mean", rtol=1e-8)
    assert list(x["niche_mean"].index) == list(x["patients"])
    assert list(x["niche_mean"].columns) == list(x["niches"])
    _close(x["loglib_mean"], r["loglib_mean"], f"{what}.loglib_mean", rtol=1e-8)
    assert list(x["loglib_mean"].index) == list(x["patients"])
    if "depth_r2" in r:
        _close(x["depth_r2"], r["depth_r2"], f"{what}.depth_r2", rtol=1e-6)
    else:
        assert x["depth_r2"] is None
    if "coef" in r:
        _frame_equal(x["coef"], r["coef"], f"{what}.coef")
        assert x["coef"].columns[0] == "test"
        assert "beta" not in x
        return
    for k in ("beta", "v_tile", "v_model"):
        _close(x[k], r[k], f"{what}.{k}")
    _close(x["pooled"], r["pooled"], f"{what}.pooled")
    _close(x["factor"], r["factor"], f"{what}.factor")
    _close(x["psi"], r["psi"], f"{what}.psi")
    assert list(x["pooled"].columns) == list(x["niches"])
    assert list(x["factor"].index) == list(x["patients"])


PARAM_NAMES = {
    "min.cells": "min_cells",
    "min.patients": "min_patients",
    "min.detect": "min_detect",
    "min.tiles": "min_tiles",
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_fit_matches_r(fits, R, name):
    fit, r = fits[name], R[name]
    assert isinstance(fit, SpiDEFit)
    assert fit.engine == r["engine"]
    assert fit.sigma == r["sigma"]
    assert (fit.condition or "") == r["condition"]
    assert list(fit.index) == strs(r["index_names"])
    # patients: columns (order, labels, values) of R's patient table
    pt = dict(r["patients"])
    assert list(fit.patients.columns) == strs(pt.pop("colnames__"))
    for c, v in pt.items():
        if np.asarray(v).dtype.kind in "fiu":
            _close(fit.patients[c], v, f"patients.{c}", rtol=1e-12)
        else:
            assert [str(s) for s in fit.patients[c]] == strs(v), c
    for k, v in r["params"].items():
        got = fit.params[PARAM_NAMES.get(k, k)]
        if k == "covariates":
            assert got == (strs(v) if v != "" else [])
        else:
            assert got == v, k
    for k in fit.index:
        _compare_index(fit.index[k], r["index"][k], f"{name}.{k}")


@pytest.mark.parametrize("name", ["slopes_all", "slopes_offset", "big_slopes_spline"])
def test_patient_slopes_table_matches_r(fits, R, name):
    from spidepy import patient_slopes

    d = patient_slopes(fits[name])
    r = R[name]["slopes"]
    _frame_equal(d, r, "slopes")
    # subsets keep R's row order; the gene / niche arguments filter in fit order
    if name == "slopes_all":
        s = patient_slopes(fits[name], gene=["G2", "G1"], niche="C")
        _frame_equal(s, R[name]["slopes_g1"], "slopes_g1")


@pytest.mark.parametrize(
    "name", ["slopes_all", "slopes_offset", "sand_cond_all", "big_slopes_spline"]
)
def test_patient_intercepts_table_matches_r(fits, R, name):
    from spidepy import patient_intercepts

    d = patient_intercepts(fits[name])
    _frame_equal(d, R[name]["intercepts"], "intercepts")
    s = patient_intercepts(fits[name], gene="G3", index="A")
    _frame_equal(s, R[name]["intercepts_g"], "intercepts_g")


# --------------------------------------------------------------------------------------
# sparse input, n_jobs
# --------------------------------------------------------------------------------------
def _same_fit(a, b, rtol=1e-10, atol=1e-12):
    assert list(a.index) == list(b.index)
    for k in a.index:
        xa, xb = a.index[k], b.index[k]
        assert xa["status"] == xb["status"]
        for key, va in xa.items():
            vb = xb[key]
            if isinstance(va, pd.DataFrame) and key == "coef":
                assert list(va["gene"]) == list(vb["gene"])
                _close(
                    va.drop(columns=["test", "gene", "niche"]),
                    vb.drop(columns=["test", "gene", "niche"]),
                    key,
                    rtol,
                    atol,
                )
            elif isinstance(va, (pd.DataFrame, pd.Series, np.ndarray)):
                _close(va, vb, key, rtol, atol)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("name", ["slopes_all", "sand_cond_all", "slopes_offset"])
def test_sparse_counts_identical_to_dense(toy, fits, fmt, name):
    a = toy.copy()
    mk = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
    a.X = mk(a.X)
    a.layers["counts"] = mk(a.layers["counts"])
    _, kw = CONFIGS[name]
    got = fit_spide(a, sigma=30, verbose=False, **kw)
    _same_fit(got, fits[name])


@pytest.mark.parametrize("name", ["slopes_all", "sand_cond_all", "big_slopes_spline"])
def test_n_jobs_does_not_change_results(toy, big, fits, name):
    ds, kw = CONFIGS[name]
    a = {"toy": toy, "big": big}[ds]
    got = fit_spide(a, sigma=30, verbose=False, n_jobs=2, **kw)
    # polish_nb sizes gene blocks by worker, so BLAS sums re-associate (~1e-13)
    _same_fit(got, fits[name], rtol=1e-9, atol=1e-11)


# --------------------------------------------------------------------------------------
# argument handling and the shape of the fit (test-fitSpiDE.R)
# --------------------------------------------------------------------------------------
def test_missing_niches_error():
    a = load_spe("toyspe_default")
    with pytest.raises(ValueError, match="niche"):
        fit_spide(a, condition="condition", verbose=False)


def test_several_bandwidths_need_sigma():
    a = build_niches(load_spe("toyspe_default"), sigma=[10, 20])
    with pytest.raises(ValueError, match="sigma"):
        fit_spide(a, index="A", min_patients=6, verbose=False)


def test_non_integer_counts_refused():
    a = build_niches(load_spe("toyspe_default"), sigma=20)
    a.layers["counts"] = a.layers["counts"] + 0.5
    with pytest.raises(ValueError, match="integer counts"):
        fit_spide(a, index="A", sigma=20, min_patients=6, verbose=False)


def test_non_finite_covariate_refused_by_name():
    a = build_niches(load_spe("toyspe_default"), sigma=20)
    a.obs["bad"] = np.r_[-np.inf, np.log(np.random.default_rng(1).uniform(size=a.n_obs - 1))]
    with pytest.raises(ValueError, match="bad"):
        fit_spide(a, sigma=20, covariates="bad", index="A", min_patients=6, verbose=False)


def test_patient_level_covariate_refused():
    a = build_niches(load_spe("toyspe_default"), sigma=20)
    with pytest.raises(ValueError, match="constant within sample"):
        fit_spide(a, sigma=20, covariates="Age", index="A", min_patients=6, verbose=False)


def test_bad_engine_and_depth_refused(toy):
    with pytest.raises(ValueError, match="engine"):
        fit_spide(toy, engine="mixed", verbose=False)
    with pytest.raises(ValueError, match="depth"):
        fit_spide(toy, depth="linear", verbose=False)


def test_fit_records_patient_level_columns():
    a = build_niches(load_spe("toyspe_default"), sigma=20)
    fit = fit_spide(a, index="A", sigma=20, min_patients=6, verbose=False)
    assert {"patient", "ncells", "condition"} <= set(fit.patients.columns)
    assert fit.condition is None


def test_nonlinear_depth_is_a_three_df_spline():
    a = build_niches(load_spe("toyspe_s8_n120_g6_seed3"), sigma=20)
    fit = fit_spide(a, index="A", sigma=20, depth="nonlinear", verbose=False)
    assert fit.params["depth"] == "nonlinear"
    assert np.isfinite(fit.index["A"]["beta"]).any()
    ct = a.obs["cell_type"].to_numpy().astype(object)
    ik = _index_cells(ct, a.obs["sample_id"].to_numpy(), "A", 10)
    Y = np.asarray(a.layers["counts"]).T
    cov = _cell_covariates(a.obs, [], "nonlinear", Y).iloc[ik]
    covk = _index_covariates(cov, "nonlinear")
    assert list(covk.columns) == ["loglib_ns1", "loglib_ns2", "loglib_ns3"]
    np.testing.assert_allclose(covk.mean(axis=0), 0, atol=1e-12)


def test_per_gene_constant_offset_is_absorbed_by_intercepts():
    # absorbed to the precision of the 1e-3 ridge on the intercepts: ~1e-4 on the toy
    a = build_niches(load_spe("toyspe_s8_n120_g6_seed3"), sigma=20)
    a.layers["shift"] = np.tile(np.linspace(-1, 1, a.n_vars), (a.n_obs, 1))
    f0 = fit_spide(a, index="A", sigma=20, verbose=False)
    f1 = fit_spide(a, index="A", sigma=20, offset="shift", verbose=False)
    for k in ("beta", "v_tile"):
        _mean_rel(f1.index["A"][k], f0.index["A"][k], 1e-3, k)
    s0 = fit_spide(a, index="A", sigma=20, engine="sandwich", verbose=False)
    s1 = fit_spide(a, index="A", sigma=20, engine="sandwich", offset="shift", verbose=False)
    for k in ("estimate", "se"):
        _mean_rel(s1.index["A"]["coef"][k], s0.index["A"]["coef"][k], 1e-3, k)
    with pytest.raises(ValueError, match="must name an assay"):
        fit_spide(a, index="A", sigma=20, offset="nope", verbose=False)


def test_log_depth_offset_replaces_depth_covariate():
    a = build_niches(load_spe("toyspe_s8_n120_g6_seed3"), sigma=20)
    ll = np.log(np.maximum(np.asarray(a.layers["counts"]).sum(axis=1), 1))
    a.layers["logdepth"] = np.tile(ll[:, None], (1, a.n_vars))
    f = fit_spide(a, index="A", sigma=20, depth="none", offset="logdepth", verbose=False)
    assert f.params["offset"] == "logdepth"
    assert np.isfinite(f.index["A"]["beta"]).any()


def test_genes_restricts_tested_genes_not_library_size():
    a = build_niches(load_spe("toyspe_s8_n120_g6_seed3"), sigma=20)
    f = fit_spide(a, index="A", sigma=20, genes=["G1", "G2"], verbose=False)
    assert set(f.index["A"]["genes"]) == {"G1", "G2"}
    full = fit_spide(a, index="A", sigma=20, verbose=False)
    g = full.index["A"]["genes"].index("G1")
    _close(f.index["A"]["beta"][0], full.index["A"]["beta"][g], "beta G1", rtol=1e-8, atol=1e-10)


def test_unknown_offset_gene_is_an_error():
    from spidepy._design import _index_offset

    a = load_spe("toyspe_s8_n120_g6_seed3")
    a.layers["o"] = np.zeros(a.shape)
    with pytest.raises(ValueError, match="subscript out of bounds"):
        _index_offset(a, "o", ["G1", "nope"], np.arange(5))


def test_verbose_reports_on_stderr_only_when_asked(toy, capsys):
    fit_spide(toy, index="A", sigma=30, verbose=False)
    assert capsys.readouterr().err == ""
    fit_spide(toy, index="A", sigma=30, verbose=True)
    assert (
        "slopes engine, index A: 20 genes, 813 cells, 16 patients, 2 niches"
        in capsys.readouterr().err
    )


def test_repr_mirrors_show(fits):
    assert repr(fits["sand_cond_all"]).splitlines() == [
        "SpiDEFit (sandwich engine), bandwidth 30",
        "  16 patients; condition: condition",
        "  index A: 20 genes x 2 niches (B, C)",
        "  index B: 20 genes x 2 niches (A, C)",
        "  index C: 20 genes x 2 niches (A, B)",
    ]
    assert "condition: none" in repr(fits["slopes_all"])


def test_nothing_to_fit_errors(toy):
    with pytest.raises(ValueError, match="no index cell type has enough cells"):
        fit_spide(toy, index="Z", sigma=30, verbose=False)
    with pytest.raises(ValueError, match="no index cell type could be fitted"):
        fit_spide(toy, index="A", genes=["nope"], sigma=30, verbose=False)
    # every patient is below min_cells: the type is skipped (and reported)
    buf = io.StringIO()
    with redirect_stderr(buf), pytest.raises(ValueError, match="could be fitted"):
        fit_spide(toy, index="A", sigma=30, min_cells=10**4, min_patients=1, verbose=True)
    assert "skipping A (0 patients)" in buf.getvalue()


def test_cells_without_counts_are_left_out_and_reported(big, capsys):
    n0 = int((np.asarray(big.layers["counts"]).sum(axis=1) == 0).sum())
    assert n0 > 0
    fit_spide(big, index="A", sigma=30, verbose=True)
    assert f"{n0} cell(s) with no counts left out" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# test-intercepts.R
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spe16():
    return build_niches(load_spe("toyspe_s16_n150_seed7"), sigma=30)


def test_slopes_engine_keeps_intercepts_niche_means_and_depth(spe16):
    xi = fit_spide(spe16, sigma=30, index="A", verbose=False).index["A"]
    assert xi["intercept"].shape == (len(xi["genes"]), len(xi["patients"]))
    assert list(xi["intercept"].index) == xi["genes"]
    assert list(xi["intercept"].columns) == xi["patients"]
    assert xi["niche_mean"].shape == (len(xi["patients"]), len(xi["niches"]))
    assert list(xi["niche_mean"].index) == xi["patients"]
    assert list(xi["loglib_mean"].index) == xi["patients"]
    assert np.isfinite(xi["intercept"].to_numpy()).all()
    assert np.isfinite(xi["loglib_mean"].to_numpy()).all()


def test_stored_intercepts_are_the_shared_fits(spe16):
    xi = fit_spide(spe16, sigma=30, index="A", verbose=False).index["A"]
    obs = spe16.obs
    Y = np.asarray(spe16.layers["counts"]).T
    cov = _cell_covariates(obs, [], "loglib", Y)
    usable = np.isfinite(cov["loglib"].to_numpy())
    ct = np.where(usable, obs["cell_type"].to_numpy(), np.nan).astype(object)
    smp = obs["sample_id"].to_numpy()
    ik = _index_cells(ct, smp, "A", 10)
    nc = _niche_columns(spe16.obsm["Niche30"], "A", ik)
    cols = nc["tested"] + [c for c in nc["cols"] if c not in nc["tested"]]
    L = pd.DataFrame(np.log1p(spe16.obsm["Niche30"].iloc[ik][cols].to_numpy()), columns=cols)
    levels = sorted(set(smp[ik]))
    codes = np.array([levels.index(s) for s in smp[ik]])
    des = _index_design(
        L, _index_covariates(cov.iloc[ik], "loglib"), codes, levels, tested=nc["tested"]
    )
    gi = [list(spe16.var_names).index(g) for g in xi["genes"]]
    f0 = _fit_index_glm(Y[np.ix_(gi, ik)], des)
    np.testing.assert_allclose(
        xi["intercept"].to_numpy(), f0["alpha"][:, des["intercept_cols"]], rtol=1e-10, atol=1e-12
    )
    want = pd.DataFrame(L).groupby(codes).mean()["B"].to_numpy()
    np.testing.assert_allclose(xi["niche_mean"]["B"].to_numpy(), want, rtol=1e-12)


def test_sandwich_engine_keeps_intercepts_and_cell_counts(spe16):
    fit = fit_spide(
        spe16, condition="condition", sigma=30, index="A", engine="sandwich", verbose=False
    )
    xi = fit.index["A"]
    assert xi["intercept"].shape == (len(xi["genes"]), len(xi["patients"]))
    assert list(xi["ncells"].index) == xi["patients"]
    assert xi["ncells"].sum() == int((spe16.obs["cell_type"] == "A").sum())


def test_patient_intercepts_long_form_and_results_dispatch(spe16):
    from spidepy import SpiDEResults, patient_intercepts

    fit = fit_spide(spe16, condition="condition", sigma=30, index="A", verbose=False)
    d = patient_intercepts(fit, gene=["G1", "G2"])
    assert len(d) == 2 * len(fit.index["A"]["patients"])
    assert {"gene", "index", "patient", "intercept", "ncells", "condition"} <= set(d.columns)
    got = d.loc[(d.gene == "G1") & (d.patient == "S1"), "intercept"].iloc[0]
    assert got == fit.index["A"]["intercept"].loc["G1", "S1"]
    res = SpiDEResults(pd.DataFrame(), "condition", "", "all", 0.05, fit)
    pd.testing.assert_frame_equal(
        patient_intercepts(res, gene="G1"), patient_intercepts(fit, gene="G1")
    )


def test_fit_without_intercepts_is_refused_with_a_pointer_to_refit(spe16):
    from spidepy import patient_intercepts

    fit = fit_spide(spe16, sigma=30, index="A", verbose=False)
    del fit.index["A"]["intercept"]
    with pytest.raises(ValueError, match="refit"):
        patient_intercepts(fit)


def test_slopes_need_the_slopes_engine(fits):
    from spidepy import patient_slopes

    with pytest.raises(ValueError, match="slopes engine"):
        patient_slopes(fits["sand_pooled"])


def test_accessors_return_empty_frame_for_no_match(fits):
    from spidepy import patient_intercepts, patient_slopes

    assert patient_slopes(fits["slopes_all"], gene="nope").empty
    assert patient_intercepts(fits["slopes_all"], index="Z").empty


# --------------------------------------------------------------------------------------
# test-engine.R / test-depth.R at the fit level
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spe_g12():
    return build_niches(load_spe("toyspe_s16_n120_g12_seed2"), sigma=30)


def _refit_design(a, genes, k="A", depth="loglib", minus=None):
    """The shared design and fit of one index type, as the R test rebuilds it."""
    obs = a.obs
    Y = np.asarray(a.layers["counts"]).T
    cov = _cell_covariates(obs, [], depth, Y)
    smp = obs["sample_id"].to_numpy()
    ik = _index_cells(obs["cell_type"].to_numpy().astype(object), smp, k, 10)
    NM = a.obsm["Niche30"]
    nc = _niche_columns(NM, k, ik)
    cols = nc["tested"] + [c for c in nc["cols"] if c not in nc["tested"]]
    L = pd.DataFrame(np.log1p(NM.iloc[ik][cols].to_numpy()), columns=cols)
    levels = sorted(set(smp[ik]))
    codes = np.array([levels.index(s) for s in smp[ik]])
    gi = [list(a.var_names).index(g) for g in genes]
    return Y[np.ix_(gi, ik)], L, cov.iloc[ik], codes, levels, nc


def test_slopes_fit_has_both_variances(spe_g12):
    fit = fit_spide(spe_g12, index="A", sigma=30, verbose=False)
    x = fit.index["A"]
    assert x["beta"].shape == (len(x["genes"]), len(x["patients"]), len(x["niches"]))
    assert (x["v_tile"][np.isfinite(x["v_tile"])] > 0).all()
    assert (x["factor"].to_numpy() >= 1).all()


def test_patient_slope_is_one_fisher_step_from_the_pooled_fit(spe_g12):
    x = fit_spide(spe_g12, index="A", sigma=30, verbose=False).index["A"]
    Y, L, cov, codes, levels, nc = _refit_design(spe_g12, x["genes"])
    des = _index_design(L, _index_covariates(cov, "loglib"), codes, levels, tested=nc["tested"])
    fit = _fit_index_glm(Y, des)
    g, s = 0, 2
    i = np.flatnonzero(codes == s)
    wr = _working_wr(Y[g], des["W"], fit["alpha"][g], fit["psi"][g])
    Ls = L.to_numpy()[i]
    Lt = Ls - (Ls * wr["w"][i, None]).sum(axis=0) / wr["w"][i].sum()
    step = np.linalg.solve((Lt * wr["w"][i, None]).T @ Lt, (Lt * wr["r"][i, None]).sum(axis=0))
    b = fit["alpha"][g, des["niche_cols"]] + step
    np.testing.assert_allclose(x["beta"][g, s, :], b[: len(nc["tested"])], rtol=1e-6)


def test_absent_niche_drops_that_slope_alone_not_the_patient():
    a = load_spe("toyspe_s16_n120_g12_seed2")
    s0 = "S3"
    ct = a.obs["cell_type"].to_numpy().astype(object)
    ct[(a.obs["sample_id"].to_numpy() == s0) & (ct == "C")] = "B"
    a.obs["cell_type"] = ct
    a = build_niches(a, sigma=30)
    x = fit_spide(a, index="A", sigma=30, verbose=False).index["A"]
    p = x["patients"].index(s0)
    b_, c_ = x["niches"].index("B"), x["niches"].index("C")
    assert np.isnan(x["beta"][:, p, c_]).all() and np.isnan(x["v_model"][:, p, c_]).all()
    assert np.isfinite(x["beta"][:, p, b_]).all() and (x["v_tile"][:, p, b_] > 0).all()
    # the kept slope is the one-step estimate with the absent niche's column left out
    Y, L, cov, codes, levels, nc = _refit_design(a, x["genes"])
    des = _index_design(L, _index_covariates(cov, "loglib"), codes, levels, tested=nc["tested"])
    fit = _fit_index_glm(Y, des)
    i = np.flatnonzero(codes == p)
    keep = [j for j, c in enumerate(L.columns) if c != "C"]
    got = []
    for g in range(len(x["genes"])):
        wr = _working_wr(Y[g], des["W"], fit["alpha"][g], fit["psi"][g])
        Ls = L.to_numpy()[np.ix_(i, keep)]
        Lt = Ls - (Ls * wr["w"][i, None]).sum(axis=0) / wr["w"][i].sum()
        step = np.linalg.solve((Lt * wr["w"][i, None]).T @ Lt, (Lt * wr["r"][i, None]).sum(axis=0))
        got.append(fit["alpha"][g, des["niche_cols"][0]] + step[0])
    np.testing.assert_allclose(x["beta"][:, p, b_], got, rtol=1e-6)


def test_serial_and_parallel_fits_agree(spe_g12):
    f1 = fit_spide(spe_g12, index="A", sigma=30, verbose=False)
    f2 = fit_spide(spe_g12, index="A", sigma=30, verbose=False, n_jobs=2)
    for k in ("beta", "v_tile"):
        _close(f2.index["A"][k], f1.index["A"][k], k, rtol=1e-8, atol=1e-10)


def test_spatial_spline_slope_is_one_fisher_step_with_block_partialled_out(big):
    from spidepy._design import _depth_blocks
    from spidepy._design import _partial_block as pb
    from spidepy._io import get_coords
    from spidepy._sandwich import _partial_block  # noqa: F401  (documents the FWL step)

    fit = fit_spide(big, index="A", sigma=30, depth="spatial_spline", verbose=False)
    x = fit.index["A"]
    assert np.nanmin(x["depth_r2"].to_numpy()) >= 0 and np.nanmax(x["depth_r2"].to_numpy()) <= 1
    obs = big.obs
    Y = np.asarray(big.layers["counts"]).T
    lib = Y.sum(axis=0)
    ct = np.where(lib > 0, obs["cell_type"].to_numpy(), np.nan).astype(object)
    smp = obs["sample_id"].to_numpy()
    ik = _index_cells(ct, smp, "A", 10)
    NM = big.obsm["Niche30"]
    nc = _niche_columns(NM, "A", ik)
    cols = nc["tested"] + [c for c in nc["cols"] if c not in nc["tested"]]
    L = pd.DataFrame(np.log1p(NM.iloc[ik][cols].to_numpy()), columns=cols)
    levels = sorted(set(smp[ik]))
    codes = np.array([levels.index(s) for s in smp[ik]])
    ell = np.log(lib[ik])
    blk = _depth_blocks(ell, get_coords(big), smp, ik, codes, levels, L)
    des = _index_design(L, None, codes, levels, tested=nc["tested"], blocks=blk)
    gi = [list(big.var_names).index(g) for g in x["genes"]]
    Yk = Y[np.ix_(gi, ik)]
    f = _fit_index_glm(Yk, des)
    g, s = 0, 1
    i = np.flatnonzero(codes == s)
    wr = _working_wr(Yk[g], des["W"], f["alpha"][g], f["psi"][g])
    Lt = pb(L.to_numpy()[i], wr["w"][i], blk["Zs"][s])
    step = np.linalg.solve((Lt * wr["w"][i, None]).T @ Lt, (Lt * wr["r"][i, None]).sum(axis=0))
    b = f["alpha"][g, des["niche_cols"]] + step
    np.testing.assert_allclose(x["beta"][g, s, :], b[: len(nc["tested"])], rtol=1e-6)


# --------------------------------------------------------------------------------------
# test-sandwich.R: strata enter the condition model only
# --------------------------------------------------------------------------------------
def test_strata_enter_the_condition_model_only_never_the_pooled_one():
    a = build_niches(load_spe("toyspe_s8_n120_g8_seed11"), sigma=20)
    pat = a.obs["sample_id"].to_numpy()
    cond = a.obs.groupby("sample_id", sort=False)["condition"].first()
    # two slides, each holding both conditions
    sl = {}
    for lv in cond.unique():
        for i, s in enumerate(cond.index[cond == lv]):
            sl[s] = "slide1" if i % 2 == 0 else "slide2"
    a.obs["slide"] = [sl[s] for s in pat]
    args = dict(condition="condition", engine="sandwich", index="A", sigma=20, min_patients=6,
                verbose=False)  # fmt: skip
    with_st = fit_spide(a, strata="slide", **args).index["A"]["coef"]
    without = fit_spide(a, **args).index["A"]["coef"]
    cols = ["estimate", "se", "df"]
    pd.testing.assert_frame_equal(
        with_st[with_st.test == "pooled"][cols].reset_index(drop=True),
        without[without.test == "pooled"][cols].reset_index(drop=True),
    )
    # a pooled-only fit (no condition) ignores strata as well
    pooled_only = fit_spide(a, strata="slide", **{**args, "condition": None}).index["A"]["coef"]
    pd.testing.assert_frame_equal(
        pooled_only[cols].reset_index(drop=True),
        without[without.test == "pooled"][cols].reset_index(drop=True),
    )
    cw = with_st[with_st.test == "condition"]
    cn = without[without.test == "condition"]
    assert not np.allclose(cw["estimate"].to_numpy(), cn["estimate"].to_numpy())


@pytest.mark.parametrize("col", ["condition", "slide"])
def test_dropped_patient_label_never_reaches_other_cells(dropped, col):
    """Cells take their own patient's condition / stratum even when a patient is dropped."""
    kw = dict(engine="sandwich", condition="condition", index="A", strata="slide",
              min_patients=3, sigma=30, verbose=False)  # fmt: skip
    base = fit_spide(dropped, **kw)
    assert "S1" not in base.index["A"]["patients"]
    b = dropped.copy()
    s1 = (b.obs["sample_id"] == "S1").to_numpy()
    cur = b.obs[col].to_numpy().astype(object)
    flip = {"Responder": "Non-responder", "Non-responder": "Responder",
            "slide1": "slide2", "slide2": "slide1"}  # fmt: skip
    cur[s1] = flip[cur[s1][0]]
    b.obs[col] = cur
    got = fit_spide(b, **kw)
    pd.testing.assert_frame_equal(got.index["A"]["coef"], base.index["A"]["coef"])
