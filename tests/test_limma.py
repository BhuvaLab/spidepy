"""limma core (lmFit, fitFDist, squeezeVar, eBayes) against limma 3.66.0 (tests/ref/make_limma.R)."""

import warnings

import numpy as np
import pytest

from spidepy import _limma as L

RTOL, ATOL = 1e-10, 1e-12
# The df2 search inside fitFDistUnequalDF1 is Brent's method on a sum of per-gene terms: R sums in
# long double, numpy in pairwise double, so the iterates differ in the last bits and the optimum
# (known only to ~1e-9 relative) moves with them. Everything downstream of that search (squeeze_var,
# e_bayes) is therefore compared at RTOL_ITER; the rest of the module at closed-form tolerance.
RTOL_ITER = 1e-8

FIT_CASES = [
    "pooled_n4", "pooled_n7", "pooled_n16", "equal_df", "array_w", "gene_w", "no_w",
    "niche_w", "niche_nw", "niche_na", "big_unequal",
]  # fmt: skip


def _close(got, want, what="", rtol=RTOL):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, what
    assert np.array_equal(np.isnan(got), np.isnan(want)), f"NA pattern differs: {what}"
    np.testing.assert_allclose(got, want, rtol=rtol, atol=ATOL, err_msg=what)


def _fit(c):
    i = c["input"]
    return L.lm_fit(i["Y"], i["X"], i.get("W"))


@pytest.mark.parametrize("case", FIT_CASES)
def test_lm_fit_matches_limma(ref, case):
    c = ref("limma")[case]
    fit = _fit(c)
    for key in ("coefficients", "stdev_unscaled", "sigma", "df_residual", "Amean"):
        _close(fit[key], c["lm"][key], f"{case}:{key}")


def test_lm_fit_handles_dropped_observations(ref):
    # na/zero weights make df differ by gene; genes with <= p observations have no sigma
    fit = _fit(ref("limma")["pooled_n7"])
    assert len(np.unique(fit["df_residual"])) > 3
    assert np.isnan(fit["sigma"][fit["df_residual"] == 0]).all()
    assert (fit["df_residual"] == 0).any()


def test_lm_fit_aliased_coefficients_are_na(ref):
    # genes observed only where the niche column is constant cannot estimate it: R gives NA
    c = ref("limma")["niche_na"]
    fit = _fit(c)
    aliased = np.isnan(fit["coefficients"][:, 1])
    assert aliased.sum() == 11
    assert np.isnan(fit["stdev_unscaled"][aliased, 1:]).all()
    assert not np.isnan(fit["coefficients"][aliased, 0]).any()


@pytest.mark.parametrize("case", FIT_CASES)
@pytest.mark.parametrize("trend", [True, False])
def test_e_bayes_matches_limma(ref, case, trend):
    c = ref("limma")[case]
    fit = _fit(c)
    fit["Amean"] = c["input"]["cov"]  # spiDE replaces Amean by the log mean expression
    eb = L.e_bayes(fit, trend=trend)
    want = c["eb_trend" if trend else "eb_flat"]
    for key in ("t", "p_value", "df_total", "s2_prior", "df_prior", "s2_post"):
        _close(eb[key], want[key], f"{case}:{key}", RTOL_ITER)


def test_e_bayes_takes_a_numeric_trend(ref):
    c = ref("limma")["pooled_n7"]
    fit = _fit(c)
    a = L.e_bayes(fit, trend=c["input"]["cov"])
    fit["Amean"] = c["input"]["cov"]
    b = L.e_bayes(fit, trend=True)
    np.testing.assert_array_equal(a["t"], b["t"])


def test_e_bayes_caps_df_total_at_pooled_df(ref):
    c = ref("limma")["equal_df"]
    fit = _fit(c)
    eb = L.e_bayes(fit)
    assert eb["df_total"].max() <= fit["df_residual"].sum()
    assert np.allclose(eb["df_total"], fit["df_residual"] + eb["df_prior"])


FDIST = ["plain", "trend", "notallok", "notallok_flat", "unequal_df1", "small_trend",
         "mid_trend", "const_cov"]  # fmt: skip


@pytest.mark.parametrize("case", FDIST)
def test_fit_f_dist_matches_limma(ref, case):
    c = ref("limma")["fdist"][case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale, df2 = L.fit_f_dist(c["x"], c["df1"], c.get("cov"))
    _close(scale, c["scale"], f"{case}:scale")
    _close(df2, c["df2"], f"{case}:df2")


def test_fit_f_dist_trend_returns_a_value_per_gene(ref):
    c = ref("limma")["fdist"]["notallok"]
    scale, _ = L.fit_f_dist(c["x"], c["df1"], c["cov"])
    assert scale.shape == np.shape(
        c["x"]
    )  # also where x is NA or df is 0: predicted from the trend


def test_fit_f_dist_degenerate_inputs():
    s, d = L.fit_f_dist(np.array([]), 3)
    assert np.isnan(s) and np.isnan(d)
    s, d = L.fit_f_dist(np.array([2.5]), 3)
    assert (s, d) == (2.5, 0.0)
    s, d = L.fit_f_dist(np.array([1.0, 2.0, 3.0]), 0.0)
    assert np.isnan(s) and np.isnan(d)
    with pytest.raises(ValueError, match="different lengths"):
        L.fit_f_dist(np.ones(4), np.ones(3))
    with pytest.raises(ValueError, match="NA covariate"):
        L.fit_f_dist(np.ones(4), 3, np.array([1.0, np.nan, 2.0, 3.0]))


def test_fit_f_dist_warns_on_zero_variances():
    x = np.array([0.0, 1.0, 1.5, 2.0, 0.8, 1.2])
    with pytest.warns(UserWarning, match="Zero sample variances"):
        L.fit_f_dist(x, 4)
    with pytest.warns(UserWarning, match="More than half"):
        L.fit_f_dist(np.array([0.0, 0.0, 0.0, 1.0, 2.0]), 4)


SQZ = ["equal_flat", "equal_trend", "unequal_flat", "unequal_trend", "df0_flat", "df0_trend",
       "df0_equal_trend"]  # fmt: skip


@pytest.mark.parametrize("case", SQZ)
def test_squeeze_var_matches_limma(ref, case):
    c = ref("limma")["sqz"][case]
    got = L.squeeze_var(c["var"], c["df"], covariate=c.get("cov"))
    _close(got["df_prior"], c["df_prior"], f"{case}:df_prior", RTOL_ITER)
    _close(got["var_prior"], c["var_prior"], f"{case}:var_prior", RTOL_ITER)
    _close(got["var_post"], c["var_post"], f"{case}:var_post", RTOL_ITER)


def test_squeeze_var_fewer_than_three_genes(ref):
    c = ref("limma")["sqz"]["tiny"]
    got = L.squeeze_var(c["var"], c["df"])
    assert got["df_prior"] == 0
    np.testing.assert_array_equal(got["var_post"], c["var_post"])
    np.testing.assert_array_equal(got["var_prior"], c["var_prior"])
    with pytest.raises(ValueError, match="empty"):
        L.squeeze_var([], [])


def test_squeeze_var_robust_is_not_implemented_yet(ref):
    c = ref("limma")["sqz"]["equal_flat"]
    with pytest.raises(NotImplementedError):
        L.squeeze_var(c["var"], c["df"], robust=True)
    with pytest.raises(NotImplementedError):
        L.squeeze_var(c["var"], np.arange(1, 301) % 7 + 1, robust=True)
    c = ref("limma")["pooled_n7"]
    with pytest.raises(NotImplementedError):
        L.e_bayes(_fit(c), robust=True)


def test_e_bayes_errors(ref):
    fit = _fit(ref("limma")["pooled_n7"])
    with pytest.raises(ValueError, match="Need Amean"):
        L.e_bayes({k: v for k, v in fit.items() if k != "Amean"}, trend=True)
    with pytest.raises(ValueError, match="length equal to the number of genes"):
        L.e_bayes(fit, trend=np.ones(3))
    zero = dict(fit, df_residual=np.zeros_like(fit["df_residual"]))
    with pytest.raises(ValueError, match="No residual degrees of freedom"):
        L.e_bayes(zero)
    nosig = dict(fit, sigma=np.full_like(fit["sigma"], np.nan))
    with pytest.raises(ValueError, match="No finite residual standard deviations"):
        L.e_bayes(nosig)
    with pytest.raises(ValueError, match="valid lmFit"):
        L.e_bayes({"coefficients": fit["coefficients"]})


def test_lm_fit_input_checks():
    Y = np.ones((3, 4))
    with pytest.raises(ValueError, match="zero rows"):
        L.lm_fit(np.empty((0, 4)), np.ones((4, 1)))
    with pytest.raises(ValueError, match="row dimension of design"):
        L.lm_fit(Y, np.ones((3, 1)))
    with pytest.raises(ValueError, match="NAs not allowed"):
        L.lm_fit(Y, np.array([[1.0], [1.0], [np.nan], [1.0]]))
    with pytest.raises(ValueError, match="unexpected size"):
        L.lm_fit(Y, np.ones((4, 1)), weights=np.ones(5))


def test_lm_fit_weight_shapes_agree():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(5, 4))
    X = np.ones((4, 1))
    w = rng.uniform(0.5, 2, 4)
    a = L.lm_fit(Y, X, weights=w)
    b = L.lm_fit(Y, X, weights=np.tile(w, (5, 1)))
    np.testing.assert_allclose(a["coefficients"], b["coefficients"], rtol=1e-14)
    g = rng.uniform(0.5, 2, 5)
    c = L.lm_fit(Y, X, weights=g)
    d = L.lm_fit(Y, X, weights=np.tile(g[:, None], (1, 4)))
    np.testing.assert_allclose(c["sigma"], d["sigma"], rtol=1e-14)


# ---- components fed R's own inputs ---------------------------------------------------------
@pytest.mark.parametrize("case", ["c1", "c2", "c3", "c4", "c5"])
def test_weighted_lowess_matches_limma(ref, case):
    c = ref("limma")["wl"][case]
    got = L._weighted_lowess(c["x"], c["y"], c["w"], c["span"], npts=200)
    # relative to the scale of the fit: the fit crosses zero, where a pointwise rtol is meaningless
    np.testing.assert_allclose(got, c["fitted"], rtol=1e-9, atol=1e-10 * np.abs(c["fitted"]).max())


def test_trigamma_inverse_matches_limma(ref):
    c = ref("limma")["tgi"]
    _close(L._trigamma_inverse(c["x"]), c["y"], "trigammaInverse")
    assert np.isnan(L._trigamma_inverse(np.array([np.nan]))[0])
    with pytest.warns(UserWarning, match="NaNs produced"):
        assert np.isnan(L._trigamma_inverse(np.array([-1.0, 2.0]))[0])


def test_logmdigamma_matches_statmod(ref):
    c = ref("limma")["lmd"]
    _close(L._logmdigamma(c["z"]), c["y"], "logmdigamma")
    assert np.isnan(L._logmdigamma(np.array([0.0, -1.0, np.nan]))).all()


def test_dqrdc2_rank_pivots_negligible_columns_last():
    x = np.column_stack([np.ones(5), np.arange(5.0), 2 * np.arange(5.0), np.arange(5.0) ** 2])
    rank, pivot = L._dqrdc2_rank(x)
    assert rank == 3
    assert list(pivot) == [0, 1, 3, 2]  # the duplicate (third) column is moved to the end
    assert L._dqrdc2_rank(np.zeros((3, 2)))[0] == 0
