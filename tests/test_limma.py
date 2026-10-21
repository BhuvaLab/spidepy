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
    # identical sigma -> infinite prior df: df.total is the pooled residual df, not Inf
    c = ref("limma")["cap"]
    fit = {k: c[k] for k in ("coefficients", "stdev_unscaled", "sigma", "df_residual")}
    eb = L.e_bayes(fit)
    assert np.isinf(eb["df_prior"])
    assert (eb["df_total"] == c["df_residual"].sum()).all()
    for key in ("t", "p_value", "df_total", "s2_post"):
        _close(eb[key], c[key], f"cap:{key}", RTOL_ITER)
    # a finite prior below the cap leaves df_total = df_residual + df_prior
    eq = _fit(ref("limma")["equal_df"])
    eb = L.e_bayes(eq)
    assert np.allclose(eb["df_total"], eq["df_residual"] + eb["df_prior"])
    assert eb["df_total"].max() <= eq["df_residual"].sum()


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
       "df0_equal_trend", "unequal_trend_ties", "unequal_trend_ties_small"]  # fmt: skip


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
    _close(got, c["fitted"], case)


WL2 = ["int_ties", "round_ties", "ties_delta", "spike_zero", "all_equal_x", "two_values", "n2",
       "n5", "n9_span", "iter4", "iter3_ties", "zero_w"]  # fmt: skip


@pytest.mark.parametrize("case", WL2)
def test_weighted_lowess_ties_and_edge_cases(ref, case):
    # heavily tied x (integer / rounded / spike of zeros), delta > 0 with ties, all-equal x,
    # tiny n, zero weights and robustness iterations, from limma's own C routine
    c = ref("limma")["wl2"][case]
    got = L._weighted_lowess(
        c["x"], c["y"], c["w"], c["span"], iterations=int(c["iterations"]), npts=int(c["npts"])
    )
    _close(got, c["fitted"], case)


FDU = ["two_acc", "two_acc_df", "two_plain", "weighted_na", "ties_trend"]


@pytest.mark.parametrize("case", FDU)
def test_fit_f_dist_unequal_df1_matches_limma(ref, case):
    # "two_acc*": exactly two informative genes plus NA / df ~ 0 genes. R then drops
    # prior.weights but keeps its flag, so scale is NaN and df2 is Brent's value on a flat objective
    c = ref("limma")["fdu"][case]
    cov = ref("limma")["sqz"]["unequal_trend_ties"]["cov"][:100] if case == "ties_trend" else None
    got = L._fit_f_dist_unequal_df1(c["x"], c["df1"], covariate=cov)
    scale, df2 = got["scale"], got["df2"]
    if case.startswith("two_acc"):
        assert np.isnan(c["scale"]) and np.isnan(scale)
    _close(scale, c["scale"], f"{case}:scale", RTOL_ITER)
    _close(df2, c["df2"], f"{case}:df2", RTOL_ITER)


def test_fit_f_dist_unequal_df1_one_informative_gene(ref):
    c = ref("limma")["fdu"]["one"]
    got = L._fit_f_dist_unequal_df1(c["x"], c["df1"])
    scale, df2 = got["scale"], got["df2"]
    assert np.isnan(scale) and np.isnan(df2)
    assert np.isnan(c["scale"]) and np.isnan(c["df2"])


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


# ---- robust empirical Bayes (tests/ref/make_limma_robust.R) --------------------------------------
# fitFDistRobustly finds df2 with uniroot (Brent, tol 1e-8) and fitFDistUnequalDF1 with optimize
# (Brent); both optima are known to ~1e-8 relative only, so everything is compared at RTOL_ITER.
ROB_KEYS = ("scale", "df2", "df2_shrunk", "tail_p_value", "prob_outlier", "df2_outlier")


def _check_robust_fit(got, want, case):
    for key in ROB_KEYS:
        if key in want:
            assert key in got, f"{case}: missing {key}"
            _close(got[key], want[key], f"{case}:{key}", RTOL_ITER)


# "small_trend_10" is in the reference file but not tested: its lowess trend interpolates four genes
# exactly, so their F statistics tie up to 1e-16 and R's own ranks (hence prob.outlier) depend on
# rounding noise.
FR = ["flat_clean", "flat_hi", "flat_lo", "flat_both", "trend_clean", "trend_hi", "trend_lo",
      "trend_both", "big_trend", "small_flat_6", "small_trend_6", "small_flat_8",
      "small_trend_8", "small_flat_10", "small_flat_12", "small_trend_12",
      "small_flat_16", "small_trend_16", "small_flat_30", "small_trend_30", "inf_flat",
      "inf_flat_out", "inf_trend", "df2_big", "ties_trend", "ties_trend_small", "const_cov",
      "wtp_small", "wtp_other", "notallok_trend", "notallok_flat", "notallok_ties", "zero_var",
      "unequal_flat", "unequal_trend", "unequal_lo", "unequal_wide"]  # fmt: skip


@pytest.mark.parametrize("case", FR)
def test_fit_f_dist_robustly_matches_limma(ref, case):
    c = ref("limma_robust")["fr"][case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = L.fit_f_dist_robustly(
            c["x"], c["df1"], c.get("cov"), winsor_tail_p=tuple(np.atleast_1d(c["wtp"]))
        )
    _check_robust_fit(got, c, case)


def test_fit_f_dist_robustly_branches(ref):
    # the reference cases must cover every return path of fitFDistRobustly
    fr = ref("limma_robust")["fr"]
    assert np.isinf(fr["inf_flat"]["df2"]) and "tail_p_value" in fr["inf_flat"]  # df2 = Inf
    assert "prob_outlier" in fr["flat_hi"] and "df2_outlier" in fr["flat_hi"]  # shrinkage
    assert "prob_outlier" not in fr["wtp_small"]  # tails below 1/n: non-robust returned
    assert "prob_outlier" not in fr["small_flat_6"]
    hi, lo = fr["flat_hi"], fr["flat_lo"]
    assert hi["df2_shrunk"].min() < hi["df2"]  # right outliers shrink the prior df
    assert lo["df2_shrunk"].min() <= lo["df2"]


def test_fit_f_dist_robustly_degenerate_inputs():
    r = L.fit_f_dist_robustly(np.array([1.0]), 3)
    assert np.isnan(r["scale"]) and np.isnan(r["df2"]) and np.isnan(r["df2_shrunk"])
    scale, df2 = L.fit_f_dist(np.array([1.0, 2.0]), 3)
    r = L.fit_f_dist_robustly(np.array([1.0, 2.0]), 3)
    assert (r["scale"], r["df2"]) == (scale, df2) and "df2_shrunk" not in r
    with pytest.raises(ValueError, match="different lengths"):
        L.fit_f_dist_robustly(np.ones(5), np.ones(4))
    with pytest.raises(ValueError, match="covariate contains"):
        L.fit_f_dist_robustly(np.ones(5), 3, np.array([1.0, 2, np.nan, 3, 4]))
    with pytest.raises(ValueError, match="mostly <= 0"):
        L.fit_f_dist_robustly(np.array([0.0, 0.0, 0.0, 1.0, 2.0]), 3)


FU = ["flat_clean", "flat_hi", "flat_lo", "flat_both", "trend_hi", "trend_both", "trend_span",
      "small_flat_6", "small_trend_6", "small_flat_8", "small_trend_8", "small_flat_12",
      "small_trend_12", "small_flat_16", "small_trend_16", "small_flat_40", "small_trend_40",
      "ties_trend", "ties_small", "na_flat", "na_trend", "scalar_df", "big_trend",
      "extreme"]  # fmt: skip


@pytest.mark.parametrize("case", FU)
def test_fit_f_dist_unequal_df1_robust_matches_limma(ref, case):
    c = ref("limma_robust")["fu"][case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = L._fit_f_dist_unequal_df1(
            c["x"], c["df1"], covariate=c.get("cov"), span=c.get("span"), robust=True
        )
    _check_robust_fit(got, c, case)


def test_fit_f_dist_unequal_df1_robust_branches(ref):
    fu = ref("limma_robust")["fu"]
    assert "df2_shrunk" in fu["flat_hi"] and "df2_outlier" in fu["flat_hi"]  # right outliers
    assert "df2_shrunk" not in fu["flat_clean"]  # no outlier: non-robust estimate returned
    assert any("df2_shrunk" not in v for v in fu.values())
    assert any("df2_shrunk" in v for v in fu.values())


SQR = ["equal_flat", "equal_trend", "unequal_flat", "unequal_trend", "df0_equal_trend",
       "df0_equal_flat", "df0_unequal_trend", "small_equal", "small_unequal",
       "span_unequal"]  # fmt: skip


@pytest.mark.parametrize("case", SQR)
def test_squeeze_var_robust_matches_limma(ref, case):
    c = ref("limma_robust")["sq"][case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = L.squeeze_var(
            c["var"], c["df"], covariate=c.get("cov"), span=c.get("span"), robust=True
        )
    for key in ("df_prior", "var_prior", "var_post"):
        _close(got[key], c[key], f"{case}:{key}", RTOL_ITER)


EBR = ["pooled_n4", "pooled_n7", "pooled_n16", "equal_df", "no_w", "few_equal_6", "few_equal_9",
       "few_equal_16", "few_na_6", "few_na_9", "few_na_16", "niche_w", "niche_nw",
       "ties_cov"]  # fmt: skip


@pytest.mark.parametrize("case", EBR)
@pytest.mark.parametrize("trend", [True, False])
def test_e_bayes_robust_matches_limma(ref, case, trend):
    # intercept-only weighted lmFit with NA observations, then eBayes(robust = TRUE) with the
    # log-mean-expression stand-in for Amean, as spiDE's pooled test does
    c = ref("limma_robust")["eb"][case]
    i = c["input"]
    fit = L.lm_fit(i["Y"], i["X"], i.get("W"))
    fit["Amean"] = i["cov"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = L.e_bayes(fit, trend=trend, robust=True)
    want = c["eb_trend" if trend else "eb_flat"]
    for key in ("t", "p_value", "df_total", "s2_prior", "df_prior", "s2_post"):
        _close(got[key], want[key], f"{case}:{trend}:{key}", RTOL_ITER)


def test_e_bayes_robust_df_prior_is_per_gene(ref):
    c = ref("limma_robust")["eb"]["equal_df"]
    i = c["input"]
    fit = L.lm_fit(i["Y"], i["X"], i.get("W"))
    fit["Amean"] = i["cov"]
    robust = L.e_bayes(fit, trend=True, robust=True)
    plain = L.e_bayes(fit, trend=True)
    assert robust["df_prior"].shape == fit["sigma"].shape  # df2.shrunk, one per gene
    assert np.ndim(plain["df_prior"]) == 0
    assert robust["df_prior"].min() < plain["df_prior"]  # outlying genes are shrunk less
