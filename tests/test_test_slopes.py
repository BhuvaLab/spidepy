"""The helpers of ``R/test-slopes.R`` on R's own inputs (``tests/ref/make_test.R``), plus the
direct-formula cases of ``test-engine.R`` and ``test-small-n-defaults.R``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from _fitref import close, fit_from_ref, frame, pvals
from _ref import load_ref
from spidepy import _limma
from spidepy._test_slopes import (
    _bh_families,
    _dl_tau2,
    _heterogeneity_p,
    _kish_df,
    _pooled_column_test,
    _robust_condition_test,
)

LIMMA = dict(rtol=1e-8, atol=1e-10)  # eBayes / fitFDistRobustly (Task 4/5 tolerance)


@pytest.fixture(scope="module")
def R():
    return load_ref("test")


def _compare(df, ref, what, rtol=1e-8, tight=("estimate", "n_patients")):
    """A helper's table against R's. ``tight`` columns are always compared at 1e-8: they do not
    pass through the robust empirical-Bayes prior, whose tie-sensitivity ``rtol`` may relax."""
    r = frame(ref)
    assert list(df.columns) == list(r.columns), what
    for c in df.columns:
        tol = 1e-8 if c in tight else rtol
        if c == "p":
            pvals(df[c], r[c], f"{what}.{c}", tol)
        else:
            close(df[c], r[c], f"{what}.{c}", rtol=tol, atol=1e-10)


# Tie noise in limma's robust prior (docs/r-discrepancies.md): the end-to-end slopes fixture has a
# 12-gene niche column in which six genes sit at the same tail probability up to 1e-16, so the
# prior df of those genes moves by up to ~1e-3 with the rounding of the weighted mean, in R and in
# Python alike. Its moderated se / t / df / p cannot be compared tighter than that.
NOISY = {("e2e_fit", "A", 0): 5e-3}


CONFIGS = ["sl16", "toy_strata", "dropped_slopes", "big", "e2e_fit"]


def _columns(R, name):
    d = R[name]
    fit = fit_from_ref(d["fit"])
    for k, x in fit.index.items():
        h = d["helpers"][k]
        vp = np.asarray(h["vpool"], float).reshape(x["beta"].shape)
        yield name, k, x, vp, h


@pytest.mark.parametrize("name", CONFIGS)
def test_tau2_and_heterogeneity_on_r_slopes(R, name):
    for _, k, x, vp, h in _columns(R, name):
        # v_model * factor is the pooled variance: closed form
        np.testing.assert_allclose(x["v_model"] * x["factor"].to_numpy()[None], vp, rtol=1e-12)
        nt = len(x["niches"])
        tau = np.asarray(h["tau"], float).reshape(len(x["genes"]), nt)
        het = np.asarray(h["het"], float).reshape(len(x["genes"]), nt)
        close(_dl_tau2(x["beta"], vp), tau, f"{name}.{k}.tau2")
        close(_heterogeneity_p(x["beta"], vp), het, f"{name}.{k}.het", rtol=1e-9)


@pytest.mark.parametrize("name", CONFIGS)
def test_pooled_column_test_on_r_slopes(R, name):
    for _, k, x, vp, h in _columns(R, name):
        nt = len(x["niches"])
        tau = np.asarray(h["tau"], float).reshape(len(x["genes"]), nt)
        me = x["mean_expr"].to_numpy()
        for j in range(nt):
            cj = h[f"col{j + 1}"]
            b, v = x["beta"][:, :, j], vp[:, :, j]
            what = f"{name}.{k}.{j}"
            tol = NOISY.get((name, k, j), 1e-8)
            _compare(
                _pooled_column_test(b, v, tau[:, j], me), cj["pooled_prop"], what + ".prop", tol
            )
            _compare(
                _pooled_column_test(b, v, tau[:, j], me, df_rule="capped"),
                cj["pooled_cap"],
                what + ".cap",
                tol,
            )
            _compare(
                _pooled_column_test(b, v, tau[:, j], me, trend=False),
                cj["pooled_flat"],
                what + ".flat",
                tol,
            )
            with np.errstate(all="ignore"):
                kd = _kish_df(1 / (v + tau[:, j][:, None]), b, 1)
            close(kd, cj["kish"], what + ".kish")


@pytest.mark.parametrize("name", CONFIGS)
def test_robust_condition_test_on_r_slopes(R, name):
    d = R[name]
    fit = fit_from_ref(d["fit"])
    from spidepy.fit import _condition_coding, _strata_by_patient

    trt_of = _condition_coding(fit.patients, "condition")
    str_of = _strata_by_patient(fit.patients, "slide") if "slide" in fit.patients else None
    n_cond = n_strata = 0
    for _, k, x, vp, h in _columns(R, name):
        nt = len(x["niches"])
        tau = np.asarray(h["tau"], float).reshape(len(x["genes"]), nt)
        trt = trt_of.reindex(x["patients"]).to_numpy()
        for j in range(nt):
            cj = h[f"col{j + 1}"]
            b, v = x["beta"][:, :, j], vp[:, :, j]
            got = _robust_condition_test(b, v, tau[:, j], trt)
            _compare(got, cj["cond"], f"{name}.{k}.{j}.cond", rtol=1e-10)
            n_cond += int(np.isfinite(got["p"]).sum())
            if "cond_strata" in cj:
                st = str_of.reindex(x["patients"]).to_numpy()
                got = _robust_condition_test(b, v, tau[:, j], trt, strata=st)
                _compare(got, cj["cond_strata"], f"{name}.{k}.{j}.cond_strata", rtol=1e-10)
                n_strata += int(np.isfinite(got["p"]).sum())
    assert n_cond > 10
    if name in ("toy_strata", "dropped_slopes"):
        assert n_strata > 10


def test_pooled_dfs_on_the_synthetic_family(R):
    f = R["family"]
    b, v, tau, me = (np.asarray(f[k], float) for k in ("b", "v", "tau", "mean_expr"))
    pro = _pooled_column_test(b, v, tau, me)
    cap = _pooled_column_test(b, v, tau, me, df_rule="capped")
    _compare(pro, f["prop"], "prop")
    _compare(cap, f["cap"], "cap")
    _compare(_pooled_column_test(b, v, tau, me, trend=False), f["flat"], "flat")
    close(_kish_df(1 / (v + tau[:, None]), b, 1), f["kish"], "kish")
    # only the df move; limma's df is the upper bound of both
    np.testing.assert_array_equal(pro["t"], cap["t"])
    np.testing.assert_array_equal(pro["estimate"], cap["estimate"])
    ok = np.isfinite(b).sum(axis=1) >= 6
    assert (pro["df"][ok] > cap["df"][ok]).mean() > 0.5
    # direct limma computation, as the testthat case. R compares at 1e-12; the robust prior is
    # an optimiser that amplifies the 1e-16 difference in how the two weight normalisations sum
    # (nanmean here, rowMeans in R), so the Python comparison is at the limma tolerance.
    w = 1 / (v + tau[:, None])
    w[~np.isfinite(b)] = np.nan
    w = w / np.nanmean(w, axis=1)[:, None]
    fit = _limma.lm_fit(b, np.ones((b.shape[1], 1)), weights=w)
    fit["Amean"] = np.log(me + 1e-3)
    eb = _limma.e_bayes(fit, robust=True, trend=True)
    neff = np.nansum(w, axis=1) ** 2 / np.nansum(w**2, axis=1)
    m = np.isfinite(w).sum(axis=1)
    np.testing.assert_allclose(pro["df"][ok], (eb["df_total"] * neff / m)[ok], rtol=1e-8)
    np.testing.assert_allclose(
        cap["df"][ok], np.minimum(eb["df_total"], np.maximum(neff - 1, 1))[ok], rtol=1e-8
    )
    assert (pro["df"][ok] <= eb["df_total"][ok] + 1e-12).all()
    # the family-level tau2 and heterogeneity
    b3, v3 = b[:, :, None], v[:, :, None]
    close(_dl_tau2(b3, v3)[:, 0], f["tau_fam"], "tau_fam")
    close(_heterogeneity_p(b3, v3)[:, 0], f["het_fam"], "het_fam", rtol=1e-9)


def test_heterogeneity_p_is_cochrans_q_and_tau2_follows(R):
    d = R["sl16"]
    fit = fit_from_ref(d["fit"])
    x = fit.index["A"]
    vp = x["v_model"] * x["factor"].to_numpy()[None]
    hp = _heterogeneity_p(x["beta"], vp)
    tau = _dl_tau2(x["beta"], vp)
    assert hp.shape == (len(x["genes"]), len(x["niches"]))
    for g in range(len(x["genes"])):
        for j in range(len(x["niches"])):
            b, w = x["beta"][g, :, j], 1 / vp[g, :, j]
            ok = np.isfinite(b) & np.isfinite(w) & (w > 0)
            if ok.sum() < 2:
                assert np.isnan(hp[g, j])
                continue
            bbar = (w[ok] * b[ok]).sum() / w[ok].sum()
            q = (w[ok] * (b[ok] - bbar) ** 2).sum()
            np.testing.assert_allclose(hp[g, j], stats.chi2.sf(q, ok.sum() - 1), rtol=1e-10)
            # tau2 is positive exactly when Q exceeds its null expectation m - 1
            assert (tau[g, j] > 0) == (q > ok.sum() - 1)
    # one patient is not enough
    b1 = x["beta"].copy()
    b1[0, 1:, 0] = np.nan
    assert np.isnan(_heterogeneity_p(b1, vp)[0, 0])
    # two near-zero variances overflow Q: NA, not p = 0
    e = R["het_edge"]
    bo, vo = (
        np.asarray(e["bo"], float).reshape(1, 3, 1),
        np.asarray(e["vo"], float).reshape(1, 3, 1),
    )
    assert np.isnan(e["p"]) and np.isnan(_heterogeneity_p(bo, vo)[0, 0])


def test_unusable_patient_drops_out_of_the_condition_test_not_the_gene(R):
    fit = fit_from_ref(R["sl16"]["fit"])
    x = fit.index["A"]
    from spidepy.fit import _condition_coding

    trt = _condition_coding(fit.patients, "condition").reindex(x["patients"]).to_numpy()
    G = len(x["genes"])
    b = x["beta"][:, :, 0]
    v = x["v_model"][:, :, 0] * x["factor"].to_numpy()[:, 0][None]
    tau = np.zeros(G)
    ref = _robust_condition_test(b, v, tau, trt)
    v2 = v.copy()
    v2[0, 0] = 0  # weight Inf for one patient of gene 1
    got = _robust_condition_test(b, v2, tau, trt)
    b3 = b.copy()
    b3[0, 0] = np.nan
    assert got["estimate"][0] == _robust_condition_test(b3, v, tau, trt)["estimate"][0]
    np.testing.assert_array_equal(got["estimate"][1:], ref["estimate"][1:])


def test_condition_test_resting_on_one_or_two_heavy_patients_is_not_made(R):
    h = R["heavy"]
    b, v, trt = (np.asarray(h[k], float) for k in ("b", "v", "trt"))
    r = _robust_condition_test(b, v, np.zeros(2), trt)
    _compare(r, h["out"], "heavy")
    assert np.isfinite(r["p"][0]) and np.isnan(r["p"][1])


def test_condition_test_is_independent_of_n_jobs(R):
    fit = fit_from_ref(R["sl16"]["fit"])
    x = fit.index["A"]
    from spidepy.fit import _condition_coding

    trt = _condition_coding(fit.patients, "condition").reindex(x["patients"]).to_numpy()
    b = x["beta"][:, :, 0]
    v = x["v_model"][:, :, 0] * x["factor"].to_numpy()[:, 0][None]
    tau = np.zeros(len(x["genes"]))
    a = _robust_condition_test(b, v, tau, trt)
    c = _robust_condition_test(b, v, tau, trt, n_jobs=2)
    assert a.equals(c)


def test_condition_confounded_with_strata_is_refused(R):
    fit = fit_from_ref(R["sl16"]["fit"])
    x = fit.index["A"]
    from spidepy.fit import _condition_coding

    trt = _condition_coding(fit.patients, "condition").reindex(x["patients"]).to_numpy()
    b = x["beta"][:, :, 0]
    v = x["v_model"][:, :, 0] * x["factor"].to_numpy()[:, 0][None]
    st = np.where(trt == 1, "s1", "s2")
    with pytest.raises(ValueError, match="confounded"):
        _robust_condition_test(b, v, np.zeros(b.shape[0]), trt, strata=st)
    # one stratum level is no strata at all
    one = _robust_condition_test(b, v, np.zeros(b.shape[0]), trt, strata=["s"] * len(trt))
    assert one.equals(_robust_condition_test(b, v, np.zeros(b.shape[0]), trt))


def test_failed_robust_prior_leaves_the_column_na_as_in_r(R, monkeypatch):
    # R retries with robust = FALSE, but its retry reads the `fit` the failed call has just
    # overwritten with NULL, so the retry fails too and every row is NA
    f = R["family"]
    b, v, tau, me = (np.asarray(f[k], float) for k in ("b", "v", "tau", "mean_expr"))
    calls = []

    def boom(fit, robust=False, **kw):
        calls.append(robust)
        raise ValueError("Could not estimate prior")

    import spidepy._test_slopes as ts

    monkeypatch.setattr(ts, "e_bayes", boom)
    out = _pooled_column_test(b, v, tau, me)
    assert calls == [True]
    assert out[["estimate", "se", "t", "df", "p"]].isna().all().all()
    assert (out["n_patients"] == np.isfinite(b).sum(axis=1)).all()


def test_bh_families_follow_the_filters():
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 40
    tab = pd.DataFrame(
        {
            "gene": [f"g{i}" for i in range(n)] * 2,
            "index": "A",
            "niche": "B",
            "test": ["pooled"] * n + ["condition"] * n,
            "p": np.r_[rng.uniform(0, 0.01, 10), rng.uniform(0, 1, n - 10), rng.uniform(0, 1, n)],
            "p_heterogeneity": np.r_[
                rng.uniform(0, 1, 5), rng.uniform(0, 1e-6, 5), rng.uniform(0, 1, 2 * n - 10)
            ],
        }
    )
    tab.loc[n + 3, "p"] = np.nan
    from spidepy._rcompat import p_adjust_bh

    for proc in ("pooled_or_heterogeneity", "heterogeneity", "filtered", "all"):
        t = _bh_families(tab, proc, 0.05)
        pl = t[t["test"] == "pooled"]
        assert (pl["in_family"] == np.isfinite(pl["p"])).all()
        np.testing.assert_allclose(pl["q"], p_adjust_bh(pl["p"]))
        c = t[t["test"] == "condition"]
        assert not c["in_family"].iloc[3]
        pooled_pass = set(pl["gene"][pl["q"] < 0.05])
        het_pass = set(pl["gene"][p_adjust_bh(pl["p_heterogeneity"]) < 0.05])
        want = {
            "pooled_or_heterogeneity": pooled_pass | het_pass,
            "heterogeneity": het_pass,
            "filtered": pooled_pass,
            "all": set(pl["gene"]),
        }[proc]
        fam = c["in_family"].to_numpy()
        assert fam.tolist() == [g in want and np.isfinite(p) for g, p in zip(c["gene"], c["p"])]
        np.testing.assert_allclose(c["q"][fam], p_adjust_bh(c["p"][fam]))
        assert c["q"][~fam].isna().all()
