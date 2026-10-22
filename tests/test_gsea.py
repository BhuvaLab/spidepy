"""``spi_gsea`` against R (``tests/ref/make_gsea.R``) and the testthat cases of
``test-spiGSEA.R``. The fits are R's own (``test.h5``), so Python tests R's slopes."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import spidepy
from _fitref import close, fit_from_ref, pvals, strs
from _ref import load_ref, load_spe
from spidepy import build_niches, spi_gsea, test_spide
from spidepy.gsea import _set_slopes

RTOL = 1e-8


@pytest.fixture(scope="module")
def R():
    return load_ref("gsea")


@pytest.fixture(scope="module")
def T():
    return load_ref("test")


def sets_of(d):
    return {k: strs(v) for k, v in d.items()}


def res_e2e(T, **kw):
    return test_spide(
        fit_from_ref(T["e2e_fit"]["fit"]), condition="condition", procedure="all", **kw
    )


def compare(got, ref, what, rtol=RTOL):
    if "empty" in ref:
        assert len(got) == 0, what
        return
    cols = strs(ref["colnames__"])
    assert list(got.columns) == cols, what
    for c in cols:
        v = np.atleast_1d(ref[c])
        assert len(got) == len(v), f"{what}.{c}: length"
        if c in ("set", "index", "niche", "test"):
            assert [str(s) for s in got[c]] == strs(v), f"{what}.{c}"
        elif c == "direction":
            ex = [
                None if str(s) == "NA" else str(s) for s in v
            ]  # R NA_character_ is written as "NA"
            assert [None if pd.isna(s) else s for s in got[c]] == ex, f"{what}.{c}"
        elif c in ("p", "q", "q.global"):
            pvals(got[c], v, f"{what}.{c}", rtol)
        elif c == "size":
            np.testing.assert_array_equal(got[c].to_numpy(), v, err_msg=f"{what}.{c}")
        else:
            close(got[c], v, f"{what}.{c}", rtol=rtol, atol=1e-10)


def test_set_slopes_on_r_inputs(R, T):
    x = fit_from_ref(T["e2e_fit"]["fit"]).index["A"]
    genes = x["genes"]
    sets = sets_of(R["e2e"]["sets"])
    mem = [np.flatnonzero(np.isin(genes, sets[n])) for n in ("with_g1", "without")]
    for key, typ in (("comp", "competitive"), ("self", "self-contained")):
        got = _set_slopes(x, mem, typ)
        close(got["b"], R["setslopes"][key]["b"], f"b {typ}", rtol=1e-10)
        close(got["v"], R["setslopes"][key]["v"], f"v {typ}", rtol=1e-10)
        close(got["mean_expr"], R["setslopes"][key]["mean_expr"], f"me {typ}", rtol=1e-10)


def test_set_slope_is_mean_of_scaled_slopes_less_background(T):
    from spidepy._test_slopes import _dl_tau2

    x = fit_from_ref(T["e2e_fit"]["fit"]).index["A"]
    j = 0
    vp = x["v_model"] * x["factor"].to_numpy()[None]
    tau = _dl_tau2(x["beta"], vp)[:, j]
    b, v = x["beta"][:, :, j], vp[:, :, j]
    sig = np.sqrt(tau + np.nanmedian(v, axis=1))
    u = b / sig[:, None]
    g = np.arange(3)
    rest = np.arange(3, len(x["genes"]))
    self_ = _set_slopes(x, [g], "self-contained")["b"][0, :, j]
    comp = _set_slopes(x, [g], "competitive")["b"][0, :, j]
    np.testing.assert_allclose(self_, u[g].mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(comp, u[g].mean(axis=0) - u[rest].mean(axis=0), rtol=1e-12)


@pytest.mark.parametrize(
    "name,kw",
    [
        ("default", dict(min_size=3)),
        ("self", dict(type="self-contained", min_size=3)),
        ("pooled", dict(test="pooled", min_size=3)),
        ("cond", dict(test="condition", min_size=3)),
        ("size_default", dict()),
        ("max2", dict(min_size=1, max_size=3)),
        ("none", dict(min_size=50)),
    ],
)
def test_e2e_fit_tables(R, T, name, kw):
    sets = sets_of(R["e2e"]["sets"])
    compare(spi_gsea(res_e2e(T), sets, **kw), R["e2e"][name], name)


def test_the_capped_df_option_is_gone(R, T):
    res = res_e2e(T)
    assert not hasattr(res, "pooled_df")
    with pytest.raises(TypeError):
        spi_gsea(res, sets_of(R["e2e"]["sets"]), pooled_df="capped")


def test_trended_prior_for_20_or_more_sets(R, T):
    sets = sets_of(R["big"]["sets"])
    assert len(sets) >= 20
    compare(spi_gsea(res_e2e(T), sets, min_size=3), R["big"]["out"], "big")


def test_sl16_condition_named_at_test_time(R, T):
    fit = fit_from_ref(T["sl16"]["fit"])
    sets = sets_of(R["sl16"]["sets"])
    res = test_spide(fit, condition="condition", procedure="all")
    compare(spi_gsea(res, sets, min_size=3), R["sl16"]["out"], "sl16")
    compare(spi_gsea(test_spide(fit), sets, min_size=3), R["sl16"]["nocond"], "nocond")


def test_toy_strata_three_index_types(R, T):
    fit = fit_from_ref(T["toy_strata"]["fit"])
    assert fit.params["strata"] == "slide"
    sets = sets_of(R["toy"]["sets"])
    res = test_spide(fit, procedure="all")
    got = spi_gsea(res, sets)
    compare(got, R["toy"]["out"], "toy")
    assert set(got["index"]) == {"A", "B", "C"}
    compare(
        spi_gsea(res, sets, type="self-contained", test="condition"), R["toy"]["self"], "toy self"
    )


def test_dropped_patient_looked_up_by_name(R, T):
    fit = fit_from_ref(T["dropped_slopes"]["fit"])
    assert len(fit.index["A"]["patients"]) == 15 and len(fit.index["B"]["patients"]) == 16
    sets = sets_of(R["dropped"]["sets"])
    compare(spi_gsea(test_spide(fit, procedure="all"), sets), R["dropped"]["out"], "dropped")


def test_dead_set_drops_without_taking_others(R, T):
    fit = fit_from_ref(T["e2e_fit"]["fit"])
    x = fit.index["A"]
    jB = x["niches"].index("B")
    x["beta"][:2, :, jB] = np.nan
    sets = {
        "dead": x["genes"][:2],
        "setA": x["genes"][3:6],
        "setB": x["genes"][6:9],
    }
    res = test_spide(fit, condition="condition", procedure="all")
    got = spi_gsea(res, sets, min_size=2, test="pooled")
    compare(got, R["dead"]["out"], "dead")
    b = got[got["niche"] == "B"].set_index("set")
    assert np.isnan(b.loc["dead", "p"])
    assert np.isfinite(b.loc[["setA", "setB"], "p"]).all()
    assert pd.isna(b.loc["dead", "direction"])
    compare(spi_gsea(res, sets, min_size=2), R["dead"]["both"], "dead both")


def test_rows_columns_and_bounds(T):
    sets = {
        "with_g1": [f"G{i}" for i in range(1, 4)],
        "without": [f"G{i}" for i in range(4, 9)],
        "rest": [f"G{i}" for i in range(9, 13)],
        "tiny": ["G1"],
    }
    gs = spi_gsea(res_e2e(T), sets, min_size=3)
    assert "tiny" not in set(gs["set"])
    assert set(gs["test"]) == {"pooled", "condition"}
    nn = len(res_e2e(T).fit.index["A"]["niches"])
    assert len(gs) == 3 * nn * 2
    ok = np.isfinite(gs["p"])
    assert ((gs["p"][ok] >= 0) & (gs["p"][ok] <= 1)).all()
    assert (gs["q"][ok] >= gs["p"][ok] - 1e-12).all()
    # pooled rows first, each block ordered by p
    assert list(gs["test"]) == sorted(gs["test"], key=lambda s: s != "pooled")
    for _, g in gs.groupby("test"):
        assert g["p"].dropna().is_monotonic_increasing


def test_pooled_set_test_never_looks_at_condition(T):
    fit = fit_from_ref(T["e2e_fit"]["fit"])
    pt = fit.patients.copy()
    rng = np.random.default_rng(4)
    pt["perm"] = pd.Categorical(rng.permutation(pt["condition"].astype(str).to_numpy()))
    fit2 = copy.copy(fit)
    fit2.patients = pt
    sets = {"a": ["G1", "G2", "G3"], "b": [f"G{i}" for i in range(4, 9)]}
    g1 = spi_gsea(
        test_spide(fit, condition="condition", procedure="all"), sets, test="pooled", min_size=3
    )
    g2 = spi_gsea(
        test_spide(fit2, condition="perm", procedure="all"), sets, test="pooled", min_size=3
    )
    np.testing.assert_allclose(g2["p"], g1["p"], rtol=1e-12)


def test_refusals(T):
    sets = {"a": ["G1", "G2", "G3", "G4", "G5"]}
    res = res_e2e(T)
    sand = test_spide(fit_from_ref(T["sand"]["fit"]))
    with pytest.raises(ValueError, match="slopes engine"):
        spi_gsea(sand, sets)
    pooled_only = test_spide(fit_from_ref(T["sl16"]["fit"]))
    assert not pooled_only.condition
    with pytest.raises(ValueError, match="condition"):
        spi_gsea(pooled_only, sets, test="condition")
    with pytest.raises(ValueError, match="named list"):
        spi_gsea(res, [["G1", "G2", "G3", "G4"]])
    with pytest.raises(ValueError, match="should be one of"):
        spi_gsea(res, sets, type="other")
    with pytest.raises(ValueError, match="should be one of"):
        spi_gsea(res, sets, test="both")
    with pytest.raises(TypeError):
        spi_gsea(res.fit, sets)


def test_n_jobs_does_not_change_results(T):
    sets = {"a": ["G1", "G2", "G3"], "b": [f"G{i}" for i in range(4, 9)]}
    res = res_e2e(T)
    pd.testing.assert_frame_equal(
        spi_gsea(res, sets, min_size=3), spi_gsea(res, sets, min_size=3, n_jobs=2)
    )


def test_end_to_end_planted_set_stands_out():
    spe = build_niches(load_spe("toyspe_s16_n150_seed7"), sigma=30)
    res = spidepy.spide(spe, condition="condition", sigma=30, index="A", procedure="all")
    sets = {
        "with_g1": [f"G{i}" for i in range(1, 4)],
        "without": [f"G{i}" for i in range(4, 9)],
        "rest": [f"G{i}" for i in range(9, 13)],
    }
    gs = spi_gsea(res, sets, test="condition", min_size=3)
    b = gs[gs["niche"] == "B"].set_index("set")
    assert b.loc["with_g1", "p"] < b.drop("with_g1")["p"].min()
    assert b.loc["with_g1", "estimate"] > 0
