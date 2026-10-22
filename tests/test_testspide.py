"""``test_spide``, ``spide``, ``SpiDEResults`` and ``results`` against R (``tests/ref/make_test.R``)
and the testthat cases of ``test-engine.R``, ``test-small-n-defaults.R``, ``test-spiDE-e2e.R`` and
the testSpiDE half of ``test-fitSpiDE.R``."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import spidepy
from _fitref import fit_from_ref, frame, strs, table_equal
from _ref import load_ref, load_spe
from spidepy import SpiDEResults, build_niches, fit_spide

PROCS = ("pooled_or_heterogeneity", "heterogeneity", "filtered", "all")
RT = dict(rtol=1e-6)  # end to end: downstream of the iterative NB fit


@pytest.fixture(scope="module")
def R():
    return load_ref("test")


def _toy():
    a = load_spe("toySpiDE")
    num = a.obs["sample_id"].str.slice(1).astype(int)
    a.obs["slide"] = np.where(num % 4 < 2, "slide1", "slide2")
    return a


@pytest.fixture(scope="module")
def toy():
    return build_niches(_toy(), sigma=30)


@pytest.fixture(scope="module")
def spe16():
    return build_niches(load_spe("toyspe_s16_n120_g12_seed2"), sigma=30)


@pytest.fixture(scope="module")
def fit16(spe16):
    return fit_spide(spe16, index="A", sigma=30, verbose=False)


@pytest.fixture(scope="module")
def dropped(toy):
    a = toy.copy()
    ct = a.obs["cell_type"].to_numpy().astype(object)
    i = np.flatnonzero((a.obs["sample_id"].to_numpy() == "S1") & (ct == "A"))
    ct[i[3:]] = "B"
    a.obs["cell_type"] = ct
    return build_niches(a, sigma=30)


def _meta(res, r, what):
    assert isinstance(res, SpiDEResults)
    assert (res.condition or "") == r["condition"], what
    assert res.contrast == r["contrast"], what
    assert res.procedure == r["procedure"], what
    assert res.fdr == r["fdr"], what


def _check(res, r, what, rtol=1e-8):
    table_equal(res.table, r["table"], what, rtol)
    _meta(res, r, what)


# --------------------------------------------------------------------------------------
# the full table on R's own fit
# --------------------------------------------------------------------------------------
def test_slopes_tables_on_r_fit_every_procedure(R):
    fit = fit_from_ref(R["sl16"]["fit"])
    d = R["sl16"]
    _check(spidepy.test_spide(fit, condition="condition"), d["default"], "default")
    _check(spidepy.test_spide(fit), d["nocond"], "nocond")
    for proc, r in d["variants"].items():
        res = spidepy.test_spide(fit, condition="condition", procedure=proc)
        _check(res, r, proc)


def test_slopes_tables_with_strata_recorded_and_named_at_test_time(R):
    d = R["toy_strata"]
    fit = fit_from_ref(d["fit"])
    assert fit.params["strata"] == "slide"
    _check(spidepy.test_spide(fit), d["default"], "recorded strata")
    for proc, r in d["variants"].items():
        res = spidepy.test_spide(fit, condition="condition", procedure=proc)
        _check(res, r, proc)
    d0 = R["toy_testtime"]
    fit0 = fit_from_ref(d0["fit"])
    res = spidepy.test_spide(fit0, condition="condition", procedure="all", strata="slide")
    _check(res, d0["all_slide"], "test-time strata")
    # the strata really enter: the condition rows differ from the unadjusted ones
    plain = spidepy.test_spide(fit0, condition="condition", procedure="all").table
    c = res.table["test"] == "condition"
    assert not np.allclose(res.table.loc[c, "estimate"], plain.loc[c, "estimate"], equal_nan=True)


def test_dropped_patient_labels_are_looked_up_by_name(R):
    # S1 is dropped from index type A only: the other index types still have all patients, and
    # the condition and strata of A's remaining patients must come from their own names
    d = R["dropped_slopes"]
    fit = fit_from_ref(d["fit"])
    assert "S1" not in fit.index["A"]["patients"]
    assert "S1" in fit.index["B"]["patients"]
    _check(spidepy.test_spide(fit), d["default"], "dropped default")
    _check(spidepy.test_spide(fit, procedure="all"), d["all"], "dropped all")


def test_sandwich_tables_on_r_fit(R):
    d = R["sand"]
    fit = fit_from_ref(d["fit"])
    _check(spidepy.test_spide(fit), d["default"], "sand default")
    _check(spidepy.test_spide(fit, procedure="all"), d["all"], "sand all")
    res0 = spidepy.test_spide(fit_from_ref(d["fit0"]))
    _check(res0, d["pooled_only"], "sand pooled only")
    assert set(res0.table["test"]) == {"pooled"}
    assert res0.table["p.heterogeneity"].isna().all()


def test_big_fixture_table_on_r_fit(R):
    d = R["big"]
    res = spidepy.test_spide(fit_from_ref(d["fit"]), condition="condition", procedure="all")
    _check(res, d["all"], "big")


def test_fit_records_patient_columns_so_a_condition_can_be_named_later(R):
    # test-fitSpiDE.R: patient-level columns are recorded, no condition, testSpiDE names one
    d = R["default_toy"]
    a = build_niches(load_spe("toyspe_default"), sigma=20)
    fit = fit_spide(a, index="A", sigma=20, min_patients=6, verbose=False)
    assert {"patient", "ncells", "condition"} <= set(fit.patients.columns)
    assert fit.condition is None
    res = spidepy.test_spide(fit, condition="condition", procedure="all")
    assert res.contrast == "Responder - Non-responder"
    assert set(res.table["test"]) == {"pooled", "condition"}
    _check(res, d["all"], "default toy", rtol=1e-6)


# --------------------------------------------------------------------------------------
# results() and show
# --------------------------------------------------------------------------------------
def test_results_accessor_matches_r(R, fit16):
    r = R["sl16"]["results"]
    res = spidepy.test_spide(fit16, condition="condition")

    def eq(got, ref, what):
        table_equal(got, ref, what, rtol=1e-6)

    eq(spidepy.results(res), r["default"], "default")
    eq(res.results(), r["default"], "method")
    eq(spidepy.results(res, test="pooled"), r["pooled"], "pooled")
    eq(spidepy.results(res, test="both"), r["both"], "both")
    eq(spidepy.results(res, test="condition", fdr=0.1), r["cond_fdr"], "cond_fdr")
    eq(spidepy.results(res, test="pooled", fdr=0.05), r["pooled_fdr"], "pooled_fdr")
    eq(spidepy.results(res, fdr=1), r["fdr1"], "fdr1")
    eq(spidepy.results(spidepy.test_spide(fit16)), r["nocond"], "nocond")
    with pytest.raises(ValueError, match="'test' should be one of"):
        spidepy.results(res, test="x")
    with pytest.raises(TypeError, match="needs a SpiDEResults"):
        spidepy.results(fit16)
    # ordered by p with missing last
    p = spidepy.results(res, test="both")["p"].to_numpy()
    fin = np.isfinite(p)
    assert (np.diff(p[fin]) >= 0).all() and fin[: fin.sum()].all()


def test_show_matches_r(R, fit16):
    r = R["sl16"]
    assert repr(spidepy.test_spide(fit16, condition="condition")) == "\n".join(strs(r["show"]))
    assert "df)" not in repr(spidepy.test_spide(fit16, condition="condition"))
    fs = fit_from_ref(R["sand"]["fit"])
    assert repr(spidepy.test_spide(fs)) == "\n".join(strs(R["sand"]["show"]))


# --------------------------------------------------------------------------------------
# end to end from AnnData
# --------------------------------------------------------------------------------------
def _e2e(res, r, what, noisy_rows=None):
    """Table at 1e-6. ``noisy_rows`` (a mask over rows) holds the pooled rows whose robust prior is
    tie-noise sensitive (see ``NOISY`` in test_test_slopes.py): se, t, df and p of those rows
    are compared at 2e-2 (0.53% seen across numpy / BLAS builds in CI) and the rest of the table, including their estimate, at 1e-6. Their
    noise reaches ``q`` of the pooled rows that rank at or above them in the BH step-up; only
    those q are relaxed. The rejected set (``in_family`` and ``q <= fdr``) must match R exactly."""
    got = res.table
    ref = frame(r["table"])
    rej = lambda t: (t["in_family"].astype(bool) & (t["q"] <= r["fdr"])).to_numpy()  # noqa: E731
    np.testing.assert_array_equal(rej(got), rej(ref), err_msg=what + " rejected set")
    if noisy_rows is not None:
        got = got.copy()
        pooled = (ref["test"] == "pooled").to_numpy()
        affected = pooled & (ref["p"].to_numpy() <= ref.loc[noisy_rows, "p"].max())
        for c, rows in (("se", noisy_rows), ("t", noisy_rows), ("df", noisy_rows),
                        ("p", noisy_rows), ("q", affected)):  # fmt: skip
            np.testing.assert_allclose(
                got.loc[rows, c], ref.loc[rows, c], rtol=2e-2, err_msg=what + c
            )
            got.loc[rows, c] = ref.loc[rows, c].to_numpy()
    table_equal(got, r["table"], what, rtol=1e-6)
    _meta(res, r, what)


def test_spide_end_to_end_matches_r(R, toy):
    sp7 = load_spe("toyspe_s16_n150_g12_seed7")
    e = R["e2e"]
    res = spidepy.spide(sp7, "condition", 30, index="A", procedure="all", verbose=False)
    t = res.table
    _e2e(res, e["slopes"], "slopes", ((t["test"] == "pooled") & (t["niche"] == "B")).to_numpy())
    _e2e(
        spidepy.spide(
            sp7, "condition", 30, index="A", engine="sandwich", procedure="all", verbose=False
        ),
        e["sandwich"],
        "sandwich",
    )
    a = load_spe("toySpiDE")
    before = set(a.obsm)
    _e2e(spidepy.spide(a, "condition", 30, verbose=False), e["toy_default"], "toy default")
    assert set(a.obsm) == before  # niches built for the run are not left on the data
    _e2e(
        spidepy.spide(
            a,
            "condition",
            30,
            index="A",
            procedure="heterogeneity",
            fdr=0.1,
            verbose=False,
        ),
        e["toy_het"],
        "toy heterogeneity",
    )


def test_spide_uses_niches_the_data_already_carries(toy):
    got = spidepy.spide(toy, "condition", 30, index="A", verbose=False)
    ref = spidepy.test_spide(fit_spide(toy, "condition", index="A", sigma=30, verbose=False))
    pd.testing.assert_frame_equal(got.table, ref.table)


def test_spide_name_reads_or_builds_niches_under_a_custom_prefix(toy):
    kw = dict(condition="condition", sigma=30, index="A", procedure="all", verbose=False)
    ref = spidepy.spide(toy, **kw)
    # niches stored under the custom prefix and no "Niche30": used as they are
    a = load_spe("toySpiDE")
    build_niches(a, sigma=30, name="Nb")
    assert [k for k in a.obsm if k.startswith("N")] == ["Nb30"]
    got = spidepy.spide(a, name="Nb", **kw)
    pd.testing.assert_frame_equal(got.table, ref.table)
    assert [k for k in a.obsm if k.startswith("N")] == ["Nb30"]
    # absent: built under the prefix for the run and not left on the data
    b = load_spe("toySpiDE")
    got2 = spidepy.spide(b, name="Nb", **kw)
    pd.testing.assert_frame_equal(got2.table, ref.table)
    assert not [k for k in b.obsm if k.startswith("N")]


def test_slopes_engine_recovers_the_planted_effect():
    sp = load_spe("toyspe_s16_n150_g12_seed7")
    res = spidepy.spide(sp, "condition", 30, index="A", procedure="all", verbose=False)
    cond = spidepy.results(res, test="condition")
    pool = spidepy.results(res, test="pooled")
    top = cond[cond["gene"] == "G1"]
    assert top["niche"].iloc[top["p"].argmin()] == "B"
    assert top["q"].min() < 0.05
    assert top.loc[top["niche"] == "B", "estimate"].iloc[0] > 0
    assert pool.loc[(pool["gene"] == "G1") & (pool["niche"] == "B"), "estimate"].iloc[0] > 0


def test_sandwich_engine_recovers_it_too():
    sp = load_spe("toyspe_s16_n150_g12_seed7")
    res = spidepy.spide(
        sp, "condition", 30, index="A", engine="sandwich", procedure="all", verbose=False
    )
    top = spidepy.results(res, test="condition")
    top = top[top["gene"] == "G1"]
    assert top["niche"].iloc[top["p"].argmin()] == "B"
    assert top["q"].min() < 0.05


def test_sandwich_refuses_a_condition_it_was_not_fitted_with():
    sp = build_niches(load_spe("toyspe_s16_n150_g12_seed7"), sigma=30)
    fs = fit_spide(sp, engine="sandwich", index="A", sigma=30, verbose=False)
    with pytest.raises(ValueError, match="fitted with"):
        spidepy.test_spide(fs, condition="condition")
    assert set(spidepy.test_spide(fs).table["test"]) == {"pooled"}


# --------------------------------------------------------------------------------------
# testthat cases on the 16-patient fit
# --------------------------------------------------------------------------------------
def test_condition_test_does_not_use_a_genes_own_tile_variance(fit16):
    import copy

    f2 = copy.deepcopy(fit16)
    rng = np.random.default_rng(9)
    f2.index["A"]["v_tile"] = fit16.index["A"]["v_tile"] * rng.uniform(
        0.2, 5, fit16.index["A"]["v_tile"].shape
    )
    r1 = spidepy.test_spide(fit16, condition="condition", procedure="all").table
    r2 = spidepy.test_spide(f2, condition="condition", procedure="all").table
    pd.testing.assert_series_equal(r2["p"], r1["p"], rtol=1e-12)
    pd.testing.assert_series_equal(r2["estimate"], r1["estimate"], rtol=1e-12)


def test_pooled_test_and_its_filter_never_look_at_the_condition(fit16):
    import copy

    r1 = spidepy.test_spide(fit16, condition="condition")
    pt = fit16.patients
    lab = pd.Series(
        np.random.default_rng(4).permutation(pt["condition"].astype(str)), index=pt["patient"]
    )
    fp = copy.deepcopy(fit16)
    fp.patients["perm"] = pd.Categorical(lab[pt["patient"]].to_numpy())
    r2 = spidepy.test_spide(fp, condition="perm")
    cols = ["gene", "niche", "p", "q", "p.heterogeneity"]
    pd.testing.assert_frame_equal(
        r1.table.query("test == 'pooled'")[cols], r2.table.query("test == 'pooled'")[cols]
    )
    for proc in ("pooled_or_heterogeneity", "heterogeneity", "filtered"):
        f1 = spidepy.test_spide(fit16, condition="condition", procedure=proc).table
        f2 = spidepy.test_spide(fp, condition="perm", procedure=proc).table

        def fam(tb):
            s = tb[(tb["test"] == "condition") & tb["in_family"]]
            return sorted(s["gene"] + " " + s["niche"])

        assert fam(f1) == fam(f2)


def test_pooled_alone_without_a_condition(fit16):
    r0 = spidepy.test_spide(fit16)
    assert set(r0.table["test"]) == {"pooled"}
    assert set(spidepy.results(r0)["test"]) == {"pooled"}
    r1 = spidepy.test_spide(fit16, condition="condition", procedure="filtered")
    cond = r1.table.query("test == 'condition'")
    pool = r1.table.query("test == 'pooled'")
    pass_ = set((pool["gene"] + " " + pool["niche"])[pool["q"] < 0.05])
    assert set((cond["gene"] + " " + cond["niche"])[cond["in_family"]]) <= pass_


def test_default_is_the_pooled_or_heterogeneity_family(fit16):
    from spidepy._rcompat import p_adjust_bh

    r = spidepy.test_spide(fit16, condition="condition")
    assert r.procedure == "pooled_or_heterogeneity"
    pd.testing.assert_frame_equal(
        r.table,
        spidepy.test_spide(fit16, condition="condition", procedure="pooled_or_heterogeneity").table,
    )
    tb = r.table
    pl = tb[(tb["test"] == "pooled") & tb["p.heterogeneity"].notna()]
    key = lambda d: list(zip(d["gene"], d["index"], d["niche"]))  # noqa: E731
    het = {k for k, q in zip(key(pl), p_adjust_bh(pl["p.heterogeneity"])) if q < r.fdr}
    pp = tb[(tb["test"] == "pooled") & tb["p"].notna()]
    pooled = {k for k, q in zip(key(pp), pp["q"]) if q < r.fdr}
    cond = tb[tb["test"] == "condition"]

    def fam(proc):
        t = spidepy.test_spide(fit16, condition="condition", procedure=proc).table
        return t[t["test"] == "condition"]

    for proc in ("pooled_or_heterogeneity", "heterogeneity", "filtered"):
        pass_ = {"pooled_or_heterogeneity": pooled | het, "heterogeneity": het, "filtered": pooled}[
            proc
        ]
        cf = fam(proc)
        want = [np.isfinite(p) and k in pass_ for k, p in zip(key(cf), cf["p"])]
        assert cf["in_family"].tolist() == want
        fm = cf["in_family"].to_numpy()
        np.testing.assert_allclose(cf["q"][fm], p_adjust_bh(cf["p"][fm]))
        assert cf["q"][~fm].isna().all()
    # the default family contains every triplet the previous default tested
    old = fam("filtered")
    assert cond["in_family"].to_numpy()[old["in_family"].to_numpy()].all()
    # each condition row carries its triplet's heterogeneity p
    pk = {
        k: h
        for k, h in zip(
            key(tb[tb["test"] == "pooled"]), tb.loc[tb["test"] == "pooled", "p.heterogeneity"]
        )
    }
    expected = np.array([pk[k] for k in key(cond)])
    np.testing.assert_array_equal(cond["p.heterogeneity"].to_numpy(), expected)


def test_previous_family_stays_available_and_only_the_family_moves(fit16):
    new = spidepy.test_spide(fit16, condition="condition").table
    old = spidepy.test_spide(fit16, condition="condition", procedure="filtered").table
    cl = (new["test"] == "condition").to_numpy()
    cols = ["gene", "niche", "estimate", "se", "t", "df", "p"]
    pd.testing.assert_frame_equal(new.loc[cl, cols], old.loc[cl, cols])
    pl = ~cl
    pd.testing.assert_series_equal(new.loc[pl, "t"], old.loc[pl, "t"])
    pk = set((old["gene"] + " " + old["niche"])[pl & (old["q"] < 0.05)])
    assert set((old["gene"] + " " + old["niche"])[cl & old["in_family"]]) <= pk


def test_sandwich_keeps_the_pooled_filter_and_refuses_the_heterogeneity_filter():
    sp = build_niches(load_spe("toyspe_s16_n120_g12_seed2"), sigma=30)
    fs = fit_spide(sp, condition="condition", index="A", sigma=30, engine="sandwich", verbose=False)
    r = spidepy.test_spide(fs)
    assert r.procedure == "filtered"
    assert r.table["p.heterogeneity"].isna().all()
    for proc in ("heterogeneity", "pooled_or_heterogeneity"):
        with pytest.raises(ValueError, match="slopes engine"):
            spidepy.test_spide(fs, procedure=proc)
    # spide() refuses the combination before it fits anything
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="slopes engine"):
        spidepy.spide(
            sp,
            "condition",
            30,
            index="A",
            engine="sandwich",
            procedure="heterogeneity",
            verbose=False,
        )
    assert time.perf_counter() - t0 < 2


def test_argument_errors(fit16, toy):
    with pytest.raises(ValueError, match="'procedure' should be one of"):
        spidepy.test_spide(fit16, procedure="nope")
    for bad in (0, 1.5, -1, np.nan, "x"):
        with pytest.raises(ValueError, match="'fdr' should be"):
            spidepy.test_spide(fit16, fdr=bad)
    with pytest.raises(ValueError, match="not a patient-level column"):
        spidepy.test_spide(fit16, condition="nope")
    with pytest.raises(ValueError, match="'sigma' must be a single bandwidth"):
        spidepy.spide(toy, "condition", verbose=False)
    with pytest.raises(ValueError, match="'sigma' must be a single bandwidth"):
        spidepy.spide(toy, "condition", [10, 30], verbose=False)
    with pytest.raises(TypeError):
        spidepy.test_spide("fit")


def test_results_identical_for_any_n_jobs(fit16):
    r1 = spidepy.test_spide(fit16, condition="condition").table
    r2 = spidepy.test_spide(fit16, condition="condition", n_jobs=2).table
    pd.testing.assert_frame_equal(r1, r2)


def test_test_spide_is_not_collected_by_pytest():
    assert spidepy.test_spide.__test__ is False
