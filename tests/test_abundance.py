"""``test_niche_abundance`` and its helpers against R (``tests/ref/make_abundance.R``) and the
testthat cases of ``test-niche-abundance.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spidepy
from _fitref import close, pvals, strs
from _ref import load_ref, load_spe
from spidepy import abundance as ab
from spidepy import build_niches
from spidepy._io import get_counts, get_obs

RT = dict(rtol=1e-10, atol=1e-12)  # closed form
RI = 1e-8  # downstream of limma's robust prior (Brent search on long-double sums in R)
FIX = "toyspe_s12_n200_g12_comp3_seed3"
GS = {
    "with_G2": ["G2", "G3", "G4", "G5", "G6"],
    "other": ["G7", "G8", "G9", "G10", "G11"],
    "tiny": ["G1", "G2"],
    "wide": [f"G{i}" for i in range(1, 13)],
}


@pytest.fixture(scope="module")
def R():
    return load_ref("abundance")


@pytest.fixture(scope="module")
def adata(R):
    """The testthat fixture with R's own niche matrix at sigma = 50."""
    a = load_spe(FIX)
    a.obsm["Niche50"] = pd.DataFrame(
        np.asarray(R["niche50"], float), index=a.obs_names, columns=strs(R["niche_cols"])
    )
    return a


@pytest.fixture(scope="module")
def adata_py():
    """The same fixture with spidepy's own niches (the testthat cases)."""
    return build_niches(load_spe(FIX), sigma=50)


def _inputs(a):
    ct = ab._sanitise_array(get_obs(a, "cell_type"))
    return get_counts(a), a.obsm["Niche50"], ct, get_obs(a, "sample_id")


# ---- .pseudobulkByIndex ---------------------------------------------------------------------
PB = {
    "all": dict(index=None, min_cells=10, prior_count=1),
    "strict": dict(index=None, min_cells=70, prior_count=1),
    "idx": dict(index=["C", "A", "zz"], min_cells=10, prior_count=0.5),
    "huge": dict(index=None, min_cells=1000, prior_count=1),
}


@pytest.mark.parametrize("case", list(PB))
def test_pseudobulk_matches_r(adata, R, case):
    r = R["pb"][case]
    got = ab._pseudobulk_by_index(*_inputs(adata), **PB[case])
    assert list(got) == [t for t in strs(r["types"]) if t]
    for k, b in got.items():
        rk = r[k]
        assert b["samples"] == strs(rk["samples"])
        assert list(b["niche"].index) == strs(rk["niche_rows"])
        close(b["Y"], rk["Y"], f"{case}.{k}.Y", **RT)
        close(b["niche"], rk["niche"], f"{case}.{k}.niche", **RT)
        close(b["ncells"], rk["ncells"], f"{case}.{k}.ncells", **RT)


def test_pseudobulk_drops_thin_samples_for_one_index_type(adata, R):
    got = ab._pseudobulk_by_index(*_inputs(adata), **PB["strict"])
    full = ab._pseudobulk_by_index(*_inputs(adata), **PB["all"])
    assert [len(got[k]["samples"]) for k in got] != [len(full[k]["samples"]) for k in got]
    assert "B" not in got  # fewer than three samples left


def test_pseudobulk_sparse_equals_dense(adata):
    import scipy.sparse as sp

    Y, NM, ct, smp = _inputs(adata)
    d = ab._pseudobulk_by_index(Y, NM, ct, smp, None, 10, 1)
    s = ab._pseudobulk_by_index(sp.csc_matrix(Y), NM, ct, smp, None, 10, 1)
    for k in d:
        np.testing.assert_allclose(s[k]["Y"], d[k]["Y"], **RT)


# ---- .patientValue --------------------------------------------------------------------------
def test_patient_value_matches_r(R):
    r = R["pv"]
    pat, pats = strs(r["pat"]), strs(r["pats"])
    num = ab._patient_value(np.asarray(r["num_x"], float), pat, pats)
    assert list(num.index) == pats
    close(num.to_numpy(float), r["num"], "num", **RT)
    chr_ = ab._patient_value(
        pd.Series([None if s == "NA" else s for s in strs(r["chr_x"])]), pat, pats
    )
    assert list(chr_) == strs(r["chr"])
    # looked up by name, never by position
    assert chr_["S1"] == "b" and chr_["S3"] == "d"


def test_patient_value_keeps_factor_and_missing_patient():
    x = pd.Series(pd.Categorical(["lo", "hi", None, "hi"], categories=["lo", "hi"]))
    v = ab._patient_value(x, ["p", "p", "q", "q"], ["q", "p", "z"])
    assert isinstance(v.dtype, pd.CategoricalDtype)
    assert list(v.index) == ["q", "p", "z"]
    assert v["q"] == "hi" and v["p"] == "lo" and pd.isna(v["z"])


# ---- stats::hat and .capLeverage ------------------------------------------------------------
def test_hat_and_cap_leverage_match_r(R):
    for name, c in R["cap"].items():
        X = np.asarray(c["X"], float)
        m = float(c["m"])
        close(ab._hat(X), c["hat"], f"{name}.hat", **RT)
        cap = ab._cap_leverage(X, m)
        close(cap["w"], c["w"], f"{name}.w", **RT)
        assert cap["leverage"] == pytest.approx(float(c["leverage"]), rel=1e-10)
        assert cap["downweighted"] == int(c["downweighted"]), name
        close(ab._hat(X * np.sqrt(cap["w"])[:, None]), c["hat_w"], f"{name}.hat_w", **RT)


def test_hat_rank_deficient_matches_r(R):
    close(ab._hat(np.asarray(R["hat_rankdef"]["X"], float)), R["hat_rankdef"]["hat"], "hat", **RT)


# ---- .abundancePair -------------------------------------------------------------------------
def _pair_df(c):
    df = pd.DataFrame({"niche": np.atleast_1d(c["niche"]).astype(float)})
    if c["has_cond"]:
        df["condition"] = pd.Categorical(
            [None if s == "NA" else s for s in strs(c["cond"])], categories=strs(c["cond_levels"])
        )
    for key in c:
        if key.startswith("cv_"):
            v = np.atleast_1d(c[key])
            df[key[3:]] = v.astype(float) if v.dtype.kind in "iuf" else v.astype(object)
    return df


@pytest.mark.parametrize(
    "case",
    ["plain", "one_cond", "cov", "cov_cond", "nocond_cap", "square", "square_pooled", "small",
     "na_cov", "na_cond", "too_few"],
)  # fmt: skip
def test_abundance_pair_matches_r(R, case):
    c = R["pair"][case]
    cov = [s for s in strs(c["cov"]) if s]
    r = ab._abundance_pair(np.asarray(c["Y"], float), _pair_df(c), cov, float(c["m"]))
    if c["null"]:
        assert r is None
        return
    assert list(r["terms"]) == strs(c["terms"])
    for tm in r["terms"]:
        t = c[f"t_{tm}"]
        close(r["terms"][tm]["estimate"], t["estimate"], f"{case}.{tm}.estimate", **RT)
        close(r["terms"][tm]["t"], t["t"], f"{case}.{tm}.t", rtol=RI, atol=1e-10)
        pvals(r["terms"][tm]["p"], t["p"], f"{case}.{tm}.p", rtol=RI)
    assert r["n"] == int(c["n"])
    assert r["leverage"] == pytest.approx(float(c["leverage"]), rel=1e-10)
    assert r["downweighted"] == int(c["downweighted"])


# ---- .abundanceSetScores --------------------------------------------------------------------
SETS = {
    "a": [f"g{i}" for i in range(1, 7)],
    "b": [f"g{i}" for i in range(4, 13)],
    "small": ["g1", "g2"],
    "absent": ["x", "y", "z", "w", "v"],
    "all": [f"g{i}" for i in range(1, 31)],
    "big": [f"g{i}" for i in range(1, 21)],
}
SC = {
    "comp": ("competitive", 5, 500),
    "self": ("self-contained", 5, 500),
    "capped": ("competitive", 5, 10),
    "none": ("competitive", 50, 500),
}


@pytest.mark.parametrize("case", list(SC))
def test_set_scores_match_r(R, case):
    Y, genes = np.asarray(R["sets_in"]["Y"], float), strs(R["sets_in"]["genes"])
    r = R["sets"][case]
    got = ab._abundance_set_scores(Y, genes, SETS, *SC[case])
    if r["null"]:
        assert got is None
        return
    assert got["sets"] == strs(r["sets"])
    close(got["S"], r["S"], case, **RT)
    np.testing.assert_array_equal(got["size"], np.atleast_1d(r["size"]))


def test_set_scores_definition():
    """Port of the testthat case: mean standardised expression, less the others' if competitive."""
    rng = np.random.default_rng(7)
    Y = rng.normal(5, 1, (30, 10))
    genes = [f"g{i}" for i in range(1, 31)]
    sets = {"a": genes[:6], "b": genes[3:12], "small": genes[:2], "absent": list("xyzwv")}
    Z = (Y - Y.mean(1, keepdims=True)) / Y.std(1, ddof=1, keepdims=True)
    sc = ab._abundance_set_scores(Y, genes, sets, "self-contained", 5, 500)
    assert sc["sets"] == ["a", "b"] and list(sc["size"]) == [6, 9]
    np.testing.assert_allclose(sc["S"][0], Z[:6].mean(0), rtol=1e-12)
    cp = ab._abundance_set_scores(Y, genes, sets, "competitive", 5, 500)
    other = np.delete(Z, np.arange(3, 12), axis=0).mean(0)
    np.testing.assert_allclose(cp["S"][1], Z[3:12].mean(0) - other, rtol=1e-12)


# ---- the public function against R ----------------------------------------------------------
PUB = {
    "cond": dict(condition="condition"),
    "nocond": dict(),
    "cov": dict(condition="condition", covariates="Age"),
    "cov_nocond": dict(covariates="Age"),
    "strict": dict(condition="condition", min_cells=70),
    "strict_nocond": dict(min_cells=70),
    "idx": dict(condition="condition", index="A", niche=["C", "B"]),
    "nocap": dict(condition="condition", max_leverage=np.inf),
    "cap15": dict(condition="condition", max_leverage=1.5),
    "prior": dict(condition="condition", prior_count=0.25),
    "gs_comp": dict(condition="condition", genesets=GS),
    "gs_self": dict(condition="condition", genesets=GS, type="self-contained"),
    "gs_nocond": dict(genesets=GS, min_size=2),
    "gs_strict": dict(condition="condition", genesets=GS, min_cells=70),
}  # fmt: skip


def _table_equal(tab, ref, what):
    cols = strs(ref["colnames__"])
    assert list(tab.columns) == cols, what
    for c in cols:
        v = np.atleast_1d(ref[c])
        assert len(tab) == len(v), f"{what}.{c}: length"
        if c in ("gene", "set", "index", "niche", "term", "direction"):
            assert [str(s) for s in tab[c]] == strs(v), f"{what}.{c}"
        elif c in ("n_patients", "downweighted", "size"):
            np.testing.assert_array_equal(tab[c].to_numpy(), v.astype(int), err_msg=f"{what}.{c}")
        elif c in ("p", "q", "q.global"):
            pvals(tab[c], v, f"{what}.{c}", RI)
        elif c == "estimate" or c == "leverage":
            close(tab[c], v, f"{what}.{c}", rtol=1e-10, atol=1e-12)
        else:
            close(tab[c], v, f"{what}.{c}", rtol=RI, atol=1e-10)


@pytest.mark.parametrize("case", list(PUB))
def test_public_matches_r(adata, R, case):
    got = spidepy.test_niche_abundance(adata, sigma=50, verbose=False, **PUB[case])
    _table_equal(got, R["pub"][case], case)


def test_public_dropped_sample_is_looked_up_by_name(adata, R):
    """min_cells = 70 leaves different samples for A and C; condition follows the sample name."""
    got = spidepy.test_niche_abundance(
        adata, condition="condition", sigma=50, min_cells=70, verbose=False
    )
    n = got.groupby("index")["n_patients"].min()
    assert n["A"] != 12 and n["C"] != 12
    _table_equal(got, R["pub"]["strict"], "strict")


def test_public_is_invariant_to_cell_order(adata):
    """Reordering cells reorders the patients' first appearance, not the answer."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(adata.n_obs)
    a = adata[perm].copy()
    for kw in (dict(condition="condition", min_cells=70), dict(covariates="Age")):
        x = spidepy.test_niche_abundance(adata, sigma=50, verbose=False, **kw)
        y = spidepy.test_niche_abundance(a, sigma=50, verbose=False, **kw)
        assert list(x.columns) == list(y.columns)
        for c in x.columns:
            if x[c].dtype.kind == "f":
                np.testing.assert_allclose(y[c], x[c], rtol=1e-9, atol=1e-12, err_msg=c)
            else:
                assert list(x[c]) == list(y[c]), c


def test_public_high_leverage_sample_matches_r(adata, R):
    a = adata.copy()
    a.obsm["Niche50"] = pd.DataFrame(
        np.asarray(R["niche50_hl"], float), index=a.obs_names, columns=strs(R["niche_cols"])
    )
    for case, kw in (
        ("cond", dict(condition="condition", max_leverage=2)),
        ("nocond", dict()),
        ("gs", dict(condition="condition", genesets=GS, max_leverage=2)),
    ):
        got = spidepy.test_niche_abundance(a, sigma=50, verbose=False, **kw)
        assert (got["downweighted"] > 0).any()
        assert (got["leverage"] > 2).any()
        _table_equal(got, R["pub_hl"][case], f"hl.{case}")


def test_public_sparse_counts_equal_dense(adata):
    import scipy.sparse as sp

    a = adata.copy()
    a.X = sp.csr_matrix(a.X)
    a.layers.pop("counts", None)
    x = spidepy.test_niche_abundance(adata, condition="condition", sigma=50, verbose=False)
    y = spidepy.test_niche_abundance(a, condition="condition", sigma=50, verbose=False)
    for c in ("estimate", "t", "p", "q"):
        np.testing.assert_allclose(y[c], x[c], rtol=1e-9, atol=1e-300)


def test_errors_match_r(adata, R):
    e = R["err"]
    f = spidepy.test_niche_abundance
    with pytest.raises(ValueError, match="no index cell type has at least three samples"):
        f(adata, sigma=50, min_cells=1000, verbose=False)
    with pytest.raises(ValueError, match="no requested niche cell types"):
        f(adata, sigma=50, niche="zz", verbose=False)
    with pytest.raises(ValueError, match="max.leverage"):
        f(adata, sigma=50, max_leverage=1, verbose=False)
    with pytest.raises(ValueError, match="genesets"):
        f(adata, sigma=50, genesets=["G1", "G2"], verbose=False)
    with pytest.raises(ValueError, match="min.size"):
        f(adata, sigma=50, genesets=GS, min_size=50, verbose=False)
    # R's messages carry the same sentences with the dotted argument names
    assert "max.leverage" in e["leverage"] and "min.size" in e["min_size"]


def test_function_is_not_collected_by_pytest():
    assert spidepy.test_niche_abundance.__test__ is False


def test_verbose_reports_on_stderr(adata, capsys):
    spidepy.test_niche_abundance(adata, sigma=50, index="A", niche="B", verbose=True)
    assert "A x B" in capsys.readouterr().err
    spidepy.test_niche_abundance(adata, sigma=50, index="A", niche="B", verbose=False)
    assert capsys.readouterr().err == ""


# ---- test-niche-abundance.R -----------------------------------------------------------------
def test_tidy_schema(adata_py):
    ct = spidepy.test_niche_abundance(adata_py, condition="condition", sigma=50, verbose=False)
    assert isinstance(ct, pd.DataFrame)
    assert {"gene", "index", "niche", "term", "estimate", "t", "p", "n_patients", "leverage",
            "downweighted", "q", "q.global"} <= set(ct.columns)  # fmt: skip
    assert set(ct["term"]) == {"niche", "condition:niche"}
    assert not (ct["index"] == ct["niche"]).any()
    assert ((ct["p"] >= 0) & (ct["p"] <= 1)).all()
    assert (ct["q"] >= ct["p"] - 1e-12).all()


def test_planted_confound_is_a_patient_level_finding(adata_py):
    ct = spidepy.test_niche_abundance(adata_py, condition="condition", sigma=50, verbose=False)
    sel = (ct["index"] == "A") & (ct["niche"] == "B") & (ct["term"] == "condition:niche")
    g2 = ct[sel & (ct["gene"] == "G2")]
    assert len(g2) == 1
    assert g2["t"].iloc[0] > 2
    ab_ = ct[sel]
    assert ab_["gene"].iloc[int(np.argmax(ab_["t"].to_numpy()))] == "G2"


def test_without_a_condition_only_the_pooled_term(adata_py):
    ct = spidepy.test_niche_abundance(adata_py, condition=None, sigma=50, verbose=False)
    assert set(ct["term"]) == {"niche"}


def test_min_cells_and_covariates(adata_py):
    kw = dict(condition="condition", sigma=50, verbose=False)
    ct_all = spidepy.test_niche_abundance(adata_py, min_cells=1, **kw)
    ct_strict = spidepy.test_niche_abundance(adata_py, min_cells=70, **kw)
    assert ct_strict["n_patients"].min() <= ct_all["n_patients"].min()
    assert (ct_strict["n_patients"] >= 3).all()
    ct_cov = spidepy.test_niche_abundance(adata_py, covariates="Age", **kw)
    assert len(ct_cov) > 0


def test_index_and_niche_restrictions(adata_py):
    ct = spidepy.test_niche_abundance(
        adata_py, condition="condition", sigma=50, index="A", niche=["B", "C"], verbose=False
    )
    assert set(ct["index"]) == {"A"}
    assert set(ct["niche"]) == {"B", "C"}


def test_niche_is_mean_of_conditions_and_interaction_their_difference():
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(5, 12))
    df = pd.DataFrame(
        {"niche": rng.normal(size=12),
         "condition": pd.Categorical(list("a" * 6 + "b" * 6), categories=["a", "b"])}
    )  # fmt: skip

    def slope(idx):
        x = df["niche"].to_numpy()[idx]
        return np.array([np.polyfit(x, y[idx], 1)[0] for y in Y])

    sa, sb = slope(np.arange(6)), slope(np.arange(6, 12))
    r = ab._abundance_pair(Y, df, max_leverage=np.inf)
    np.testing.assert_allclose(r["terms"]["niche"]["estimate"], (sa + sb) / 2, rtol=1e-10)
    np.testing.assert_allclose(r["terms"]["condition:niche"]["estimate"], sb - sa, rtol=1e-10)
    # one condition with a single sample: only the pooled association, across all samples
    df1 = df.assign(condition=pd.Categorical(list("a" * 11 + "b"), categories=["a", "b"]))
    r1 = ab._abundance_pair(Y, df1, max_leverage=np.inf)
    assert list(r1["terms"]) == ["niche"]
    pooled = np.array([np.polyfit(df["niche"], y, 1)[0] for y in Y])
    np.testing.assert_allclose(r1["terms"]["niche"]["estimate"], pooled, rtol=1e-10)


def test_q_global_is_one_bh_family_per_term(adata_py):
    from spidepy._rcompat import p_adjust_bh

    ct = spidepy.test_niche_abundance(adata_py, condition="condition", sigma=50, verbose=False)
    for tm in ct["term"].unique():
        i = (ct["term"] == tm).to_numpy()
        np.testing.assert_allclose(ct["q.global"][i], p_adjust_bh(ct["p"][i]), rtol=1e-12)


def test_leverage_cap_downweights_an_influential_sample():
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(20), np.append(rng.normal(size=19), 12)])
    cap = ab._cap_leverage(X, 3)
    assert cap["leverage"] > 3 and cap["downweighted"] == 1 and cap["w"][19] < 1
    hw = ab._hat(X * np.sqrt(cap["w"])[:, None])
    assert np.all((hw <= 3 * X.shape[1] / 20 * 1.001) | (cap["w"] <= 0.05 + 1e-12))
    X2 = np.column_stack([np.ones(20), np.append(rng.normal(size=19), 6)])
    c2 = ab._cap_leverage(X2, 3)
    assert c2["w"].min() > 0.05
    assert ab._hat(X2 * np.sqrt(c2["w"])[:, None]).max() <= 3 * 2 / 20 * 1.001
    np.testing.assert_array_equal(ab._cap_leverage(X, np.inf)["w"], np.ones(20))
    # the weighted fit equals weighted least squares with those weights
    Y = rng.normal(size=(3, 20))
    r = ab._abundance_pair(Y, pd.DataFrame({"niche": X[:, 1]}), max_leverage=3)
    assert r["downweighted"] == 1
    want = [np.polyfit(X[:, 1], y, 1, w=np.sqrt(cap["w"]))[0] for y in Y]
    np.testing.assert_allclose(r["terms"]["niche"]["estimate"], want, rtol=1e-10)
    # no influential sample: nothing is down-weighted
    flat = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)])
    assert ab._cap_leverage(flat, 3)["downweighted"] == 0


def test_a_pair_the_cap_leaves_without_support_drops_out():
    rng = np.random.default_rng(6)
    Y = rng.normal(size=(4, 14))
    df = pd.DataFrame(
        {"niche": np.append(rng.normal(size=12), [0, 25]),
         "condition": pd.Categorical(list("a" * 12 + "bb"), categories=["a", "b"])}
    )  # fmt: skip
    r = ab._abundance_pair(Y, df, max_leverage=2)  # must not raise
    if r is not None:
        assert all(np.isfinite(v["p"]).all() for v in r["terms"].values())


def test_gene_sets_tested_like_genes(adata_py):
    sets = {k: GS[k] for k in ("with_G2", "other")}
    t = spidepy.test_niche_abundance(
        adata_py, condition="condition", sigma=50, genesets=sets, verbose=False
    )
    from spidepy._rcompat import p_adjust_bh

    assert {"set", "size", "index", "niche", "term", "estimate", "t", "p", "n_patients",
            "leverage", "downweighted", "q", "q.global", "direction"} <= set(t.columns)  # fmt: skip
    assert "gene" not in t.columns
    assert set(t["term"]) == {"niche", "condition:niche"}
    for tm in t["term"].unique():
        i = (t["term"] == tm).to_numpy()
        np.testing.assert_allclose(t["q.global"][i], p_adjust_bh(t["p"][i]), rtol=1e-12)
    x = t[(t["index"] == "A") & (t["niche"] == "B") & (t["term"] == "condition:niche")]
    tt = dict(zip(x["set"], x["t"]))
    assert tt["with_G2"] > tt["other"]
    with pytest.raises(ValueError, match="min.size"):
        spidepy.test_niche_abundance(adata_py, sigma=50, genesets=sets, min_size=50, verbose=False)
