"""Input glue (_io) and argument checkers (_checks); messages are R's (checkers.R)."""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spidepy import _checks as ck
from spidepy import _io

# ----------------------------------------------------------------------------- _io


def test_get_counts_orientation_and_assay(toy_spide):
    a = toy_spide
    Y = _io.get_counts(a)
    assert Y.shape == (a.n_vars, a.n_obs)
    np.testing.assert_array_equal(Y, a.X.T)
    a.layers["raw"] = a.X * 2
    np.testing.assert_array_equal(_io.get_counts(a, "raw"), a.X.T * 2)
    np.testing.assert_array_equal(_io.get_counts(a, "counts"), a.layers["counts"].T)
    del a.layers["counts"]  # "counts" falls back to X
    np.testing.assert_array_equal(_io.get_counts(a, "counts"), a.X.T)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_get_counts_sparse_stays_sparse(toy_spide, fmt):
    a = toy_spide
    dense = a.X.copy()
    a.X = sp.csr_matrix(dense) if fmt == "csr" else sp.csc_matrix(dense)
    a.layers["counts"] = a.X.copy()
    Y = _io.get_counts(a)
    assert sp.issparse(Y) and Y.format == "csc" and Y.shape == (a.n_vars, a.n_obs)
    np.testing.assert_array_equal(Y.toarray(), dense.T)


def test_get_coords(toy_spide):
    a = toy_spide
    xy = _io.get_coords(a)
    assert xy.shape == (a.n_obs, 2) and xy.dtype == float
    a.obsm["spatial"] = pd.DataFrame(
        np.arange(3 * a.n_obs).reshape(a.n_obs, 3), index=a.obs_names
    )  # DataFrame, int, 3 columns: first two kept
    xy = _io.get_coords(a)
    assert xy.dtype == float and xy.shape == (a.n_obs, 2)
    np.testing.assert_array_equal(xy[:, 1], np.arange(a.n_obs) * 3 + 1)


def test_get_coords_errors(toy_spide):
    a = toy_spide
    a.obsm["spatial"] = np.ones((a.n_obs, 1))
    with pytest.raises(ValueError, match="at least 2 columns"):
        _io.get_coords(a)
    bad = np.ones((a.n_obs, 2))
    bad[3, 1] = np.nan
    a.obsm["spatial"] = bad
    with pytest.raises(ValueError, match="non-finite"):
        _io.get_coords(a)
    del a.obsm["spatial"]
    with pytest.raises(ValueError, match="not found"):
        _io.get_coords(a)


def test_get_obs_strings_and_nan(toy_spide):
    a = toy_spide
    s = _io.get_obs(a, "sample_id")
    assert s.dtype == object and all(isinstance(v, str) for v in s)
    a.obs["num"] = [1.0, 2.5, np.nan] + [3.0] * (a.n_obs - 3)
    n = _io.get_obs(a, "num")
    assert list(n[:3][[0, 1]]) == ["1", "2.5"] and pd.isna(n[2]) and n[3] == "3"
    a.obs["cat"] = pd.Categorical(["u", None] + ["v"] * (a.n_obs - 2))
    c = _io.get_obs(a, "cat")
    assert c[0] == "u" and pd.isna(c[1]) and c[2] == "v"
    with pytest.raises(ValueError, match="'nope' not found"):
        _io.get_obs(a, "nope")


def test_niche_roundtrip_and_keys(toy_spide):
    a = toy_spide
    assert _io.niche_key("Niche", 20) == "Niche20"
    assert _io.niche_key("Niche", 2.5) == "Niche2.5"
    assert _io.niche_key("N", 1e5) == "N1e+05"
    df = pd.DataFrame(np.ones((a.n_obs, 2)), index=a.obs_names, columns=["A", "B"])
    _io.set_niche(a, "Niche20", df)
    _io.set_niche(a, "Niche2.5", df)
    _io.set_niche(a, "Other30", df)
    assert _io.niche_keys(a, "Niche") == ["Niche20", "Niche2.5"]
    assert _io.get_niche(a, "Niche20").equals(df)
    with pytest.raises(ValueError, match="not found"):
        _io.get_niche(a, "Niche99")
    with pytest.raises(ValueError, match="one row per cell"):
        _io.set_niche(a, "Niche5", df.iloc[:5])
    with pytest.raises(ValueError, match="one row per cell"):
        _io.set_niche(a, "Niche5", df.set_axis(df.index[::-1]))


def test_set_niche_on_view_does_not_touch_parent(toy_spide):
    a = toy_spide
    v = a[: a.n_obs // 2]
    df = pd.DataFrame(np.ones((v.n_obs, 1)), index=v.obs_names, columns=["A"])
    _io.set_niche(v, "Niche20", df)
    assert not v.is_view and "Niche20" in v.obsm and "Niche20" not in a.obsm


def test_group_map(toy_spide):
    a = toy_spide
    assert _io.get_group_map(a, "Niche20") is None
    _io.set_group_map(a, "Niche20", {"BC": ["B", "C"]})
    _io.set_group_map(a, "Niche30", {"DE": ["D", "E"]})
    assert _io.get_group_map(a, "Niche20") == {"BC": ["B", "C"]}
    assert _io.get_group_map(a, "Niche30") == {"DE": ["D", "E"]}
    _io.set_group_map(a, "Niche20", None)
    assert _io.get_group_map(a, "Niche20") is None
    assert _io.get_group_map(a, "Niche30") == {"DE": ["D", "E"]}


# ------------------------------------------------------------------- check_adata


def test_check_adata(toy_spide):
    a = toy_spide
    assert ck.check_adata(a) is True
    with pytest.raises(TypeError, match="should be an AnnData"):
        ck.check_adata(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="assay 'nope' not found in 'adata'"):
        ck.check_adata(a, assay="nope")
    a.layers["raw"] = a.X
    assert ck.check_adata(a, assay="raw")
    del a.layers["counts"]
    assert ck.check_adata(a)  # "counts" falls back to X
    with pytest.raises(ValueError, match="cell type column 'ct' not found"):
        ck.check_adata(a, cell_type="ct")
    with pytest.raises(ValueError, match="sample id column 'sid' not found"):
        ck.check_adata(a, sample_id="sid")


# ------------------------------------------------------------------ check_counts


def test_check_counts_nonneg_dense_and_sparse():
    Y = np.array([[0.0, 1, 2], [3, 4, 5]])
    assert ck.check_counts(Y)
    assert ck.check_counts(sp.csc_matrix(Y))
    for bad in (Y - 1, np.full((2, 3), np.nan)):
        with pytest.raises(ValueError, match="counts should be non-negative"):
            ck.check_counts(bad)
        with pytest.raises(ValueError, match="counts should be non-negative"):
            ck.check_counts(sp.csc_matrix(np.nan_to_num(bad, nan=-1.0)))
    with pytest.raises(ValueError, match="non-negative"):
        ck.check_counts(np.empty((0, 3)))
    # NaN entries are ignored as long as something finite remains
    Y2 = Y.copy()
    Y2[0, 0] = np.nan
    assert ck.check_counts(Y2)
    # a negative value stored in a sparse matrix is found
    S = sp.csc_matrix(np.array([[0.0, 0, 2], [0, -3, 0]]))
    with pytest.raises(ValueError, match="non-negative"):
        ck.check_counts(S)
    # fully dense-stored sparse matrix (no structural zeros)
    assert ck.check_counts(sp.csc_matrix(np.ones((2, 2))))


def test_check_counts_integer_rule_first_20_genes():
    rng = np.random.default_rng(0)
    Y = rng.poisson(3, size=(30, 40)).astype(float)
    assert ck.check_counts(Y, integer_only=True)
    Y_bad = Y.copy()
    Y_bad[5, 7] += 0.5
    with pytest.raises(ValueError, match="integer counts"):
        ck.check_counts(Y_bad, integer_only=True)
    assert ck.check_counts(Y_bad)  # only checked on request
    with pytest.raises(ValueError, match=r'Supply raw counts \(assay = "counts"\)'):
        ck.check_counts(sp.csc_matrix(Y_bad), integer_only=True)
    Y_late = Y.copy()
    Y_late[20, 0] += 0.5  # gene 21 is beyond the rule's reach
    assert ck.check_counts(Y_late, integer_only=True)
    assert ck.check_counts(sp.csc_matrix(Y_late), integer_only=True)
    Y_tol = Y.copy()
    Y_tol[0, 0] += 1e-9  # within 1e-8
    assert ck.check_counts(Y_tol, integer_only=True)
    Y_nan = Y.copy()
    Y_nan[0, 0] = np.nan  # non-finite values are skipped by the integer rule
    assert ck.check_counts(Y_nan, integer_only=True)


# ------------------------------------------------------------------ obs checkers


def test_check_condition(toy_spide):
    a = toy_spide
    assert ck.check_condition(a, "condition")
    with pytest.raises(ValueError, match="condition column 'cc' not found in adata.obs"):
        ck.check_condition(a, "cc")
    a.obs["three"] = (["a", "b", "c"] * a.n_obs)[: a.n_obs]
    with pytest.raises(ValueError, match="'three' should have exactly two levels, found 3"):
        ck.check_condition(a, "three")
    a.obs["one"] = "x"
    with pytest.raises(ValueError, match="found 1"):
        ck.check_condition(a, "one")
    # NA is not a level
    a.obs["two_na"] = pd.Series(["a", "b", None] * (a.n_obs // 3), index=a.obs_names)
    assert ck.check_condition(a, "two_na")


def test_check_covariates(toy_spide):
    a = toy_spide
    assert ck.check_covariates(a, [])
    assert ck.check_covariates(a, None)
    assert ck.check_covariates(a, ["Area", "Age"])
    with pytest.raises(ValueError, match=r"not found in adata.obs: zz, yy"):
        ck.check_covariates(a, ["Area", "zz", "yy"])
    a.obs["bad"] = np.r_[-np.inf, np.ones(a.n_obs - 1)]  # what log(0) gives
    a.obs["nan"] = np.r_[np.nan, np.ones(a.n_obs - 1)]
    a.obs["ok"] = 1.0
    assert ck.check_covariates(a, ["bad"])  # opt-in rule
    with pytest.raises(ValueError, match=r"non-finite values: bad, nan\. Every cell"):
        ck.check_covariates(a, ["bad", "ok", "nan"], finite_only=True)
    assert ck.check_covariates(a, ["ok", "Area"], finite_only=True)
    # non-numeric columns are not subject to the finiteness rule
    a.obs["lab"] = pd.Series([None] + ["u"] * (a.n_obs - 1), index=a.obs_names)
    assert ck.check_covariates(a, ["lab"], finite_only=True)


def test_check_sample(toy_spide):
    a = toy_spide
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ck.check_sample(a, "condition", "sample_id", ["Area"])
        assert ck.check_sample(a)  # no condition, no covariates
    # condition varying within samples; names listed in R's sort order
    a.obs["flip"] = np.where(np.arange(a.n_obs) % 2 == 0, "u", "v")
    with pytest.raises(ValueError, match=r"'flip' varies within sample\(s\): S1, S10, S11"):
        ck.check_sample(a, "flip")
    # patient-level covariate (R test: "constant within sample")
    with pytest.raises(ValueError, match=r"constant within sample: Age"):
        ck.check_sample(a, "condition", covariates=["Area", "Age"])
    with pytest.raises(ValueError, match="sample id column 'sid' not found"):
        ck.check_sample(a, sample_id="sid")
    # NA within a sample does not count as a level
    c = a.obs["condition"].astype(object).copy()
    c.iloc[0] = None
    a.obs["cond_na"] = c
    assert ck.check_sample(a, "cond_na")


def test_check_sample_warns_below_three_samples(spe):
    a = spe("toyspe_s2_n20_g2_seed1")
    with pytest.warns(UserWarning, match=r"only 2 sample\(s\); the between-patient tests"):
        ck.check_sample(a, "condition")
    b = spe("toyspe_s8_n120_g6_seed3")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ck.check_sample(b, "condition")


def test_check_column(toy_spide):
    a = toy_spide
    assert ck.check_column(a, None, "strata")
    assert ck.check_column(a, "cell_type", "strata")
    with pytest.raises(ValueError, match="strata column 'nope' not found in adata.obs"):
        ck.check_column(a, "nope", "strata")
    with pytest.raises(ValueError, match="section column 'a, b' not found"):
        ck.check_column(a, ["a", "b"], "section")
    with pytest.raises(ValueError, match="strata column '1' not found"):
        ck.check_column(a, 1, "strata")


# --------------------------------------------------------------- scalar checkers


@pytest.mark.parametrize("ok", [0.05, 1, 1.0, np.float64(0.5), 1e-12])
def test_check_fdr_ok(ok):
    assert ck.check_fdr(ok)


@pytest.mark.parametrize("bad", [0, 0.0, -0.1, 1.01, np.nan, None, "0.05", True, [0.1, 0.2]])
def test_check_fdr_errors(bad):
    with pytest.raises(ValueError, match=r"single numeric value in \(0, 1\]"):
        ck.check_fdr(bad)


def test_check_procedure():
    for proc in ("filtered", "all"):
        assert ck.check_procedure(proc, "sandwich")
    for proc in ("heterogeneity", "pooled_or_heterogeneity", "filtered"):
        assert ck.check_procedure(proc, "slopes")
    for proc in ("heterogeneity", "pooled_or_heterogeneity"):
        with pytest.raises(ValueError, match="slopes engine"):
            ck.check_procedure(proc, "sandwich")
    with pytest.raises(ValueError, match=r'procedure = "heterogeneity" needs the slopes engine'):
        ck.check_procedure("heterogeneity", "sandwich")


def test_check_niche(toy_spide):
    a = toy_spide
    with pytest.raises(ValueError, match="niche.*Niche20.*build_niches"):
        ck.check_niche(a, 20)
    for s in (20, 2.5):
        _io.set_niche(a, f"Niche{s}", pd.DataFrame({"A": 1.0}, index=a.obs_names))
    assert ck.check_niche(a, 20) and ck.check_niche(a, [20, 2.5])
    with pytest.raises(ValueError, match=r"not found in adata.obsm: Niche30, Niche40\."):
        ck.check_niche(a, [20, 30, 40])
    _io.set_niche(a, "Mine20", pd.DataFrame({"A": 1.0}, index=a.obs_names))
    assert ck.check_niche(a, 20, name="Mine")


# ------------------------------------------------------------ fit-object checkers


def _fit(a, depth="loglib", sigma=20.0, index=("A",)):
    """A minimal fit: the index cells per patient as fit_spide() would record them."""
    ct = a.obs["cell_type"].to_numpy(dtype=object)
    smp = a.obs["sample_id"].to_numpy(dtype=object)
    usable = (
        np.asarray(a.layers["counts"].sum(axis=1)).ravel() > 0
        if depth != "none"
        else np.ones(a.n_obs, bool)
    )
    idx = {}
    for k in index:
        pts = sorted(set(smp[ct == k]))
        nc = {p: int(((smp == p) & (ct == k) & usable).sum()) for p in pts}
        idx[k] = {
            "patients": pts,
            "ncells": nc,
            "niches": ["B", "C"],
            "genes": ["G1", "G2"],
            "intercept": np.zeros((2, len(pts))),
        }
    return SimpleNamespace(
        engine="slopes",
        sigma=sigma,
        index=idx,
        params={"cell_type": "cell_type", "sample_id": "sample_id", "depth": depth},
    )


def test_check_intercepts():
    assert ck.check_intercepts({"intercept": np.zeros(2)}, "f()")
    for xi in ({}, {"intercept": None}):
        with pytest.raises(ValueError, match="f\\(\\) needs the patient intercepts"):
            ck.check_intercepts(xi, "f()")


def test_check_results():
    class SpiDEResults:  # stand-ins named as the later task's classes
        pass

    class SpiDEFit:
        pass

    assert ck.check_results(SpiDEResults(), "f()")
    assert ck.check_results(SpiDEFit(), "f()", fit_ok=True)
    with pytest.raises(TypeError, match=r"f\(\) needs a SpiDEResults from test_spide\(\) or spide"):
        ck.check_results(SpiDEFit(), "f()")
    with pytest.raises(TypeError, match="or a SpiDEFit from fit_spide"):
        ck.check_results(object(), "f()", fit_ok=True)


def test_check_slopes_engine(toy_spide):
    fit = _fit(toy_spide)
    assert ck.check_slopes_engine(fit, "f()")
    fit.engine = "sandwich"
    with pytest.raises(ValueError, match=r'f\(\) needs the slopes engine.*engine = "slopes"'):
        ck.check_slopes_engine(fit, "f()")


def test_check_triplet(toy_spide):
    fit = _fit(toy_spide)
    assert ck.check_triplet(fit, "A")
    assert ck.check_triplet(fit, "A", "B", ["G1", "G2"])
    assert ck.check_triplet(fit, "A", gene="G1")
    with pytest.raises(ValueError, match="fitted index types: A"):
        ck.check_triplet(fit, "Z")
    with pytest.raises(ValueError, match="fitted index types"):
        ck.check_triplet(fit, ["A"])
    with pytest.raises(ValueError, match="niches tested in A: B, C"):
        ck.check_triplet(fit, "A", "Z")
    with pytest.raises(ValueError, match="not tested in A: nope"):
        ck.check_triplet(fit, "A", "B", ["G1", "nope"])
    with pytest.raises(ValueError, match="'gene' must be a character vector"):
        ck.check_triplet(fit, "A", "B", [])
    with pytest.raises(ValueError, match="'gene' must be a character vector"):
        ck.check_triplet(fit, "A", "B", [1, 2])


def test_check_fit_adata(toy_spide):
    a = toy_spide
    fit = _fit(a)
    assert ck.check_fit_adata(a, fit, "A")
    # a foreign object: other sample labels
    other = a.copy()
    other.obs["sample_id"] = "X" + other.obs["sample_id"].astype(str)
    with pytest.raises(ValueError, match="'adata' does not hold the fit's A cells.*made from"):
        ck.check_fit_adata(other, fit, "A")
    # half of the A cells removed
    keep = ~((a.obs["cell_type"] == "A") & (np.arange(a.n_obs) % 2 == 0)).to_numpy()
    with pytest.raises(ValueError, match="made from"):
        ck.check_fit_adata(a[keep].copy(), fit, ["A"])
    # missing columns
    bad = a.copy()
    bad.obs = bad.obs.drop(columns="sample_id")
    with pytest.raises(ValueError, match="not found in adata.obs"):
        ck.check_fit_adata(bad, fit, "A")


def test_check_fit_adata_depth_rule_and_ncells_forms(toy_spide):
    a = toy_spide
    # zero out one A cell's counts: usable under depth 'none' only
    i = int(np.flatnonzero((a.obs["cell_type"] == "A").to_numpy())[0])
    a.layers["counts"][i] = 0
    fit_loglib, fit_none = _fit(a), _fit(a, depth="none")
    assert ck.check_fit_adata(a, fit_loglib, "A") and ck.check_fit_adata(a, fit_none, "A")
    # the fit recorded with depth 'none' counts the empty cell; under loglib it is not usable
    fit_mixed = SimpleNamespace(**{**vars(fit_none), "params": fit_loglib.params})
    with pytest.raises(ValueError, match="made from"):
        ck.check_fit_adata(a, fit_mixed, "A")
    # ncells as a Series and as an aligned array
    xi = fit_loglib.index["A"]
    xi["ncells"] = pd.Series(xi["ncells"])
    assert ck.check_fit_adata(a, fit_loglib, "A")
    xi["ncells"] = np.array([xi["ncells"][p] for p in xi["patients"]])
    assert ck.check_fit_adata(a, fit_loglib, "A")
    # no ncells recorded: every patient must have cells
    xi["ncells"] = None
    assert ck.check_fit_adata(a, fit_loglib, "A")
    xi["patients"] = [*xi["patients"], "S99"]
    with pytest.raises(ValueError, match="made from"):
        ck.check_fit_adata(a, fit_loglib, "A")


def test_check_fit_adata_niches(toy_spide):
    a = toy_spide
    fit = _fit(a)
    with pytest.raises(ValueError, match="niche\\(s\\) B are not columns of Niche20 in 'adata'"):
        ck.check_fit_adata(a, fit, "A", niches="B")
    _io.set_niche(a, "Niche20", pd.DataFrame({"B": 1.0, "C": 2.0}, index=a.obs_names))
    assert ck.check_fit_adata(a, fit, "A", niches=["B", "C"])
    with pytest.raises(ValueError, match="niche\\(s\\) Z are not columns"):
        ck.check_fit_adata(a, fit, "A", niches=["B", "Z"])


def test_check_fit_adata_sparse_counts(toy_spide):
    a = toy_spide
    fit = _fit(a)
    a.layers["counts"] = sp.csr_matrix(a.layers["counts"])
    assert ck.check_fit_adata(a, fit, "A")


# ----------------------------------------------------------------- check_genesets


def test_check_genesets():
    assert ck.check_genesets({"s1": ["G1", "G2"], "s2": np.array(["G3"]), "s3": "G4"})
    assert ck.check_genesets({"empty": []})
    msg = "named list of character vectors with unique names"
    for bad in (
        {},
        [["G1"]],
        {"s1": [1, 2]},
        {"s1": ["G1"], "s2": None},
        {1: ["G1"]},
        {"s1": 5},
    ):
        with pytest.raises(ValueError, match=msg):
            ck.check_genesets(bad)
