import numpy as np
import pandas as pd
import pytest

from _ref import load_ref

ALL_FIXTURES = [
    "toySpiDE",
    "toyspe_default",
    "toyspe_comp3",
    "toyspe_s12_comp3",
    "toyspe_s16_n120_g12_seed2",
    "toyspe_s16_n150_g12_seed7",
    "toyspe_s16_n150_g6_seed5",
    "toyspe_s16_n150_seed7",
    "toyspe_s16_n150_seed99",
    "toyspe_s2_n20_g2_seed1",
    "toyspe_s8_n120_g6_seed3",
    "toyspe_s8_n120_g8_seed11",
    "toyspe_s8_n650_g4_seed11",
]


def test_roundtrip():
    ref = load_ref("harness")
    assert ref["m"].shape == (2, 3)
    assert ref["m"][1, 2] == 6.5
    assert ref["m"][0, 1] == 3  # column-major fill, R shape kept
    assert np.isnan(ref["v"][1])
    assert list(ref["s"]) == ["a", "b"]
    assert isinstance(ref["x"], float) and ref["x"] == 3.25
    assert list(ref["i"]) == [1, 2, 3]
    assert list(ref["l"]) == [1, 0, 1]


def test_factor_keeps_levels():
    f = load_ref("harness")["f"]
    assert isinstance(f, pd.Categorical)
    assert list(f) == ["u", "v", "u"]
    assert list(f.categories) == ["v", "u", "w"]  # R's level order, incl. unused level


def test_length_one_values():
    ref = load_ref("harness")
    assert isinstance(ref["s1"], str) and ref["s1"] == "only"
    assert ref["m11"].shape == (1, 1) and ref["m11"][0, 0] == 7.5


def test_toy_spide_matches_r_report(toy_spide):
    a = toy_spide
    n_genes, n_cells = load_ref("toySpiDE")["dim"]  # dims as R printed them: 20 x 2400
    assert (a.n_vars, a.n_obs) == (n_genes, n_cells) == (20, 2400)
    assert list(a.obs.columns) == [
        "sample_id", "condition", "x", "y", "cell_type", "Age", "cell_id", "Area", "nCount",
    ]  # fmt: skip
    assert a.var_names[0] == "G1" and a.var_names[-1] == "G20"
    assert a.obs_names[0] == "cell1" and a.obs_names[-1] == "cell2400"
    assert a.obsm["spatial"].shape == (2400, 2)
    # spatial coordinates are the x / y colData columns
    np.testing.assert_array_equal(a.obsm["spatial"], a.obs[["x", "y"]].to_numpy())
    assert a.obs["sample_id"].nunique() == 16
    assert sorted(a.obs["condition"].unique()) == ["Non-responder", "Responder"]
    assert sorted(a.obs["cell_type"].unique()) == ["A", "B", "C"]
    # counts: R's double storage, genes x cells sums agree with the nCount column
    assert a.layers["counts"].dtype == np.float64
    np.testing.assert_array_equal(a.layers["counts"].sum(axis=1), a.obs["nCount"].to_numpy())
    assert a.layers["counts"].min() >= 0


def test_toy_spide_is_fresh_per_test(toy_spide):
    toy_spide.X[:] = -1  # mutating must not leak into the next test
    toy_spide.obs.loc[:, "Age"] = 0


def test_toy_spide_is_fresh_per_test_again(toy_spide):
    assert toy_spide.X.min() >= 0 and toy_spide.obs["Age"].abs().max() > 0


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_fixture_loads_with_r_dims(name, spe):
    a = spe(name)
    n_genes, n_cells = a.uns["r_dim"]
    assert a.shape == (n_cells, n_genes)
    assert a.layers["counts"].shape == a.shape
    assert a.obsm["spatial"].shape == (n_cells, 2)
    assert a.uns["r_call"]


def test_fixture_dims_follow_call(spe):
    a = spe("toyspe_s8_n120_g6_seed3")
    assert a.shape == (8 * 120, 6) and a.obs["sample_id"].nunique() == 8
    assert spe("toyspe_default").shape == (6 * 80, 20)
    assert spe("toyspe_s2_n20_g2_seed1").shape == (40, 2)


def test_bundled_dataset_equals_fixture(toy_spide):
    import anndata

    from spidepy import datasets

    path = datasets.__path__[0] + "/toySpiDE.h5ad"
    b = anndata.read_h5ad(path)
    assert b.shape == toy_spide.shape
    np.testing.assert_array_equal(b.layers["counts"], toy_spide.layers["counts"])
    np.testing.assert_array_equal(b.obsm["spatial"], toy_spide.obsm["spatial"])
