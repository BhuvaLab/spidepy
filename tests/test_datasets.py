"""Toy-data loader and simulators (port of test-toydata.R plus fixture-shape checks)."""

import numpy as np
import pandas as pd
import pytest

import spidepy
from _ref import load_spe
from spidepy.datasets import simulate_clustered, simulate_niche, simulate_toy, toy_spide

CALLS = {
    "toyspe_default": {},
    "toyspe_comp3": {"composition": 3},
    "toyspe_s12_comp3": {"n_samples": 12, "composition": 3},
    "toyspe_s2_n20_g2_seed1": {"n_samples": 2, "n_per": 20, "n_genes": 2, "seed": 1},
    "toyspe_s8_n120_g6_seed3": {"n_samples": 8, "n_per": 120, "n_genes": 6, "seed": 3},
}


def _is_str(s):
    return s.map(lambda v: isinstance(v, str)).all()


def test_toy_spide_matches_the_r_object():
    a, r = toy_spide(), load_spe("toySpiDE")
    assert a.shape == (2400, 20) and list(a.var_names) == [f"G{i}" for i in range(1, 21)]
    assert list(a.obs.columns) == list(r.obs.columns)
    for c in r.obs.columns:
        if r.obs[c].dtype == float:
            assert a.obs[c].dtype == float
        else:
            assert _is_str(a.obs[c])
            assert list(a.obs[c]) == list(r.obs[c])
    np.testing.assert_array_equal(a.layers["counts"], r.layers["counts"])
    np.testing.assert_array_equal(a.X, a.layers["counts"])
    np.testing.assert_array_equal(a.obsm["spatial"], r.obsm["spatial"])
    assert a.obs["condition"].nunique() == 2 and a.obs["sample_id"].nunique() == 16
    assert spidepy.toy_spide().shape == (2400, 20)


@pytest.mark.parametrize("topic", list(CALLS))
def test_simulate_toy_layout_matches_r_fixture(topic):
    r = load_spe(topic)
    a = simulate_toy(**CALLS[topic])
    assert a.shape == r.shape
    assert list(a.obs.columns) == list(r.obs.columns)
    for c in r.obs.columns:
        if r.obs[c].dtype == float:
            assert a.obs[c].dtype == float, c
        else:
            assert _is_str(a.obs[c]), c
    assert a.X.dtype == r.X.dtype and a.layers["counts"].dtype == r.X.dtype
    assert a.obsm["spatial"].shape == r.obsm["spatial"].shape
    assert list(a.var_names) == list(r.var_names) and list(a.obs_names) == list(r.obs_names)
    # same sample / condition / cell-type vocabularies and ordering as R
    for c in ("sample_id", "condition"):
        assert list(pd.unique(a.obs[c])) == list(pd.unique(r.obs[c]))
    assert set(a.obs["cell_type"]) == set(r.obs["cell_type"]) == {"A", "B", "C"}
    assert (a.obs["sample_id"].value_counts() == (a.n_obs // len(set(a.obs["sample_id"])))).all()
    np.testing.assert_array_equal(a.obs["nCount"], np.asarray(a.X).sum(axis=1))
    assert (a.X >= 0).all() and (a.X == np.round(a.X)).all()
    assert (a.obs["Area"] > 0).all()
    # Age is a sample-level covariate
    assert (a.obs.groupby("sample_id")["Age"].nunique() == 1).all()


def test_simulated_counts_are_low_count_like_r():
    # the zero fraction depends on the random gene means, so only the regime is checked (R ~0.70)
    for seed in (1, 2, 3):
        assert 0.4 < (simulate_toy(seed=seed).X == 0).mean() < 0.8


def test_seed_is_reproducible_and_matters():
    a, b, c = simulate_toy(seed=4), simulate_toy(seed=4), simulate_toy(seed=5)
    np.testing.assert_array_equal(a.X, b.X)
    pd.testing.assert_frame_equal(a.obs, b.obs)
    assert not np.array_equal(a.obs["x"], c.obs["x"])


def test_composition_zero_is_the_default():
    a, b = simulate_toy(), simulate_toy(composition=0)
    np.testing.assert_array_equal(a.X, b.X)


def test_composition_plants_a_between_sample_effect():
    a = simulate_toy()
    cs = simulate_toy(n_samples=12, composition=3)
    o = cs.obs
    y = pd.Series(np.asarray(cs.X)[:, 1], index=o.index)  # G2
    sel = (o["cell_type"] == "A") & (o["condition"] == "Responder")
    m = y[sel].groupby(o["sample_id"][sel]).mean()
    assert m.max() / m.min() > 1.5
    mx = o["x"][sel].groupby(o["sample_id"][sel]).mean()
    assert m.corr(mx[m.index], method="spearman") > 0.5
    o0 = a.obs
    mins = o0["x"][o0["cell_type"] == "A"].groupby(o0["sample_id"]).min()
    assert (mins < 0.3 * 500).all()


def test_global_rng_is_untouched():
    np.random.seed(3)
    x1 = np.random.uniform()
    np.random.seed(3)
    simulate_toy(n_samples=2, n_per=20, n_genes=2, seed=1)
    simulate_niche(n_samples=2, n_per=20, n_genes=2)
    simulate_clustered(n_samples=2, n_per=20, n_genes=2, sd_nested=0.3)
    assert np.random.uniform() == x1


def test_simulate_niche_has_no_condition_and_a_planted_x_effect():
    a = simulate_niche(n_samples=8, n_per=200, n_genes=6, seed=2)
    assert list(a.obs.columns) == [
        "sample_id",
        "x",
        "y",
        "cell_type",
        "Age",
        "cell_id",
        "Area",
        "nCount",
    ]
    o = a.obs
    isa = (o["cell_type"] == "A").to_numpy()
    g1 = np.asarray(a.X)[:, 0]
    assert np.corrcoef(o["x"][isa], g1[isa])[0, 1] > 0.1


def test_simulate_clustered_layout_and_patient_effect():
    a = simulate_clustered(sd_nested=0.3, seed=2)
    assert a.shape == (8 * 80, 30)
    assert list(a.obs.columns) == [
        "sample_id",
        "condition",
        "x",
        "y",
        "cell_type",
        "cell_id",
        "Area",
        "nCount",
    ]
    # condition is a sample-level label
    assert (a.obs.groupby("sample_id")["condition"].nunique() == 1).all()
    big = simulate_clustered(sd_patient=1.5, seed=2)
    small = simulate_clustered(sd_patient=0.0, seed=2)

    def spread(d):
        lg = np.log1p(np.asarray(d.X))
        per = pd.DataFrame(lg).groupby(d.obs["sample_id"].to_numpy()).mean()
        return per.std(axis=0).mean()

    assert spread(big) > 2 * spread(small)


def test_spide_recovers_the_planted_effect_on_simulated_data():
    # Detection of G1 / A / B is draw-dependent (about 15% of seeds at R's default design, as
    # in R; about 65% here with 24 samples x 300 cells, since the B niche is collinear with the
    # other niches). Require it in at least 3 of 8 seeds, so no single draw decides the test.
    hits = 0
    for seed in range(1, 9):
        sp = simulate_toy(n_samples=24, n_per=300, n_genes=12, seed=seed)
        res = spidepy.spide(sp, "condition", 30, index="A", procedure="all", verbose=False)
        pool = spidepy.results(res, test="pooled")
        g1 = pool[(pool["gene"] == "G1") & (pool["niche"] == "B")]
        hits += int(g1["q"].iloc[0] < 0.05 and g1["estimate"].iloc[0] > 0)
    assert hits >= 3
