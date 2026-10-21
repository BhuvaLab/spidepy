"""build_niches / merge_niches / _is_self_niche against R (tests/ref/make_niches.R) and the
ported testthat cases of test-buildNiches.R and test-mergeNiches.R."""

import numpy as np
import pandas as pd
import pytest

from _ref import load_ref, load_spe
from spidepy import niches as nz
from spidepy._io import get_group_map, niche_key
from spidepy.niches import _is_self_niche, _resolve_members, build_niches, merge_niches

RT = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def R():
    return load_ref("niches")


def _arr(df):
    return df.to_numpy()


def test_build_parity_toy(toy_spide, R):
    sg = (10, 30, 2.5)
    out = build_niches(toy_spide, sigma=sg)
    assert [k for k in out.obsm.keys() if k != "spatial"] == list(R["build_names"])
    for s in sg:
        key = niche_key("Niche", s)
        df = out.obsm[key]
        assert list(df.columns) == list(R["build_cols"])
        assert df.index.equals(out.obs_names)
        np.testing.assert_allclose(_arr(df), R["build"][key], **RT)


def test_build_shuffled_absent_type_and_options(spe, R):
    a = spe("niches_shuffled")
    s = R["shuf"]
    build_niches(a, sigma=[15, 20])
    for key, ref_key in (("Niche15", "n15"), ("Niche20", "n20")):
        assert list(a.obsm[key].columns) == list(s["cols"])
        np.testing.assert_allclose(_arr(a.obsm[key]), s[ref_key], **RT)
    # B is absent from S2 -> exactly zero there, not NA
    in_s2 = (a.obs["sample_id"] == "S2").to_numpy()
    assert (a.obsm["Niche20"].loc[in_s2, "B"] == 0).all()
    for kw, ref_key in (
        (dict(edge=False, diggle=False), "n20_ff"),
        (dict(edge=True, diggle=False), "n20_tf"),
    ):
        b = build_niches(a.copy(), sigma=20, **kw)
        np.testing.assert_allclose(_arr(b.obsm["Niche20"]), s[ref_key], **RT)
    c = build_niches(a.copy(), sigma=20, name="Nb")
    np.testing.assert_allclose(_arr(c.obsm["Nb20"]), s["named"], **RT)


def test_shape_nonneg_and_no_na(toy_spide):
    out = build_niches(toy_spide, sigma=[10, 20])
    assert {"Niche10", "Niche20"} <= set(out.obsm.keys())
    n = out.obsm["Niche20"]
    assert n.shape[0] == out.n_obs
    assert set(n.columns) == set(toy_spide.obs["cell_type"].unique())
    assert (_arr(n) >= 0).all() and not np.isnan(_arr(n)).any()


def test_planted_gradient(toy_spide):
    out = build_niches(toy_spide, sigma=15)
    x = out.obsm["spatial"][:, 0]
    assert (
        np.corrcoef(_arr(out.obsm["Niche15"])[:, list(out.obsm["Niche15"].columns).index("B")], x)[
            0, 1
        ]
        > 0.4
    )


def test_rows_align_with_cells_when_interleaved(spe):
    a = spe("niches_shuffled")  # samples interleaved
    build_niches(a, sigma=15)
    sid = a.obs["sample_id"].iloc[0]
    m = (a.obs["sample_id"] == sid).to_numpy()
    sub = a[m].copy()
    ref = nz._effective_niche(
        sub.obsm["spatial"], sub.obs["cell_type"].astype(str).to_numpy(), 15.0, True, True
    )
    ref = ref.reindex(columns=a.obsm["Niche15"].columns).fillna(0)
    np.testing.assert_allclose(_arr(a.obsm["Niche15"])[m], _arr(ref), rtol=1e-8)


def test_serial_equals_parallel(toy_spide):
    s1 = build_niches(toy_spide.copy(), sigma=20, n_jobs=1)
    s2 = build_niches(toy_spide.copy(), sigma=20, n_jobs=2)
    pd.testing.assert_frame_equal(s1.obsm["Niche20"], s2.obsm["Niche20"])


def test_copy_flag_and_inplace(toy_spide):
    c = build_niches(toy_spide, sigma=20, copy=True)
    assert "Niche20" in c.obsm and "Niche20" not in toy_spide.obsm
    assert build_niches(toy_spide, sigma=20) is toy_spide and "Niche20" in toy_spide.obsm


def test_build_checks_columns(toy_spide):
    with pytest.raises(ValueError, match="cell type column"):
        build_niches(toy_spide, sigma=10, cell_type="nope")


# ------------------------------- mergeNiches -------------------------------


@pytest.fixture(scope="module")
def base(R):
    """Niches built by R from toySpiDE (sigma 10, 20), loaded as the starting point."""

    def make():
        a = load_spe("toySpiDE")
        for s, k in ((10, "before10"), (20, "before20")):
            a.obsm[f"Niche{s}"] = pd.DataFrame(
                R["merge"][k], index=a.obs_names, columns=list(R["merge"]["before_cols"])
            )
        return a

    return make


def _same(df, ref, cols):
    assert list(df.columns) == list(np.atleast_1d(cols))
    np.testing.assert_allclose(_arr(df), ref, **RT)


def test_merge_values_and_metadata(base, R):
    m = R["merge"]
    a = base()
    merge_niches(a, {"AC": ["A", "C"]}, sigma=20)
    _same(a.obsm["Niche20"], m["m1_20"], m["m1_cols"])
    b = a.obsm["Niche20"]
    before = base().obsm["Niche20"]
    np.testing.assert_allclose(b["AC"], before["A"] + before["C"], **RT)
    np.testing.assert_array_equal(b["B"], before["B"])
    assert set(a.uns["spiDE_niche_groups"]) == {"Niche20"}  # Niche10 not touched
    assert list(base().obsm["Niche10"].columns) == list(a.obsm["Niche10"].columns)
    gm = get_group_map(a, "Niche20")
    assert gm == {k: list(np.atleast_1d(v)) for k, v in m["m1_gm"].items()}


def test_remerge_composes_to_fine_types(base, R):
    m = R["merge"]
    a = base()
    merge_niches(a, {"AC": ["A", "C"]}, sigma=20)
    merge_niches(a, {"ACB": ["AC", "B"]}, sigma=20)
    _same(a.obsm["Niche20"], m["m2_20"], m["m2_cols"])
    gm = get_group_map(a, "Niche20")
    assert list(gm) == ["ACB"] and gm["ACB"] == list(m["m2_gm"]["ACB"])


def test_merge_all_bandwidths_and_unknown_members(base, R):
    m = R["merge"]
    a = base()
    merge_niches(a, {"AC": ["A", "C"]})
    _same(a.obsm["Niche10"], m["all_10"], m["all_cols"])
    assert set(a.uns["spiDE_niche_groups"]) == {"Niche10", "Niche20"}
    z = base()
    merge_niches(z, {"AC": ["A", "C", "Zzz"], "Q": ["Nope"]}, sigma=10)
    _same(z.obsm["Niche10"], m["z_10"], m["z_cols"])  # Q is an all-zero column
    assert (z.obsm["Niche10"]["Q"] == 0).all()
    assert get_group_map(z, "Niche10") == {"AC": ["A", "C"], "B": ["B"], "Q": []}


def test_merge_missing_bandwidth_errors(base):
    a = base()
    with pytest.raises(ValueError, match="not found"):
        merge_niches(a, {"AC": ["A", "C"]}, sigma=99)


def test_merge_requires_groups(base):
    with pytest.raises(ValueError):
        merge_niches(base(), {})
    with pytest.raises(ValueError):
        merge_niches(base(), [["A", "C"]])


def test_merge_prefix_quirk(base, R):
    """R greps ^Niche, so any obsm key starting with the name is merged (not just Niche<num>)."""
    m = R["merge"]
    a = base()
    a.obsm["NicheFoo"] = a.obsm["Niche10"].copy()
    merge_niches(a, {"AC": ["A", "C"]})
    assert list(a.obsm["NicheFoo"].columns) == list(m["quirk_foo_cols"])
    assert set(a.uns["spiDE_niche_groups"]) == set(m["quirk_gm_names"])


def test_merge_copy_flag(base):
    a = base()
    b = merge_niches(a, {"AC": ["A", "C"]}, sigma=20, copy=True)
    assert list(a.obsm["Niche20"].columns) == ["A", "B", "C"]
    assert list(b.obsm["Niche20"].columns) == ["AC", "B"]


def test_merge_group_name_shared_with_leftover(base):
    """Group "B" absorbing A and C while column B is left over: two "B" columns, as in R."""
    a = base()
    merge_niches(a, {"B": ["A", "C"]}, sigma=20)
    assert list(a.obsm["Niche20"].columns) == ["B", "B"]
    assert get_group_map(a, "Niche20") == {"B": ["A", "C"]}


def test_resolve_members():
    assert _resolve_members(["A", "B"], None) == ["A", "B"]
    prior = {"AC": ["A", "C"], "B": ["B"], "Q": []}
    assert _resolve_members(["AC", "B", "Q"], prior) == ["A", "C", "B", "Q"]


# ------------------------------- _is_self_niche -------------------------------


def _na(v):
    return [None if s == "NA" else s for s in np.atleast_1d(v)]


def test_is_self_niche_cases(R):
    s = R["selfn"]
    gmap = {"AC": ["A", "C"], "B": ["B"], "T m": ["T-cell", "x y"]}
    c1 = s["c1"]
    np.testing.assert_array_equal(
        _is_self_niche(_na(c1["index"]), _na(c1["niche"])), np.asarray(c1["res"], bool)
    )
    c2 = s["c2"]
    np.testing.assert_array_equal(
        _is_self_niche(c2["index"], c2["niche"], gmap), np.asarray(c2["res"], bool)
    )
    np.testing.assert_array_equal(
        _is_self_niche(c2["index"], c2["niche"]), np.asarray(c2["res_null"], bool)
    )
    c3 = s["c3"]  # niche name is itself a member; names and members sanitised
    np.testing.assert_array_equal(
        _is_self_niche(c3["index"], c3["niche"], gmap), np.asarray(c3["res"], bool)
    )
    assert list(s["gmap_names"]) == list(gmap)


def test_is_self_niche_nan_and_empty():
    out = _is_self_niche(["A", np.nan, "B"], ["A", "A", None])
    assert out.tolist() == [True, False, False]
    assert _is_self_niche([], []).shape == (0,)
