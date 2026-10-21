"""Per-index design helpers against R (``tests/ref/make_design.R``) and the testthat cases of
``test-depth.R`` (design / block parts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from spanormpy._block_design import NBBlockDesign

from _design_ref import CONFIGS, a1, idx0, r_design, ref, strs
from spidepy._design import (
    _cell_covariates,
    _dense_x,
    _depth_blocks,
    _index_cells,
    _index_covariates,
    _index_design,
    _index_offset,
    _lin_pred,
    _niche_columns,
    _partial_block,
    _patient_blocks,
    _patient_table,
    _pick_sigma,
)
from spidepy._rcompat import r_sort_str

RT = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def R():
    return ref()


@pytest.fixture
def adata(spe, R):
    a = spe("toyspe_s8_n650_g4_seed11")
    i = R["inputs"]
    a.obs["section"] = a1(i["section"])
    a.obs["slide"] = a1(i["slide"])
    a.obs["qc"] = a1(i["qc"]).astype(float)
    a.layers["off"] = np.asarray(i["off_all"]).T.copy()
    a.obsm["Niche30"] = pd.DataFrame(
        np.asarray(i["NM"]), index=a.obs_names, columns=strs(i["NM_cols"])
    )
    return a


def _smp(a):
    return a.obs["sample_id"].astype(str).to_numpy()


def _ct_usable(a):
    ct = a.obs["cell_type"].astype(str).to_numpy().astype(object)
    ct[np.asarray(a.layers["counts"]).sum(axis=1) <= 0] = np.nan
    return ct


def test_columns_and_order_match_fixture(adata, R):
    assert strs(R["inputs"]["obs_columns"]) == list(adata.obs.columns)


def test_index_cells(adata, R):
    ct, smp = _ct_usable(adata), _smp(adata)
    c = R["cells"]
    assert (~pd.isna(pd.Series(ct))).sum() == adata.n_obs - int(c["n_unusable"])
    for k, key, mc in [
        ("A", "ik", 10),
        ("B", "ik_b", 10),
        ("A", "ik_hi", 245),
        ("A", "ik_none", 10**5),
    ]:
        np.testing.assert_array_equal(_index_cells(ct, smp, k, mc), idx0(c[key]))
    ct_na = ct.copy()
    ct_na[(smp == "S2") & (ct_na == "A")] = np.nan
    np.testing.assert_array_equal(_index_cells(ct_na, smp, "A", 10), idx0(c["ik_na"]))


def test_niche_columns(adata, R):
    NM = adata.obsm["Niche30"]
    ct, smp = _ct_usable(adata), _smp(adata)
    ik = _index_cells(ct, smp, "A", 10)
    r = R["niche_cols"]

    def chk(got, key):
        assert got["cols"] == strs(r[key]["cols"]), key
        assert got["tested"] == strs(r[key]["tested"]), key

    chk(_niche_columns(NM, "A", ik), "all")
    chk(_niche_columns(NM, "A", ik, niche="C"), "only_C")
    ikb = _index_cells(ct, smp, "B", 10)
    chk(_niche_columns(NM, "B", ikb), "b_all")
    chk(_niche_columns(NM, "A", ik, group_map={"B": ["A", "B"]}), "grp")
    ikc = _index_cells(ct, smp, "C", 10)
    chk(_niche_columns(NM, "C", ikc, niche=["B", "A"], group_map={"A": ["A", "B"]}), "grp_niche")
    chk(
        _niche_columns(pd.DataFrame(R["inputs"]["NM_const_C"], columns=NM.columns), "A", ik),
        "const",
    )
    chk(_niche_columns(pd.DataFrame(R["inputs"]["NM_na_B"], columns=NM.columns), "A", ik), "na")


def test_pick_sigma(adata, R):
    r = R["pick_sigma"]
    a = adata
    del a.obsm["Niche30"]
    a.obsm["Niche10"] = np.zeros((a.n_obs, 1))
    assert _pick_sigma(a, None, "Niche") == r["only"]
    a.obsm["Niche30"] = np.zeros((a.n_obs, 1))
    a.obsm["Niche2.5"] = np.zeros((a.n_obs, 1))
    assert _pick_sigma(a, 2.5, "Niche") == r["given"]
    with pytest.raises(ValueError) as e:
        _pick_sigma(a, None, "Niche")
    assert str(e.value) == r["multi_msg"]
    with pytest.raises(ValueError) as e:
        _pick_sigma(a, [1, 2], "Niche")
    assert str(e.value) == r["len_msg"]
    with pytest.raises(ValueError) as e:
        _pick_sigma(a, None, "Foo")
    assert str(e.value) == r["none_msg"]


def test_patient_table(adata, R):
    r = R["patient_table"]
    pt = _patient_table(adata.obs, _smp(adata))
    assert list(pt.columns) == strs(r["names"])
    assert pt["patient"].tolist() == strs(r["patient"])
    np.testing.assert_array_equal(pt["ncells"].to_numpy(), a1(r["ncells"]))
    for c in ["sample_id", "condition", "slide"]:
        assert pt[c].astype(str).tolist() == strs(r[c]), c
    np.testing.assert_allclose(pt["Age"], r["Age"], **RT)
    np.testing.assert_allclose(
        pt["qc"], r["qc"], **RT, equal_nan=True
    )  # NaN on S3: first cell is NA
    assert np.isnan(pt.loc[pt["patient"] == "S3", "qc"]).all()


def test_patient_table_orders_patients_like_r(adata):
    smp = _smp(adata).astype(object)
    smp[smp == "S1"] = "S10"
    pt = _patient_table(adata.obs.assign(sample_id=smp), smp)
    assert pt["patient"].tolist() == r_sort_str(set(smp))
    assert pt["patient"].tolist()[0] == "S10"


def test_cell_covariates(adata, R):
    Y = np.asarray(adata.layers["counts"]).T
    for key, cvs, depth in [
        ("loglib", ["Area"], "loglib"),
        ("none", ["Area"], "none"),
        ("nonlinear", [], "nonlinear"),
        ("two", ["Area", "Age"], "loglib"),
    ]:
        got = _cell_covariates(adata.obs, cvs, depth, Y)
        assert list(got.columns) == strs(R["cell_cov"][key]["cols"]), key
        np.testing.assert_allclose(got.to_numpy(), R["cell_cov"][key]["m"], **RT, equal_nan=True)
        assert got.index.equals(adata.obs.index)
    e = _cell_covariates(adata.obs, [], "none", Y)
    assert e.shape == (adata.n_obs, 0)
    assert list(a1(R["cell_cov"]["none_empty"]["dim"])) == [adata.n_obs, 0]


def test_cell_covariates_sparse_counts_and_no_counts_cell(adata):
    import scipy.sparse as sp

    Y = np.asarray(adata.layers["counts"]).T
    a = _cell_covariates(adata.obs, [], "loglib", sp.csc_matrix(Y))
    b = _cell_covariates(adata.obs, [], "loglib", Y)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    zero = np.flatnonzero(Y.sum(axis=0) == 0)
    assert len(zero) and np.isnan(b["loglib"].to_numpy()[zero]).all()


def test_index_covariates(adata, R):
    Y = np.asarray(adata.layers["counts"]).T
    ik = idx0(R["cells"]["ik"])
    cov = {
        "loglib": _cell_covariates(adata.obs, ["Area"], "loglib", Y),
        "nonlinear": _cell_covariates(adata.obs, [], "nonlinear", Y),
        "two": _cell_covariates(adata.obs, ["Area", "Age"], "loglib", Y),
        "none": _cell_covariates(adata.obs, ["Area"], "none", Y),
        "none_empty": _cell_covariates(adata.obs, [], "none", Y),
    }
    cases = [
        ("loglib", "loglib", "loglib"),
        ("nonlinear", "nonlinear", "two"),
        ("spatial", "spatial_spline", "two"),
        ("spatial_only", "spatial_spline", "nonlinear"),
        ("none", "none", "none"),
        ("none_empty", "none", "none_empty"),
    ]
    for key, depth, cv in cases:
        r = R["index_cov"][key]
        got = _index_covariates(cov[cv].iloc[ik], depth)
        if int(r["null"]):
            assert got is None, key
            continue
        assert list(got.columns) == strs(r["cols"]), key
        np.testing.assert_allclose(got.to_numpy(), r["m"], **RT)
        assert np.abs(got.mean()).max() < 1e-12


def test_index_covariates_nonlinear_names(adata, R):
    Y = np.asarray(adata.layers["counts"]).T
    ik = idx0(R["cells"]["ik"])
    got = _index_covariates(
        _cell_covariates(adata.obs, ["Area"], "loglib", Y).iloc[ik], "nonlinear"
    )
    assert list(got.columns) == ["loglib_ns1", "loglib_ns2", "loglib_ns3", "Area"]


def test_index_offset(adata, R):
    r = R["index_offset"]
    ik = idx0(R["cells"]["ik"])
    got = _index_offset(adata, "off", strs(r["genes"]), ik)
    assert got.shape == (2, len(ik))
    np.testing.assert_allclose(got, r["m"], **RT)
    adata.layers["off"][ik[6], 1] = np.inf
    with pytest.raises(ValueError) as e:
        _index_offset(adata, "off", list(adata.var_names), ik)
    assert str(e.value) == r["bad_msg"]


# ---- depth blocks ---------------------------------------------------------------------
def _fit_inputs(adata, R):
    ik = idx0(R["cells"]["ik"])
    smp = _smp(adata)
    levels = r_sort_str(set(smp[ik]))
    codes = np.array([levels.index(s) for s in smp[ik]])
    Y = np.asarray(adata.layers["counts"]).T
    ell = _cell_covariates(adata.obs, [], "loglib", Y)["loglib"].to_numpy()
    return ik, codes, levels, ell


def test_depth_blocks(adata, R):
    ik, codes, levels, ell = _fit_inputs(adata, R)
    d = R["depth"]
    L = pd.DataFrame(d["L"], columns=["B", "C"])
    blk = _depth_blocks(ell[ik], adata.obsm["spatial"], adata.obs["section"], ik, codes, levels, L)
    assert list(blk["Zc"].columns) == strs(d["Zc_cols"])
    np.testing.assert_allclose(blk["Zc"].to_numpy(), d["Zc"], **RT)
    for s in range(len(levels)):
        np.testing.assert_allclose(blk["Zs"][s], d["Zs"][f"s{s + 1}"], **RT)
    np.testing.assert_allclose(blk["r2"].to_numpy(), d["r2"], rtol=1e-9, atol=1e-12)
    assert blk["r2"].index.tolist() == levels
    assert [z.shape[1] for z in blk["Zs"]] == a1(d["ncols"]).tolist()
    # constant niche column: tss == 0, r2 stays NA
    L2 = L.copy()
    L2["B"] = 1.0
    r2c = _depth_blocks(
        ell[ik], adata.obsm["spatial"], adata.obs["section"], ik, codes, levels, L2
    )["r2"]
    np.testing.assert_allclose(r2c.to_numpy(), d["r2_const"], rtol=1e-9, atol=1e-12, equal_nan=True)


def test_depth_blocks_df_by_section_size_and_padding(adata, R):
    """testthat 'depth blocks: df by section size, padded compactly, R2 diagnostic' on the
    fixture with sections added: S1 two sections (df 2), S3 one tiny (df 0) and one df 2."""
    ik, codes, levels, ell = _fit_inputs(adata, R)
    L = pd.DataFrame(R["depth"]["L"], columns=["B", "C"])
    sec = adata.obs["section"].to_numpy()
    blk = _depth_blocks(ell[ik], adata.obsm["spatial"], sec, ik, codes, levels, L)
    widths = [z.shape[1] for z in blk["Zs"]]
    expect = []
    for s in range(len(levels)):
        i = np.flatnonzero(codes == s)
        w = 1
        for sg in dict.fromkeys(sec[ik][i]):
            n = int((sec[ik][i] == sg).sum())
            w += 1 + (9 if n >= 200 else 4 if n >= 100 else 0)
        expect.append(w)
    assert widths == expect
    sizes = [int((sec[ik] == sg).sum()) for sg in dict.fromkeys(sec[ik])]
    assert any(n >= 200 for n in sizes) and any(100 <= n < 200 for n in sizes)  # df 3 and df 2
    assert any(n < 100 for n in sizes)  # df 0: the log library size alone
    assert max(len(dict.fromkeys(sec[ik][codes == s])) for s in range(len(levels))) == 2
    assert blk["Zc"].shape[1] == max(widths)
    for s, z in enumerate(blk["Zs"]):
        i = codes == s
        w = z.shape[1]
        np.testing.assert_array_equal(blk["Zc"].to_numpy()[i, :w], z)
        assert np.all(blk["Zc"].to_numpy()[i, w:] == 0)
        assert np.all(z[:, 0] == 1)
    r2 = blk["r2"].to_numpy()
    assert np.all((r2 >= 0) & (r2 <= 1))


def test_depth_blocks_nonfinite_depth_is_an_error(adata, R):
    ik, codes, levels, ell = _fit_inputs(adata, R)
    L = pd.DataFrame(R["depth"]["L"], columns=["B", "C"])
    bad = ell[ik].copy()
    bad[-1] = np.nan
    with pytest.raises(ValueError) as e:
        _depth_blocks(bad, adata.obsm["spatial"], adata.obs["section"], ik, codes, levels, L)
    assert str(e.value) == R["depth"]["bad_msg"]


# ---- the design -----------------------------------------------------------------------
def _config_inputs(adata, R, name):
    """L, cov, trt and strata of a config, built the way fitSpiDE builds them."""
    ik, codes, levels, _ = _fit_inputs(adata, R)
    fi = R["fit_inputs"]
    L = pd.DataFrame(fi["L"], columns=strs(fi["L_cols"]))
    depth = str(R[name]["depth"])
    cvs = {"c1": ["Area"], "c3": ["Area"], "c6": ["Area"]}.get(name, [])
    Y = np.asarray(adata.layers["counts"]).T
    cov = _cell_covariates(adata.obs, cvs, depth, Y)
    covk = _index_covariates(cov.iloc[ik], depth)
    blocks = None
    if depth == "spatial_spline":
        blocks = _depth_blocks(
            cov["loglib"].to_numpy()[ik],
            adata.obsm["spatial"],
            adata.obs["section"].to_numpy(),
            ik,
            codes,
            levels,
            L,
        )
    trt = a1(R["inputs"]["trt_cell"]) if name in ("c2", "c4") else None
    strata = a1(R["inputs"]["slide_cell"]) if name in ("c2", "c4") else None
    if name == "c5":
        strata = np.array(["X"] * len(ik))
    tested = strs(fi["L_cols"]) if name == "c6" else ["C"]
    return L, covk, codes, levels, trt, strata, tested, blocks


@pytest.mark.parametrize("name", CONFIGS)
def test_index_design_matches_r(adata, R, name):
    L, covk, codes, levels, trt, strata, tested, blocks = _config_inputs(adata, R, name)
    got = _index_design(
        L, covk, codes, levels, trt=trt, strata=strata, tested=tested, blocks=blocks
    )
    cfg = R[name]
    want = r_design(cfg)
    assert got["colnames"] == want["colnames"]
    for k in ["patient", "dense", "intercept_cols", "tested", "niche_cols", "start", "pen"]:
        np.testing.assert_allclose(got[k], want[k], **RT, err_msg=k)
    assert got["patients"] == want["patients"] and got["npat"] == want["npat"]
    assert got["tested_niche"] == want["tested_niche"]
    if want["absorb"] is None:
        assert got["absorb"] is None and got["Zs"] is not None
        assert isinstance(got["W"], NBBlockDesign)
        np.testing.assert_allclose(got["W"].X.numpy(), cfg["X"], **RT)
        np.testing.assert_allclose(got["W"].Z.numpy(), cfg["Zc"], **RT)
        for s in range(got["npat"]):
            np.testing.assert_allclose(got["Zs"][s], cfg["Zs"][f"s{s + 1}"], **RT)
    else:
        np.testing.assert_array_equal(got["absorb"], want["absorb"])
        assert got["Zs"] is None
        np.testing.assert_allclose(got["W"], cfg["W"], **RT)
    assert got["W"].shape[1] == len(got["colnames"])


def test_index_design_names_and_zero_stratum_columns(adata, R):
    L, covk, codes, levels, trt, strata, tested, blocks = _config_inputs(adata, R, "c2")
    got = _index_design(L, covk, codes, levels, trt=trt, strata=strata, tested=["C"])
    cn = got["colnames"]
    assert cn[:3] == ["patient:S1", "patient:S2", "patient:S3"]
    assert "strata:Y:C" in cn and "strata:Z:B" in cn and "condition:C" in cn
    assert cn.index("strata:Y:C") < cn.index("strata:Z:C") < cn.index("strata:Y:B")
    assert cn[got["tested"][0]] == "condition:C"
    # zero strata x niche products are dropped
    s = R["strata_zero"]
    Lz = pd.DataFrame(s["L"], columns=["C", "B"])
    d = _index_design(Lz, None, codes, levels, strata=a1(R["inputs"]["slide_cell"]), tested=["C"])
    assert d["colnames"] == strs(s["colnames"])
    np.testing.assert_allclose(d["W"], s["W"], **RT)
    assert "strata:Z:B" not in d["colnames"]


def test_index_design_minimal_and_categorical_strata(adata, R):
    m = R["d_min"]
    ik, codes, levels, _ = _fit_inputs(adata, R)
    d = _index_design(pd.DataFrame(m["L"], columns=["C"]), None, codes, levels)
    assert d["colnames"] == strs(m["colnames"])
    np.testing.assert_allclose(d["W"], m["W"], **RT)
    assert d["dense"].tolist() == [int(m["dense"]) - 1] and d["tested"].tolist() == [
        int(m["tested"]) - 1
    ]
    # an already-factor strata keeps its levels (and drops unused ones)
    st = pd.Categorical(a1(R["inputs"]["slide_cell"]), categories=["X", "Y", "Z", "Q"])
    d2 = _index_design(pd.DataFrame(m["L"], columns=["C"]), None, codes, levels, strata=st)
    assert d2["colnames"] == ["patient:" + p for p in levels] + ["strata:Y:C", "strata:Z:C", "C"]


# ---- small linear-algebra helpers ------------------------------------------------------
@pytest.mark.parametrize("name", CONFIGS)
def test_lin_pred_dense_x_partial_block(R, name):
    cfg = R[name]
    des = r_design(cfg)
    alpha = np.asarray(cfg["alpha"])
    for g in (1, 3):
        np.testing.assert_allclose(_lin_pred(des["W"], alpha[g - 1]), cfg[f"lp{g}"], **RT)
    X = _dense_x(des)
    np.testing.assert_allclose(X, cfg["dense_x"], **RT)
    rows_of = [np.flatnonzero(des["patient"] == s) for s in range(des["npat"])]
    Zs = _patient_blocks(des, rows_of)
    assert len(Zs) == des["npat"] and Zs[1].shape[1] == int(cfg["z2_ncol"])
    i = rows_of[1]
    w = np.asarray(cfg["wr1"]["w"])[i]
    got = _partial_block(X[i], w, Zs[1])
    np.testing.assert_allclose(got, cfg["partial"], rtol=1e-9, atol=1e-10)


def test_partial_block_misc(R):
    m = R["partial_misc"]
    X, Z, w = (np.asarray(m[k]) for k in ("X", "Z", "w"))
    assert int(m["singular_null"]) == 1 and _partial_block(X, w, Z) is None
    np.testing.assert_allclose(_partial_block(X, w, Z[:, :2]), m["ok"], **RT)
    np.testing.assert_allclose(_partial_block(X, w, np.ones((10, 1))), m["centre"], **RT)
    # the residual is orthogonal (weighted) to the block
    r = _partial_block(X, w, Z[:, :2])
    np.testing.assert_allclose(Z[:, :2].T @ (r * w[:, None]), 0, atol=1e-12)


# --------------------------------------------------------------------------------------
# carry-overs from the Task 6 / 8 reviews (fixed in Task 9)
# --------------------------------------------------------------------------------------
def test_levels_sort_numbers_numerically_like_r_factor():
    from spidepy._design import _factor_codes, _levels

    # factor(c(10, 9, 10, NA)) has levels "9" "10"; strings sort as strings
    assert _levels(np.array([10.0, 9.0, 10.0, np.nan])) == ["9", "10"]
    lv, codes = _factor_codes(np.array([10, 9, 10]))
    assert lv == ["9", "10"] and codes.tolist() == [1, 0, 1]
    lv, codes = _factor_codes(np.array(["10", "9", None, "10"], dtype=object))
    assert lv == ["10", "9"] and codes.tolist() == [0, 1, -1, 0]
    cat = pd.Categorical(["b", "a", None, "b"], categories=["c", "b", "a"])
    lv, codes = _factor_codes(cat)  # category order kept, unused "c" dropped
    assert lv == ["b", "a"] and codes.tolist() == [0, 1, -1, 0]


def test_numeric_strata_use_numeric_level_order():
    L = pd.DataFrame({"B": np.linspace(0.1, 1, 6)})
    codes = np.zeros(6, dtype=int)
    des = _index_design(L, None, codes, ["p"], strata=np.array([10, 9, 10, 2, 9, 2]), tested=["B"])
    # levels 2, 9, 10: treatment coding drops "2" and names the others in numeric order
    assert [c for c in des["colnames"] if c.startswith("strata:")] == ["strata:9:B", "strata:10:B"]


def test_pick_sigma_unparseable_key_is_na_like_r(adata):
    a = adata
    del a.obsm["Niche30"]
    a.obsm["Niche1.2.3"] = np.zeros((a.n_obs, 1))  # matches ^Niche[0-9.]+$, as.numeric -> NA
    assert np.isnan(_pick_sigma(a, None, "Niche"))
    a.obsm["Niche5"] = np.zeros((a.n_obs, 1))
    with pytest.raises(ValueError, match=r"\(5\): choose one with 'sigma'"):
        _pick_sigma(a, None, "Niche")
    assert _pick_sigma(a, 5, "Niche") == 5.0


def test_index_offset_unknown_gene_is_out_of_bounds(adata):
    ik = np.arange(5)
    with pytest.raises(ValueError, match="subscript out of bounds"):
        _index_offset(adata, "off", ["G1", "nope"], ik)


def test_index_cells_and_patient_table_handle_missing_labels():
    ct = np.array(["A", "A", np.nan, "A", "A", "B", "A"], dtype=object)
    smp = np.array(["p", "p", "p", np.nan, "q", "q", "q"], dtype=object)
    # R: table() drops the NA sample; q has 2 "A" cells (4, 6), p has 2 (0, 1)
    np.testing.assert_array_equal(_index_cells(ct, smp, "A", 2), [0, 1, 4, 6])
    np.testing.assert_array_equal(_index_cells(ct, smp, "A", 3), [])
    cd = pd.DataFrame(
        {
            "g": pd.Categorical(["x", "x", "x", "y", "z", "z", "z"], categories=["z", "y", "x"]),
            "num": [1.0, 1.0, 2.0, 5.0, 3.0, 3.0, np.nan],
            "vary": [1, 2, 1, 1, 1, 1, 1],
        }
    )
    pt = _patient_table(cd, smp)
    assert pt["patient"].tolist() == ["p", "q"]
    assert pt["ncells"].tolist() == [3, 3]
    assert list(pt.columns) == ["patient", "ncells", "g"]  # num varies in p, vary in p
    assert pt["g"].tolist() == ["x", "z"]
    assert list(pt["g"].cat.categories) == ["z", "x"]
