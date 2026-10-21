"""Task 14: the plot theme, colours, labels and shared data builders (R/plot-theme.R,
R/plot-helpers.R; tests ``test-plot-helpers.R``)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from _fitref import close, frame, strs  # noqa: E402
from _plotref import fit_from_dump, plot_ref, plot_spe, res_from_dump  # noqa: E402
from spidepy._checks import check_results, check_triplet  # noqa: E402
from spidepy._io import get_counts  # noqa: E402
from spidepy._test_slopes import _dl_tau2  # noqa: E402
from spidepy.pl import _helpers as H  # noqa: E402
from spidepy.pl import _theme as T  # noqa: E402
from spidepy.pl import spide_colours, style  # noqa: E402


@pytest.fixture(scope="module")
def R():
    return plot_ref()


@pytest.fixture(scope="module")
def spe():
    return plot_spe()


@pytest.fixture(scope="module")
def fit(R):
    return fit_from_dump(R["fit"])


@pytest.fixture(scope="module")
def res(R):
    return res_from_dump(R["res"])


# --- colours and theme ------------------------------------------------------------------------


def test_fixed_colours_are_rs_in_rs_order(R):
    got = spide_colours()
    assert list(got) == ["up", "down", "first", "second", "mid", "faint", "light", "grey"]
    assert list(got.values()) == strs(R["cols"]["fixed"])
    assert spide_colours("direction") == ["#B8791A", "#5B4B9A"] == strs(R["cols"]["direction"])
    assert spide_colours("condition") == ["#0A9396", "#AE2012"] == strs(R["cols"]["condition"])


@pytest.mark.parametrize("n", [1, 2, 5, 9, 11, 20])
def test_ramps_equal_rs_hex_values(R, n):
    assert spide_colours("magnitude", n) == strs(R["cols"][f"magnitude{n}"])
    assert spide_colours("diverging", n) == strs(R["cols"][f"diverging{n}"])


@pytest.mark.parametrize("n", [1, 5, 8, 9, 10, 12, 20])
def test_nuisance_is_okabe_ito_then_the_hue_wheel(R, n):
    got = spide_colours("nuisance", n)
    assert got == strs(R["cols"][f"nuisance{n}"])
    assert len(got) == n


def test_internal_ramp_with_skip_equals_r(R):
    assert T._magnitude_ramp(256, skip=0.1) == strs(R["cols"]["ramp256_skip"])
    assert T._magnitude_ramp(7, skip=0.3) == strs(R["cols"]["ramp7_skip"])


def test_magnitude_ramp_darkens():
    lum = [
        mpl.colors.to_rgb(c) @ np.array([0.2126, 0.7152, 0.0722])
        for c in spide_colours("magnitude", 9)
    ]
    assert np.all(np.diff(lum) < 0)


def test_colour_arguments_are_checked():
    with pytest.raises(ValueError, match="'role' should be one of"):
        spide_colours("nope")
    for bad in (0, -1, float("nan"), "3", None):
        with pytest.raises(ValueError, match="'n' must be a positive integer"):
            spide_colours("magnitude", bad)


def test_condition_colours_by_level_count():
    assert T._condition_colours(["a", "b"]) == {"a": "#0A9396", "b": "#AE2012"}
    assert T._condition_colours(["all patients"]) == {"all patients": "black"}
    assert T._direction_colours() == {"up": "#B8791A", "down": "#5B4B9A"}
    cmap, norm = T._t_cmap_norm(4.0)
    assert mpl.colors.to_hex(cmap(norm(0.0))) == "#f7f7f7"
    assert mpl.colors.to_hex(cmap(norm(99.0))) == "#b8791a"  # beyond the limit: squished


def test_style_is_a_context_that_restores_rcparams():
    before = dict(mpl.rcParams)
    with style():
        assert mpl.rcParams["axes.edgecolor"] == "black"
        assert mpl.rcParams["axes.grid"] is False
        assert mpl.rcParams["xtick.color"] == "black"
        assert mpl.rcParams["xtick.labelsize"] == pytest.approx(11 * 1.1)
        assert mpl.rcParams["axes.labelsize"] == pytest.approx(11 * 1.1 * 1.1)
        fig = Figure()
        ax = fig.subplots()
        assert all(sp.get_visible() for sp in ax.spines.values())
        assert ax.spines["left"].get_edgecolor()[:3] == (0.0, 0.0, 0.0)
    assert dict(mpl.rcParams) == before
    with style(rl=2):
        assert mpl.rcParams["xtick.labelsize"] == pytest.approx(22)
    for bad in (-1, 0, float("nan"), "a", True):
        with pytest.raises(ValueError, match="'rl' must be a single positive number"):
            with style(bad):
                pass


def test_legend_titles_are_italic():
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], label="x")
    leg = ax.legend(title="title")
    T._style_legend(leg)
    assert leg.get_title().get_fontstyle() == "italic"


# --- labels -----------------------------------------------------------------------------------


def test_plotmath_strings_equal_rs(R):
    assert T._it_triplet('a"b\\c', "T cell", "B|x") == str(R["lab"]["trip"])
    assert T._it_gene('a"b\\c') == str(R["lab"]["gene"])
    assert T._it_gene("G1") == 'italic("G1")'


def test_fmt_q_equals_r_including_its_padding_of_one(R):
    got = T._fmt_q(R["lab"]["fmtq_in"])
    assert list(got) == strs(R["lab"]["fmtq"])
    assert list(T._fmt_q([0.0234, 2e-8, np.nan])) == ["0.023", "2e-08", "NA"]
    assert T._fmt_q(1.0)[0] == "  1"


NASTY = [
    'a"b\\c',
    "HLA-DR",
    "a_b",
    "x y",
    "a$b",
    "{x}",
    "50%",
    "a#b&c",
    "a^b~c",
    "µ-α",
    "",
    "a'b|c/d",
]


@pytest.mark.parametrize("gene", NASTY)
def test_mathtext_gene_labels_render_literally(gene):
    from matplotlib.mathtext import MathTextParser

    p = MathTextParser("path")
    p.parse(T._mt_gene(gene), dpi=100)  # raises on markup mathtext cannot read
    p.parse(T._mt_triplet(gene, "T cell", gene), dpi=100)
    fig = Figure()
    ax = fig.subplots()
    ax.set_title(T._mt_gene(gene))
    fig.savefig(__import__("io").BytesIO(), format="png")


def test_mathtext_gene_is_italic_and_hyphen_is_not_a_spaced_minus():
    from matplotlib.mathtext import MathTextParser

    p = MathTextParser("path")
    w_hyphen = p.parse(T._mt_gene("HLA-DR"), dpi=100)[0]
    w_plain = p.parse(T._mt_gene("HLADR"), dpi=100)[0]
    assert w_hyphen - w_plain < 12  # one glyph, not a minus with operator spacing
    assert T._mt_triplet("G1", "A", "B") == "$\\mathit{G1}\\ \\ \\mathrm{A\\ \\text{|}\\ B}$"


def test_wrap_dims_are_ggplots():
    got = {n: T._wrap_dims(n) for n in (1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 17)}
    assert got == {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (3, 3),
                   9: (3, 3), 10: (3, 4), 12: (3, 4), 13: (4, 4), 14: (4, 4), 17: (4, 5)}  # fmt: skip


# --- data builders ----------------------------------------------------------------------------


def test_slope_weights_are_the_pooled_tests_over_the_whole_gene_family(R, fit):
    v, tau = H._slope_weights(fit.index["A"])
    close(v, R["helpers"]["slope_v"], "v")
    close(tau, R["helpers"]["slope_tau"], "tau")
    xi = fit.index["A"]
    fac = xi["factor"].to_numpy()[None, :, :]
    close(v, xi["v_model"] * fac, "v direct")
    close(tau, _dl_tau2(xi["beta"], v), "tau direct")


def test_slope_weights_join_the_factor_by_patient_name(fit):
    xi = fit.index["A"]
    shuffled = dict(xi)
    shuffled["factor"] = xi["factor"].iloc[::-1]  # same values, rows in another order
    v0, t0 = H._slope_weights(xi)
    v1, t1 = H._slope_weights(shuffled)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_array_equal(t0, t1)


def test_cell_type_means_are_the_dense_cp10k_means(R, spe):
    M, n = H._cell_type_means(spe, ["G1", "G2"], "counts", "cell_type")
    close(M.to_numpy(), R["helpers"]["ctm"], "means")
    assert list(M.index) == ["G1", "G2"] and list(M.columns) == strs(R["helpers"]["ctm_types"])
    np.testing.assert_array_equal(n.to_numpy(), R["helpers"]["ctm_n"])
    # an independent dense computation
    Y = np.asarray(get_counts(spe, "counts"))
    lib = Y.sum(axis=0)
    ct = spe.obs["cell_type"].to_numpy()
    for k in "ABC":
        i = (lib > 0) & (ct == k)
        ref = (Y[:2][:, i] / lib[i] * 1e4).mean(axis=1)
        np.testing.assert_allclose(M[k].to_numpy(), ref, rtol=1e-10)
        assert n[k] == i.sum()


def test_cell_type_means_name_missing_genes(spe):
    with pytest.raises(ValueError, match="not in 'adata': nope"):
        H._cell_type_means(spe, ["G1", "nope"])


def test_cell_type_means_sparse_equals_dense(spe):
    import scipy.sparse as sp

    sparse = spe.copy()
    sparse.X = sp.csr_matrix(sparse.X)
    sparse.layers["counts"] = sp.csr_matrix(sparse.layers["counts"])
    a, na = H._cell_type_means(spe, ["G3", "G1"])
    b, nb = H._cell_type_means(sparse, ["G3", "G1"])
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)
    np.testing.assert_array_equal(na, nb)


def test_niche_fold_equals_rs(R, spe, fit, res):
    tab = res.table
    close(H._niche_fold(tab, spe, fit), R["helpers"]["fold"], "fold")
    close(
        H._niche_fold(tab[tab["gene"] == "G1"], spe, fit),
        np.atleast_1d(R["helpers"]["fold_g1"]),
        "fold g1",
    )


@pytest.mark.parametrize("topic, key", [("plot_spe_bc", "bc"), ("plot_spe_b", "b")])
def test_niche_fold_pools_the_members_of_a_merged_niche(R, topic, key):
    sm = plot_spe(topic)
    rm = res_from_dump(R[key]["res"])
    got = H._niche_fold(rm.table, sm, rm.fit)
    close(got, R[key]["fold"], f"fold {key}")
    close(
        H._niche_fold(rm.table[rm.table["gene"] == "G1"], sm, rm.fit),
        np.atleast_1d(R[key]["fold_g1"]),
        "g1",
    )
    # by hand: pooled members' CP10k mean over the index type's, pseudo count on both
    M, n = H._cell_type_means(sm, ["G1"])
    pooled = (M.loc["G1", "B"] * n["B"] + M.loc["G1", "C"] * n["C"]) / (n["B"] + n["C"])
    t = rm.table[rm.table["gene"] == "G1"]
    first = (t["niche"] == ("BC" if key == "bc" else "B")).to_numpy().argmax()
    assert got[rm.table.index.get_loc(t.index[first])] == pytest.approx(
        (pooled + 0.01) / (M.loc["G1", "A"] + 0.01), rel=1e-10
    )


def test_niche_fold_refuses_a_foreign_adata_and_names_missing_genes(R, spe, fit, res):
    tab = res.table[res.table["gene"] == "G1"].copy()
    tab["gene"] = "nope"
    with pytest.raises(ValueError, match="not in 'adata': nope"):
        H._niche_fold(tab, spe, fit)
    rm = res_from_dump(R["bc"]["res"])  # a merged-niche table against the unmerged adata
    with pytest.raises(ValueError, match="made from"):
        H._niche_fold(rm.table, spe, rm.fit)


def test_expression_bands_split_the_genes_into_fifths(R, fit):
    for k, key, lev in ((5, "bands", "bands_levels"), (3, "bands3", "bands3_levels")):
        b = H._expression_bands(fit, k)
        ref = frame(R["helpers"][key])
        assert list(b.columns) == ["gene", "index", "band"]
        assert [str(x) for x in b["gene"]] == strs(ref["gene"])
        assert [str(x) for x in b["band"]] == strs(ref["band"])
        assert list(b["band"].cat.categories) == strs(R["helpers"][lev])
    b = H._expression_bands(fit)
    assert b["band"].nunique() == 5
    assert b["band"].value_counts().max() - b["band"].value_counts().min() <= 1
    assert list(b["band"].cat.categories)[0].endswith("(dimmest)")


def test_index_cell_data_are_the_fits_cells(R, spe, fit):
    d = H._index_cell_data(spe, fit, "A", "B", ["G2", "G1"])
    h = R["helpers"]
    close(d["counts"], h["icd_counts"], "counts")
    close(d["lib"], h["icd_lib"], "lib")
    close(d["density"], h["icd_density"], "density")
    assert list(d["patient"]) == strs(h["icd_patient"])
    assert d["counts"].shape == (2, len(d["patient"]))
    assert set(d["patient"]) == set(fit.index["A"]["patients"])


def test_index_cell_data_refuse_an_adata_that_is_not_the_fits(spe, fit):
    other = spe[spe.obs["sample_id"].isin(["S1", "S2"])].copy()
    other.obs["sample_id"] = "X" + other.obs["sample_id"].astype(str)
    with pytest.raises(ValueError, match="the object the fit was made from"):
        H._index_cell_data(other, fit, "A", "B", ["G1"])
    ct = spe.obs["cell_type"].to_numpy()
    keep = ~((ct == "A") & (np.arange(len(ct)) % 2 == 0))
    with pytest.raises(ValueError, match="made from"):
        H._index_cell_data(spe[keep].copy(), fit, "A", "B", ["G1"])
    with pytest.raises(ValueError, match="not in 'adata': zzz"):
        H._index_cell_data(spe, fit, "A", "B", ["G1", "zzz"])


def test_group_levels_and_direction_labels(R, spe, fit, res):
    g = H._group_levels(fit, "condition")
    assert list(g.index) == strs(R["helpers"]["grp_names"])
    assert [str(x) for x in g] == strs(R["helpers"]["grp"])
    assert (
        list(g.cat.categories) == strs(R["helpers"]["grp_levels"]) == ["Non-responder", "Responder"]
    )
    assert list(H._group_levels(fit, None).cat.categories) == ["all patients"]
    assert list(H._group_levels(fit, None).cat.categories) == strs(R["helpers"]["grp_none"])
    assert list(H._direction_labels(res, "condition").values()) == strs(R["helpers"]["dir_cond"])
    assert list(H._direction_labels(res, "condition").values()) == [
        "higher slope in Responder",
        "higher slope in Non-responder",
    ]
    assert list(H._direction_labels(res, "pooled").values()) == strs(R["helpers"]["dir_pooled"])


def test_group_levels_join_the_condition_by_patient_name(fit):
    shuffled = fit.patients.sample(frac=1, random_state=1).reset_index(drop=True)
    f2 = type(fit)(fit.engine, fit.sigma, fit.condition, shuffled, fit.index, fit.params)
    a, b = H._group_levels(fit, "condition"), H._group_levels(f2, "condition")
    assert (a.reindex(b.index).astype(str) == b.astype(str)).all()


def test_resolve_test(R, res, fit):
    assert H._resolve_test(res, None) == str(R["helpers"]["test_cond"]) == "condition"
    rnc = res_from_dump(R["rnc"])
    assert H._resolve_test(rnc, None) == str(R["helpers"]["test_pooled"]) == "pooled"
    assert H._resolve_test(res, "pooled") == "pooled"
    with pytest.raises(ValueError, match="no condition-specific test"):
        H._resolve_test(rnc, "condition")


def test_triplet_rows_follow_the_genes_order(R, res):
    got = H._triplet_rows(res.table, "pooled", ["G3", "G1"], "A", "B")
    ref = frame(R["helpers"]["trip_rows"])
    assert list(got["gene"]) == strs(ref["gene"]) == ["G3", "G1"]
    close(got["estimate"], ref["estimate"], "estimate")
    miss = H._triplet_rows(res.table, "pooled", ["G1", "nope"], "A", "B")
    assert miss["estimate"].isna().tolist() == [False, True]


def test_checkers_name_what_is_wrong(spe, fit, res):
    with pytest.raises(TypeError, match="SpiDEResults"):
        check_results(fit, "f()")
    assert check_results(fit, "f()", fit_ok=True) is True
    with pytest.raises(ValueError, match="fitted index types: A"):
        check_triplet(fit, "Z")
    with pytest.raises(ValueError, match="niches tested in A"):
        check_triplet(fit, "A", "Z")
    with pytest.raises(ValueError, match="not tested in A: nope"):
        check_triplet(fit, "A", "B", ["G1", "nope"])
