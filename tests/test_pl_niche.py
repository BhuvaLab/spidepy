"""Task 14: plot_niche_map and plot_niche_response (R/plot-niche.R; tests
``test-plots-niche.R`` and ``test-plots-devices.R``).

The data builders are compared with the data frames of R's ggplot objects (``p$data`` and the
layers' data) on R's own fits; the figures are checked for structure (panels, titles, labels,
legends, plotted points) and R's error messages, and one PNG per plot is written to a tmp dir.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import spidepy  # noqa: E402
from _fitref import frame  # noqa: E402
from _plotref import fit_from_dump, plot_ref, plot_spe, res_from_dump  # noqa: E402
from spidepy import pl  # noqa: E402
from spidepy.pl import _niche as N  # noqa: E402
from spidepy.pl import _theme as T  # noqa: E402


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


def same_frame(got: pd.DataFrame, ref: dict, what: str, rtol=1e-10, atol=1e-12):
    """A DataFrame against R's data frame: column names, row order, labels, numbers at 1e-10."""
    ref = frame(ref)
    assert list(got.columns) == list(ref.columns), what
    assert len(got) == len(ref), f"{what}: {len(got)} rows vs {len(ref)}"
    for c in ref.columns:
        a, b = got[c].to_numpy(), ref[c].to_numpy()
        if b.dtype.kind in "fiu":
            np.testing.assert_array_equal(np.isnan(a.astype(float)), np.isnan(b.astype(float)), c)
            ok = ~np.isnan(b.astype(float))
            np.testing.assert_allclose(
                a.astype(float)[ok],
                b.astype(float)[ok],
                rtol=rtol,
                atol=atol,
                err_msg=f"{what}.{c}",
            )
        elif b.dtype.kind == "b" or a.dtype.kind == "b":
            np.testing.assert_array_equal(a.astype(bool), b.astype(bool), err_msg=f"{what}.{c}")
        else:
            assert [str(x) for x in a] == [str(x) for x in b], f"{what}.{c}"


def panels(fig):
    """The data panels: not the colour bar."""
    return [a for a in fig.axes if a.get_label() != "<colorbar>"]


def n_points(ax):
    return sum(len(c.get_offsets()) for c in ax.collections)


# --- plot_niche_map: data ---------------------------------------------------------------------


def test_map_data_equal_rs(R, spe):
    m = R["maps"]
    same_frame(N._niche_map_data(spe, "B", "S1"), m["plain"]["data"], "plain")
    d = N._niche_map_data(spe, "B", "S1", "A")
    same_frame(d, m["index"]["data"], "index")
    same_frame(N._scale_bar(d, "µm"), m["index"]["bar"], "bar")
    same_frame(N._niche_map_data(spe, "C", "S3"), m["small"]["data"], "small")


def test_map_data_one_row_per_cell_of_the_sample(spe):
    d = N._niche_map_data(spe, "B", "S1")
    assert len(d) == (spe.obs["sample_id"] == "S1").sum()
    assert list(d.columns) == ["x", "y", "cell_type", "density", "section", "focal"]
    nm = spe.obsm["Niche30"]["B"].to_numpy()[(spe.obs["sample_id"] == "S1").to_numpy()]
    np.testing.assert_allclose(d["density"], np.log1p(nm), rtol=1e-12)
    d = N._niche_map_data(spe, "B", "S1", "A")
    assert (
        d["focal"].sum() == ((spe.obs["sample_id"] == "S1") & (spe.obs["cell_type"] == "A")).sum()
    )


def test_each_section_of_a_patient_sits_on_its_own_origin(R, spe):
    s2 = spe.copy()
    s2.obs["section"] = np.atleast_1d(R["maps"]["section_vec"])
    d = N._niche_map_data(s2, "B", "S1", "A", section="section")
    same_frame(d, R["maps"]["section"]["data"], "section")
    same_frame(N._scale_bar(d, None), R["maps"]["section"]["bar"], "section bar")
    assert set(d["section"]) == {"left", "right"}
    np.testing.assert_array_equal(d.groupby("section")["x"].min().to_numpy(), [0, 0])
    np.testing.assert_array_equal(d.groupby("section")["y"].min().to_numpy(), [0, 0])


def test_map_errors_are_rs(spe):
    with pytest.raises(ValueError, match="not a column"):
        pl.plot_niche_map(spe, niche="Z", sample="S1")
    with pytest.raises(ValueError, match="nope"):
        pl.plot_niche_map(spe, niche="B", sample="nope")
    with pytest.raises(ValueError, match="no Z cells in sample S1"):
        pl.plot_niche_map(spe, niche="B", sample="S1", index="Z")
    with pytest.raises(ValueError, match="section column 'zz' not found"):
        pl.plot_niche_map(spe, niche="B", sample="S1", section="zz")
    bare = spe.copy()
    del bare.obsm["Niche30"]
    with pytest.raises(ValueError, match="no niche reducedDims found"):
        pl.plot_niche_map(bare, "B", "S1")


def test_map_needs_sigma_when_several_bandwidths(spe):
    two = spe.copy()
    two.obsm["Niche60"] = two.obsm["Niche30"].copy()
    with pytest.raises(ValueError, match="several niche bandwidths"):
        pl.plot_niche_map(two, niche="B", sample="S1")
    fig = pl.plot_niche_map(two, niche="B", sample="S1", sigma=60)
    assert isinstance(fig, Figure)


# --- plot_niche_map: figure -------------------------------------------------------------------


def test_map_figure_colours_one_samples_cells(spe, tmp_path):
    fig = pl.plot_niche_map(spe, niche="B", sample="S1")
    assert isinstance(fig, Figure)
    ax = panels(fig)
    assert len(ax) == 1
    assert n_points(ax[0]) == (spe.obs["sample_id"] == "S1").sum()
    cb = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cb) == 1 and cb[0].get_ylabel() == "B density\n(log1p)"
    # no tick labels or axis titles, but the panel border stays
    assert ax[0].get_xticks().size == 0 and ax[0].get_xlabel() == ""
    assert all(s.get_visible() for s in ax[0].spines.values())
    fig.savefig(tmp_path / "niche_map.png", dpi=60)
    assert (tmp_path / "niche_map.png").stat().st_size > 1000


def test_map_with_index_greys_the_other_cells_and_draws_the_scale_bar(spe, tmp_path):
    fig = pl.plot_niche_map(spe, niche="B", sample="S1", index="A", unit="µm")
    ax = panels(fig)[0]
    is_s1 = (spe.obs["sample_id"] == "S1").to_numpy()
    n_a = (is_s1 & (spe.obs["cell_type"] == "A").to_numpy()).sum()
    sizes = [len(c.get_offsets()) for c in ax.collections]
    assert sizes == [is_s1.sum() - n_a, n_a]  # grey first, the index type's cells on top
    grey = ax.collections[0].get_facecolor()[0][:3]
    assert tuple(np.round(grey, 3)) == tuple(np.round(matplotlib.colors.to_rgb("#EBEBEB"), 3))
    labels = [t.get_text() for t in ax.texts]
    assert labels and labels[0].endswith(" µm") and labels[0].split()[0].isdigit()
    # the scale bar's label sits below the bar: the panel must leave room for it
    d = N._niche_map_data(spe, "B", "S1", "A")
    dy = d["y"].max() - d["y"].min()
    assert ax.get_ylim()[0] < d["y"].min() - 0.1 * dy
    fig.savefig(tmp_path / "niche_map_index.png", dpi=60)
    no_bar = pl.plot_niche_map(spe, niche="B", sample="S1", index="A", scale_bar=False)
    assert not panels(no_bar)[0].texts


def test_map_one_panel_per_section_titled_by_section(R, spe, tmp_path):
    s2 = spe.copy()
    s2.obs["section"] = np.atleast_1d(R["maps"]["section_vec"])
    fig = pl.plot_niche_map(s2, niche="B", sample="S1", section="section")
    ax = panels(fig)
    assert [a.get_title() for a in ax] == ["left", "right"]
    d = N._niche_map_data(s2, "B", "S1", section="section")
    assert [n_points(a) for a in ax] == [(d["section"] == s).sum() for s in ("left", "right")]
    fig.savefig(tmp_path / "niche_map_sections.png", dpi=60)


def test_map_default_point_size_follows_the_number_of_cells(spe):
    n = (spe.obs["sample_id"] == "S1").sum()
    big = pl.plot_niche_map(spe, "B", "S1", point_size=1.0)
    small = pl.plot_niche_map(spe, "B", "S1", point_size=0.1)
    assert (
        panels(big)[0].collections[0].get_sizes()[0]
        > panels(small)[0].collections[0].get_sizes()[0]
    )
    default = pl.plot_niche_map(spe, "B", "S1")
    want = T._point_area(min(1.5, max(0.05, 50 / np.sqrt(n))))
    assert panels(default)[0].collections[0].get_sizes()[0] == pytest.approx(want)


# --- plot_niche_response: data ----------------------------------------------------------------


def _fits(R):
    return {
        "res": res_from_dump(R["res"]),
        "rs": res_from_dump(R["rs"]),
        "fsp": fit_from_dump(R["fsp"]),
        "fnc": fit_from_dump(R["fnc"]),
    }


RESP = {
    "adj": ("res", dict(gene=["G1", "G2"], niche="B")),
    "adj4": ("res", dict(gene=["G3", "G1", "G2", "G1"], niche="C", bins=4)),
    "raw": ("res", dict(gene=["G1", "G2"], niche="B", adjust=False)),
    "sand": ("rs", dict(gene="G1", niche="B")),
    "sand_raw": ("rs", dict(gene="G1", niche="B", adjust=False)),
    "spline": ("fsp", dict(gene="G1", niche="B")),
    "nocond": ("fnc", dict(gene="G1", niche="B")),
}


@pytest.mark.parametrize("name", list(RESP))
def test_response_data_equal_rs(R, spe, name):
    which, kw = RESP[name]
    D = N._niche_response_data(spe, _fits(R)[which], index="A", **kw)
    ref = R["resp"][name]
    same_frame(D["data"], ref["data"], f"{name}.data")
    for part in ("summary", "lines", "centroids"):
        assert (part in D) == (part in ref), f"{name}: {part}"
        if part in ref:
            same_frame(D[part], ref[part], f"{name}.{part}")


def test_response_data_have_rs_columns_and_one_row_per_gene_patient_bin(R, spe, res):
    D = N._niche_response_data(spe, res, ["G1", "G2"], "A", "B")
    pb = D["data"]
    assert list(pb.columns) == [
        "patient",
        "bin",
        "expr",
        "density",
        "gene",
        "condition",
        "gene_lab",
    ]
    assert not pb.duplicated(["gene", "patient", "bin"]).any()
    assert np.isfinite(pb["expr"]).all()
    assert D["labels"] == {"G1": 'italic("G1")', "G2": 'italic("G2")'}


def test_response_joins_conditions_by_patient_name_not_position(R, spe, res):
    base = N._niche_response_data(spe, res, "G1", "A", "B")["data"]
    shuffled = res.fit.patients.sample(frac=1, random_state=3).reset_index(drop=True)
    f2 = type(res.fit)(res.fit.engine, res.fit.sigma, res.fit.condition, shuffled, res.fit.index,
                       res.fit.params)  # fmt: skip
    r2 = type(res)(
        res.table, res.condition, res.contrast, res.procedure, res.fdr, res.pooled_df, f2
    )
    again = N._niche_response_data(spe, r2, "G1", "A", "B")["data"]
    pd.testing.assert_frame_equal(base.reset_index(drop=True), again.reset_index(drop=True))
    truth = res.fit.patients.set_index("patient")["condition"]
    assert (
        base["condition"].astype(str).to_numpy() == truth.reindex(base["patient"]).to_numpy()
    ).all()


def test_response_drops_a_gene_the_fit_could_not_fit(R, spe):
    f2 = fit_from_dump(R["f2"])
    with pytest.warns(UserWarning, match="G2 not fitted in A, left out"):
        D = N._niche_response_data(spe, f2, ["G1", "G2"], "A", "B")
    assert set(D["data"]["gene"]) == {"G1"}
    same_frame(D["data"], R["resp"]["dropped"]["data"], "dropped")
    # the raw display needs no intercepts: both genes stay
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw = N._niche_response_data(spe, f2, ["G1", "G2"], "A", "B", adjust=False)
    assert set(raw["data"]["gene"]) == {"G1", "G2"}
    same_frame(raw["centroids"], R["resp"]["dropped_raw"]["centroids"], "dropped raw")
    with pytest.warns(UserWarning, match="G2"):
        with pytest.raises(ValueError, match="no gene left to plot"):
            N._niche_response_data(spe, f2, "G2", "A", "B")


def test_response_references_the_median_intercept_robust_to_one_extreme(R, spe, fit):
    f3 = fit_from_dump(R["f3"])  # patient S2's intercept of G1 pinned at -50
    p1 = N._niche_response_data(spe, fit, "G1", "A", "B")["data"]
    p2 = N._niche_response_data(spe, f3, "G1", "A", "B")["data"]
    same_frame(p2, R["resp"]["extreme"]["data"], "extreme")
    s1 = lambda p: p.loc[p["patient"] == "S1", "expr"].to_numpy()  # noqa: E731
    assert np.max(np.abs(np.log(s1(p2) / s1(p1)))) < 0.5


def test_adjusted_display_does_not_remove_depth_twice_under_spatial_spline(R, spe):
    fsp = fit_from_dump(R["fsp"])
    p = N._niche_response_data(spe, fsp, "G1", "A", "B")["data"]
    p = p[p["expr"] > 0]
    lev = np.log(p["expr"]).groupby(p["patient"]).mean()
    d = fsp.index["A"]["loglib_mean"].reindex(lev.index)
    slope = np.polyfit(d.to_numpy(), lev.to_numpy(), 1)[0]
    assert abs(slope) < 0.5


def test_response_slopes_follow_the_engine(R, spe, res):
    xi = res.fit.index["A"]
    patient, group = N._response_slopes(res.fit, xi, "B")
    j = xi["niches"].index("B")
    pats = ["S3", "S1"]  # by name, in the order asked
    want = [xi["beta"][0, xi["patients"].index(p), j] for p in pats]
    np.testing.assert_array_equal(patient("G1", pats), want)
    assert np.isnan(patient("G1", ["nobody"])[0])
    rs = res_from_dump(R["rs"])
    patient, group = N._response_slopes(rs.fit, rs.fit.index["A"], "B")
    cf = rs.fit.index["A"]["coef"]
    est = cf[(cf["test"] == "pooled") & (cf["niche"] == "B") & (cf["gene"] == "G1")][
        "estimate"
    ].iloc[0]
    assert group("G1", ["S1"]) == est
    np.testing.assert_array_equal(patient("G1", pats), [est, est])


def test_response_errors_are_rs(R, spe, fit, res):
    f = fit_from_dump(R["fit"])
    for xi in f.index.values():
        xi["intercept"] = None
    with pytest.raises(ValueError, match="refit"):
        pl.plot_niche_response(spe, f, "G1", "A", "B")
    N._niche_response_data(spe, f, "G1", "A", "B", adjust=False)  # raw display needs none
    pl.plot_niche_response(spe, f, "G1", "A", "B", adjust=False)
    other = spe[spe.obs["sample_id"].isin(["S1", "S2"])].copy()
    with pytest.raises(ValueError, match="the object the fit was made from"):
        pl.plot_niche_response(other, res, "G1", "A", "B")
    with pytest.raises(TypeError, match="SpiDEResults"):
        pl.plot_niche_response(spe, "not results", "G1", "A", "B")
    with pytest.raises(ValueError, match="fitted index types: A"):
        pl.plot_niche_response(spe, res, "G1", "Z", "B")
    with pytest.raises(ValueError, match="niches tested in A"):
        pl.plot_niche_response(spe, res, "G1", "A", "Z")
    with pytest.raises(ValueError, match="not tested in A: nope"):
        pl.plot_niche_response(spe, res, ["G1", "nope"], "A", "B")


# --- plot_niche_response: figure --------------------------------------------------------------


def legend_labels(fig):
    return [[t.get_text() for t in leg.get_texts()] for leg in fig.legends]


def test_adjusted_response_one_panel_per_gene(spe, res, tmp_path):
    fig = pl.plot_niche_response(spe, res, gene=["G1", "G2"], index="A", niche="B")
    assert isinstance(fig, Figure)
    ax = panels(fig)
    assert len(ax) == 2
    assert [a.get_title() for a in ax] == [T._mt_gene("G1"), T._mt_gene("G2")]
    assert fig.get_supxlabel() == "B density (log1p)"
    assert fig.get_supylabel() == "expression relative to patient (CP10k)"
    assert legend_labels(fig) == [["Non-responder", "Responder"],
                                  ["fitted slope", "mean across patients"]]  # fmt: skip
    D = N._niche_response_data(spe, res, ["G1", "G2"], "A", "B")
    pb = D["data"]
    for a, g in zip(ax, ["G1", "G2"], strict=True):
        assert a.get_yscale() == "log"
        # faint patient-bin points first, then the points of the mean
        assert len(a.collections[0].get_offsets()) == ((pb["gene"] == g) & (pb["expr"] > 0)).sum()
        # per condition: a fitted slope, a mean line and the means' markers (the bars are collections)
        assert len(a.lines) == 3 * 2
    fig.canvas.draw()
    labels = [t.get_text() for a in ax for t in a.get_yticklabels()]
    assert not any(__import__("re").search(r"[0-9] [0-9]", s) for s in labels)  # comma, not space
    fig.savefig(tmp_path / "niche_response.png", dpi=60)
    assert (tmp_path / "niche_response.png").stat().st_size > 1000


def test_raw_response_legend_and_layers(spe, res, tmp_path):
    fig = pl.plot_niche_response(spe, res, "G1", "A", "B", adjust=False)
    ax = panels(fig)
    assert len(ax) == 1
    assert fig.get_supylabel() == "expression (CP10k)"
    assert legend_labels(fig) == [
        ["Non-responder", "Responder"],
        ["between patients", "within patient"],
    ]
    D = N._niche_response_data(spe, res, "G1", "A", "B", adjust=False)
    n_pat = len(D["centroids"])
    # the patients' within-patient segments, plus one between-patient line
    assert len(ax[0].lines) == n_pat + 1
    # faint cell-bin points, then the patients' open circles
    assert [len(c.get_offsets()) for c in ax[0].collections] == [
        ((D["data"]["gene"] == "G1") & (D["data"]["expr"] > 0)).sum(),
        n_pat,
    ]
    fig.savefig(tmp_path / "niche_response_raw.png", dpi=60)


def test_response_with_the_sandwich_engine_and_without_a_condition(R, spe, tmp_path):
    rs = res_from_dump(R["rs"])
    fig = pl.plot_niche_response(spe, rs, "G1", "A", "B")
    assert legend_labels(fig)[0] == ["Non-responder", "Responder"]
    pl.plot_niche_response(spe, rs, "G1", "A", "B", adjust=False)
    fig = pl.plot_niche_response(spe, fit_from_dump(R["fnc"]), "G1", "A", "B")
    assert legend_labels(fig)[0] == ["all patients"]  # one group, drawn in black
    fig.savefig(tmp_path / "niche_response_nocond.png", dpi=60)


def test_response_lays_out_many_genes_like_facet_wrap(spe, res):
    fig = pl.plot_niche_response(spe, res, ["G1", "G2", "G3", "G4", "G5"], "A", "B")
    ax = panels(fig)
    assert len([a for a in ax if a.get_visible()]) == 5
    assert ax[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)
    # the free y axes differ between genes
    assert ax[0].get_ylim() != ax[1].get_ylim()


def test_response_drops_unfitted_genes_with_a_message(R, spe):
    f2 = fit_from_dump(R["f2"])
    with pytest.warns(UserWarning, match="G2 not fitted in A, left out"):
        fig = pl.plot_niche_response(spe, f2, ["G1", "G2"], "A", "B")
    assert [a.get_title() for a in panels(fig)] == [T._mt_gene("G1")]


def test_gene_with_markup_characters_plots(R, spe, tmp_path):
    odd = 'G_1-"x"\\y'
    res = res_from_dump(R["res"])  # a private copy: the gene is renamed in place
    for xi in res.fit.index.values():
        xi["genes"] = [odd if g == "G1" else g for g in xi["genes"]]
        xi["intercept"].index = xi["genes"]
        xi["mean_expr"].index = xi["genes"]
    a2 = spe.copy()
    a2.var_names = [odd if g == "G1" else g for g in a2.var_names]
    fig = pl.plot_niche_response(a2, res, odd, "A", "B")
    assert panels(fig)[0].get_title() == T._mt_gene(odd)
    fig.savefig(tmp_path / "odd.png", dpi=50)


def test_plots_draw_to_pdf_without_warnings(spe, res, tmp_path):
    figs = [
        pl.plot_niche_map(spe, niche="B", sample="S1", index="A", unit="µm"),
        pl.plot_niche_response(spe, res, gene="G1", index="A", niche="B"),
        pl.plot_niche_response(spe, res, gene="G1", index="A", niche="B", adjust=False),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i, fig in enumerate(figs):
            fig.savefig(tmp_path / f"p{i}.pdf")
            fig.savefig(tmp_path / f"p{i}.png", dpi=40)
            assert (tmp_path / f"p{i}.pdf").stat().st_size > 1000


# --- end to end from AnnData (Python's own fit) -----------------------------------------------


def test_end_to_end_on_the_toy_data(tmp_path):
    adata = spidepy.toy_spide()
    spidepy.build_niches(adata, sigma=30)
    fit = spidepy.fit_spide(adata, condition="condition", sigma=30, index="A")
    res = spidepy.test_spide(fit, procedure="all")
    fig = spidepy.pl.plot_niche_response(adata, res, gene=["G1", "G2"], index="A", niche="B")
    assert len(panels(fig)) == 2 and len(fig.legends) == 2
    fig.savefig(tmp_path / "e2e_response.png", dpi=50)
    fig = spidepy.pl.plot_niche_map(adata, niche="B", sample="S1", index="A")
    assert n_points(panels(fig)[0]) == (adata.obs["sample_id"] == "S1").sum()
    fig = spidepy.pl.plot_niche_response(adata, fit, "G1", "A", "B", adjust=False)
    assert isinstance(fig, Figure)
