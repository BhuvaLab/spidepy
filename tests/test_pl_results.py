"""Task 15: the results plots (R/plot-results.R; tests ``test-plots-results.R``).

The data builders are compared with R's own on R's fits and results (``plot_data.h5``): the call
counts and graph edges and nodes, the top triplets, spillover points and labels, the triplet
heatmap's rows and ordering, and the p-value bands. The figures are checked for structure and R's
error messages.
"""

from __future__ import annotations

import copy
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from _fitref import frame, strs  # noqa: E402
from _plotref import plot_ref, plot_spe, res_from_dump, same_frame  # noqa: E402
from _ref import load_ref  # noqa: E402
from spidepy import pl, spi_gsea  # noqa: E402
from spidepy.pl import _results as Q  # noqa: E402


@pytest.fixture(scope="module")
def R():
    return load_ref("plot_data")


@pytest.fixture(scope="module")
def res(R):
    return res_from_dump(R["res"])


@pytest.fixture(scope="module")
def spe():
    return plot_spe()


def render(fig):
    """Draw the figure with warnings as errors."""
    assert isinstance(fig, Figure)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig.canvas.draw()
    return fig


def texts(ax):
    return [t.get_text() for t in ax.texts]


# --- plotCallMap -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, test, fdr",
    [("pool", "pooled", 0.05), ("cond", "condition", 0.05), ("none", "pooled", 1e-12)],
)
def test_call_heatmap_data_match_r(R, res, key, test, fdr):
    n, tested = Q._call_counts(res, test, fdr)
    from spidepy.pl._helpers import _direction_labels

    lab = _direction_labels(res, test)
    g = Q._call_heatmap(n, tested, lab)
    same_frame(g, R["calls"][key]["data"], key)
    if "dir_levels" in R["calls"][key]:
        assert list(g["direction"].cat.categories) == strs(R["calls"][key]["dir_levels"])
    assert not (g["tested"] & (g["index"] == g["niche"])).any()


def test_call_counts_sum_to_the_calls(res):
    n, _ = Q._call_counts(res, "pooled", 0.05)
    tab = res.table[(res.table["test"] == "pooled") & (res.table["q"] <= 0.05)]
    assert n["calls"].sum() == len(tab)
    assert list(n.columns) == ["index", "niche", "direction", "calls"]


@pytest.mark.parametrize(
    "key, test, fdr, min_calls",
    [
        ("g_cond", "condition", 0.9, 1),
        ("g_pool", "pooled", 0.05, 1),
        ("g_pool2", "pooled", 0.2, 2),
    ],
)
def test_call_graph_data_match_r(R, res, key, test, fdr, min_calls):
    from spidepy.pl._helpers import _direction_labels

    n, tested = Q._call_counts(res, test, fdr)
    G = Q._call_graph(n, tested, _direction_labels(res, test), min_calls)
    ref = R["calls"][key]
    same_frame(G["edges"], ref["edges"], f"{key}.edges")
    same_frame(G["nodes"], ref["nodes"], f"{key}.nodes")


def test_call_graph_without_edges_and_wrapped_labels(R, res):
    from spidepy.pl._helpers import _direction_labels

    n, tested = Q._call_counts(res, "pooled", 1e-12)
    G = Q._call_graph(n, tested, _direction_labels(res, "pooled"), 1)
    assert len(G["edges"]) == 0
    same_frame(G["nodes"], R["calls"]["g_none"]["nodes"], "none.nodes")
    r3 = copy.copy(res)
    r3.table = frame(R["calls"]["r3_table"])
    r3.table["in_family"] = r3.table["in_family"].astype(bool)
    n, tested = Q._call_counts(r3, "pooled", 0.05)
    G = Q._call_graph(n, tested, _direction_labels(r3, "pooled"), 1)
    same_frame(G["nodes"], R["calls"]["g_wrap"]["nodes"], "wrap.nodes")
    assert all(len(w) <= 12 for lab in G["nodes"]["label"] for w in lab.split("\n"))
    r = np.hypot(G["nodes"]["x"], G["nodes"]["y"])
    np.testing.assert_allclose(r, 1.0, rtol=1e-12)  # nodes on the unit circle, labels at 1.3


def test_call_map_figures(res):
    fig = render(pl.plot_call_map(res, test="pooled", fdr=0.05))
    panels = [a for a in fig.axes]
    assert len(panels) == 2
    assert {a.get_title() for a in panels} == {"rises with density", "falls with density"}
    assert [t.get_text() for t in panels[0].get_xticklabels()] == ["A", "B", "C"]
    # exact counts are written on the tested cells only: 6 of 9 per panel
    assert all(len(texts(a)) == 6 for a in panels)
    g = render(pl.plot_call_map(res, test="condition", style="graph", fdr=0.9))
    assert len(g.axes) == 2
    lab = [t.get_text() for t in g.axes[0].texts]
    assert sorted(lab) == ["A", "B", "C"]
    # labels sit outside their nodes, anchored away from the circle's centre
    for t in g.axes[0].texts:
        x, y = t.get_position()
        assert np.hypot(x, y) >= 1.25
        if y > 0.5:
            assert t.get_va() == "bottom"
        if y < -0.5:
            assert t.get_va() == "top"
    assert len(g.axes[0].patches) > 0


def test_call_map_draws_when_nothing_is_called(res):
    render(pl.plot_call_map(res, test="pooled", fdr=1e-12))
    g = render(pl.plot_call_map(res, test="pooled", fdr=1e-12, style="graph"))
    assert all(len(a.patches) == 0 for a in g.axes)


def test_call_map_defaults_and_errors(res):
    # the condition test is the default when the results have one
    f = pl.plot_call_map(res)
    assert {a.get_title() for a in f.axes} == {
        "higher slope in Responder",
        "higher slope in Non-responder",
    }
    with pytest.raises(ValueError, match="'style'"):
        pl.plot_call_map(res, style="bar")
    with pytest.raises(ValueError, match="fdr"):
        pl.plot_call_map(res, fdr=0)
    with pytest.raises(TypeError, match="SpiDEResults"):
        pl.plot_call_map(pd.DataFrame())
    r0 = copy.copy(res)
    r0.table = res.table.copy()
    r0.table.loc[r0.table["test"] == "condition", ["p", "q"]] = np.nan
    with pytest.raises(ValueError, match="no tested triplet"):
        pl.plot_call_map(r0)
    pl.plot_call_map(r0, test="pooled")


# --- plotTopTriplets -------------------------------------------------------------------------


def top_same(got, ref, levels, what):
    same_frame(got, ref, what)
    assert list(got["label"].cat.categories) == strs(levels)
    assert got["marker"].dtype == bool


def test_top_triplets_data_match_r(R, res, spe):
    T = R["top"]
    top_same(Q._top_triplets_data(res, "pooled", 10), T["pooled"], T["pooled_levels"], "pooled")
    top_same(
        Q._top_triplets_data(res, gene=["G1", "G3"], adata=spe),
        T["genes"],
        T["genes_levels"],
        "genes",
    )
    top_same(
        Q._top_triplets_data(res, n=12, adata=spe, fold=2), T["cond"], T["cond_levels"], "cond"
    )
    same_frame(
        Q._top_triplets_data(res, "pooled", 5, index="B", niche="A", fdr=0.2), T["idx"], "idx"
    )


def test_top_triplets_markers_need_the_object(res, spe):
    d = Q._top_triplets_data(res, "pooled", 10)
    assert not d["marker"].any()
    d = Q._top_triplets_data(res, gene=["G1", "G3"], adata=spe)
    assert set(d["gene"]) == {"G1", "G3"} and d["marker"].dtype == bool


def test_top_triplets_figure(res, spe):
    fig = render(pl.plot_top_triplets(res, test="pooled", n=10))
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 10
    assert ax.get_xlabel() == "slope on niche density (log1p)"
    assert len(ax.collections) >= 1
    fig = render(pl.plot_top_triplets(res, gene=["G1", "G3"], adata=spe))
    assert "slope difference (Responder - Non-responder)" == fig.axes[0].get_xlabel()
    with pytest.raises(ValueError, match="no triplet"):
        pl.plot_top_triplets(res, gene="nope")
    with pytest.raises(ValueError, match="fdr"):
        pl.plot_top_triplets(res, fdr=2)


# --- plotSpillover ---------------------------------------------------------------------------


def test_spillover_data_match_r(R, res, spe):
    S = R["spill"]
    D = Q._spillover_data(res, spe, "pooled", fold=0.4, label=4, fdr=0.2)
    same_frame(D["data"], S["pooled"], "spill")
    assert list(D["data"]["status"].cat.categories) == strs(S["pooled_status_levels"])
    same_frame(D["labels"], S["pooled_labels"], "labels")
    n = int(((res.table["test"] == "pooled") & np.isfinite(res.table["p"])).sum())
    assert len(D["data"]) == n
    D = Q._spillover_data(res, spe, "condition", fold=1.2, label=3, fdr=0.9)
    same_frame(D["data"], S["cond"], "spill cond")
    same_frame(D["labels"], S["cond_labels"], "labels cond")


def test_spillover_figure_and_errors(res, spe):
    fig = render(pl.plot_spillover(res, spe, fold=0.4, label=4, fdr=0.2))
    ax = fig.axes[0]
    assert (
        ax.get_xlabel() == "log2 fold (niche type / index type)" and ax.get_ylabel() == "signed z"
    )
    assert sum(len(c.get_offsets()) for c in ax.collections) > 0
    assert any("calls" in t.get_text() for t in ax.texts)
    assert len([t for t in ax.texts if t.get_text().startswith("$")]) == 3  # the labelled markers
    render(pl.plot_spillover(res, spe))
    r0 = copy.copy(res)
    r0.table = res.table.copy()
    r0.table.loc[r0.table["test"] == "condition", ["p", "q"]] = np.nan
    with pytest.raises(ValueError, match="no tested triplet"):
        pl.plot_spillover(r0, spe, test="condition")


def test_merged_niches_plots_accept_the_group():
    """plot_spillover / plot_top_triplets / plot_call_map on a merged niche (members pooled)."""
    R = plot_ref()
    sm = plot_spe("plot_spe_bc")
    rmg = res_from_dump(R["bc"]["res"])
    render(pl.plot_spillover(rmg, sm))
    render(pl.plot_top_triplets(rmg, adata=sm, n=5))
    render(pl.plot_call_map(rmg))


# --- plotTripletHeatmap ----------------------------------------------------------------------


def test_triplet_heatmap_data_match_r(R, res):
    H = R["heat"]
    D = Q._triplet_heatmap_data(res, test="pooled", n=8)
    same_frame(D["data"], H["main"], "main")
    assert list(D["data"]["feature"].cat.categories) == strs(H["main_levels"])
    assert D["data"]["feature"].nunique() == 8
    assert D["limit"] == pytest.approx(H["lim_main"], rel=1e-12)
    D = Q._triplet_heatmap_data(res, n=12, cluster=False)
    same_frame(D["data"], H["noclust"], "noclust")
    assert list(D["data"]["feature"].cat.categories) == strs(H["noclust_levels"])


def test_triplet_heatmap_untested_cells_and_clustering(R, res):
    H = R["heat"]
    r2 = copy.copy(res)
    r2.table = res.table[~((res.table["gene"] == "G2") & (res.table["index"] == "B"))]
    D = Q._triplet_heatmap_data(r2, features=["G1", "G2"], test="pooled")
    same_frame(D["data"], H["untested"], "untested")
    same_frame(D["untested"], H["untested_nt"], "untested.nt")
    D = Q._triplet_heatmap_data(r2, features=["G2", "G1", "G5", "G9"], test="pooled", limit=3)
    same_frame(D["data"], H["u3"], "u3")
    assert list(D["data"]["feature"].cat.categories) == strs(H["u3_levels"])
    same_frame(D["untested"], H["u3_nt"], "u3.nt")
    assert D["limit"] == 3


def test_triplet_heatmap_gsea_sets(R, res):
    H = R["heat"]
    gs = frame(H["gsea_in"])
    D = Q._triplet_heatmap_data(gs, test="pooled")
    same_frame(D["data"], H["gsea"], "gsea")
    assert list(D["data"]["feature"].cat.categories) == strs(H["gsea_levels"])
    assert D["qcol"] == "q.global" and D["fdr"] == 0.05
    assert set(D["data"]["feature"].astype(str)) == {"one", "two", "three"}


def test_hclust_order_matches_r(R):
    for k, h in R["hc"].items():
        m = np.asarray(h["m"], dtype=float)
        got = [i + 1 for i in Q._hclust_order(m)]
        assert got == [int(i) for i in np.atleast_1d(h["order"])], k


def test_triplet_heatmap_figures_and_errors(R, res):
    fig = render(pl.plot_triplet_heatmap(res, test="pooled", n=8))
    assert len(fig.axes) == 4  # one panel per index type, plus the colour bar
    labs = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labs) == 8 and all(s.startswith("$\\mathit{") for s in labs)
    r2 = copy.copy(res)
    r2.table = res.table[~((res.table["gene"] == "G2") & (res.table["index"] == "B"))]
    fig = render(pl.plot_triplet_heatmap(r2, features=["G1", "G2"], test="pooled"))
    assert any("not tested" in t.get_text() for leg in fig.legends for t in leg.get_texts())
    gs = frame(R["heat"]["gsea_in"])
    fig = render(pl.plot_triplet_heatmap(gs, test="pooled"))
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == strs(R["heat"]["gsea_levels"])
    with pytest.raises(ValueError, match="SpiDEResults or a spiGSEA"):
        pl.plot_triplet_heatmap(pd.DataFrame({"a": [1]}))
    r0 = copy.copy(res)
    r0.table = res.table.copy()
    r0.table["p"] = np.nan
    with pytest.raises(ValueError, match="no tested triplet"):
        pl.plot_triplet_heatmap(r0, test="pooled")
    with pytest.raises(ValueError, match="none of the features"):
        pl.plot_triplet_heatmap(res, features=["nope"], test="pooled")


def test_triplet_heatmap_of_spi_gsea_runs_end_to_end(res):
    gs = spi_gsea(
        res,
        {"one": [f"G{i}" for i in range(1, 6)], "two": [f"G{i}" for i in range(6, 13)]},
        min_size=3,
    )
    render(pl.plot_triplet_heatmap(gs, test="pooled"))


# --- plotPValues -----------------------------------------------------------------------------


def test_p_value_bands_match_r(R, res):
    V = R["pvals"]
    D = Q._p_values_data(res)
    same_frame(D["data"], V["data"], "data")
    same_frame(D["uni"], V["uni"], "uni")
    assert list(D["data"]["band"].cat.categories) == strs(V["band_levels"])
    assert list(D["data"]["test"].cat.categories) == strs(V["test_levels"])
    D = Q._p_values_data(res, bands=3)
    same_frame(D["data"], V["data3"], "data3")
    same_frame(D["uni"], V["uni3"], "uni3")


def test_p_values_figure_and_errors(res):
    fig = render(pl.plot_p_values(res))
    assert len(fig.axes) == 10  # two tests x five expression bands
    assert [a.get_title() for a in fig.axes[:5]][0].startswith("Q1")
    # each panel counts its triplets in 20 bins
    tot = sum(sum(p.get_height() for p in a.patches) for a in fig.axes)
    assert tot == np.isfinite(res.table["p"]).sum()
    D = Q._p_values_data(res)
    assert D["data"]["band"].nunique() == 5
    assert set(D["data"]["test"].astype(str)) == {"pooled", "condition"}
    r0 = copy.copy(res)
    r0.table = res.table.copy()
    r0.table["p"] = np.nan
    with pytest.raises(ValueError, match="no tested triplet"):
        pl.plot_p_values(r0)
    with pytest.raises(TypeError, match="SpiDEResults"):
        pl.plot_p_values(res.table)


def test_results_plots_are_independent_of_row_order(res, spe):
    """Rows are joined by name: shuffling the results table leaves the plotted data unchanged."""
    r2 = copy.copy(res)
    r2.table = res.table.sample(frac=1, random_state=5).reset_index(drop=True)
    a = Q._top_triplets_data(res, "pooled", 10, adata=spe)
    b = Q._top_triplets_data(r2, "pooled", 10, adata=spe)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    n1, _ = Q._call_counts(res, "pooled", 0.05)
    n2, _ = Q._call_counts(r2, "pooled", 0.05)
    pd.testing.assert_frame_equal(n1, n2)
    assert re.fullmatch(r"italic.*", a["label"].astype(str).iloc[0])


def test_end_to_end_on_the_toy_data(tmp_path):
    """Python's own fit and tests through every results and patient plot (PNG written)."""
    import spidepy

    adata = spidepy.toy_spide()
    spidepy.build_niches(adata, sigma=30)
    res = spidepy.spide(adata, condition="condition", sigma=30, procedure="all")
    figs = {
        "call_map": pl.plot_call_map(res, test="pooled"),
        "call_graph": pl.plot_call_map(res, style="graph"),
        "top": pl.plot_top_triplets(res, n=10, adata=adata),
        "spill": pl.plot_spillover(res, adata),
        "heat": pl.plot_triplet_heatmap(res, test="pooled", n=8),
        "pvals": pl.plot_p_values(res),
        "slopes": pl.plot_patient_slopes(res, ["G1", "G2"], "A", "B"),
        "effects": pl.plot_patient_effects(res, "A"),
    }
    for k, f in figs.items():
        assert isinstance(f, Figure), k
        f.savefig(tmp_path / f"{k}.png", dpi=40)
