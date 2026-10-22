"""The results views (`R/plot-results.R`): where the calls are, the strongest triplets, the
spillover signature, triplet heatmaps and p-value bands."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from .._checks import check_adata, check_fdr, check_results
from .._classes import SpiDEResults
from .._rcompat import r_num_str, r_sort_str
from ..fit import _match_arg
from ._helpers import _direction_labels, _expression_bands, _niche_fold, _resolve_test
from ._theme import (
    _PT,
    _SPIDE_COLS,
    _direction_colours,
    _it_triplet,
    _magnitude_ramp,
    _mt_gene,
    _mt_triplet,
    _point_area,
    _stack_legends,
    _strip,
    _strwrap,
    _style_legend,
    _t_cmap_norm,
    _wrap_dims,
    r_colour_hex,
    spide_colours,
    style,
)

_FAINT = r_colour_hex(_SPIDE_COLS["faint"])
_FS = 11 * 1.1  # base text size of theme_spiDE


def _strip_box(ax, text: str, side: str = "right") -> None:
    """A boxed facet strip beside the panel (the row labels of `facet_grid`)."""
    ax.text(1.03, 0.5, text, transform=ax.transAxes, rotation=270, ha="left", va="center",
            fontsize=_FS * 0.8,
            bbox={"boxstyle": "square,pad=0.3", "fc": "none", "ec": "black", "lw": 0.5})  # fmt: skip


# --- plotCallMap -----------------------------------------------------------------------------


def _sort_by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows ordered by string columns with `r_sort_str`'s collation (first column slowest)."""
    keys = [
        df[c].map({v: i for i, v in enumerate(r_sort_str(set(df[c])))}).to_numpy() for c in cols
    ]
    return df.iloc[np.lexsort(keys[::-1])].reset_index(drop=True)


def _call_counts(res, test: str, fdr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts of genes with `q <= fdr` per (index, niche, direction) and the tested pairs
    (the inputs of R `plotCallMap`'s builders).

    `n` has `index, niche, direction, calls` ordered as R's `aggregate` (direction slowest,
    index fastest); `tested` the distinct (index, niche) with a finite p.
    """
    tab = res.table[res.table["test"] == test]
    fin = np.isfinite(tab["p"].to_numpy(dtype=float))
    if not fin.any():
        raise ValueError("no tested triplet has a p-value")
    tested = tab.loc[fin, ["index", "niche"]].drop_duplicates().reset_index(drop=True)
    q = tab["q"].to_numpy(dtype=float)
    est = tab["estimate"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        hit = tab[(~np.isnan(q)) & (q <= fdr) & ~np.isnan(est)]
    if len(hit):
        d = pd.DataFrame(
            {
                "index": hit["index"].to_numpy(),
                "niche": hit["niche"].to_numpy(),
                "direction": np.where(hit["estimate"].to_numpy(dtype=float) > 0, "up", "down"),
            }
        )
        n = d.groupby(["direction", "niche", "index"], sort=False).size().reset_index(name="calls")
        n = _sort_by(n, ["direction", "niche", "index"])
        n = n[["index", "niche", "direction", "calls"]]
    else:
        n = pd.DataFrame(
            {"index": [], "niche": [], "direction": [], "calls": np.array([], dtype=int)}
        )
    return n, tested


def _call_heatmap(n: pd.DataFrame, tested: pd.DataFrame, lab: dict) -> pd.DataFrame:
    """The grid behind the call heatmap (R `.callHeatmap`'s data): one row per (index, niche,
    direction) with `tested`, `calls` and `signed` (+/- log10(1 + calls)); `direction` is
    a categorical of the legend labels."""
    idx = r_sort_str(set(tested["index"]))
    nic = r_sort_str(set(tested["niche"]))
    grid = pd.DataFrame(
        [(i, j, d) for d in ("up", "down") for j in nic for i in idx],
        columns=["index", "niche", "direction"],
    )
    pairs = set(zip(tested["index"], tested["niche"], strict=True))
    grid["tested"] = [(i, j) in pairs for i, j in zip(grid["index"], grid["niche"], strict=True)]
    # merge(grid, n, all.x = TRUE) sorts by the key columns
    grid = grid.merge(n, how="left", on=["index", "niche", "direction"])
    grid = _sort_by(grid, ["index", "niche", "direction"])
    grid["calls"] = grid["calls"].astype(float)
    grid.loc[grid["calls"].isna() & grid["tested"], "calls"] = 0.0
    sign = np.where(grid["direction"] == "up", 1.0, -1.0)
    grid["signed"] = sign * np.log10(1 + grid["calls"])
    grid["direction"] = pd.Categorical(
        grid["direction"].map(lab), categories=[lab["up"], lab["down"]]
    )
    return grid


def _call_graph(n: pd.DataFrame, tested: pd.DataFrame, lab: dict, min_calls: float) -> dict:
    """The edges and nodes of the call graph (R `.callGraph`'s layer data).

    Cell types sit on the unit circle at angle `pi/2 - 2 pi (i - 1) / n` in sorted order; an
    edge runs from the niche type to the index type, shortened by 0.13 at both ends. `nodes`
    holds each type once per panel, with its label wrapped to 12 characters and the label's
    anchors.
    """
    types = r_sort_str(set(tested["index"]) | set(tested["niche"]))
    ang = math.pi / 2 - 2 * math.pi * np.arange(len(types)) / len(types)
    pos = pd.DataFrame({"type": types, "x": np.cos(ang), "y": np.sin(ang)}).set_index(
        "type", drop=False
    )
    e = n[n["calls"] >= min_calls].reset_index(drop=True).copy()
    e["x0"] = pos["x"].reindex(e["niche"]).to_numpy()
    e["y0"] = pos["y"].reindex(e["niche"]).to_numpy()
    e["x1"] = pos["x"].reindex(e["index"]).to_numpy()
    e["y1"] = pos["y"].reindex(e["index"]).to_numpy()
    r = 0.13
    ln = np.sqrt((e["x1"] - e["x0"]) ** 2 + (e["y1"] - e["y0"]) ** 2)
    e["xs"] = e["x0"] + r * (e["x1"] - e["x0"]) / ln
    e["ys"] = e["y0"] + r * (e["y1"] - e["y0"]) / ln
    e["xe"] = e["x1"] - r * (e["x1"] - e["x0"]) / ln
    e["ye"] = e["y1"] - r * (e["y1"] - e["y0"]) / ln
    panels = [lab["up"], lab["down"]]
    e["panel"] = pd.Categorical(e["direction"].map(lab), categories=panels)
    nodes = pd.DataFrame(
        [(t, pos.loc[t, "x"], pos.loc[t, "y"], p) for p in panels for t in types],
        columns=["type", "x", "y", "panel"],
    )
    nodes["panel"] = pd.Categorical(nodes["panel"], categories=panels)
    nodes["label"] = [_strwrap(t, 12) for t in nodes["type"]]
    # labels outside the nodes, anchored away from the circle's centre
    nodes["hj"] = np.where(nodes["x"].abs() < 0.2, 0.5, np.where(nodes["x"] > 0, 0.0, 1.0))
    nodes["vj"] = np.where(nodes["y"] > 0.3, 0.0, np.where(nodes["y"] < -0.3, 1.0, 0.5))
    return {"edges": e, "nodes": nodes, "types": types}


def plot_call_map(
    res, test: str | None = None, fdr: float | None = None, style: str = "heatmap",
    min_calls: int = 1,
) -> Figure:  # fmt: skip
    """Where the calls are (R `plotCallMap`).

    The number of genes called for every (index, niche) pair, split by direction: for the pooled
    test, a slope that rises or falls with the niche's density; for the condition-specific test,
    a slope higher in one condition than in the other. The heatmap gives exact counts and marks
    untested pairs (an index type against its own niche) in grey; the graph draws cell types on a
    fixed circle with an arrow from niche to index type, its width the number of genes. It reads
    best when calls are few (the condition test), the heatmap when every pair has some.

    Read it for which cell-type pairs carry the signal, and for pairs whose calls all go one way
    toward one niche type (the shape segmentation spillover gives; see
    `plot_spillover`). An arrow is an association, not signalling.

    Parameters
    ----------
    res : SpiDEResults
    test : {"pooled", "condition"}, optional
        Default: the condition-specific test if run, else the pooled test.
    fdr : float, optional
        The q-value threshold of a call (default: the results').
    style : {"heatmap", "graph"}
    min_calls : int
        The graph's smallest drawn edge.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels, one per direction.
    """
    check_results(res, "plot_call_map()")
    st = _match_arg(style, ("heatmap", "graph"), "style")
    test = _resolve_test(res, test)
    fdr = res.fdr if fdr is None else fdr
    check_fdr(fdr)
    n, tested = _call_counts(res, test, fdr)
    lab = _direction_labels(res, test)
    if st == "heatmap":
        return _draw_call_heatmap(_call_heatmap(n, tested, lab), lab)
    return _draw_call_graph(_call_graph(n, tested, lab, min_calls), lab)


def _draw_call_heatmap(grid: pd.DataFrame, lab: dict) -> Figure:
    idx = r_sort_str(set(grid["index"]))
    nic = r_sort_str(set(grid["niche"]))
    lim = max(1.0, float(np.nanmax(np.abs(grid["signed"].to_numpy()))))
    cmap = ListedColormap(spide_colours("diverging", 257))
    panels = [lab["up"], lab["down"]]
    nrow, ncol = _wrap_dims(2)
    cell = 0.5
    W = ncol * (len(nic) * cell + 0.9) + 0.4
    H = nrow * (len(idx) * cell + 1.4) + 0.5
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        axs = fig.subplots(nrow, ncol, squeeze=False, sharey=True)
        for k, ax in enumerate(axs.ravel()):
            g = grid[grid["direction"].astype(object) == panels[k]]
            M = np.full((len(idx), len(nic)), np.nan)
            T = np.zeros((len(idx), len(nic)), dtype=bool)
            C = np.full((len(idx), len(nic)), np.nan)
            for _, r in g.iterrows():
                i, j = idx.index(r["index"]), nic.index(r["niche"])
                T[i, j], M[i, j], C[i, j] = r["tested"], r["signed"], r["calls"]
            ax.pcolormesh(np.ma.masked_array(np.ones(T.shape), mask=T),
                          cmap=ListedColormap([_FAINT]), edgecolors="white", linewidth=0.6)  # fmt: skip
            ax.pcolormesh(np.ma.masked_invalid(np.where(T, M, np.nan)), cmap=cmap, vmin=-lim,
                          vmax=lim, edgecolors="white", linewidth=0.6)  # fmt: skip
            for i in range(len(idx)):
                for j in range(len(nic)):
                    if T[i, j]:
                        ax.text(j + 0.5, i + 0.5, f"{int(C[i, j])}", ha="center", va="center",
                                fontsize=3.4 * _PT,
                                color="white" if abs(M[i, j]) > 0.55 * lim else "black")  # fmt: skip
            ax.set_xlim(0, len(nic))
            ax.set_ylim(0, len(idx))
            ax.set_aspect("equal")
            ax.set_xticks(np.arange(len(nic)) + 0.5, nic, rotation=40, ha="right")
            ax.set_yticks(np.arange(len(idx)) + 0.5, idx)
            ax.tick_params(length=0)
            _strip(ax, panels[k])
        fig.supxlabel("niche type", fontsize=_FS)
        fig.supylabel("index type", fontsize=_FS)
        fig.set_layout_engine("constrained")
    return fig


def _line_mm(calls, lo: float, hi: float) -> np.ndarray:
    """`scale_linewidth(range = c(0.3, 2.6))`: calls rescaled linearly to millimetres."""
    calls = np.asarray(calls, dtype=float)
    if hi <= lo:
        return np.full(len(calls), (0.3 + 2.6) / 2)
    return 0.3 + (calls - lo) / (hi - lo) * 2.3


def _draw_call_graph(G: dict, lab: dict) -> Figure:
    e, nodes = G["edges"], G["nodes"]
    panels = [lab["up"], lab["down"]]
    dcol = _direction_colours()
    nrow, ncol = _wrap_dims(2)
    lo = float(e["calls"].min()) if len(e) else 1.0
    hi = float(e["calls"].max()) if len(e) else 1.0
    pw = 3.9
    W, H = ncol * pw + 1.6, nrow * pw * 0.8 + 0.5
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        fig.subplots_adjust(left=0.01, right=1 - 1.5 / W, top=0.93, bottom=0.02, wspace=0.05)
        axs = fig.subplots(nrow, ncol, squeeze=False)
        for k, ax in enumerate(axs.ravel()):
            p = panels[k]
            ek = e[e["panel"].astype(object) == p]
            lw = _line_mm(ek["calls"], lo, hi)
            for (_, r), w in zip(ek.iterrows(), lw, strict=True):
                ax.add_patch(
                    FancyArrowPatch(
                        (r["xs"], r["ys"]), (r["xe"], r["ye"]), connectionstyle="arc3,rad=0.22",
                        arrowstyle="-|>,head_length=0.5,head_width=0.2", mutation_scale=2 * _PT * 2,
                        lw=w * _PT, color=dcol[r["direction"]], capstyle="round", shrinkA=0,
                        shrinkB=0, zorder=2,
                    )
                )  # fmt: skip
            nd = nodes[nodes["panel"].astype(object) == p]
            ax.scatter(nd["x"], nd["y"], s=_point_area(6) * 0.9, facecolors="white",
                       edgecolors="black", linewidths=0.6, zorder=3)  # fmt: skip
            for _, r in nd.iterrows():
                ax.text(1.3 * r["x"], 1.3 * r["y"], r["label"],
                        ha={0.0: "left", 0.5: "center", 1.0: "right"}[r["hj"]],
                        va={0.0: "bottom", 0.5: "center", 1.0: "top"}[r["vj"]],
                        fontsize=3.6 * _PT, linespacing=1.0)  # fmt: skip
            ax.set_xlim(-1 - 0.8, 1 + 0.8)
            ax.set_ylim(-1 - 0.45, 1 + 0.45)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(p, fontsize=_FS * 1.2 * 0.8,
                         bbox={"boxstyle": "square,pad=0.3", "fc": "none", "ec": "black",
                               "lw": 0.5})  # fmt: skip
        if len(e):
            brk = [
                b for b in MaxNLocator(nbins=4, integer=True).tick_values(lo, hi) if lo <= b <= hi
            ] or [lo]
            handles = [
                Line2D([], [], color=r_colour_hex(_SPIDE_COLS["grey"]), lw=w * _PT)
                for w in _line_mm(brk, lo, hi)
            ]
            _stack_legends(fig, [("genes", handles, [f"{b:g}" for b in brk])], 1 - 1.4 / W,
                           y_top=0.7)  # fmt: skip
    return fig


# --- plotTopTriplets -------------------------------------------------------------------------


def _top_triplets_data(
    res, test=None, n: int = 30, gene=None, index=None, niche=None, adata=None, fold: float = 4,
    fdr=None, assay: str = "counts", name: str = "Niche",
) -> pd.DataFrame:  # fmt: skip
    """The plotted rows of `plot_top_triplets` (R `plotTopTriplets`' `p$data`): the
    results table's columns plus `lo`, `hi`, `called`, `direction`, `marker` and
    `label` (R's plotmath string); `called` and `direction` are categoricals."""
    check_results(res, "plot_top_triplets()")
    test = _resolve_test(res, test)
    fdr = res.fdr if fdr is None else fdr
    check_fdr(fdr)
    tab = res.results(test=test)
    for col, val in (("gene", gene), ("index", index), ("niche", niche)):
        if val is not None:
            keep = [val] if isinstance(val, str) else [str(v) for v in val]
            tab = tab[tab[col].isin(keep)]
    tab = tab[
        np.isfinite(tab["p"].to_numpy(dtype=float)) & np.isfinite(tab["se"].to_numpy(dtype=float))
    ]
    if not len(tab):
        raise ValueError("no triplet of the chosen genes, index and niche types was tested")
    tab = tab.iloc[np.argsort(tab["p"].to_numpy(dtype=float), kind="stable")].head(n)
    tab = tab.reset_index(drop=True)
    tq = stats.t.ppf(0.975, tab["df"].to_numpy(dtype=float))
    tab["lo"] = tab["estimate"] - tq * tab["se"]
    tab["hi"] = tab["estimate"] + tq * tab["se"]
    q = tab["q"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        called = ~np.isnan(q) & (q <= fdr)
    tab["called"] = pd.Categorical(np.where(called, "yes", "no"), categories=["yes", "no"])
    tab["direction"] = pd.Categorical(
        np.where(tab["estimate"].to_numpy(dtype=float) > 0, "up", "down"), categories=["up", "down"]
    )
    if adata is None:
        tab["marker"] = False
    else:
        tab["marker"] = _niche_fold(tab, adata, res.fit, assay, name) >= fold
    lab = [
        _it_triplet(g, i, j)
        for g, i, j in zip(tab["gene"], tab["index"], tab["niche"], strict=True)
    ]
    order = np.argsort(tab["estimate"].to_numpy(dtype=float), kind="stable")
    tab["label"] = pd.Categorical(lab, categories=[lab[i] for i in order])
    return tab


def plot_top_triplets(
    res, test=None, n: int = 30, gene=None, index=None, niche=None, adata=None, fold: float = 4,
    fdr=None, assay: str = "counts", name: str = "Niche",
) -> Figure:  # fmt: skip
    """The strongest triplets, or those of chosen genes (R `plotTopTriplets`).

    A forest plot: each triplet's estimate (the pooled slope, or the slope difference between
    conditions) with its 95% confidence interval, faded where it is not called. `gene`,
    `index` and `niche` filter the table before the `n` smallest p-values are taken, so
    `gene=["CD74", "HLA-DRA"]` shows those genes' triplets. With `adata`, a hollow triangle
    marks a gene at least `fold`-fold higher in the niche type than in the index type (see
    `plot_spillover`).

    Parameters
    ----------
    res : SpiDEResults
    test : {"pooled", "condition"}, optional
        Default: the condition-specific test if run, else the pooled test.
    n : int
        The number of triplets.
    gene, index, niche : str or sequence of str, optional
        Filters.
    adata : anndata.AnnData, optional
        The object the results came from, for the niche-marker flag.
    fold : float
        The niche-marker threshold.
    fdr : float, optional
        The q-value threshold of a call (default: the results').
    assay, name : str
        The counts layer and the niche `obsm` prefix.

    Returns
    -------
    matplotlib.figure.Figure
    """
    check_results(res, "plot_top_triplets()")
    test = _resolve_test(res, test)
    tab = _top_triplets_data(res, test, n, gene, index, niche, adata, fold, fdr, assay, name)
    fdr = res.fdr if fdr is None else fdr
    lab = _direction_labels(res, test)
    dcol = _direction_colours()
    cats = list(tab["label"].cat.categories)
    ypos = {c: i for i, c in enumerate(cats)}
    y = np.array([ypos[c] for c in tab["label"].astype(object)], dtype=float)
    nrow = len(tab)
    W, H = 8.6, nrow * 0.27 + 3.1
    xlab = (
        "slope on niche density (log1p)"
        if test == "pooled"
        else f"slope difference ({res.contrast})"
    )
    ylab = [
        _mt_triplet(g, i, j)
        for g, i, j in sorted(
            zip(tab["gene"], tab["index"], tab["niche"], strict=True),
            key=lambda t: ypos[_it_triplet(*t)],
        )
    ]
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        fig.subplots_adjust(left=2.4 / W, right=0.97, top=1 - 0.2 / H, bottom=2.3 / H)
        ax = fig.subplots()
        ax.axvline(0, color="black", ls=":", lw=0.3 * _PT)
        for cl, alpha in (("yes", 1.0), ("no", 0.35)):
            for dr in ("up", "down"):
                m = (
                    (tab["called"].astype(object) == cl) & (tab["direction"].astype(object) == dr)
                ).to_numpy()
                if not m.any():
                    continue
                ax.hlines(
                    y[m], tab["lo"][m], tab["hi"][m], colors=dcol[dr], lw=0.6 * _PT, alpha=alpha
                )
                for mk, mm in (
                    (False, m & ~tab["marker"].to_numpy(dtype=bool)),
                    (True, m & tab["marker"].to_numpy(dtype=bool)),
                ):
                    if mm.any():
                        ax.scatter(tab["estimate"][mm], y[mm], s=_point_area(2) if not mk else _point_area(2) * 1.4,
                                   marker="^" if mk else "o", facecolors="white" if mk else dcol[dr],
                                   edgecolors=dcol[dr], linewidths=0.8 if mk else 0, alpha=alpha, zorder=3)  # fmt: skip
        ax.set_yticks(range(len(cats)), ylab)
        ax.set_ylim(-0.7, len(cats) - 0.3)
        ax.set_xlabel(xlab)
        ax.margins(x=0.05)
        h1 = [Line2D([], [], marker="o", ls="none", color=dcol[k], ms=4.5) for k in ("up", "down")]
        h2 = [
            Line2D([], [], marker="o", ls="none", color="black", ms=4.5, alpha=a) for a in (1, 0.35)
        ]
        h3 = [Line2D([], [], marker="^", ls="none", mfc="white", mec="black", ms=5.5)]
        qlab = ["yes", "no"]
        legs = [
            (None, h1, [lab["up"], lab["down"]]),
            (f"q ≤ {r_num_str(fdr)}", h2, qlab),
            (None, h3, [f"≥ {r_num_str(fold)}-fold higher in the niche type"]),
        ]
        _bottom_legends(fig, legs, 0.01, 1.35 / H)
    return fig


def _bottom_legends(fig, groups, x0: float, y_top: float) -> None:
    """Legends laid left to right along the bottom of the figure, each with its title."""
    renderer = fig.canvas.get_renderer()
    x = x0
    for title, handles, labels in groups:
        leg = fig.legend(handles, labels, title=title, loc="upper left", bbox_to_anchor=(x, y_top),
                         frameon=False, alignment="left")  # fmt: skip
        _style_legend(leg)
        box = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        x = box.x1 + 0.02


# --- plotSpillover ---------------------------------------------------------------------------


def _spillover_data(
    res, adata, test: str = "pooled", fold: float = 4, label: int = 6, fdr=None,
    assay: str = "counts", name: str = "Niche",
) -> dict:  # fmt: skip
    """The data behind `plot_spillover` (R `plotSpillover`).

    Returns `data` (R's `p$data`: the test's tested rows with `fold` (log2), `z` (signed)
    and `status`), `labels` (the strongest `label` calls at least `fold`-fold in the
    niche type, with R's plotmath `text`), `n_marker` and `n_called`.
    """
    check_results(res, "plot_spillover()")
    test = _resolve_test(res, test)
    fdr = res.fdr if fdr is None else fdr
    check_fdr(fdr)
    check_adata(adata, assay, res.fit.params["cell_type"], res.fit.params["sample_id"])
    tab = res.table[(res.table["test"] == test) & np.isfinite(res.table["p"].to_numpy(dtype=float))]
    if not len(tab):
        raise ValueError("no tested triplet has a p-value")
    tab = tab.reset_index(drop=True).copy()
    tab["fold"] = np.log2(_niche_fold(tab, adata, res.fit, assay, name))
    p = np.clip(tab["p"].to_numpy(dtype=float), 1e-300, 1.0)
    tab["z"] = np.sign(tab["t"].to_numpy(dtype=float)) * stats.norm.isf(p / 2)
    q = tab["q"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        called = ~np.isnan(q) & (q <= fdr)
        up = tab["estimate"].to_numpy(dtype=float) > 0
        mk = (
            called
            & np.isfinite(tab["fold"].to_numpy())
            & (tab["fold"].to_numpy() >= math.log2(fold))
        )
    tab["status"] = pd.Categorical(
        np.where(called, np.where(up, "up", "down"), "not called"),
        categories=["up", "down", "not called"],
    )
    lab = tab[mk]
    lab = lab.iloc[np.argsort(-np.abs(lab["z"].to_numpy()), kind="stable")].head(label).copy()
    lab["text"] = [
        _it_triplet(g, i, j)
        for g, i, j in zip(lab["gene"], lab["index"], lab["niche"], strict=True)
    ]
    return {"data": tab, "labels": lab, "n_marker": int(mk.sum()), "n_called": int(called.sum())}


def _spread(y: np.ndarray, gap: float, lo: float, hi: float) -> np.ndarray:
    """Spread values so neighbours are at least `gap` apart, keeping their order and staying
    inside `[lo, hi]` (labels pushed apart in y, as `ggrepel`'s `direction = "y"`)."""
    order = np.argsort(y, kind="stable")
    out = np.asarray(y, dtype=float)[order].copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    over = out[-1] - hi if len(out) else 0.0
    if over > 0:
        out -= over
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - gap)
    res = np.empty_like(out)
    res[order] = np.maximum(out, lo)
    return res


def plot_spillover(
    res, adata, test: str = "pooled", fold: float = 4, label: int = 6, fdr=None,
    assay: str = "counts", name: str = "Niche",
) -> Figure:  # fmt: skip
    """Are the calls the niche type's own markers? (R `plotSpillover`).

    Every tested triplet, placed by how much higher the gene is in the niche type than in the
    index type (log2, from each cell type's mean counts per 10,000) against the signed z of its
    test; calls are coloured by direction. A gene several-fold higher in the niche type whose
    measured level in the index cells rises with that niche's density is what transcripts
    leaking across segmentation boundaries would produce: the association is real in the data,
    but it needs orthogonal evidence (nuclear counts, a spillover-corrected segmentation) before
    it is read as biology.

    Parameters
    ----------
    res : SpiDEResults
    adata : anndata.AnnData
        The object the results came from.
    test : {"pooled", "condition"}
    fold : float
        The niche-marker threshold (shaded).
    label : int
        The number of markers labelled (strongest first).
    fdr : float, optional
        The q-value threshold of a call (default: the results').
    assay, name : str
        The counts layer and the niche `obsm` prefix.

    Returns
    -------
    matplotlib.figure.Figure
    """
    D = _spillover_data(res, adata, test, fold, label, fdr, assay, name)
    test = _resolve_test(res, test)
    tab, lab = D["data"], D["labels"]
    dlab = _direction_labels(res, test)
    cols = {**_direction_colours(), "not called": r_colour_hex(_SPIDE_COLS["light"])}
    names = {"up": dlab["up"], "down": dlab["down"], "not called": "not called"}
    lf = math.log2(fold)
    W, H = 8.0, 4.6
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        fig.subplots_adjust(left=0.9 / W, right=1 - 2.8 / W, top=1 - 0.2 / H, bottom=0.8 / H)
        ax = fig.subplots()
        for st_, size in (("not called", 0.7), ("up", 1.1), ("down", 1.1)):
            m = (tab["status"].astype(object) == st_).to_numpy()
            ax.scatter(tab["fold"][m], tab["z"][m], s=_point_area(size), c=cols[st_], linewidths=0,
                       zorder=2)  # fmt: skip
        ax.axhline(0, color="black", lw=0.3 * _PT, zorder=1)
        ax.axvline(lf, color="black", ls="--", lw=0.4 * _PT, zorder=1)
        ax.margins(0.04)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.axvspan(lf, max(x1, lf), color=_FAINT, lw=0, zorder=0)
        ax.set_xlim(x0, max(x1, lf))
        # right-aligned at the fold threshold, at the foot of the panel
        ax.text(lf, 0.03, f"{D['n_marker']} of {D['n_called']} calls\n≥ {r_num_str(fold)}-fold in niche type",
                transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=3.4 * _PT)  # fmt: skip
        if len(lab):
            xl = lf - 0.4
            gap = (y1 - y0) * (8.5 * 1.7) / (H * 72 * (1 - 1.0 / H))
            ys = _spread(lab["z"].to_numpy(dtype=float), gap, y0, y1)
            for (_, r), yl in zip(lab.iterrows(), ys, strict=True):
                ax.plot([xl, r["fold"]], [yl, r["z"]], color="black", lw=0.3 * _PT, zorder=1)
                ax.text(xl, yl, _mt_triplet(r["gene"], r["index"], r["niche"]), ha="right",
                        va="center", fontsize=3 * _PT,
                        bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"},
                        zorder=4)  # fmt: skip
        ax.set_xlim(min(x0, xl) if len(lab) else x0, max(x1, lf))
        ax.set_ylim(y0, y1)
        ax.set_xlabel("log2 fold (niche type / index type)")
        ax.set_ylabel("signed z")
        h = [Line2D([], [], marker="o", ls="none", color=cols[k], ms=5) for k in names]
        _stack_legends(fig, [(None, h, list(names.values()))], 1 - 2.7 / W, y_top=0.95)
    return fig


# --- plotTripletHeatmap ----------------------------------------------------------------------


def _hclust_order(m: np.ndarray) -> list[int]:
    """The row order of R `hclust(dist(m))$order` (complete linkage, Euclidean distance).

    A merge puts a singleton before a cluster and the lower-numbered of two clusters first (R's
    `hcass2`); the order is the leaves read left to right from the last merge.
    """
    n = m.shape[0]
    if n < 3:
        return list(range(n))
    z = linkage(pdist(m), method="complete")
    left, right = {}, {}
    for k, (a, b, _, _) in enumerate(z):
        a, b = int(a), int(b)
        left[n + k], right[n + k] = a, b

    def walk(node: int) -> list[int]:
        if node < n:
            return [node]
        return walk(left[node]) + walk(right[node])

    return walk(2 * n - 2)


def _triplet_heatmap_data(
    x, features=None, test=None, n: int = 30, fdr=None, limit=None, cluster: bool = True
) -> dict:  # fmt: skip
    """The data behind `plot_triplet_heatmap` (R `plotTripletHeatmap`).

    Returns `data` (R's `p$data`: the plotted rows plus `feature` (categorical in the
    clustered order) and `called`), `order` (features, bottom to top), `untested` (the
    feature x column cells with no test), `limit`, `fdr`, `qcol` and `feature_col`.
    """
    if isinstance(x, SpiDEResults):
        tab, fcol, qcol = x.table, "gene", "q"
        fdr = x.fdr if fdr is None else fdr
        test = _resolve_test(x, test)
    elif isinstance(x, pd.DataFrame) and {
        "set",
        "index",
        "niche",
        "test",
        "t",
        "p",
        "q.global",
    } <= set(x.columns):
        tab, fcol, qcol = x, "set", "q.global"
        fdr = 0.05 if fdr is None else fdr
        if test is None:
            test = "condition" if (tab["test"] == "condition").any() else "pooled"
        test = _match_arg(test, ("pooled", "condition"), "test")
    else:
        raise ValueError("'x' must be a SpiDEResults or a spiGSEA() table")
    check_fdr(fdr)
    tab = pd.DataFrame(tab)
    ok = (
        (tab["test"] == test)
        & np.isfinite(tab["t"].to_numpy(dtype=float))
        & np.isfinite(tab["p"].to_numpy(dtype=float))
    )
    tab = tab[ok].reset_index(drop=True)
    if not len(tab):
        raise ValueError("no tested triplet has a p-value")
    tab["feature"] = tab[fcol].astype(str)
    if features is None:
        best = tab.groupby("feature", sort=False)["p"].min()
        best = best.reindex(r_sort_str(best.index))
        best = best.iloc[np.argsort(best.to_numpy(dtype=float), kind="stable")]
        features = list(best.index[:n])
    else:
        features = [features] if isinstance(features, str) else [str(f) for f in features]
    tab = tab[tab["feature"].isin(features)].reset_index(drop=True)
    if not len(tab):
        raise ValueError("none of the features was tested")
    ord_ = list(dict.fromkeys(tab["feature"]))
    if cluster and len(ord_) > 2:
        col = tab["index"] + "\r" + tab["niche"]
        m = tab.assign(col=col).pivot_table(
            index="feature", columns="col", values="t", aggfunc="sum", fill_value=0.0
        )
        rows = r_sort_str(m.index)
        cols = r_sort_str(m.columns)
        mm = m.loc[rows, cols].to_numpy(dtype=float)
        ord_ = [rows[i] for i in _hclust_order(mm)]
    tab["feature"] = pd.Categorical(tab["feature"], categories=ord_)
    lim = (
        max(float(np.quantile(np.abs(tab["t"].to_numpy(dtype=float)), 0.98)), 1.0)
        if limit is None
        else limit
    )
    q = tab[qcol].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        tab["called"] = ~np.isnan(q) & (q <= fdr)
    cols_ = tab[["index", "niche"]].drop_duplicates()
    have = set(zip(tab["feature"].astype(str), tab["index"], tab["niche"], strict=True))
    nt = pd.DataFrame(
        [(f, i, j) for i, j in zip(cols_["index"], cols_["niche"], strict=True) for f in ord_],
        columns=["feature", "index", "niche"],
    )
    nt = nt[
        [
            (f, i, j) not in have
            for f, i, j in zip(nt["feature"], nt["index"], nt["niche"], strict=True)
        ]
    ]
    nt["feature"] = pd.Categorical(nt["feature"], categories=ord_)
    return {"data": tab, "order": ord_, "untested": nt.reset_index(drop=True), "limit": lim, "fdr": fdr,
            "qcol": qcol, "feature_col": fcol}  # fmt: skip


def plot_triplet_heatmap(
    x, features=None, test=None, n: int = 30, fdr=None, limit=None, cluster: bool = True
) -> Figure:  # fmt: skip
    """Niche-response profiles of genes or gene sets (R `plotTripletHeatmap`).

    Features (genes, or gene sets from `spi_gsea`) against every (index, niche)
    column: the fill is the test's t (gold: the slope rises with the niche's density, or is
    higher in the second condition; violet: the reverse), a dot marks a call, grey a feature not
    tested in that index type. Rows are clustered on their t profiles. Read it for a gene's or
    pathway's response across all cell-type pairs, and for pairs that share responses.

    Parameters
    ----------
    x : SpiDEResults or pandas.DataFrame
        Results (genes) or the table `spi_gsea` returns (sets; calls by
        `q.global`).
    features : sequence of str, optional
        `None` (the `n` features with the smallest p-value in any column) or the features to
        show.
    test : {"pooled", "condition"}, optional
        Default: the condition-specific test if present, else the pooled test.
    n : int
        The number of features when `features` is `None`.
    fdr : float, optional
        The q-value threshold of a call (default: the results', or 0.05 for sets).
    limit : float, optional
        The |t| at which the fill saturates (default: the 98th percentile, at least 1).
    cluster : bool
        Order rows by hierarchical clustering of their t profiles.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per index type.
    """
    D = _triplet_heatmap_data(x, features, test, n, fdr, limit, cluster)
    tab, ord_, nt, lim = D["data"], D["order"], D["untested"], D["limit"]
    cmap, norm = _t_cmap_norm(lim)
    idx = r_sort_str(set(tab["index"]))
    niches = {i: r_sort_str(set(tab.loc[tab["index"] == i, "niche"])) for i in idx}
    yp = {f: k for k, f in enumerate(ord_)}
    is_gene = D["feature_col"] == "gene"
    W = sum(len(v) for v in niches.values()) * 0.42 + len(idx) * 0.25 + 3.4
    H = len(ord_) * 0.24 + 1.7
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(1, len(idx), width_ratios=[len(niches[i]) for i in idx], left=1.2 / W,
                              right=1 - 2.2 / W, bottom=1.0 / H, top=1 - 0.7 / H, wspace=0.08)  # fmt: skip
        axs = []
        for k, i in enumerate(idx):
            ax = fig.add_subplot(gs[0, k], sharey=axs[0] if axs else None)
            axs.append(ax)
            nc = niches[i]
            xp = {j: c for c, j in enumerate(nc)}
            s = tab[tab["index"] == i]
            xs = np.array([xp[j] for j in s["niche"]]) + 0.5
            ys = np.array([yp[f] for f in s["feature"].astype(str)]) + 0.5
            ax.scatter(xs, ys, marker="s", s=0, c=[0] * len(xs))  # set the data range
            for xx, yy, tv in zip(xs, ys, s["t"], strict=True):
                ax.add_patch(_tile(xx - 0.5, yy - 0.5, cmap(norm(tv)), 0.4))
            u = nt[nt["index"] == i]
            for j, f in zip(u["niche"], u["feature"].astype(str), strict=True):
                ax.add_patch(_tile(xp[j], yp[f], r_colour_hex(_SPIDE_COLS["light"]), 0.4))
            cm = s["called"].to_numpy(dtype=bool)
            ax.scatter(xs[cm], ys[cm], s=_point_area(0.9), c="black", linewidths=0, zorder=3)
            ax.set_xlim(0, len(nc))
            ax.set_ylim(0, len(ord_))
            ax.set_xticks(np.arange(len(nc)) + 0.5, nc, rotation=45, ha="right")
            ax.tick_params(length=0)
            if k == 0:
                ax.set_yticks(
                    np.arange(len(ord_)) + 0.5, [_mt_gene(f) if is_gene else f for f in ord_]
                )
            else:
                ax.tick_params(labelleft=False)
            _strip(ax, _strwrap(i, 10), fontsize=_FS * 0.85)
        fig.supxlabel("niche type", x=(1.2 + (W - 3.4) / 2) / W, fontsize=_FS)
        cax = fig.add_axes([1 - 1.9 / W, 0.55, 0.16 / W, 0.3])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.outline.set_linewidth(0)
        cb.ax.set_title("t", loc="left", fontstyle="italic", fontsize=_FS)
        groups = []
        if tab["called"].any():
            groups.append((None, [Line2D([], [], marker="o", ls="none", color="black", ms=3)],
                           [f"{D['qcol']} ≤ {r_num_str(D['fdr'])}"]))  # fmt: skip
        if len(nt):
            groups.append(
                (None, [Patch(fc=r_colour_hex(_SPIDE_COLS["light"]), ec="white")], ["not tested"])
            )
        if groups:
            _stack_legends(fig, groups, 1 - 1.9 / W, y_top=0.5)
    return fig


def _tile(x: float, y: float, colour, lw: float):
    """A unit tile with a white border (`geom_tile(colour = "white")`)."""
    return Rectangle((x, y), 1, 1, fc=colour, ec="white", lw=lw * _PT)


# --- plotPValues -----------------------------------------------------------------------------


def _isort_with_index(x: np.ndarray) -> np.ndarray:
    """The order R's `isort_with_index` (a shell sort, not stable) gives `x`: the indices
    (0-based) of `x` in sorted order. `merge()` orders its matches with it."""
    x = np.asarray(x, dtype=int).copy()
    n = len(x)
    idx = np.arange(n)
    h = 1
    while h <= n // 9:
        h = 3 * h + 1
    while h > 0:
        for i in range(h, n):
            v, iv, j = x[i], idx[i], i
            while j >= h and x[j - h] > v:
                x[j], idx[j] = x[j - h], idx[j - h]
                j -= h
            x[j], idx[j] = v, iv
        h //= 3
    return idx


def _p_values_data(res, bands: int = 5) -> dict:
    """The data behind `plot_p_values` (R `plotPValues`): `data` (R's `p$data`, one
    row per tested triplet: `gene, index, niche, test, p, band`), and `uni` (the level of a
    uniform p-value histogram, `length / 20`, per test and band)."""
    check_results(res, "plot_p_values()")
    tab = res.table.loc[
        np.isfinite(res.table["p"].to_numpy(dtype=float)), ["gene", "index", "niche", "test", "p"]
    ]
    if not len(tab):
        raise ValueError("no tested triplet has a p-value")
    b = _expression_bands(res.fit, int(bands))
    bkey = (b["gene"].astype(str) + "\r" + b["index"].astype(str)).to_numpy()
    xkey = (tab["gene"].astype(str) + "\r" + tab["index"].astype(str)).to_numpy()
    # merge(tab, bands, by = c("gene", "index")): the rows of tab are ordered by the matching
    # row of the bands with R's (unstable) shell sort, then stably by the pasted key
    first = pd.Series(np.arange(1, len(bkey) + 1), index=bkey)
    first = first[~first.index.duplicated()]
    ix = first.reindex(xkey).fillna(0).to_numpy(dtype=int)
    perm = _isort_with_index(ix)
    keep = perm[ix[perm] > 0]
    order = keep[np.argsort(xkey[keep], kind="stable")]
    band = b["band"].to_numpy()[ix[order] - 1]
    tab = tab.iloc[order].reset_index(drop=True)
    tab["band"] = pd.Categorical(band, categories=list(b["band"].cat.categories))
    lv = [t for t in ("pooled", "condition") if t in set(tab["test"])]
    tab["test"] = pd.Categorical(tab["test"], categories=lv)
    g = (
        tab.groupby(["band", "test"], observed=True, sort=True)["p"]
        .size()
        .reset_index(name="level")
    )
    g["level"] = g["level"] / 20
    uni = g[["test", "band", "level"]]
    return {"data": tab, "uni": uni}


def plot_p_values(res, bands: int = 5) -> Figure:
    """p-values by test and expression band (R `plotPValues`).

    Histograms of every triplet's p-value, one row per test and one column per fifth of each
    index type's genes by mean expression, with the level of a uniform distribution dashed. A
    spike at 0 on a flat floor is signal. A floor that rises toward 0 only in the brightest genes,
    or a hump at 1 (low power, conservative p-values), deserves a look at the calibration
    vignette. This is a sanity check, not a calibration test: real signal also moves p-values
    toward 0, and calibration needs data where there is nothing to find.

    Parameters
    ----------
    res : SpiDEResults
    bands : int
        The number of expression bands.

    Returns
    -------
    matplotlib.figure.Figure
        One row per test, one column per expression band.
    """
    D = _p_values_data(res, bands)
    tab, uni = D["data"], D["uni"]
    tests = list(tab["test"].cat.categories)
    bl = list(tab["band"].cat.categories)
    brk = np.round(np.arange(0, 21) * 0.05, 10)
    fill = _magnitude_ramp(9)[4]
    W, H = len(bl) * 1.9 + 1.7, len(tests) * 1.7 + 1.3
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        fig.subplots_adjust(left=0.8 / W, right=1 - 0.9 / W, top=1 - 0.55 / H, bottom=0.75 / H,
                            wspace=0.35, hspace=0.3)  # fmt: skip
        axs = fig.subplots(len(tests), len(bl), squeeze=False, sharey="row")
        for ri, t in enumerate(tests):
            for ci, b in enumerate(bl):
                ax = axs[ri, ci]
                p = tab.loc[
                    (tab["test"].astype(object) == t) & (tab["band"].astype(object) == b), "p"
                ].to_numpy(dtype=float)
                # ggplot2 bins are (a, b] with the first closed
                code = np.maximum(np.searchsorted(brk, p, side="left"), 1) - 1
                cnt = np.bincount(code, minlength=20)[:20]
                ax.bar(
                    brk[:-1],
                    cnt,
                    width=0.05,
                    align="edge",
                    color=fill,
                    edgecolor="white",
                    linewidth=0.2 * _PT,
                )
                u = uni[(uni["test"].astype(object) == t) & (uni["band"].astype(object) == b)]
                if len(u):
                    ax.axhline(float(u["level"].iloc[0]), color="black", ls="--", lw=0.4 * _PT)
                ax.set_xlim(-0.03, 1.03)
                ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
                ax.set_ylim(0, None)
                ax.margins(y=0.05)
                ax.set_ylim(bottom=0)
                if ri == 0:
                    _strip(ax, _strwrap(b, 9), fontsize=_FS * 0.8 * 0.85)
                if ci == len(bl) - 1:
                    _strip_box(ax, _strwrap(t, 9))
        fig.supxlabel("p-value", x=(0.8 + (W - 1.7) / 2) / W, fontsize=_FS)
        fig.supylabel("triplets", fontsize=_FS)
        fig.legend([Line2D([], [], color="black", ls="--", lw=1)], ["uniform"], loc="lower right",
                   bbox_to_anchor=(1.0, 0.0), frameon=False)  # fmt: skip
    return fig
