"""The niche views (``R/plot-niche.R``): the niche covariate in tissue, and the index cells'
expression against it."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from .._checks import (
    check_adata,
    check_column,
    check_intercepts,
    check_niche,
    check_results,
    check_triplet,
)
from .._classes import SpiDEResults
from .._design import _pick_sigma
from .._io import get_coords, get_niche, get_obs, niche_key
from .._rcompat import r_num_str, r_sort_str
from ._helpers import _group_levels, _index_cell_data, _slope_weights
from ._theme import (
    _PT,
    _SPIDE_COLS,
    _blank_panel,
    _condition_colours,
    _it_gene,
    _log10_axis,
    _magnitude_cmap,
    _mt_gene,
    _point_area,
    _strip,
    _wrap_dims,
    r_colour_hex,
    style,
)

# --- plotNicheMap ----------------------------------------------------------------------------


def _niche_map_data(
    adata,
    niche,
    sample,
    index=None,
    sigma=None,
    cell_type: str = "cell_type",
    sample_id: str = "sample_id",
    section=None,
    name: str = "Niche",
) -> pd.DataFrame:
    """One sample's cells for :func:`plot_niche_map` (the data of R's ``plotNicheMap`` ggplot).

    One row per cell of the sample: ``x``, ``y`` (from each section's own origin), ``cell_type``,
    ``density`` (log1p of the niche column), ``section`` and ``focal`` (coloured).
    """
    check_adata(adata, cell_type=cell_type, sample_id=sample_id)
    check_column(adata, section, "section")
    sigma = _pick_sigma(adata, sigma, name)
    check_niche(adata, sigma, name)
    key = niche_key(name, sigma)
    NM = get_niche(adata, key)
    if not isinstance(niche, str) or niche not in NM.columns:
        shown = niche if isinstance(niche, str) else ", ".join(map(str, np.atleast_1d(niche)))
        raise ValueError(f"niche '{shown}' is not a column of {key}")
    smp = get_obs(adata, sample_id)
    if isinstance(sample, str) is False or not (smp == sample).any():
        shown = sample if isinstance(sample, str) else ", ".join(map(str, np.atleast_1d(sample)))
        raise ValueError(f"sample '{shown}' not found in adata.obs['{sample_id}']")
    i = np.flatnonzero(smp == sample)
    ct = get_obs(adata, cell_type)[i]
    if index is not None and not (ct == index).any():
        raise ValueError(f"no {index} cells in sample {sample}")
    xy = get_coords(adata)[i]
    d = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cell_type": ct,
            "density": np.log1p(NM[niche].to_numpy(dtype=float)[i]),
            "section": sample if section is None else get_obs(adata, section)[i],
        }
    )
    d["focal"] = True if index is None else (d["cell_type"] == index).to_numpy()
    # each section on its own origin: sections' coordinates are unrelated
    d["x"] = d["x"] - d.groupby("section")["x"].transform("min")
    d["y"] = d["y"] - d.groupby("section")["y"].transform("min")
    return d


def _scale_bar(d: pd.DataFrame, unit) -> pd.DataFrame:
    """A scale bar of about a fifth of the width, under the bottom-right corner (R ``.scaleBar``).

    One row: ``x``, ``xend``, ``y`` (below the cells) and ``label``.
    """
    xr = (d["x"].min(), d["x"].max())
    yr = (d["y"].min(), d["y"].max())
    length = float(f"{(xr[1] - xr[0]) / 5:.0e}")  # signif(., 1)
    lab = r_num_str(length)
    return pd.DataFrame(
        {
            "x": [xr[1] - length],
            "xend": [xr[1]],
            "y": [yr[0] - 0.04 * (yr[1] - yr[0])],
            "label": [lab if unit is None else f"{lab} {unit}"],
        }
    )


def plot_niche_map(
    adata,
    niche,
    sample,
    index=None,
    sigma=None,
    cell_type: str = "cell_type",
    sample_id: str = "sample_id",
    section=None,
    name: str = "Niche",
    point_size=None,
    scale_bar: bool = True,
    unit=None,
) -> Figure:
    """The niche covariate in tissue (R ``plotNicheMap``).

    One sample's cells, coloured by the log1p density of a niche type: the covariate spiDE
    regresses on. With ``index``, only the index type's cells are coloured (the covariate as the
    index type's model reads it) and the others are light grey.

    Read it to judge the bandwidth: a speckled field that follows single cells is too small; a
    field nearly flat within a section is too large (the patient intercept then absorbs it). It
    also shows where the index cells sit relative to the niche.

    Parameters
    ----------
    adata : anndata.AnnData
        With niche matrices in ``obsm`` (:func:`~spidepy.build_niches`).
    niche : str
        A niche column (a cell type or a :func:`~spidepy.merge_niches` group).
    sample : str
        The sample (patient) to show.
    index : str, optional
        An index cell type to highlight.
    sigma : float, optional
        The bandwidth (needed only if ``adata`` carries several).
    cell_type, sample_id : str
        The ``obs`` columns of cell type and sample.
    section : str, optional
        The ``obs`` column of the section; a sample with several sections gets one panel each.
    name : str
        The niche ``obsm`` prefix.
    point_size : float, optional
        The point size in ggplot2 units (default: from the number of cells).
    scale_bar : bool
        Draw a scale bar of about a fifth of the width.
    unit : str, optional
        The coordinates' unit, for the scale bar's label (e.g. ``"um"``).

    Returns
    -------
    matplotlib.figure.Figure
        One panel per section, plus the colour bar's axes.
    """
    d = _niche_map_data(adata, niche, sample, index, sigma, cell_type, sample_id, section, name)
    if point_size is None:
        point_size = min(1.5, max(0.05, 50 / math.sqrt(len(d))))
    sections = r_sort_str(d["section"].unique())
    nrow, ncol = _wrap_dims(len(sections))
    xr = (d["x"].min(), d["x"].max())
    yr = (d["y"].min(), d["y"].max())
    bar = _scale_bar(d, unit) if scale_bar else None
    # ggplot2 expands a continuous scale by 5% (the scale bar widens y by its own offset and
    # leaves room below for its label)
    ex = 0.05 * (xr[1] - xr[0])
    ylo = yr[0] if bar is None else min(yr[0], float(bar["y"].iloc[0]))
    lo_mult, hi_mult = (0.05, 0.05) if bar is None else (0.16, 0.04)
    ylim = (ylo - lo_mult * (yr[1] - ylo), yr[1] + hi_mult * (yr[1] - ylo))
    asp = (ylim[1] - ylim[0]) / ((xr[1] - xr[0]) + 2 * ex) if xr[1] > xr[0] else 1.0
    pw = 3.6
    ph = float(np.clip(pw * asp, 1.8, 6.0))
    foc = d[d["focal"]]
    vmin, vmax = foc["density"].min(), foc["density"].max()
    with style():
        fig = Figure(figsize=(ncol * pw + 2.4, nrow * ph + 0.4), layout="constrained")
        axs = fig.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True)
        sc = None
        for k, ax in enumerate(axs.ravel()):
            if k >= len(sections):
                ax.set_visible(False)
                continue
            s = d[d["section"] == sections[k]]
            if index is not None:
                g = s[~s["focal"]]
                ax.scatter(
                    g["x"],
                    g["y"],
                    s=_point_area(point_size * 0.7),
                    c=r_colour_hex(_SPIDE_COLS["faint"]),
                    linewidths=0,
                    marker="o",
                )
            f = s[s["focal"]]
            sc = ax.scatter(
                f["x"], f["y"], s=_point_area(point_size), c=f["density"],
                cmap=_magnitude_cmap(256, 0.1), vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1,
                linewidths=0, marker="o",
            )  # fmt: skip
            if bar is not None:
                b = bar.iloc[0]
                ax.plot([b["x"], b["xend"]], [b["y"], b["y"]], color="black", lw=0.8 * _PT,
                        solid_capstyle="butt")  # fmt: skip
                ax.annotate(b["label"], ((b["x"] + b["xend"]) / 2, b["y"]), xytext=(0, -4),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=3.2 * _PT)  # fmt: skip
            ax.set_xlim(xr[0] - ex, xr[1] + ex)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            _blank_panel(ax)
            if len(sections) > 1:
                _strip(ax, sections[k])
        cb = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.6, aspect=18)
        cb.set_label(f"{niche} density\n(log1p)", fontstyle="italic", fontsize=11)
        cb.outline.set_linewidth(0)
    return fig


# --- plotNicheResponse -----------------------------------------------------------------------


def _intercept_at_own_depth(fit) -> bool:
    """Whether the patient intercepts sit at each patient's own mean depth (R
    ``.interceptAtOwnDepth``).

    True under ``depth = "spatial_spline"`` (log depth centred per section) and under ``"none"``
    with no offset or covariates (no depth term). Under ``"loglib"`` and ``"nonlinear"`` the depth
    covariate is centred over the index type, a reference shared by every patient.
    """
    p = fit.params
    return p.get("depth") == "spatial_spline" or (
        p.get("depth") == "none" and not p.get("offset") and not p.get("covariates")
    )


def _response_slopes(fit, xi: dict, niche: str):
    """The slopes drawn by :func:`plot_niche_response` (R ``.responseSlopes``).

    Each patient's own slope and each group's weighted mean (the slopes engine's pooled-test
    weights), or, under the sandwich engine, the pooled slope for every patient and group.
    Returns ``(patient, group)``: ``patient(g, pats)`` the slopes of these patients (matched by
    name; missing patients are NaN), ``group(g, pats)`` the weighted mean slope of the patients.
    """
    if fit.engine == "slopes":
        j = list(xi["niches"]).index(niche)
        v, tau = _slope_weights(xi)
        genes = [str(g) for g in xi["genes"]]
        pats = [str(p) for p in xi["patients"]]

        def get(g):
            gi = genes.index(g)
            with np.errstate(divide="ignore", invalid="ignore"):
                w = 1.0 / (v[gi, :, j] + tau[gi, j])
            return pd.Series(np.asarray(xi["beta"])[gi, :, j], index=pats), pd.Series(w, index=pats)

        def patient(g, ps):
            return get(g)[0].reindex(list(ps)).to_numpy(dtype=float)

        def group(g, ps):
            b, w = get(g)
            k = [p for p in ps if p in b.index]
            k = [p for p in k if np.isfinite(b[p]) and np.isfinite(w[p]) and w[p] > 0]
            if not k:
                return np.nan
            return float((w[k] * b[k]).sum() / w[k].sum())

        return patient, group
    cf = xi["coef"]
    cf = cf[(cf["test"] == "pooled") & (cf["niche"] == niche)].drop_duplicates("gene")
    est = cf.set_index("gene")["estimate"]

    def pooled(g):
        return float(est.get(g, np.nan))

    return (
        lambda g, ps: np.full(len(ps), pooled(g)),
        lambda g, ps: pooled(g),
    )


def _cut_codes(v: np.ndarray, br: np.ndarray) -> np.ndarray:
    """R ``cut(v, br, include.lowest = TRUE, labels = FALSE)``: the bin (1-based) of each value;
    bins are ``[b0, b1], (b1, b2], ...``."""
    return np.maximum(np.searchsorted(br, v, side="left"), 1)


def _bins_of(v: np.ndarray, n: int) -> np.ndarray:
    """Bins of ``n`` quantile bands of ``v`` (R ``unique(quantile(v, seq(0, 1, length = n + 1)))``
    then ``cut``); one bin if the quantiles collapse."""
    br = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)))
    return np.ones(len(v), dtype=int) if len(br) < 2 else _cut_codes(v, br)


def _aggregate_mean(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """R ``aggregate(cbind(expr, density) ~ by, data, mean)``: rows with a missing value are
    dropped, groups sorted by the grouping variables with the *last* varying slowest."""
    df = df.dropna(subset=["expr", "density"])
    g = df.groupby(by[::-1], sort=True)[["expr", "density"]].mean().reset_index()
    return g[by + ["expr", "density"]]


def _niche_response_data(
    adata, x, gene, index, niche, adjust: bool = True, bins: int = 8, assay: str = "counts",
    name: str = "Niche",
) -> dict:  # fmt: skip
    """The data behind :func:`plot_niche_response` (R ``plotNicheResponse``'s ggplot layers).

    Returns a dict: ``data`` (R's ``p$data``: one row per (gene, patient, bin) with ``patient``,
    ``bin``, ``expr`` (mean CP10k), ``density`` (mean log1p density), ``gene``, ``condition`` and
    ``gene_lab``); with ``adjust`` also ``summary`` (the mean across patients per group and bin
    with its 95% interval) and ``lines`` (each group's fitted slope), else ``centroids`` (each
    patient's mean and its within-patient segment); ``genes`` (the plotted genes), ``levels`` (the
    groups, in contrast order) and ``labels`` (plotmath label by gene).
    """
    check_results(x, "plot_niche_response()", fit_ok=True)
    is_res = isinstance(x, SpiDEResults)
    fit = x.fit if is_res else x
    condition = x.condition if is_res else fit.condition
    check_adata(adata, assay, fit.params["cell_type"], fit.params["sample_id"])
    gene = list(dict.fromkeys([gene] if isinstance(gene, str) else [str(g) for g in gene]))
    check_triplet(fit, index, niche, gene)
    xi = fit.index[index]
    if adjust:
        check_intercepts(xi, "plot_niche_response(adjust = True)")
        # a gene the fit could not fit has no intercept: it drops out, not the plot
        fitted = np.isfinite(xi["intercept"].loc[gene].to_numpy(dtype=float)).sum(axis=1) > 0
        if not fitted.all():
            lost = ", ".join(g for g, ok in zip(gene, fitted, strict=True) if not ok)
            warnings.warn(
                f"plot_niche_response(): {lost} not fitted in {index}, left out",
                UserWarning,
                stacklevel=3,
            )
        gene = [g for g, ok in zip(gene, fitted, strict=True) if ok]
        if not gene:
            raise ValueError("no gene left to plot: none of them was fitted")
    cells = _index_cell_data(adata, fit, index, niche, gene, assay, name)
    grp = _group_levels(fit, condition)
    lv = [str(c) for c in grp.cat.categories]
    lab = {g: _it_gene(g) for g in gene}
    s_patient, s_group = _response_slopes(fit, xi, niche)
    dens = cells["density"]
    pat = cells["patient"]
    if adjust:
        bin_ = _bins_of(dens, bins)
    else:
        bin_ = np.ones(len(dens))
        for p in pd.unique(pat):
            m = pat == p
            bin_[m] = _bins_of(dens[m], 5)
    pieces = []
    for k, g in enumerate(gene):
        e = cells["counts"][k] / cells["lib"] * 1e4
        if adjust:
            a = xi["intercept"].loc[g].astype(float)
            # CP10k has divided out each cell's depth already; where the intercepts sit at each
            # patient's own mean depth, that depth comes off them too
            if _intercept_at_own_depth(fit):
                ll = xi["loglib_mean"].astype(float)
                a = a - (ll.reindex(a.index) - np.nanmedian(ll.to_numpy()))
            fin = a[np.isfinite(a)]
            # the reference is the patients' median intercept: one patient pinned at an extreme
            # value by the intercepts' ridge would move a mean
            ref = float(np.median(fin)) if len(fin) else np.nan
            e = e * np.exp(-(a - ref)).reindex(pat).to_numpy()  # by patient name
        d = _aggregate_mean(
            pd.DataFrame({"expr": e, "density": dens, "patient": pat, "bin": bin_}),
            ["patient", "bin"],
        )
        d["gene"] = g
        pieces.append(d)
    pb = pd.concat(pieces, ignore_index=True)
    cond = pd.Categorical(
        grp.reindex(pb["patient"].to_numpy()).astype(object), categories=lv
    )  # by patient name
    pb["condition"] = cond
    pb["gene_lab"] = pd.Categorical(pb["gene"].map(lab), categories=list(lab.values()))
    out = {"data": pb, "genes": gene, "levels": lv, "labels": lab}
    genes_sorted = r_sort_str(set(pb["gene"]))
    if adjust:
        rows = []
        for (b, c, g), s in pb.groupby(
            ["bin", "condition", "gene"], sort=False, observed=True
        ):  # fmt: skip
            n = len(s)
            se = s["expr"].std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
            rows.append(
                {
                    "gene": g,
                    "condition": c,
                    "bin": b,
                    "density": s["density"].mean(),
                    "mean": s["expr"].mean(),
                    "half": stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan,
                }  # fmt: skip
            )
        sm = pd.DataFrame(rows, columns=["gene", "condition", "bin", "density", "mean", "half"])
        # split(drop = TRUE) orders the groups by bin, then condition level, then gene
        sm["_c"] = pd.Categorical(sm["condition"], categories=lv).codes
        sm["_g"] = sm["gene"].map({g: i for i, g in enumerate(genes_sorted)})
        sm = sm.sort_values(["bin", "_c", "_g"], kind="stable").drop(columns=["_c", "_g"])
        sm = sm[sm["mean"] > 0].reset_index(drop=True)
        if not len(sm):
            raise ValueError("no counts of the chosen genes in these cells")
        floor_y = sm["mean"].min() / 2
        sm["lo"] = np.fmax(sm["mean"] - sm["half"], floor_y).where(sm["half"].notna())
        sm["hi"] = sm["mean"] + sm["half"]
        sm["gene_lab"] = pd.Categorical(sm["gene"].map(lab), categories=list(lab.values()))
        sm["condition"] = pd.Categorical(sm["condition"].astype(object), categories=lv)
        out["summary"] = sm
        lrows = []
        for (c, g), s in sorted(
            ((k, v) for k, v in sm.groupby(["condition", "gene"], observed=True)),
            key=lambda kv: (lv.index(kv[0][0]), genes_sorted.index(kv[0][1])),
        ):
            pats = [p for p, cv in grp.items() if cv == c]
            b = s_group(g, pats)
            x0, y0 = s["density"].mean(), np.log(s["mean"]).mean()
            xs = (s["density"].min(), s["density"].max())
            with np.errstate(invalid="ignore"):
                lrows.append(
                    {
                        "gene_lab": lab[g],
                        "condition": c,
                        "x": xs[0],
                        "xend": xs[1],
                        "y": np.exp(y0 + b * (xs[0] - x0)),
                        "yend": np.exp(y0 + b * (xs[1] - x0)),
                    }  # fmt: skip
                )
        ln = pd.DataFrame(lrows, columns=["gene_lab", "condition", "x", "xend", "y", "yend"])
        ln["gene_lab"] = pd.Categorical(ln["gene_lab"], categories=list(lab.values()))
        ln["condition"] = pd.Categorical(ln["condition"].astype(object), categories=lv)
        out["lines"] = ln
    else:
        pieces = []
        for k, g in enumerate(gene):
            e = cells["counts"][k] / cells["lib"] * 1e4
            d = _aggregate_mean(
                pd.DataFrame({"expr": e, "density": dens, "patient": pat}), ["patient"]
            )
            d["gene"] = g
            d["beta"] = s_patient(g, d["patient"])
            q = np.array([np.quantile(dens[pat == p], [0.05, 0.95]) for p in d["patient"]]).reshape(
                -1, 2
            )
            d["x"], d["xend"] = q[:, 0], q[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                d["y"] = np.exp(np.log(d["expr"]) + d["beta"] * (d["x"] - d["density"]))
                d["yend"] = np.exp(np.log(d["expr"]) + d["beta"] * (d["xend"] - d["density"]))
            pieces.append(d)
        cen = pd.concat(pieces, ignore_index=True)
        cen = cen[np.isfinite(cen["beta"]) & (cen["expr"] > 0)].reset_index(drop=True)
        cen["condition"] = pd.Categorical(
            grp.reindex(cen["patient"].to_numpy()).astype(object), categories=lv
        )
        cen["gene_lab"] = pd.Categorical(cen["gene"].map(lab), categories=list(lab.values()))
        out["centroids"] = cen
    return out


def plot_niche_response(
    adata, res, gene, index, niche, adjust: bool = True, bins: int = 8, assay: str = "counts",
    name: str = "Niche",
) -> Figure:  # fmt: skip
    """Expression against niche density: the cells behind a slope (R ``plotNicheResponse``).

    The index type's cells' expression of each gene (counts per 10,000, log axis) against the
    log1p density of the niche type, binned, one panel per gene.

    With ``adjust=True`` (the default) each cell's expression is divided by its patient's fitted
    intercept, so the patients share one reference level and overlay: faint points are each
    patient's mean in a density bin, points and bars the mean across patients and its 95%
    interval, per condition, and the dashed line each group's fitted slope. This is what the
    tests compare. With ``adjust=False`` the data are raw: each patient's own slope is a segment
    through its mean (open circle), and the dashed line through the patients' means is the
    between-patient association that the intercepts absorb (:func:`~spidepy.test_niche_abundance`'s
    question), not a niche slope.

    Read it for the data behind a slope, whether the log1p scale is roughly linear, and whether a
    few extreme cells drive it. The display normalises depth by library size (the model fits a
    slope on it).

    Parameters
    ----------
    adata : anndata.AnnData
        The object the fit was made from.
    res : SpiDEResults or SpiDEFit
        ``adjust=True`` needs a fit that kept the patient intercepts. Under the sandwich engine
        every patient's and group's line has the pooled slope.
    gene : str or sequence of str
        One or more genes, one panel each.
    index, niche : str
        The index and niche cell types.
    adjust : bool
        Divide out each patient's intercept (see above).
    bins : int
        The number of density bins (``adjust=True``; bins are quantiles over all patients). Raw
        data use fifths of each patient's own density range.
    assay : str
        The counts assay.
    name : str
        The niche ``obsm`` prefix.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per gene (a free log y axis each), with the condition and line legends.
    """
    D = _niche_response_data(adata, res, gene, index, niche, adjust, bins, assay, name)
    pb, genes, lv = D["data"], D["genes"], D["levels"]
    col = _condition_colours(lv)
    present = [c for c in lv if (pb["condition"].astype(object) == c).any()]
    nrow, ncol = _wrap_dims(len(genes))
    pw, ph = 3.3, 2.9
    left, right, bottom, top = 1.2, 2.7, 0.8, 0.3
    W = ncol * pw + left + right
    H = nrow * ph + bottom + top
    xlab = f"{niche} density (log1p)"
    ylab = "expression relative to patient (CP10k)" if adjust else "expression (CP10k)"
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)  # a renderer, to place the second legend
        fig.subplots_adjust(
            left=left / W, right=1 - right / W, bottom=bottom / H, top=1 - top / H,
            wspace=0.8 / pw, hspace=0.75 / ph,
        )  # fmt: skip
        axs = fig.subplots(nrow, ncol, squeeze=False, sharex=True)
        for k, ax in enumerate(axs.ravel()):
            if k >= len(genes):
                ax.set_visible(False)
                continue
            g = genes[k]
            lab = D["labels"][g]
            P = pb[(pb["gene"] == g) & (pb["expr"] > 0)]
            pcol = [col[str(c)] for c in P["condition"].astype(object)]
            if adjust:
                ax.scatter(P["density"], P["expr"], s=_point_area(0.6), c=pcol, alpha=0.3,
                           linewidths=0, marker="o")  # fmt: skip
                L = D["lines"]
                for _, r in L[L["gene_lab"].astype(str) == lab].iterrows():
                    ax.plot([r["x"], r["xend"]], [r["y"], r["yend"]], color=col[str(r["condition"])],
                            lw=0.8 * _PT, ls=(0, (2, 2)))  # fmt: skip
                S = D["summary"]
                S = S[S["gene"] == g]
                for c in lv:
                    s = S[S["condition"].astype(object) == c].sort_values("density")
                    if len(s):
                        ax.plot(s["density"], s["mean"], color=col[c], lw=0.5 * _PT, ls="-")
                s = S.dropna(subset=["lo", "hi"])
                for c in lv:
                    t = s[s["condition"].astype(object) == c]
                    if len(t):
                        ax.errorbar(t["density"], t["mean"],
                                    yerr=[t["mean"] - t["lo"], t["hi"] - t["mean"]], fmt="o",
                                    color=col[c], ms=0.3 * 4 * _PT, lw=0.5 * _PT, elinewidth=0.5 * _PT,
                                    capsize=0)  # fmt: skip
            else:
                ax.scatter(P["density"], P["expr"], s=_point_area(0.6), c=pcol, alpha=0.35,
                           linewidths=0, marker="o")  # fmt: skip
                C = D["centroids"]
                C = C[C["gene"] == g]
                for _, r in C.iterrows():
                    ax.plot([r["x"], r["xend"]], [r["y"], r["yend"]], color=col[str(r["condition"])],
                            lw=0.35 * _PT, ls="-")  # fmt: skip
                if len(C) > 1 and C["density"].nunique() > 1:
                    fit_ = np.polyfit(C["density"], np.log10(C["expr"]), 1)
                    xs = np.array([C["density"].min(), C["density"].max()])
                    ax.plot(xs, 10 ** np.polyval(fit_, xs), color="black", lw=0.7 * _PT,
                            ls=(0, (2, 2)))  # fmt: skip
                ax.scatter(C["density"], C["expr"], s=_point_area(1.8), facecolors="white",
                           edgecolors=[col[str(c)] for c in C["condition"]], linewidths=0.6,
                           marker="o", zorder=3)  # fmt: skip
            ax.set_yscale("log")
            ax.margins(x=0.05)
            _log10_axis(ax)
            _strip(ax, _mt_gene(g))
            if k + ncol >= len(genes):  # the last panel of its column carries the x axis
                ax.tick_params(labelbottom=True)
        # the axis titles are centred on the panels, not on the figure with its legend
        fig.supxlabel(xlab, x=(left + ncol * pw / 2) / W, fontsize=11 * 1.1 * 1.1)
        fig.supylabel(ylab, y=(bottom + nrow * ph / 2) / H, fontsize=11 * 1.1 * 1.1)
        _legends(fig, present, col, adjust, right / W, H)
    return fig


def _legends(fig: Figure, present, col, adjust: bool, width_frac: float, height: float) -> None:
    """The condition (colour) legend above the line-type legend, right of the panels."""
    ms = 3.0 if adjust else 7.0
    h1 = [Line2D([], [], marker="o", ls="none", color=col[c], ms=ms) for c in present]
    names = (
        ("fitted slope", "mean across patients")
        if adjust
        else ("between patients", "within patient")
    )
    dash = (0, (2, 2))
    # legend entries are alphabetical (ggplot2 sorts the discrete linetype scale)
    h2 = [Line2D([], [], color="black", ls=s, lw=1.4) for s in (dash, "-")]
    x0 = 1 - width_frac + 0.02
    leg1 = fig.legend(h1, present, loc="upper left", bbox_to_anchor=(x0, 0.9), frameon=False)
    # stack the second guide under the first
    box = leg1.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    y2 = box.y0 - 0.03
    fig.legend(h2, list(names), loc="upper left", bbox_to_anchor=(x0, y2), frameon=False)
