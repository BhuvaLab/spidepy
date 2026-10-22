"""The patient-level views (``R/plot-patients.R``): each patient's slopes for a triplet, and what
the patients' intercepts (or slopes) capture."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy import stats

from .._checks import check_intercepts, check_results, check_slopes_engine, check_triplet
from .._classes import SpiDEResults
from .._io import as_character_array
from .._rcompat import r_sort_str
from ..fit import _condition_coding, _match_arg
from ._helpers import _group_levels, _slope_weights, _triplet_rows
from ._theme import (
    _PT,
    _SPIDE_COLS,
    _condition_colours,
    _fmt_q,
    _it_gene,
    _magnitude_cmap,
    _mt_gene,
    _point_area,
    _stack_legends,
    _strip,
    r_colour_hex,
    spide_colours,
    style,
)

# --- plotPatientSlopes -----------------------------------------------------------------------


def _slope_strip_labels(res, gene, index: str, niche: str) -> dict[str, str]:
    """Each gene's strip label: the italic symbol over the condition test's slope difference and
    q (or, without a condition, the pooled slope and q), as R's plotmath string
    (R ``.slopeStripLabels``)."""
    gene = [str(g) for g in np.atleast_1d(gene)]
    cond = bool(res.condition)
    r = _triplet_rows(res.table, "condition" if cond else "pooled", gene, index, niche)
    est = r["estimate"].to_numpy(dtype=float)
    qs = _fmt_q(r["q"].to_numpy(dtype=float))
    out = {}
    for g, e, q in zip(gene, est, qs, strict=True):
        es = f"{e:+.2f}" if np.isfinite(e) else "NA"
        stat = f" = {es}, q = {q}"
        stat = f'Delta * "{stat}"' if cond else f'"slope{stat}"'
        out[g] = f"atop({_it_gene(g)}, scriptstyle({stat}))"
    return out


def _strip_text(res, gene: str, index: str, niche: str) -> str:
    """The two-line mathtext strip of one gene (the counterpart of the plotmath label)."""
    cond = bool(res.condition)
    r = _triplet_rows(res.table, "condition" if cond else "pooled", [gene], index, niche)
    e = float(r["estimate"].iloc[0])
    es = f"{e:+.2f}" if np.isfinite(e) else "NA"
    q = _fmt_q(r["q"].to_numpy(dtype=float))[0]
    return f"{_mt_gene(gene)}\n{'Δ' if cond else 'slope'} = {es}, q = {q}"


def _patient_slopes_data(res, gene, index: str, niche: str, ci: float = 0.95) -> dict:
    """The data behind :func:`plot_patient_slopes` (R ``plotPatientSlopes``'s layers).

    Returns a dict: ``data`` (R's ``p$data``: one row per (gene, patient) with a usable slope,
    ``gene, patient, slope, se, weight, condition, gene_lab``; ``patient`` and ``gene_lab`` are
    categoricals in R's order), ``band`` (the pooled estimate and its interval per gene),
    ``gm`` (each group's weighted mean slope per gene), ``labels`` (plotmath strip label by
    gene), ``levels`` (the groups, in contrast order) and ``z`` (the interval's normal quantile).
    """
    check_results(res, "plot_patient_slopes()")
    fit = res.fit
    check_slopes_engine(fit, "plot_patient_slopes()")
    gene = list(dict.fromkeys([gene] if isinstance(gene, str) else [str(g) for g in gene]))
    check_triplet(fit, index, niche, gene)
    if (
        isinstance(ci, bool)
        or not isinstance(ci, (int, float, np.integer, np.floating))
        or not np.isfinite(ci)
        or ci <= 0
        or ci >= 1
    ):
        raise ValueError("'ci' must be a single number in (0, 1)")
    xi = fit.index[index]
    j = list(xi["niches"]).index(niche)
    v_all, tau_all = _slope_weights(xi)
    z = float(stats.norm.ppf(1 - (1 - ci) / 2))
    grp = _group_levels(fit, res.condition)
    lv = [str(c) for c in grp.cat.categories]
    genes_all = [str(g) for g in xi["genes"]]
    pats = [str(p) for p in xi["patients"]]
    pieces = []
    for g in gene:
        gi = genes_all.index(g)
        b = np.asarray(xi["beta"], dtype=float)[gi, :, j]
        v = v_all[gi, :, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / (v + tau_all[gi, j])
            ok = np.isfinite(b) & np.isfinite(w) & (w > 0)
            pieces.append(
                pd.DataFrame(
                    {
                        "gene": g,
                        "patient": pats,
                        "slope": b,
                        "se": np.sqrt(v),
                        "weight": np.where(ok, w / w[ok].sum(), np.nan),
                    }
                )[ok]
            )
    d = pd.concat(pieces, ignore_index=True)
    if not len(d):
        raise ValueError("no patient has a usable slope for these genes")
    d["condition"] = pd.Categorical(
        grp.reindex(d["patient"].to_numpy()).astype(object), categories=lv
    )
    # patients ordered by group, then by their mean slope over the genes shown (the mean is
    # taken over rows with a group, as aggregate() drops the others)
    ok = d["condition"].notna()
    ord_ = d[ok].groupby(["condition", "patient"], observed=True)["slope"].mean().reset_index()
    ord_["_p"] = ord_["patient"].map({p: i for i, p in enumerate(r_sort_str(set(ord_["patient"])))})
    ord_ = ord_.sort_values(["condition", "_p"], kind="stable")
    ord_ = ord_.sort_values(["condition", "slope"], kind="stable")
    plev = list(dict.fromkeys(ord_["patient"]))
    d["patient"] = pd.Categorical(d["patient"], categories=plev)
    lab = _slope_strip_labels(res, gene, index, niche)
    labs = [lab[g] for g in gene]
    d["gene_lab"] = pd.Categorical(d["gene"].map(lab), categories=labs)
    pr = _triplet_rows(res.table, "pooled", gene, index, niche)
    est = pr["estimate"].to_numpy(dtype=float)
    tq = stats.t.ppf(1 - (1 - ci) / 2, pr["df"].to_numpy(dtype=float))
    se = pr["se"].to_numpy(dtype=float)
    band = pd.DataFrame(
        {
            "gene_lab": pd.Categorical(labs, categories=labs),
            "est": est,
            "lo": est - tq * se,
            "hi": est + tq * se,
        }
    )
    # split(d, list(gene_lab, condition), drop = TRUE): condition slowest, gene_lab fastest
    rows = []
    for c in lv:
        for gl in labs:
            s = d[(d["gene_lab"] == gl) & (d["condition"] == c)]
            if len(s):
                rows.append(
                    {
                        "gene_lab": gl,
                        "condition": c,
                        "mean": float((s["weight"] * s["slope"]).sum() / s["weight"].sum()),
                    }
                )
    gm = pd.DataFrame(rows, columns=["gene_lab", "condition", "mean"])
    gm["gene_lab"] = pd.Categorical(gm["gene_lab"], categories=labs)
    gm["condition"] = pd.Categorical(gm["condition"], categories=lv)
    return {"data": d, "band": band, "gm": gm, "labels": lab, "levels": lv, "z": z, "genes": gene}


def _size_breaks(top: float) -> list[float]:
    """Up to four round, positive breaks of a 0..``top`` size scale."""
    from matplotlib.ticker import MaxNLocator

    b = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, top)
    return [x for x in b if 0 < x <= top * 1.0000001]


def plot_patient_slopes(res, gene, index: str, niche: str, ci: float = 0.95) -> Figure:
    """Each patient's niche slope for a triplet (R ``plotPatientSlopes``).

    The evidence the tests weigh: every patient's own slope of the gene's expression in the
    index type on the niche type's density (log1p), with an interval from its within-patient
    variance (the variance the tests use) and a point sized by the patient's weight in them. The
    pooled estimate and its confidence interval are the grey band. With a condition the patients
    are split by it, each group's weighted mean is a coloured line, and each strip gives the
    condition-specific slope difference and its q-value.

    Read it for whether a call rests on patients that agree or on one or two heavily weighted
    ones, and for why a condition test is missing (each group needs three effective patients).
    Patients are ordered by group, then by their mean slope over the genes shown, so a row is one
    patient across genes.

    Parameters
    ----------
    res : SpiDEResults
        From the slopes engine.
    gene : str or sequence of str
        One or more genes, one column each.
    index, niche : str
        The index and niche cell types.
    ci : float
        The confidence level of the intervals.

    Returns
    -------
    matplotlib.figure.Figure
        One column per gene and one row per condition group.
    """
    D = _patient_slopes_data(res, gene, index, niche, ci)
    d, band, gm, lab, lv, z, genes = (
        D["data"],
        D["band"],
        D["gm"],
        D["labels"],
        D["levels"],
        D["z"],
        D["genes"],
    )
    col = _condition_colours(lv)
    rows = [c for c in lv if (d["condition"].astype(object) == c).any()]
    plev = list(d["patient"].cat.categories)
    pat_group = (
        d.drop_duplicates("patient").set_index("patient", drop=True)["condition"].astype(object)
    )
    row_pats = {c: [p for p in plev if pat_group.get(p) == c] for c in rows}
    ncol = len(genes)
    maxw = float(d["weight"].max())
    pw, row_h = 3.0, 0.2
    left, right, bottom, top = 1.0, 3.6, 0.8, 0.9
    heights = [len(row_pats[c]) * row_h + 0.35 for c in rows]
    W = ncol * pw + left + right
    H = sum(heights) + bottom + top + 0.15 * (len(rows) - 1)
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(
            len(rows), ncol, height_ratios=heights, left=left / W, right=1 - right / W,
            bottom=bottom / H, top=1 - top / H, wspace=0.12, hspace=0.15 / (sum(heights) / len(rows)),
        )  # fmt: skip
        first_in_col: dict[int, object] = {}
        for ri, c in enumerate(rows):
            pats = row_pats[c]
            ypos = {p: i for i, p in enumerate(pats)}
            for ci_, g in enumerate(genes):
                ax = fig.add_subplot(gs[ri, ci_], sharex=first_in_col.get(ci_))
                first_in_col.setdefault(ci_, ax)
                gl = lab[g]
                b = band[band["gene_lab"] == gl].iloc[0]
                if np.isfinite(b["lo"]) and np.isfinite(b["hi"]):
                    ax.axvspan(b["lo"], b["hi"], color=r_colour_hex(_SPIDE_COLS["faint"]), lw=0)
                if np.isfinite(b["est"]):
                    ax.axvline(b["est"], color=r_colour_hex(_SPIDE_COLS["grey"]), lw=0.4 * _PT)
                ax.axvline(0, color="black", ls=":", lw=0.3 * _PT)
                m = gm[(gm["gene_lab"] == gl) & (gm["condition"].astype(object) == c)]
                if len(m):
                    ax.axvline(float(m["mean"].iloc[0]), color=col[c], lw=0.8 * _PT)
                s = d[(d["gene_lab"] == gl) & (d["condition"].astype(object) == c)]
                y = np.array([ypos[p] for p in s["patient"].astype(object)], dtype=float)
                ax.hlines(y, s["slope"] - z * s["se"], s["slope"] + z * s["se"], colors=col[c],
                          lw=0.4 * _PT)  # fmt: skip
                ax.scatter(s["slope"], y, s=[_point_area(2.8 * math.sqrt(w / maxw)) for w in s["weight"]],
                           c=col[c], linewidths=0, zorder=3)  # fmt: skip
                ax.set_ylim(-0.6, len(pats) - 0.4)
                ax.set_yticks(range(len(pats)))
                if ci_ == 0:
                    ax.set_yticklabels(pats, fontsize=11 * 1.1 * 0.65)
                else:
                    ax.tick_params(labelleft=False)
                ax.tick_params(labelbottom=ri == len(rows) - 1)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))  # breaks_extended(n = 3)
                if ri == 0:
                    _strip(ax, _strip_text(res, g, index, niche), fontsize=11 * 1.1 * 0.8 * 0.85)
                if ci_ == ncol - 1:
                    ax.text(1.03, 0.5, _break_label(c), transform=ax.transAxes, rotation=270,
                            ha="left", va="center", fontsize=11 * 1.1 * 0.8 * 0.95,
                            bbox={"boxstyle": "square,pad=0.3", "fc": "none", "ec": "black",
                                  "lw": 0.5})  # fmt: skip
        fig.supxlabel(f"slope on {niche} density (log1p)", x=(left + ncol * pw / 2) / W,
                      fontsize=11 * 1.1 * 1.1)  # fmt: skip
        brk = _size_breaks(maxw)
        grey = r_colour_hex(_SPIDE_COLS["grey"])
        size_h = [
            Line2D([], [], marker="o", ls="none", color=grey, ms=2.8 * math.sqrt(b_ / maxw) * _PT)
            for b_ in brk
        ]
        groups = [
            ("weight", size_h, [f"{100 * b_:g}%" for b_ in brk]),
            (None, [Patch(fc=r_colour_hex(_SPIDE_COLS["faint"]), ec="none")],
             [f"pooled estimate ({100 * ci:g}% CI)"]),
        ]  # fmt: skip
        _stack_legends(fig, groups, 1 - right / W + 0.04, y_top=0.95)
        if res.condition:
            fig.text(1 - 0.01, 0.01, f"Δ: slope difference, {res.contrast}", ha="right",
                     va="bottom", fontsize=11 * 1.1 * 0.8)  # fmt: skip
    return fig


def _break_label(label: str) -> str:
    """A label longer than 9 characters broken onto two lines, at its first hyphen or, failing
    that, the space nearest its middle (``"Non-\\nresponder"``; R ``.breakLabel``)."""
    label = str(label)
    if len(label) <= 9:
        return label
    if "-" in label:
        return label.replace("-", "-\n", 1)
    sp = [i for i, ch in enumerate(label) if ch == " "]
    if not sp:
        return label
    k = sp[int(np.argmin([abs(i + 1 - len(label) / 2) for i in sp]))]
    return label[:k] + "\n" + label[k + 1 :]


# --- plotPatientEffects ----------------------------------------------------------------------


def _patient_effect_matrix(xi: dict, niche: str | None, ntop: int) -> pd.DataFrame:
    """Genes x patients, gene-centred: the intercepts, or the slopes on one niche scaled by each
    gene's between-patient spread (R ``.patientEffectMatrix``).

    A gene missing in most patients (one the fit could not fit) drops out first, then a patient
    missing for more than a tenth of the remaining genes, then any gene not finite in every
    remaining patient; the ``ntop`` most expressed genes enter.
    """
    genes = [str(g) for g in xi["genes"]]
    pats = [str(p) for p in xi["patients"]]
    if niche is None:
        M = xi["intercept"].reindex(index=genes, columns=pats).to_numpy(dtype=float).copy()
        # a patient with no counts of a gene has an intercept held only by the 1e-3 ridge, far
        # below the other patients': it is missing, not a patient effect
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a gene missing in every patient
            med = np.nanmedian(M, axis=1)
            spread = np.fmax(1.4826 * np.nanmedian(np.abs(M - med[:, None]), axis=1), 0.1)
            M[~np.isnan(M) & (M < (med - 5 * spread)[:, None])] = np.nan
    else:
        j = list(xi["niches"]).index(niche)
        v, tau = _slope_weights(xi)
        b = np.asarray(xi["beta"], dtype=float)[:, :, j]
        vj = v[:, :, j]
        with np.errstate(invalid="ignore"):
            sig = np.sqrt(tau[:, j] + np.nanmedian(vj, axis=1))
            M = b / sig[:, None]
    fin = np.isfinite(M)
    r_ok = fin.mean(axis=1) >= 0.5
    M, genes = M[r_ok], [g for g, k in zip(genes, r_ok, strict=True) if k]
    c_ok = np.isfinite(M).mean(axis=0) >= 0.9
    M, pats = M[:, c_ok], [p for p, k in zip(pats, c_ok, strict=True) if k]
    all_ok = (~np.isfinite(M)).sum(axis=1) == 0
    ok = {g for g, k in zip(genes, all_ok, strict=True) if k}
    me = xi["mean_expr"]
    me = me[me.notna()]
    # sort(mean_expr, decreasing = TRUE): ties keep their order
    ranked = me.index[np.argsort(-me.to_numpy(dtype=float), kind="stable")]
    keep = [g for g in dict.fromkeys(ranked) if g in ok][:ntop]
    M = pd.DataFrame(M, index=genes, columns=pats).loc[keep]
    return M - M.mean(axis=1).to_numpy()[:, None]


def _patient_covariates(fit, xi: dict, covariates=None) -> dict:
    """The covariates of :func:`plot_patient_effects` (R ``.patientCovariates``): every
    patient-level column that varies and is not an identifier, plus the derived ones.

    Returns ``cov`` (a DataFrame by patient; numeric columns as floats, the rest categorical)
    and ``group`` (a Series ``"patient"`` / ``"derived"`` by column name).
    """
    pats = [str(p) for p in xi["patients"]]
    pt = fit.patients.assign(patient=fit.patients["patient"].astype(str))
    pt = pt.set_index("patient", drop=False).reindex(pats)
    cand = [c for c in pt.columns if c not in ("patient", "ncells")]
    cols = {}
    for cn in cand:
        v = pt[cn]
        u = v.dropna().nunique()
        is_num = pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)
        if u > 1 and (is_num or u < len(pt)):
            if is_num:
                cols[cn] = v.to_numpy(dtype=float)
            else:
                ch = as_character_array(v)
                cols[cn] = pd.Categorical(ch, categories=r_sort_str(set(ch[pd.notna(ch)])))
    n_pt = len(cols)
    ncells = xi["ncells"].reindex(pats).to_numpy(dtype=float)
    cols["cells (log10)"] = np.log10(ncells)
    cols["depth (mean log)"] = xi["loglib_mean"].reindex(pats).to_numpy(dtype=float)
    nm = xi["niche_mean"].reindex(pats)
    for nc in nm.columns:
        cols[f"{nc} density"] = nm[nc].to_numpy(dtype=float)
    cov = pd.DataFrame(cols, index=pats)
    group = pd.Series(["patient"] * n_pt + ["derived"] * (len(cols) - n_pt), index=list(cols))
    if covariates is not None:
        covariates = [covariates] if isinstance(covariates, str) else [str(c) for c in covariates]
        miss = [c for c in covariates if c not in cov.columns]
        if miss:
            raise ValueError(
                f"covariate(s) not available: {', '.join(miss)} "
                f"(available: {', '.join(cov.columns)})"
            )
        cov, group = cov[covariates], group[covariates]
    return {"cov": cov, "group": group}


def _column(cov: pd.DataFrame, name: str):
    """One covariate as a ``Categorical`` or a float array."""
    v = cov[name].values
    return v if isinstance(v, pd.Categorical) else cov[name].to_numpy(dtype=float)


def _adj_r2(y: np.ndarray, v) -> float:
    """Adjusted R^2 of ``y`` on one covariate, floored at 0; NaN when it cannot be fitted (R
    ``.adjR2``: ``summary(lm(y ~ v))$adj.r.squared``)."""
    y = np.asarray(y, dtype=float)
    if isinstance(v, pd.Categorical):
        ok = np.isfinite(y) & ~pd.isna(v)
        v = pd.Categorical(np.asarray(v, dtype=object)[ok])  # droplevels
        v = v.remove_unused_categories()
        y = y[ok]
        n_lv = len(v.categories)
        if n_lv < 2 or len(y) < 4:
            return np.nan
        X = np.column_stack(
            [np.ones(len(y))] + [(v.codes == k).astype(float) for k in range(1, n_lv)]
        )
    else:
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(y) & ~np.isnan(v)
        y, v = y[ok], v[ok]
        if len(np.unique(v)) < 2 or len(y) < 4:
            return np.nan
        X = np.column_stack([np.ones(len(y)), v])
    n, p = X.shape
    if n - p <= 0:
        return np.nan
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ coef) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    if tss == 0:
        return np.nan
    adj = 1 - (rss / tss) * (n - 1) / (n - p)
    return max(0.0, adj) if np.isfinite(adj) else np.nan


def _patient_effects_data(
    x, index: str, niche=None, covariates=None, type: str = "association", colour_by=None,
    n_pcs: int = 5, ntop: int = 500,
) -> dict:  # fmt: skip
    """The data behind :func:`plot_patient_effects` (R ``plotPatientEffects``'s ggplot).

    Returns a dict with ``data`` (R's ``p$data``: for ``"association"`` ``pc, covariate, r2,
    group``; for ``"pca"`` ``patient, PC1, PC2`` and the covariate), ``type``, ``xlab``,
    ``ylab`` (pca), ``pc_levels`` / ``covariate_levels`` (association), ``colour_by`` and
    ``levels`` (the condition's levels when it is the colouring).
    """
    check_results(x, "plot_patient_effects()", fit_ok=True)
    type = _match_arg(type, ("association", "pca"), "type")
    is_res = isinstance(x, SpiDEResults)
    fit = x.fit if is_res else x
    condition = x.condition if is_res else fit.condition
    check_triplet(fit, index, niche)
    xi = fit.index[index]
    check_intercepts(xi, "plot_patient_effects()")
    if niche is not None:
        check_slopes_engine(fit, "plot_patient_effects(niche = )")
    M = _patient_effect_matrix(xi, niche, ntop)
    if M.shape[0] < 2 or M.shape[1] < 3:
        raise ValueError("too few genes or patients for principal components")
    # prcomp(t(M), center = TRUE, scale. = FALSE)
    Xc = M.to_numpy().T
    Xc = Xc - Xc.mean(axis=0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = pd.DataFrame(U * s, index=list(M.columns))
    sdev = s / math.sqrt(Xc.shape[0] - 1)
    ve = sdev**2 / (sdev**2).sum()
    n_pcs = max(1, min(int(n_pcs), scores.shape[1]))
    cv = _patient_covariates(fit, xi, covariates)
    cvm = cv["cov"].loc[list(M.columns)]
    what = "the patient intercepts" if niche is None else f"the patient slopes on {niche}"
    out = {"type": type, "ve": ve, "scores": scores, "cov": cvm}
    if type == "pca":
        if colour_by is None:
            colour_by = condition if condition and condition in cvm.columns else cvm.columns[0]
        if colour_by not in cvm.columns:
            raise ValueError(f"'colour_by' must be one of: {', '.join(cvm.columns)}")
        d = pd.DataFrame(
            {"patient": list(M.columns), "PC1": scores[0].to_numpy(), "PC2": scores[1].to_numpy()}
        )
        vals = _column(cvm, colour_by)
        levels = None
        if condition and colour_by == condition:
            levels = _condition_coding(fit.patients, condition).attrs["levels"]
            ch = as_character_array(pd.Series(vals))
            d[colour_by] = pd.Categorical(ch, categories=levels)
        else:
            d[colour_by] = vals
        out.update(data=d, colour_by=colour_by, levels=levels)
        out["xlab"] = f"PC1 ({100 * ve[0]:.0f}%)"
        out["ylab"] = f"PC2 ({100 * ve[1]:.0f}%)"
        return out
    pcl = [f"{k + 1}\n{100 * ve[k]:.0f}%" for k in range(n_pcs)]
    names = list(cvm.columns)
    rec = [
        (k, c, _adj_r2(scores[k].to_numpy(), _column(cvm, c))) for c in names for k in range(n_pcs)
    ]
    a = pd.DataFrame(rec, columns=["pc", "covariate", "r2"])
    a["pc"] = pd.Categorical([pcl[k] for k in a["pc"]], categories=pcl)
    a["group"] = pd.Categorical(
        [cv["group"][c] for c in a["covariate"]], categories=["patient", "derived"]
    )
    a["covariate"] = pd.Categorical(a["covariate"], categories=names[::-1])
    out.update(
        data=a, xlab=f"principal component of {what} (% variance)", pc_levels=pcl,
        covariate_levels=names[::-1],
    )  # fmt: skip
    return out


def plot_patient_effects(
    x, index: str, niche=None, covariates=None, type: str = "association", colour_by=None,
    n_pcs: int = 5, ntop: int = 500,
) -> Figure:  # fmt: skip
    """What the patient intercepts, or slopes, capture (R ``plotPatientEffects``).

    Every index type's shared fit gives each patient an intercept per gene
    (:func:`~spidepy.patient_intercepts`): they absorb every difference between patients (the
    condition's main effect, composition, batch), which is what makes the niche slopes
    within-patient slopes. This relates the principal components of the patients' gene-centred
    intercepts to patient covariates: every patient-level ``obs`` column, plus the patients'
    index cells (log10), mean log library size, and mean log1p density of each tested niche
    type. Each tile is the adjusted R^2 of a component on a covariate.

    Structure in the intercepts is harmless to the tests: it is absorbed by design. With
    ``niche``, the same analysis runs on the patients' slopes on that niche (each gene's slopes
    scaled by its between-patient spread, as in :func:`~spidepy.spi_gsea`): a slide or batch that
    organises the slopes can reach the condition test, and belongs in ``strata``.

    Parameters
    ----------
    x : SpiDEResults or SpiDEFit
    index : str
        The index cell type.
    niche : str, optional
        ``None`` (the intercepts) or a niche type (its slopes; slopes engine).
    covariates : sequence of str, optional
        ``None`` (all) or the covariates to show, by the names in the plot.
    type : {"association", "pca"}
        The R^2 tiles, or the first two components coloured by ``colour_by``.
    colour_by : str, optional
        For ``type="pca"``: a covariate (default the condition, else the first).
    n_pcs : int
        The number of components shown.
    ntop : int
        The number of genes, most expressed first, whose values enter the components (a gene
        with any non-finite value is left out).

    Returns
    -------
    matplotlib.figure.Figure
    """
    D = _patient_effects_data(x, index, niche, covariates, type, colour_by, n_pcs, ntop)
    d = D["data"]
    if D["type"] == "pca":
        return _pca_figure(D, x)
    pcs, covs = D["pc_levels"], D["covariate_levels"]
    r2 = d.pivot(index="covariate", columns="pc", values="r2").reindex(index=covs, columns=pcs)
    grp = d.drop_duplicates("covariate").set_index("covariate")["group"].astype(str)
    kinds = [g for g in ("patient", "derived") if (grp == g).any()]
    heights = [(grp == g).sum() for g in kinds]
    cmap = _magnitude_cmap(256).with_extremes(bad=r_colour_hex(_SPIDE_COLS["faint"]))
    W = len(pcs) * 0.9 + 3.6
    H = len(covs) * 0.38 + 1.9
    with style():
        fig = Figure(figsize=(W, H))
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(len(kinds), 1, height_ratios=heights, left=2.0 / W, right=1 - 1.6 / W,
                              bottom=0.95 / H, top=1 - 0.3 / H, hspace=0.12)  # fmt: skip
        im = None
        for gi, kd in enumerate(kinds):
            ax = fig.add_subplot(gs[gi, 0])
            names = [c for c in covs if grp[c] == kd]  # bottom to top
            sub = r2.loc[names].to_numpy(dtype=float)
            im = ax.pcolormesh(np.ma.masked_invalid(sub), cmap=cmap, vmin=0, vmax=1,
                               edgecolors="white", linewidth=0.6)  # fmt: skip
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub[i, j]
                    if np.isfinite(v) and v >= 0.05:
                        ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                                color="white" if v > 0.45 else "black", fontsize=11 * 1.1 * 0.8)  # fmt: skip
            ax.set_xlim(0, len(pcs))
            ax.set_ylim(0, len(names))
            ax.set_yticks(np.arange(len(names)) + 0.5, names)
            ax.set_xticks(
                np.arange(len(pcs)) + 0.5, pcs if gi == len(kinds) - 1 else [""] * len(pcs)
            )
            ax.tick_params(length=0)
            ax.text(1.03, 0.5, kd, transform=ax.transAxes, rotation=270, ha="left", va="center",
                    fontsize=11 * 1.1 * 0.8,
                    bbox={"boxstyle": "square,pad=0.3", "fc": "none", "ec": "black", "lw": 0.5})  # fmt: skip
        fig.supxlabel(D["xlab"], x=(2.0 + (W - 3.6) / 2) / W, fontsize=11 * 1.1 * 1.1)
        cax = fig.add_axes([1 - 1.15 / W, 0.35, 0.18 / W, 0.35])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(0)
        cb.ax.set_title("adjusted $R^2$", fontsize=11 * 1.1, fontstyle="italic", loc="left")
    return fig


def _pca_figure(D: dict, x) -> Figure:
    """The PC1 / PC2 scatter of :func:`plot_patient_effects`, coloured by one covariate."""
    d, cb_ = D["data"], D["colour_by"]
    v = d[cb_]
    with style():
        fig = Figure(figsize=(6.4, 4.6), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        if D["levels"] is not None:
            col = _condition_colours(D["levels"])
            for lvl in D["levels"]:
                m = (v.astype(object) == lvl).to_numpy()
                ax.scatter(
                    d["PC1"][m],
                    d["PC2"][m],
                    s=_point_area(2.2),
                    c=col[lvl],
                    linewidths=0,
                    label=lvl,
                )
            _legend_cb(ax, cb_)
        elif pd.api.types.is_numeric_dtype(v) and not isinstance(v.dtype, pd.CategoricalDtype):
            sc = ax.scatter(
                d["PC1"],
                d["PC2"],
                c=v,
                s=_point_area(2.2),
                cmap=_magnitude_cmap(256),
                linewidths=0,
            )
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(cb_, fontstyle="italic")
            cb.outline.set_linewidth(0)
        else:
            cats = (
                list(v.cat.categories)
                if isinstance(v.dtype, pd.CategoricalDtype)
                else r_sort_str(set(v.dropna().astype(str)))
            )
            cols = spide_colours("nuisance", len(cats))
            for c, k in zip(cats, cols, strict=True):
                m = (v.astype(object) == c).to_numpy()
                ax.scatter(d["PC1"][m], d["PC2"][m], s=_point_area(2.2), c=k, linewidths=0, label=c)
            _legend_cb(ax, cb_)
        ax.set_xlabel(D["xlab"])
        ax.set_ylabel(D["ylab"])
    return fig


def _legend_cb(ax, title: str) -> None:
    leg = ax.legend(title=title, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    leg.get_title().set_fontstyle("italic")
