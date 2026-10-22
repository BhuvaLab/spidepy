"""Accessors of a fit: each patient's niche slopes and intercepts (`R/accessors.R`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._checks import check_intercepts, check_results
from ._classes import SpiDEFit, SpiDEResults


def _as_fit(obj) -> SpiDEFit:
    """The fit of a `SpiDEFit` or a `SpiDEResults`."""
    if isinstance(obj, SpiDEResults):
        return obj.fit
    if not isinstance(obj, SpiDEFit):
        raise TypeError("'obj' should be a SpiDEFit or a SpiDEResults")
    return obj


def _as_list(x) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else [str(v) for v in x]


def _pick(labels, wanted) -> np.ndarray:
    """Positions of `labels` that are in `wanted` (R `which(x %in% wanted)`), all if None."""
    if wanted is None:
        return np.arange(len(labels))
    w = set(wanted)
    return np.array([i for i, x in enumerate(labels) if x in w], dtype=int)


def _with_patients(
    d: pd.DataFrame, fit: SpiDEFit, keys: list[str], order: list[str]
) -> pd.DataFrame:
    """Left-join the patient-level columns recorded by the fit, then order rows by `keys`.

    R: `merge(d, pt[, setdiff(colnames(pt), "ncells")], by = "patient", all.x = TRUE)` and
    `d[order(order), c(keys, rest)]`. Strings compare in code-point order.
    """
    pt = fit.patients.drop(columns="ncells").set_index("patient", drop=False)
    extra = pt.drop(columns="patient").reindex(d["patient"].to_numpy()).reset_index(drop=True)
    d = pd.concat([d.reset_index(drop=True), extra], axis=1)
    d = d.sort_values(order, kind="stable")
    rest = [c for c in d.columns if c not in keys]
    return d[keys + rest].reset_index(drop=True)


def patient_slopes(obj, gene=None, index=None, niche=None) -> pd.DataFrame:
    """Each patient's niche slopes in long form (R `patientSlopes`).

    The slopes engine estimates every patient's own niche slopes (a one-step negative binomial
    estimate from the shared fit); this returns them with their variances, e.g. to plot how a
    gene's slope varies between patients and conditions.

    Parameters
    ----------
    obj : SpiDEFit or SpiDEResults
        A fit (or the results of one) from the slopes engine.
    gene, index, niche : str or sequence of str, optional
        Subsets.

    Returns
    -------
    pandas.DataFrame
        `gene, index, niche, patient, slope, var_spatial` (the within-patient spatial
        sandwich variance), `var_model, ncells` plus the patient-level columns recorded by
        `fit_spide`; ordered by index, gene, niche, patient.
    """
    fit = _as_fit(obj)
    if fit.engine != "slopes":
        raise ValueError(
            'per-patient slopes are estimated by the slopes engine (fit_spide(engine = "slopes"))'
        )
    gene, index, niche = _as_list(gene), _as_list(index), _as_list(niche)
    out = []
    for k in [k for k in (list(fit.index) if index is None else index) if k in fit.index]:
        x = fit.index[k]
        gi, ni = _pick(x["genes"], gene), _pick(x["niches"], niche)
        si = np.arange(len(x["patients"]))
        if not len(gi) or not len(ni):
            continue
        # expand.grid(g, s, j): gene varies fastest, then patient, then niche
        g = np.tile(gi, len(si) * len(ni))
        s = np.tile(np.repeat(si, len(gi)), len(ni))
        j = np.repeat(ni, len(gi) * len(si))
        genes, pats, niches = map(np.asarray, (x["genes"], x["patients"], x["niches"]))
        out.append(
            pd.DataFrame(
                {
                    "gene": genes[g],
                    "index": k,
                    "niche": niches[j],
                    "patient": pats[s],
                    "slope": x["beta"][g, s, j],
                    "var_spatial": x["v_tile"][g, s, j],
                    "var_model": x["v_model"][g, s, j],
                    "ncells": np.asarray(x["ncells"])[s],
                }
            )
        )
    if not out:
        return pd.DataFrame()
    keys = ["gene", "index", "niche", "patient"]
    return _with_patients(
        pd.concat(out, ignore_index=True), fit, keys, ["index", "gene", "niche", "patient"]
    )


def patient_intercepts(obj, gene=None, index=None) -> pd.DataFrame:
    """Each patient's intercept per gene and index type, in long form (R `patientIntercepts`).

    Every index type's shared fit has an intercept per patient and gene: the patient's
    expression level of the gene in that type, at zero niche density and a reference depth (the
    index type's mean log library size under `depth = "loglib"` or `"nonlinear"`, each
    patient's own mean under `"spatial_spline"`). The intercepts absorb every difference
    between patients, which is why the niche slopes are within-patient slopes.

    Parameters
    ----------
    obj : SpiDEFit or SpiDEResults
    gene, index : str or sequence of str, optional
        Subsets.

    Returns
    -------
    pandas.DataFrame
        `gene, index, patient, intercept` (log scale), `ncells` (the patient's cells of
        the index type) plus the patient-level columns recorded by
        `fit_spide`; ordered by index, gene, patient.
    """
    fit = _as_fit(obj)
    gene, index = _as_list(gene), _as_list(index)
    out = []
    for k in [k for k in (list(fit.index) if index is None else index) if k in fit.index]:
        x = fit.index[k]
        check_intercepts(x, "patient_intercepts()")
        gi = _pick(x["genes"], gene)
        if not len(gi):
            continue
        si = np.arange(len(x["patients"]))
        g, s = np.tile(gi, len(si)), np.repeat(si, len(gi))
        genes, pats = np.asarray(x["genes"]), np.asarray(x["patients"])
        out.append(
            pd.DataFrame(
                {
                    "gene": genes[g],
                    "index": k,
                    "patient": pats[s],
                    "intercept": np.asarray(x["intercept"])[g, s],
                    "ncells": np.asarray(x["ncells"])[s],
                }
            )
        )
    if not out:
        return pd.DataFrame()
    return _with_patients(
        pd.concat(out, ignore_index=True),
        fit,
        ["gene", "index", "patient"],
        ["index", "gene", "patient"],
    )


def results(res: SpiDEResults, test: str | None = None, fdr: float | None = None) -> pd.DataFrame:
    """Extract spiDE's test results (R `results`); see `SpiDEResults.results`."""
    check_results(res, "results()")
    return res.results(test=test, fdr=fdr)
