"""Argument checkers: fail early with informative messages so the user-facing functions can
assume well-formed inputs. One function per checker of R `R/checkers.R`.

Messages are R's, with the R object names mapped to the Python ones (`spe` -> `adata`,
`colData(spe)` -> `adata.obs`, `buildNiches()` -> `build_niches()`). Every checker
returns `True` on success (R `invisible(TRUE)`). Fit and result objects are inspected by
attribute (`fit.engine`, `fit.index`, `fit.params`, `fit.sigma`) and result classes by
name, so this module does not import the classes defined in later tasks.
"""

from __future__ import annotations

import numbers
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ._io import get_counts, get_niche, get_obs, niche_key
from ._rcompat import r_sort_str


def _is_str_seq(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index)) and all(
        isinstance(s, str) for s in x
    )


def _as_str_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str) or not hasattr(x, "__iter__"):
        return [str(x)]
    return [str(s) for s in x]


def _has_class(x, name: str) -> bool:
    return any(c.__name__ == name for c in type(x).__mro__)


def check_adata(
    adata, assay: str = "counts", cell_type: str = "cell_type", sample_id: str = "sample_id"
) -> bool:
    """R `checkSPE`: an AnnData with the assay and the cell type / sample id columns.

    As R's `assayNames` the assay must be a layer; the name `"counts"` additionally
    refers to `X` when no such layer exists (`get_counts` then reads `X`).
    """
    if not isinstance(adata, AnnData):
        raise TypeError("'adata' should be an AnnData object")
    if assay not in adata.layers and assay != "counts":
        raise ValueError(f"assay '{assay}' not found in 'adata'")
    if cell_type not in adata.obs.columns:
        raise ValueError(f"cell type column '{cell_type}' not found in adata.obs")
    if sample_id not in adata.obs.columns:
        raise ValueError(f"sample id column '{sample_id}' not found in adata.obs")
    return True


def check_counts(Y, integer_only: bool = False) -> bool:
    """R `checkCounts`: counts (genes x cells) are non-negative, optionally integer-valued.

    Only a minimum is computed (sparse input is never densified). The integer rule looks at
    the first 20 genes (rows) only, ignoring non-finite values, tolerance `1e-8`.
    """
    if sp.issparse(Y):
        vals = Y.data[~np.isnan(Y.data)]
        # a structural zero is a value too: it caps the minimum at 0
        has_zero = Y.nnz < Y.shape[0] * Y.shape[1]
        mn = min(vals.min() if vals.size else np.inf, 0.0 if has_zero else np.inf)
    else:
        Y = np.asarray(Y)
        ok = Y[~np.isnan(Y)] if Y.dtype.kind == "f" else Y
        mn = ok.min() if ok.size else np.inf
    if not np.isfinite(mn) or mn < 0:
        raise ValueError("counts should be non-negative")
    if integer_only:
        head = Y[:20]
        ss = np.asarray(head.data if sp.issparse(head) else head, dtype=float).ravel()
        ss = ss[np.isfinite(ss)]
        # a stored zero or a missing entry is an integer, so only stored values matter
        if ss.size and np.max(np.abs(ss - np.round(ss))) > 1e-8:
            raise ValueError(
                "fit_spide() requires integer counts, and this assay is not "
                "integer-valued: the negative binomial likelihood is undefined off "
                "the integers, so every gene's dispersion would collapse "
                'to its upper bound.\n  Supply raw counts (assay = "counts").'
            )
    return True


def check_condition(adata: AnnData, condition: str) -> bool:
    """R `checkCondition`: the condition column exists and has exactly two levels."""
    if condition not in adata.obs.columns:
        raise ValueError(f"condition column '{condition}' not found in adata.obs")
    vals = adata.obs[condition]
    n = len(pd.unique(vals[vals.notna()]))
    if n != 2:
        raise ValueError(f"condition '{condition}' should have exactly two levels, found {n}")
    return True


def check_covariates(adata: AnnData, covariates, finite_only: bool = False) -> bool:
    """R `checkCovariates`: covariates exist in `obs`; with `finite_only` also finite.

    The finiteness rule applies to numeric columns only. It is opt-in because only the GLM
    design breaks on a missing value (the design matrix would lose rows); a patient-level
    estimator drops such patients on purpose.
    """
    covariates = _as_str_list(covariates)
    if not covariates:
        return True
    missing = [c for c in dict.fromkeys(covariates) if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"covariate(s) not found in adata.obs: {', '.join(missing)}")
    if not finite_only:
        return True
    bad = []
    for cv in covariates:
        x = adata.obs[cv]
        if x.dtype.kind in "iuf" and not np.all(np.isfinite(x.to_numpy(dtype=float))):
            bad.append(cv)
    if bad:
        raise ValueError(
            f"covariate(s) with missing or non-finite values: {', '.join(bad)}. "
            "Every cell needs a finite value (log() of a zero-valued column is the usual "
            "cause -- use log1p, or drop the affected cells)."
        )
    return True


def _n_unique_by_sample(smp: np.ndarray, x: np.ndarray) -> dict[str, int]:
    """R `tapply(x, smp, function(v) length(unique(v[!is.na(v)])))` (NA samples dropped)."""
    df = pd.DataFrame({"s": smp, "x": x}).dropna(subset=["s"])
    n = df.groupby("s")["x"].nunique()  # nunique ignores NaN
    return {k: int(n[k]) for k in r_sort_str(n.index)}


def check_sample(
    adata: AnnData, condition: str | None = None, sample_id: str = "sample_id", covariates=()
) -> bool:
    """R `checkSample`: the condition (and no covariate) is constant within a sample.

    The condition must be a patient-level variable; a covariate constant within samples is
    confounded with the per-patient intercepts. Fewer than 3 samples is a warning.
    """
    if sample_id not in adata.obs.columns:
        raise ValueError(f"sample id column '{sample_id}' not found in adata.obs")
    smp = get_obs(adata, sample_id)
    # a condition-free (niche-only) design has no condition to be patient-level
    if condition is not None:
        n_lvl = _n_unique_by_sample(smp, get_obs(adata, condition))
        varies = [k for k, v in n_lvl.items() if v > 1]
        if varies:
            raise ValueError(
                f"condition '{condition}' varies within sample(s): {', '.join(varies)}. "
                f"spiDE needs a patient-level condition (constant within '{sample_id}')."
            )
    const = [
        cv
        for cv in _as_str_list(covariates)
        if all(v <= 1 for v in _n_unique_by_sample(smp, get_obs(adata, cv)).values())
    ]
    if const:
        raise ValueError(
            f"covariate(s) constant within sample: {', '.join(const)}. The per-patient "
            "intercepts already absorb all between-patient effects, so drop these "
            "patient-level covariates."
        )
    n_samples = len(pd.unique(pd.Series(smp, dtype=object)))  # R unique() counts NA once
    if n_samples < 3:
        warnings.warn(
            f"only {n_samples} sample(s); the between-patient tests need several patients "
            "per condition",
            UserWarning,
            stacklevel=2,
        )
    return True


def check_column(adata: AnnData, column, what: str) -> bool:
    """R `checkColumn`: an optional `obs` column named by an argument (`None` = unused)."""
    if column is None:
        return True
    if not isinstance(column, str) or column not in adata.obs.columns:
        shown = ", ".join(_as_str_list(column)) if not isinstance(column, str) else column
        raise ValueError(f"{what} column '{shown}' not found in adata.obs")
    return True


def check_fdr(fdr) -> bool:
    """R `checkFdr`: a single number in (0, 1]; 1 is the "show everything" threshold."""
    if (
        isinstance(fdr, bool)
        or not isinstance(fdr, numbers.Real)
        or np.isnan(fdr)
        or fdr <= 0
        or fdr > 1
    ):
        raise ValueError("'fdr' should be a single numeric value in (0, 1]")
    return True


def check_procedure(procedure: str, engine: str) -> bool:
    """R `checkProcedure`: the heterogeneity filters need the slopes engine's slopes."""
    if procedure in ("heterogeneity", "pooled_or_heterogeneity") and engine != "slopes":
        raise ValueError(
            f'procedure = "{procedure}" needs the slopes engine\'s per-patient slopes; '
            'use procedure = "filtered" or "all" with the sandwich engine'
        )
    return True


def check_niche(adata: AnnData, sigma, name: str = "Niche") -> bool:
    """R `checkNiche`: the niche matrix of every requested bandwidth has been built."""
    sigmas = [sigma] if isinstance(sigma, numbers.Real) else list(np.atleast_1d(sigma))
    missing = [k for k in (niche_key(name, s) for s in sigmas) if k not in adata.obsm]
    if missing:
        raise ValueError(
            f"niche matrix/matrices not found in adata.obsm: {', '.join(missing)}. "
            "Run build_niches() first."
        )
    return True


def check_intercepts(xi: dict, what: str) -> bool:
    """R `checkIntercepts`: the index type's fit kept the patient intercepts."""
    if xi.get("intercept") is None:
        raise ValueError(
            f"{what} needs the patient intercepts that fit_spide() keeps from spiDE 0.99.34 "
            "on: refit this object with the installed spidepy"
        )
    return True


def check_results(x, what: str, fit_ok: bool = False) -> bool:
    """R `checkResults`: a `SpiDEResults` (or, with `fit_ok`, a `SpiDEFit`).

    R's legacy-object guard (`.assertCurrent`) has no counterpart: there are no legacy
    spidepy objects.
    """
    if not (_has_class(x, "SpiDEResults") or (fit_ok and _has_class(x, "SpiDEFit"))):
        alt = ", or a SpiDEFit from fit_spide()" if fit_ok else ""
        raise TypeError(f"{what} needs a SpiDEResults from test_spide() or spide(){alt}")
    return True


def check_slopes_engine(fit, what: str) -> bool:
    """R `checkSlopesEngine`: per-patient slope views exist only for the slopes engine."""
    if fit.engine != "slopes":
        raise ValueError(
            f'{what} needs the slopes engine\'s per-patient slopes: fit with engine = "slopes"'
        )
    return True


def check_triplet(fit, index, niche=None, gene=None) -> bool:
    """R `checkTriplet`: an index type of the fit, one of its tested niches, genes it tested."""
    if not isinstance(index, str) or index not in fit.index:
        raise ValueError(
            f"'index' must be one of the fitted index types: {', '.join(map(str, fit.index))}"
        )
    xi = fit.index[index]
    if niche is not None and (not isinstance(niche, str) or niche not in list(xi["niches"])):
        raise ValueError(
            f"'niche' must be one of the niches tested in {index}: "
            f"{', '.join(map(str, xi['niches']))}"
        )
    if gene is not None:
        if not (isinstance(gene, str) or _is_str_seq(gene)) or not len(_as_str_list(gene)):
            raise ValueError("'gene' must be a character vector")
        have = set(map(str, xi["genes"]))
        miss = [g for g in dict.fromkeys(_as_str_list(gene)) if g not in have]
        if miss:
            raise ValueError(f"gene(s) not tested in {index}: {', '.join(miss)}")
    return True


def check_fit_adata(
    adata: AnnData, fit, index, niches=None, assay: str = "counts", name: str = "Niche"
) -> bool:
    """R `checkFitSPE`: `adata` is the object the fit was made from.

    It must hold the same index cells per patient (the fit's usable rule: cells with counts
    when depth enters, every cell under `depth = "none"`) and every niche column named.
    """
    hint = "is it the object the fit was made from?"
    p = fit.params
    ct = get_obs(adata, p["cell_type"])
    smp = get_obs(adata, p["sample_id"])
    if not len(ct) or not len(smp):
        raise ValueError(f"'adata' lacks the fit's cell type or sample columns: {hint}")
    if p["depth"] != "none":
        usable = np.asarray(get_counts(adata, assay).sum(axis=0)).ravel() > 0
    else:
        usable = np.ones(len(ct), dtype=bool)
    for k in _as_str_list(index):
        xi = fit.index[k]
        patients = [str(s) for s in xi["patients"]]
        sel = pd.Series(smp[pd.notna(ct) & (ct == k) & usable], dtype=object)
        # table(factor(smp[...], levels = patients)): other samples are dropped
        n = sel.value_counts().reindex(patients, fill_value=0).to_numpy()
        nc = xi.get("ncells")
        if nc is not None:
            want = (
                [nc[q] for q in patients]  # R: xi$ncells[xi$patients] (named by patient)
                if isinstance(nc, (dict, pd.Series))
                else nc
            )
            same = np.array_equal(n, np.asarray(want, dtype=int))
        else:
            same = bool(np.all(n > 0))
        if not same:
            raise ValueError(f"'adata' does not hold the fit's {k} cells: {hint}")
    niches = _as_str_list(niches)
    if niches:
        rd = niche_key(name, fit.sigma)
        have = list(map(str, get_niche(adata, rd).columns)) if rd in adata.obsm else []
        miss = [x for x in dict.fromkeys(niches) if x not in have]
        if miss:
            raise ValueError(
                f"niche(s) {', '.join(miss)} are not columns of {rd} in 'adata': {hint}"
            )
    return True


def check_genesets(genesets) -> bool:
    """R `checkGenesets`: a non-empty dict of gene lists (str sequences) keyed by name."""
    if (
        not isinstance(genesets, dict)
        or not genesets
        or not all(isinstance(k, str) for k in genesets)
        or not all(isinstance(v, str) or _is_str_seq(v) for v in genesets.values())
    ):
        raise ValueError("'genesets' must be a named list of character vectors with unique names")
    return True
