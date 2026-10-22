"""`test_spide` and `spide`: the tests across patients and the end-to-end run
(`R/testSpiDE.R`, `R/spiDE.R`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._checks import check_fdr, check_procedure
from ._classes import SpiDEFit, SpiDEResults
from ._io import niche_key
from ._test_slopes import _bh_families, _slopes_tests
from .fit import _condition_coding, _match_arg, _strata_by_patient, fit_spide
from .niches import build_niches

_PROCEDURES = ("pooled_or_heterogeneity", "heterogeneity", "filtered", "all")
_SANDWICH_COLS = ["gene", "niche", "test", "estimate", "se", "t", "df", "p"]


def test_spide(
    fit: SpiDEFit,
    condition: str | None = None,
    procedure: str | None = None,
    fdr: float = 0.05,
    strata: str | None = None,
    n_jobs: int = 1,
) -> SpiDEResults:
    """Test niche-dependent differential expression (R `testSpiDE`).

    Always runs the pooled test: for every (gene, index cell type, niche cell type), whether
    the gene's expression in the index type changes with the density of the niche type, within
    patients, consistently across patients. If a `condition` is given (here, or recorded by
    `fit_spide`), it also runs the condition-specific test: whether that slope
    differs between the two conditions, a between-patient contrast of within-patient slopes.

    The condition test is carried out within a family of triplets chosen without looking at the
    condition, so the choice is exact under any relabelling of patients:

    - `"pooled_or_heterogeneity"` (the slopes engine's default): the triplets that pass either
      filter below.
    - `"heterogeneity"`: the triplets whose slopes vary between patients more than their
      sampling variances allow (Cochran's Q; Benjamini-Hochberg over every triplet at `fdr`).
    - `"filtered"` (the sandwich engine's default): the triplets whose pooled test passes at
      `fdr`.
    - `"all"`: every triplet.

    The heterogeneity filters need each patient's slopes, so they are the slopes engine's only.

    Parameters
    ----------
    fit : SpiDEFit
        A fit from `fit_spide`.
    condition : str, optional
        Default: the condition recorded by the fit, if any. Otherwise a two-level, patient-level
        column of `fit.patients`. With the sandwich engine it must be the condition the model
        was fitted with. The contrast is second level minus first.
    procedure : {"pooled_or_heterogeneity", "heterogeneity", "filtered", "all"}, optional
        The condition test's family; default by engine.
    fdr : float, default 0.05
        The FDR level of the filter.
    strata : str, optional
        Default: the fit's. A patient-level column adjusted for in the slopes engine's
        condition test (e.g. slide, where the condition is confounded with it).
    n_jobs : int, default 1
        The slopes engine's condition test is run over genes in blocks on this many workers.

    Returns
    -------
    SpiDEResults
        Read it with `SpiDEResults.results`.

    See Also
    --------
    fit_spide : Produces the fit that is tested here.
    spide : Build, fit and test in one call.
    results : Extract the results table.

    Examples
    --------
    >>> from spidepy import build_niches, fit_spide, test_spide
    >>> from spidepy.datasets import toy_spide
    >>> adata = build_niches(toy_spide(), sigma=30)
    >>> fit = fit_spide(adata, condition="condition", verbose=False)
    >>> res = test_spide(fit)
    >>> res.contrast
    'Responder - Non-responder'
    >>> res.procedure
    'pooled_or_heterogeneity'
    >>> sorted(res.results(test="both")["test"].unique())
    ['condition', 'pooled']
    """
    if not isinstance(fit, SpiDEFit):
        raise TypeError("'fit' should be a SpiDEFit from fit_spide()")
    slopes = fit.engine == "slopes"
    if procedure is None:
        procedure = "pooled_or_heterogeneity" if slopes else "filtered"
    else:
        procedure = _match_arg(procedure, _PROCEDURES, "procedure")
    check_procedure(procedure, fit.engine)
    check_fdr(fdr)
    if condition is None and fit.condition:
        condition = fit.condition
    if not slopes and condition is not None and condition != fit.condition:
        raise ValueError(
            "the sandwich engine tests the condition it was fitted with; refit with "
            f'fit_spide(condition = "{condition}", engine = "sandwich")'
        )
    if strata is None:
        strata = fit.params.get("strata")
    pt = fit.patients
    trt_of = _condition_coding(pt, condition) if condition is not None else None
    contrast = " - ".join(reversed(trt_of.attrs["levels"])) if trt_of is not None else ""
    str_of = _strata_by_patient(pt, strata) if trt_of is not None and strata is not None else None
    tabs = []
    for k, x in fit.index.items():
        if slopes:
            # per-patient labels by patient NAME for this index type's surviving patients
            pats = list(x["patients"])
            trt = None if trt_of is None else trt_of.reindex(pats).to_numpy()
            st = None if str_of is None else str_of.reindex(pats).to_numpy()
            tk = _slopes_tests(x, trt=trt, strata=st, n_jobs=n_jobs)
        else:
            tk = x["coef"]
            if condition is None:
                tk = tk[tk["test"] == "pooled"]
            tk = tk[_SANDWICH_COLS].copy()
            tk["n_patients"] = float(len(x["patients"]))
            tk["p.heterogeneity"] = np.nan
        tk = tk.reset_index(drop=True)
        tk.insert(1, "index", k)
        tabs.append(tk)
    tab = pd.concat(tabs, ignore_index=True)
    tab = tab[
        ["gene", "index", "niche"] + [c for c in tab.columns if c not in ("gene", "index", "niche")]
    ]
    tab = _bh_families(tab, procedure=procedure, fdr=fdr)
    return SpiDEResults(
        table=tab,
        condition=condition,
        contrast=contrast,
        procedure=procedure,
        fdr=fdr,
        fit=fit,
    )


# a pytest run collecting test_*.py modules must not take this public function for a test
test_spide.__test__ = False


def spide(
    adata: AnnData,
    condition: str | None = None,
    sigma: float | None = None,
    engine: str = "slopes",
    procedure: str | None = None,
    fdr: float = 0.05,
    cell_type: str = "cell_type",
    sample_id: str = "sample_id",
    name: str = "Niche",
    n_jobs: int = 1,
    verbose: bool = True,
    **fit_kwargs,
) -> SpiDEResults:
    """Run spiDE end to end (R `spiDE`).

    Builds the niche covariates at bandwidth `sigma` (unless `adata` already carries them),
    fits the per-patient niche models (`fit_spide`) and tests them
    (`test_spide`): the pooled niche test always, and the condition-specific
    test when a `condition` is given.

    Parameters
    ----------
    adata : AnnData
        Cells x genes with raw counts. Niches built here are not left on `adata`.
    condition : str, optional
        The `obs` column of a two-level, patient-level condition.
    sigma : float, default None
        The niche bandwidth (a single value). Required: `None` raises an error.
    engine : {"slopes", "sandwich"}, default "slopes"
        See `fit_spide`.
    procedure : str, optional
        Passed to `test_spide`; defaults to the engine's default.
    fdr : float, default 0.05
        Passed to `test_spide`.
    cell_type : str, default "cell_type"
        The `obs` column of the cell type.
    sample_id : str, default "sample_id"
        The `obs` column of the patient.
    name : str, default "Niche"
        The niche key prefix: the niches are read from, or built into,
        `adata.obsm[name + sigma]`.
    n_jobs : int, default 1
        Number of workers.
    verbose : bool, default True
        Report progress on stderr.
    **fit_kwargs
        Further arguments to `fit_spide`.

    Returns
    -------
    SpiDEResults
        The tests of the fit.

    See Also
    --------
    build_niches : The first step.
    fit_spide : The second step.
    test_spide : The third step.

    Examples
    --------
    >>> from spidepy import spide
    >>> from spidepy.datasets import toy_spide
    >>> res = spide(toy_spide(), condition="condition", sigma=30, verbose=False)
    >>> res.fit.engine
    'slopes'
    >>> res.results().columns.tolist()[:4]
    ['gene', 'index', 'niche', 'test']
    """
    engine = _match_arg(engine, ("slopes", "sandwich"), "engine")
    if procedure is not None:
        procedure = _match_arg(procedure, _PROCEDURES, "procedure")
    if procedure is not None:
        check_procedure(procedure, engine)  # before the fit, not after it
    if sigma is None or np.ndim(sigma) != 0:
        raise ValueError("'sigma' must be a single bandwidth")
    built = False
    if niche_key(name, sigma) not in adata.obsm:
        build_niches(
            adata, sigma=sigma, cell_type=cell_type, sample_id=sample_id, name=name, n_jobs=n_jobs
        )
        built = True
    try:
        fit = fit_spide(
            adata,
            condition=condition,
            engine=engine,
            sigma=sigma,
            cell_type=cell_type,
            sample_id=sample_id,
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            **fit_kwargs,
        )
    finally:
        if built:
            del adata.obsm[niche_key(name, sigma)]
    return test_spide(
        fit,
        condition=condition,
        procedure=procedure,
        fdr=fdr,
        n_jobs=n_jobs,
    )


__all__ = ["spide", "test_spide"]
