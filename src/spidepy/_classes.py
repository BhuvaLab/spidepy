"""The two result classes of the per-patient engines (`R/AllClasses.R`).

`SpiDEFit` holds, per index cell type, what `spidepy.fit_spide` estimated;
`SpiDEResults` holds the tidy table of the tests and the fit it came from. R's `$` slot
accessor is plain attribute access here, and R's legacy-object guard (`.assertCurrent`) has no counterpart: there are no legacy objects.

Contents of `SpiDEFit.index[k]` (keys are R's list names):

- both engines: `genes`, `niches` (tested), `adjusted`, `patients` (lists of str),
  `mean_expr` (Series by gene), `status` (dict `genes, not_fitted, psi_at_bound,
  runaway`), `depth_r2` (DataFrame patients x niches, or `None`), `intercept`
  (DataFrame genes x patients), `niche_mean` (DataFrame patients x niches), `loglib_mean`
  (Series by patient), `ncells` (Series by patient, index cells).
- slopes engine adds `beta`, `v_tile`, `v_model` (ndarrays genes x patients x niches),
  `pooled` (DataFrame genes x niches), `factor` (DataFrame patients x niches) and `psi`
  (Series by gene).
- sandwich engine adds `coef` (DataFrame `test, gene, niche, estimate, se, df, se_model,
  psi, t, p`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._rcompat import r_num_str


@dataclass
class SpiDEFit:
    """The fitted per-patient niche models of spiDE, returned by `spidepy.fit_spide`.

    Attributes
    ----------
    engine : str
        `"slopes"` or `"sandwich"`.
    sigma : float
        The niche bandwidth the fit used.
    condition : str or None
        The condition the fit was given (R: a character of length 0 or 1). The sandwich
        engine fits condition-specific slopes only if it was given one; the slopes engine only
        records it.
    patients : pandas.DataFrame
        One row per patient: `patient`, `ncells` and every `obs` column constant within
        patients (so a condition can be named at test time).
    index : dict
        One entry per index cell type holding the engine's estimates (see the module docs).
    params : dict
        The settings the fit used.

    See Also
    --------
    fit_spide : Creates a `SpiDEFit`.
    SpiDEResults : The tests of a fit.

    Examples
    --------
    >>> from spidepy import build_niches, fit_spide
    >>> from spidepy.datasets import toy_spide
    >>> fit = fit_spide(build_niches(toy_spide(), sigma=30), verbose=False)
    >>> print(fit)
    SpiDEFit (slopes engine), bandwidth 30
      16 patients; condition: none
      index A: 20 genes x 2 niches (B, C)
      index B: 20 genes x 2 niches (A, C)
      index C: 20 genes x 2 niches (A, B)
    >>> fit.patients.shape[0]
    16
    """

    engine: str
    sigma: float
    condition: str | None
    patients: pd.DataFrame
    index: dict[str, dict]
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.engine not in ("slopes", "sandwich"):
            raise ValueError('\'engine\' must be "slopes" or "sandwich"')

    def __repr__(self) -> str:
        """R `show`: engine, bandwidth, patients, condition and one line per index type."""
        lines = [
            f"SpiDEFit ({self.engine} engine), bandwidth {r_num_str(self.sigma)}",
            f"  {len(self.patients)} patients; condition: "
            f"{self.condition if self.condition else 'none'}",
        ]
        for k, x in self.index.items():
            lines.append(
                f"  index {k}: {len(x['genes'])} genes x {len(x['niches'])} niches "
                f"({', '.join(x['niches'])})"
            )
        return "\n".join(lines)


@dataclass
class SpiDEResults:
    """The results of spiDE's niche tests, returned by `spidepy.test_spide` and
    `spidepy.spide`; read it with `results`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per (gene, index, niche, test): `gene, index, niche, test, estimate, se, t,
        df, p, n_patients, p.heterogeneity` (R `p.heterogeneity`; NA for the sandwich
        engine), `q` and `in_family`.
    condition : str or None
        The condition tested.
    contrast : str
        E.g. `"Responder - Non-responder"` (second level minus first; `""` without a
        condition).
    procedure : str
        The condition test's family.
    fdr : float
        The FDR level of the filter.
    fit : SpiDEFit
        The fit that was tested.

    See Also
    --------
    test_spide : Creates a `SpiDEResults`.
    results : Function form of `SpiDEResults.results`.

    Examples
    --------
    >>> from spidepy import build_niches, fit_spide, test_spide
    >>> from spidepy.datasets import toy_spide
    >>> fit = fit_spide(build_niches(toy_spide(), sigma=30), condition="condition", verbose=False)
    >>> res = test_spide(fit)
    >>> print(res)
    SpiDEResults (slopes engine)
      pooled across patients: 120 tests, 2 at FDR 0.05
      condition-specific (Responder - Non-responder; pooled_or_heterogeneity family): 2 tests, 1 at FDR 0.05
    >>> res.table.shape[1]
    13
    """

    table: pd.DataFrame
    condition: str | None
    contrast: str
    procedure: str
    fdr: float
    fit: SpiDEFit

    def results(self, test: str | None = None, fdr: float | None = None) -> pd.DataFrame:
        """The results table (R `results`), ordered by p-value (missing last).

        Parameters
        ----------
        test : {"condition", "pooled", "both"}, optional
            Which test's rows (default: the condition test if the table has one, else pooled).
        fdr : float, optional
            Keep only the rows in a BH family with `q <= fdr`.

        Returns
        -------
        pandas.DataFrame
            The `SpiDEResults.table` rows of the chosen test, ordered by `p` (missing last).

        See Also
        --------
        results : Function form.

        Examples
        --------
        >>> from spidepy import build_niches, fit_spide, test_spide
        >>> from spidepy.datasets import toy_spide
        >>> fit = fit_spide(build_niches(toy_spide(), sigma=30), condition="condition", verbose=False)
        >>> res = test_spide(fit)
        >>> res.results(test="pooled").shape
        (120, 13)
        >>> res.results(fdr=0.05)[["gene", "index", "niche"]].values.tolist()
        [['G1', 'A', 'B']]
        """
        tb = self.table
        if test is None:
            test = "condition" if (tb["test"] == "condition").any() else "pooled"
        if test not in ("condition", "pooled", "both"):
            raise ValueError("'test' should be one of 'condition', 'pooled', 'both'")
        if test != "both":
            tb = tb[tb["test"] == test]
        if fdr is not None:
            tb = tb[tb["q"].notna() & (tb["q"] <= fdr)]
        # order(p, na.last = TRUE): stable, missing last
        tb = tb.iloc[np.argsort(tb["p"].to_numpy(dtype=float), kind="stable")]
        return tb.reset_index(drop=True)

    def __repr__(self) -> str:
        """R `show`: one line per test, its size and the discoveries at the FDR."""
        tb = self.table
        lines = [f"SpiDEResults ({self.fit.engine} engine)"]
        for tt in pd.unique(tb["test"]):
            q = tb.loc[(tb["test"] == tt) & tb["q"].notna(), "q"]
            if tt == "condition":
                lab = f"condition-specific ({self.contrast}; {self.procedure} family)"
            else:
                lab = "pooled across patients"
            lines.append(
                f"  {lab}: {len(q)} tests, {int((q <= self.fdr).sum())} at FDR "
                f"{r_num_str(self.fdr)}"
            )
        return "\n".join(lines)
