"""The two result classes of the per-patient engines (``R/AllClasses.R``).

``SpiDEFit`` holds, per index cell type, what :func:`spidepy.fit_spide` estimated;
``SpiDEResults`` holds the tidy table of the tests and the fit it came from (its methods
arrive with the test functions). R's ``$`` slot accessor is plain attribute access here, and
R's legacy-object guard (``.assertCurrent``) has no counterpart: there are no legacy objects.

Contents of ``SpiDEFit.index[k]`` (keys are R's list names):

- both engines: ``genes``, ``niches`` (tested), ``adjusted``, ``patients`` (lists of str),
  ``mean_expr`` (Series by gene), ``status`` (dict ``genes, not_fitted, psi_at_bound,
  runaway``), ``depth_r2`` (DataFrame patients x niches, or ``None``), ``intercept``
  (DataFrame genes x patients), ``niche_mean`` (DataFrame patients x niches), ``loglib_mean``
  (Series by patient), ``ncells`` (Series by patient, index cells).
- slopes engine adds ``beta``, ``v_tile``, ``v_model`` (ndarrays genes x patients x niches),
  ``pooled`` (DataFrame genes x niches), ``factor`` (DataFrame patients x niches) and ``psi``
  (Series by gene).
- sandwich engine adds ``coef`` (DataFrame ``test, gene, niche, estimate, se, df, se_model,
  psi, t, p``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ._rcompat import r_num_str


@dataclass
class SpiDEFit:
    """The fitted per-patient niche models of spiDE, returned by :func:`spidepy.fit_spide`.

    Attributes
    ----------
    engine : str
        ``"slopes"`` or ``"sandwich"``.
    sigma : float
        The niche bandwidth the fit used.
    condition : str or None
        The condition the fit was given (R: a character of length 0 or 1). The sandwich
        engine fits condition-specific slopes only if it was given one; the slopes engine only
        records it.
    patients : pandas.DataFrame
        One row per patient: ``patient``, ``ncells`` and every ``obs`` column constant within
        patients (so a condition can be named at test time).
    index : dict
        One entry per index cell type holding the engine's estimates (see the module docs).
    params : dict
        The settings the fit used.
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
        """R ``show``: engine, bandwidth, patients, condition and one line per index type."""
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
    """The results of spiDE's niche tests (fields only; the methods come with the tests).

    Attributes
    ----------
    table : pandas.DataFrame
        One row per (gene, index, niche, test).
    condition : str or None
        The condition tested.
    contrast : str
        E.g. ``"Responder - Non-responder"``.
    procedure : str
        The condition test's family.
    fdr : float
        The FDR level of the filter.
    pooled_df : str
        The slopes engine's pooled-test degrees of freedom rule (R slot ``pooled.df``).
    fit : SpiDEFit
        The fit that was tested.
    """

    table: pd.DataFrame
    condition: str | None
    contrast: str
    procedure: str
    fdr: float
    pooled_df: str
    fit: SpiDEFit
