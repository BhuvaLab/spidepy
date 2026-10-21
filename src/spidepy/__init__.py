"""spidepy: context-specific, neighbourhood-dependent differential expression (port of spiDE)."""

__version__ = "0.1.0.dev0"

from ._classes import SpiDEFit, SpiDEResults  # noqa: E402
from .abundance import test_niche_abundance  # noqa: E402
from .accessors import patient_intercepts, patient_slopes, results  # noqa: E402
from .fit import fit_spide  # noqa: E402
from .gsea import spi_gsea  # noqa: E402
from .niches import build_niches, merge_niches  # noqa: E402
from .test import spide, test_spide  # noqa: E402

__all__ = [
    "SpiDEFit",
    "SpiDEResults",
    "build_niches",
    "fit_spide",
    "merge_niches",
    "patient_intercepts",
    "patient_slopes",
    "results",
    "spi_gsea",
    "spide",
    "test_niche_abundance",
    "test_spide",
]
