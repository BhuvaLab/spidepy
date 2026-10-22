"""Plots of spiDE results (matplotlib). Each plot returns a `matplotlib.figure.Figure`."""

from ._niche import plot_niche_map, plot_niche_response
from ._patients import plot_patient_effects, plot_patient_slopes
from ._results import (
    plot_call_map,
    plot_p_values,
    plot_spillover,
    plot_top_triplets,
    plot_triplet_heatmap,
)
from ._theme import spide_colours, style

__all__ = [
    "plot_call_map",
    "plot_niche_map",
    "plot_niche_response",
    "plot_patient_effects",
    "plot_p_values",
    "plot_patient_slopes",
    "plot_spillover",
    "plot_top_triplets",
    "plot_triplet_heatmap",
    "spide_colours",
    "style",
]
