"""Plots of spiDE results (matplotlib). Each plot returns a :class:`matplotlib.figure.Figure`."""

from ._niche import plot_niche_map, plot_niche_response
from ._theme import spide_colours, style

__all__ = ["plot_niche_map", "plot_niche_response", "spide_colours", "style"]
