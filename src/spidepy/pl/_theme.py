"""The spiDE plot theme and colours (``R/plot-theme.R``).

One colour per role, the same in every plot: amber / violet for the direction of a slope (and
the diverging t scale), teal / rufous for the first / second condition level, the scico "lapaz"
ramp for magnitudes and the Okabe-Ito colours for a nuisance factor. :func:`style` approximates
``theme_spiDE``: a black panel border, no grid, black ticks, boxed facet strips and italic legend
titles. Matplotlib has no theme object, so the pieces a style context cannot carry (italic
legend titles, boxed strips, facet layout) are small helpers here, shared by every plot.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl
import numpy as np

# fixed colours by name (R .spideCols), in R's order
_SPIDE_COLS = {
    "up": "#B8791A",
    "down": "#5B4B9A",
    "first": "#0A9396",
    "second": "#AE2012",
    "mid": "#F7F7F7",
    "faint": "grey92",
    "light": "grey80",
    "grey": "grey45",
}
# scico::scico(11, palette = "lapaz"), reversed to run light to dark (R .lapaz)
_LAPAZ = ["#FEF2F2", "#F1D5C4", "#CAB79D", "#A0A695", "#7B9A9E", "#5B8BA2", "#4177A1",
          "#315E98", "#27468B", "#212A78", "#190C64"]  # fmt: skip
_OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
              "#999999"]  # fmt: skip
_ROLES = ("direction", "condition", "magnitude", "diverging", "nuisance")

# matplotlib's CSS grey names are not R's: R's grey92 is #EBEBEB and so on
_R_GREYS = {
    f"grey{i}": "#{0:02X}{0:02X}{0:02X}".format(int(i * 255 / 100 + 0.5)) for i in range(101)
}


def r_colour_hex(col: str) -> str:
    """A colour as a hex string, reading R's ``greyNN`` names (``grey92`` -> ``#EBEBEB``)."""
    return _R_GREYS.get(col, col)


# --- R's colour ramps (grDevices::colorRampPalette(space = "Lab"), scales::hue_pal) ------------

# sRGB -> XYZ of grDevices::convertColor: primaries and white point of R's colorspaces$sRGB
_PRIMARIES = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
_WHITE_XY = (0.3137, 0.3291)  # R's "D65" (white.points), which is not the CIE D65


def _srgb_matrix() -> tuple[np.ndarray, np.ndarray]:
    """R ``make.rgb``: the primaries' matrix ``M`` (row vector convention) and the white XYZ."""
    rgb = np.column_stack([_PRIMARIES, 1 - _PRIMARIES.sum(axis=1)])
    white = np.array([_WHITE_XY[0] / _WHITE_XY[1], 1.0, (1 - sum(_WHITE_XY)) / _WHITE_XY[1]])
    return (white @ np.linalg.inv(rgb) * rgb.T).T, white


_SRGB_M, _LAB_WHITE = _srgb_matrix()
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def _col2rgb(cols: list[str]) -> np.ndarray:
    return np.array([mpl.colors.to_rgb(r_colour_hex(c)) for c in cols])


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyzr = (lin @ _SRGB_M) / _LAB_WHITE
    f = np.where(xyzr <= _EPS, (_KAPPA * xyzr + 16) / 116, np.cbrt(xyzr))
    return np.column_stack(
        [116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]
    )


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    L = lab[:, 0]
    yr = np.where(L < _KAPPA * _EPS, L / _KAPPA, ((L + 16) / 116) ** 3)
    fy = (np.where(yr <= _EPS, _KAPPA * yr, L) + 16) / 116
    fx = lab[:, 1] / 500 + fy
    fz = fy - lab[:, 2] / 200
    zr = np.where(fz**3 <= _EPS, (116 * fz - 16) / _KAPPA, fz**3)
    xr = np.where(fx**3 <= _EPS, (116 * fx - 16) / _KAPPA, fx**3)
    xyz = np.column_stack([xr, yr, zr]) * _LAB_WHITE
    lin = xyz @ np.linalg.inv(_SRGB_M)
    with np.errstate(invalid="ignore"):
        rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)
    return np.clip(np.round(rgb, 5), 0, 1)  # convertColor's trim()


def _hex(rgb: np.ndarray, truncate: bool = False) -> list[str]:
    """Hex strings of 0-1 channels: rounded half up, or truncated as R ``rgb(maxColorValue = 255)``
    does with the ramp's already-trimmed values (so 91 / 255 reads back as 90)."""
    v = np.floor(np.asarray(rgb) * 255 + (0.0 if truncate else 0.5)).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*c) for c in v]


def _color_ramp_lab(colors: list[str], n: int) -> list[str]:
    """R ``colorRampPalette(colors, space = "Lab")(n)``: linear interpolation in CIE Lab."""
    lab = _srgb_to_lab(_col2rgb(colors))
    xs = np.linspace(0.0, 1.0, len(colors))
    at = np.linspace(0.0, 1.0, n)
    out = np.column_stack([np.interp(at, xs, lab[:, k]) for k in range(3)])
    return _hex(_lab_to_srgb(out), truncate=True)


# farver (scales >= 1.2) encodes hue_pal's polar-Luv colours against this white and matrix
_FARVER_WHITE = np.array([0.95047, 1.0, 1.08883])
_FARVER_RGB = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])  # fmt: skip


def _hue_pal(n: int, h=(15.0, 375.0), c: float = 100.0, l: float = 65.0) -> list[str]:
    """``scales::hue_pal()(n)``: ``n`` evenly spaced hues of equal chroma and lightness."""
    h0, h1 = h
    if (h1 - h0) % 360 < 1:
        h1 -= 360 / n
    hues = np.linspace(h0, h1, n) % 360
    u = c * np.cos(np.radians(hues))
    v = c * np.sin(np.radians(hues))
    yr = (l + 16) / 116
    y = yr**3 if l > _EPS * _KAPPA else l / _KAPPA
    den = _FARVER_WHITE[0] + 15 * _FARVER_WHITE[1] + 3 * _FARVER_WHITE[2]
    u0, v0 = 4 * _FARVER_WHITE[0] / den, 9 * _FARVER_WHITE[1] / den
    a = (52 * l / (u + 13 * l * u0) - 1) / 3
    b = -5 * y
    d = y * (39 * l / (v + 13 * l * v0) - 5)
    x = (d - b) / (a + 1 / 3)
    z = x * a + b
    xyz = np.column_stack([x, np.full(n, y), z]) * 1.0
    lin = xyz @ np.linalg.inv(_FARVER_RGB).T
    srgb = np.where(lin > 0.0031308, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055, 12.92 * lin)
    return _hex(np.clip(srgb, 0, 1))


def _magnitude_ramp(n: int = 256, skip: float = 0.0) -> list[str]:
    """The magnitude ramp, light to dark (R ``.magnitudeRamp``).

    ``skip`` drops that fraction of the light end: on a white panel the lowest values must stay
    visible (the tissue map).
    """
    r = _color_ramp_lab(_LAPAZ, 1000)
    r = r[int(math.floor(skip * 999)) :]
    pos = np.round(np.linspace(1, len(r), n)).astype(int) - 1  # R round(): half to even
    return [r[i] for i in pos]


def spide_colours(role: str | None = None, n: int = 9):
    """The spiDE colours (R ``spiDEColours``).

    spiDE gives each role one colour and uses it in every plot: amber and violet for a slope that
    rises or falls with niche density (and for the diverging t scale), teal and rufous for the
    first and second level of the condition, the scico "lapaz" ramp for magnitudes, and the
    Okabe-Ito colours for a nuisance factor such as slide. All pairs stay distinct under the
    common forms of colour blindness.

    Parameters
    ----------
    role : {"direction", "condition", "magnitude", "diverging", "nuisance"}, optional
        ``None`` returns the named fixed colours.
    n : int
        The number of colours of a ramp, or of nuisance levels.

    Returns
    -------
    dict or list of str
        ``{name: colour}`` for ``role=None``, else a list of colours (hex, or R's ``greyNN``
        names for the fixed ones).
    """
    if role is None:
        return dict(_SPIDE_COLS)
    if role not in _ROLES:
        raise ValueError(f"'role' should be one of {', '.join(repr(r) for r in _ROLES)}")
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, float, np.integer, np.floating))
        or not np.isfinite(n)
        or n < 1
    ):
        raise ValueError("'n' must be a positive integer")
    n = int(n)
    if role == "direction":
        return [_SPIDE_COLS["up"], _SPIDE_COLS["down"]]
    if role == "condition":
        return [_SPIDE_COLS["first"], _SPIDE_COLS["second"]]
    if role == "magnitude":
        return _magnitude_ramp(n)
    if role == "diverging":
        return _color_ramp_lab([_SPIDE_COLS["down"], _SPIDE_COLS["mid"], _SPIDE_COLS["up"]], n)
    return _OKABE_ITO[:n] if n <= len(_OKABE_ITO) else _hue_pal(n)


# --- colour roles as matplotlib objects (R .scaleDirection / .scaleCondition / .scaleT) --------


def _direction_colours() -> dict[str, str]:
    """``{"up": amber, "down": violet}``."""
    return {"up": _SPIDE_COLS["up"], "down": _SPIDE_COLS["down"]}


def _condition_colours(levels) -> dict[str, str]:
    """A colour per condition level: teal / rufous for two levels, else black (``.scaleCondition``)."""
    levels = [str(x) for x in levels]
    if len(levels) == 2:
        return dict(zip(levels, (_SPIDE_COLS["first"], _SPIDE_COLS["second"]), strict=True))
    return {x: "black" for x in levels}


def _magnitude_cmap(n: int = 256, skip: float = 0.0, name: str = "spide_magnitude"):
    """The magnitude ramp as a matplotlib colormap."""
    return mpl.colors.ListedColormap(_magnitude_ramp(n, skip), name=name)


def _t_cmap_norm(limit: float):
    """The diverging t scale: violet - near-white - amber, symmetric about 0 (``.scaleT``).

    Returns ``(cmap, norm)``; values beyond ``+-limit`` are squished onto the end colours.
    """
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "spide_t", [_SPIDE_COLS["down"], _SPIDE_COLS["mid"], _SPIDE_COLS["up"]], N=257
    )
    return cmap, mpl.colors.Normalize(vmin=-limit, vmax=limit, clip=True)


# --- the theme -------------------------------------------------------------------------------


@contextmanager
def style(rl: float = 1.1) -> Iterator[None]:
    """A matplotlib style context approximating ``theme_spiDE`` (R ``theme_spiDE``).

    A black panel border (all four spines), no grid, black axis ticks, no top / right ticks,
    text sizes scaled by ``rl`` against a base of 11 pt as ggplot2's ``theme_minimal`` does.
    Use it as ``with spidepy.pl.style(): ...`` around your own plotting code. Boxed facet strips
    and italic legend titles are drawn by the spiDE plots themselves (matplotlib has no rcParams
    for them).

    Parameters
    ----------
    rl : float
        A positive number, the text size relative to ggplot2's default.
    """
    if (
        isinstance(rl, bool)
        or not isinstance(rl, (int, float, np.integer, np.floating))
        or not np.isfinite(rl)
        or rl <= 0
    ):
        raise ValueError("'rl' must be a single positive number")
    base = 11.0
    text = base * rl
    rc = {
        "font.size": text,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.titlesize": base * rl * 1.2,
        "axes.labelsize": base * rl * 1.1,
        "xtick.labelsize": text,
        "ytick.labelsize": text,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.top": False,
        "ytick.right": False,
        "legend.fontsize": text,
        "legend.title_fontsize": text,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with mpl.rc_context(rc):
        yield


def _style_legend(leg) -> None:
    """Italic legend title (``legend.title = element_text(face = "italic")``)."""
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontstyle("italic")


# --- facet layout ----------------------------------------------------------------------------


def _wrap_dims(n: int) -> tuple[int, int]:
    """``(nrow, ncol)`` of ``facet_wrap`` for ``n`` panels (ggplot2 ``wrap_dims``)."""
    if n <= 3:
        n_a, n_b = n, 1
    elif n <= 6:
        n_a, n_b = (n + 1) // 2, 2
    elif n <= 12:
        n_a, n_b = (n + 2) // 3, 3
    else:
        n_a = math.ceil(math.sqrt(n))
        n_b = math.ceil(n / n_a)
    # n2mfrow() gives (nrow, ncol) of a base-graphics layout, the longer side first; ggplot2's
    # wrap_dims swaps them, so the longer side is the number of columns here
    return n_b, n_a


# --- plotmath labels -> mathtext -------------------------------------------------------------


def _pm_string(x: str) -> str:
    """Escape quotes and backslashes for an R plotmath string (R ``.pmString``)."""
    return re.sub(r'(["\\])', r"\\\1", str(x))


def _it_gene(g: str) -> str:
    """R's plotmath label of an italic gene name (``.itGene``), kept for parity with R's data."""
    return f'italic("{_pm_string(g)}")'


def _it_triplet(gene: str, index: str, niche: str) -> str:
    """R's plotmath label ``italic(gene) "index | niche"`` (``.itTriplet``)."""
    return f'italic("{_pm_string(gene)}")~~"{_pm_string(index)} | {_pm_string(niche)}"'


def _mt_body(s: str) -> str:
    """The characters of ``s`` as mathtext that renders them literally.

    Letters and digits pass through; a space is ``\\ ``; any other character is set in
    ``\\text{}``, escaping the few that mathtext reads as markup (``$ \\ { } # % & _ ^``), so a
    gene such as ``HLA-DR``, ``a_b``, ``a"b`` or ``a\\b`` renders as written (a bare ``-`` would
    become a spaced minus sign).
    """
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        elif ch == " ":
            out.append(r"\ ")
        elif ch in "$\\{}#%&_^":
            out.append("\\text{\\" + ch + "}")
        else:
            out.append(f"\\text{{{ch}}}")
    return "".join(out)


def _mt_gene(g: str) -> str:
    """An italic gene name as mathtext, the counterpart of :func:`_it_gene`."""
    return "$\\mathit{" + _mt_body(g) + "}$"


def _mt_triplet(gene: str, index: str, niche: str) -> str:
    """Mathtext of :func:`_it_triplet`: an italic gene, two spaces, ``index | niche``."""
    return (
        "$\\mathit{" + _mt_body(gene) + "}\\ \\ \\mathrm{" + _mt_body(f"{index} | {niche}") + "}$"
    )


def _fmt_q(q) -> np.ndarray:
    """q-values as labels (R ``.fmtq``): ``2e-08`` below 1e-3, else two significant digits.

    R's ``formatC(q, digits = 2, format = "g")`` pads a short result to three characters, so
    ``q = 1`` reads ``"  1"``; kept (see ``docs/r-discrepancies.md``).
    """
    out = []
    for v in np.atleast_1d(np.asarray(q, dtype=float)):
        if not np.isfinite(v):
            out.append("NA")
        elif v < 1e-3:
            out.append(f"{v:.0e}")
        else:
            out.append(f"{v:.2g}".rjust(3))
    return np.array(out, dtype=object)


# --- figure pieces shared by the plots -------------------------------------------------------

# ggplot2 sizes are millimetres; matplotlib's are points
_PT = 72.27 / 25.4


def _point_area(size: float) -> float:
    """``scatter`` area (pt^2) of a ggplot2 ``geom_point`` of this ``size``, stroke 0.5 included."""
    return (size * _PT + 0.5 * 3.78 / 2) ** 2


def _strip(ax, text: str, fontsize: float | None = None) -> None:
    """A boxed facet strip above the panel (``strip.background = element_rect(colour = "black")``).

    ``text`` may be mathtext. It is the axes' title, in a black-outlined box.
    """
    ax.set_title(
        text,
        fontsize=fontsize,
        pad=6,
        bbox={"boxstyle": "square,pad=0.3", "fc": "none", "ec": "black", "lw": 0.5},
    )


def _thousands(v: float, _pos=None) -> str:
    """Axis labels as ``scales::label_number(drop0trailing = TRUE, big.mark = ",")``."""
    return f"{v:,.10g}"


def _log10_axis(ax) -> None:
    """A log10 y axis labelled with plain numbers (``scale_y_log10(labels = label_number(...))``).

    Labelled breaks are decades, thinned to ``1, 3`` or ``1, 2, 5`` multiples when the range
    spans few decades, so the axis never shows an unlabelled stretch.
    """
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    span = np.log10(hi / lo) if lo > 0 else 10.0
    subs = (1.0,) if span > 3 else (1.0, 3.0) if span > 1.5 else (1.0, 2.0, 5.0)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=subs, numticks=20))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_thousands))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def _blank_panel(ax) -> None:
    """Hide ticks, tick labels and axis titles but keep the panel border."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _strwrap(text: str, width: int) -> str:
    """Lines of fewer than ``width`` characters joined by newlines (R ``strwrap`` pasted with
    ``"\\n"``).

    As in R, words are split on whitespace only (never at a hyphen), a word longer than a line is
    kept whole, runs of spaces collapse, and a sentence end (a word ending in ``.``, ``?`` or
    ``!``, optionally followed by quotes or brackets) that the text follows with two spaces keeps
    both.
    """
    tokens = re.split(r"[ \t\n]", str(text))
    words: list[tuple[str, int]] = []  # (word, spaces after it)
    for k, w in enumerate(tokens):
        if not w:
            # an empty token right after a sentence end is that sentence's second space
            if words and k > 0 and tokens[k - 1] and re.search(r"[.?!][)\"']*$", tokens[k - 1]):
                words[-1] = (words[-1][0], 2)
            continue
        words.append((w, 1))
    lines: list[str] = []
    cur, cur_len, gap = [], 0, 1
    for w, sp in words:
        if cur and cur_len + gap + len(w) > width - 1:
            lines.append("".join(cur))
            cur, cur_len = [], 0
        if cur:
            cur.append(" " * gap)
            cur_len += gap
        cur.append(w)
        cur_len += len(w)
        gap = sp
    if cur:
        lines.append("".join(cur))
    return "\n".join(lines)


def _stack_legends(fig, groups, x0: float, y_top: float = 0.9, gap: float = 0.03) -> None:
    """Stack legends down the right of ``fig``, each below the previous one.

    ``groups`` is a list of ``(title, handles, labels)`` (a title may be ``None``); titles are
    italic, as in ``theme_spiDE``. Needs a renderer on the figure (an Agg canvas) to measure.
    """
    y = y_top
    renderer = fig.canvas.get_renderer()
    for title, handles, labels in groups:
        leg = fig.legend(handles, labels, title=title, loc="upper left", bbox_to_anchor=(x0, y),
                         frameon=False, alignment="left")  # fmt: skip
        _style_legend(leg)
        box = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        y = box.y0 - gap
