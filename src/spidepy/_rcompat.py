"""Small R-compatibility helpers: number formatting, name sanitising, p-value adjustment, sorting."""

from __future__ import annotations

import math
import re

import numpy as np


def r_num_str(x: float) -> str:
    """Format a double as R's ``as.character()`` / ``paste0()`` does.

    Ports R's number-to-string conversion at ``scipen = 0``: the value is written with the
    fewest significant digits (at most 15) that reproduce it to 15 digits, then fixed notation
    is used unless scientific notation is strictly shorter (``1e+05``, ``1e-04``, ``1e+15``).
    The 15-digit rounding only decides the notation and the number of decimals.
    ``NaN``/``Inf``/``-Inf`` print as in R; a missing value (``None``) prints ``"NA"``.
    """
    if x is None:
        return "NA"
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0:
        return "0"
    neg = x < 0
    # 15 significant digits, then drop trailing zeros to get the minimal digit string
    mant, exp = f"{abs(x):.14e}".split("e")
    digits = mant.replace(".", "").rstrip("0") or "0"
    e = int(exp)
    nsig = len(digits)
    # width of each representation (sign excluded: it is common to both)
    exp_w = 2 if abs(e) < 100 else 3
    sci_w = (nsig + 1 if nsig > 1 else 1) + 2 + exp_w  # mantissa, "e", sign, exponent
    if e >= 0:
        left, rgt = e + 1, max(0, nsig - e - 1)
    else:
        left, rgt = 1, nsig - e - 1
    fix_w = left + (rgt + 1 if rgt > 0 else 0)
    if fix_w <= sci_w:
        # fixed notation prints the double itself (sprintf "%.*f"), not the 15-digit rounding:
        # as.character(12345678901234567) is "12345678901234568"
        s = f"{abs(x):.{rgt}f}"
    else:
        s = digits[0] + ("." + digits[1:] if nsig > 1 else "")
        s += f"e{'-' if e < 0 else '+'}{abs(e):0{2}d}"
    return ("-" if neg else "") + s


def sanitise(x):
    """Replace spaces and hyphens by ``.`` (R ``gsub(" |-", ".", x)``).

    Accepts a string or an iterable of strings and returns the same kind.
    """
    if isinstance(x, str):
        return re.sub(r" |-", ".", x)
    return [re.sub(r" |-", ".", s) for s in x]


def p_adjust_bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjustment as R ``p.adjust(p, method = "BH")``.

    ``NaN`` entries stay ``NaN`` and do not count towards ``n`` (R's ``na.rm``-style handling:
    ``n`` is the number of non-missing p-values).
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    n = int(ok.sum())
    if n == 0:
        return out
    q = p[ok]
    o = np.argsort(-q, kind="stable")  # decreasing, as order(p, decreasing = TRUE)
    i = np.arange(n, 0, -1)
    adj = np.minimum(1.0, np.minimum.accumulate(n / i * q[o]))
    res = np.empty(n)
    res[o] = adj
    out[ok] = res
    return out


def r_sort_str(xs) -> list[str]:
    """Sort strings as R ``sort()`` does under the C collation (code-point order).

    R's ``sort()`` on character vectors uses the session locale (``en_US.UTF-8`` interleaves
    ``a A b B``; ``C`` puts every capital first). Python ``sorted`` is code-point order, which
    equals R's ``method = "radix"`` and ``LC_COLLATE=C``. The two agree for labels such as
    ``S1, S10, S2`` and ``Non-responder, Responder`` (same case, no punctuation clashes); where
    a locale would differ, see ``docs/r-discrepancies.md``.
    """
    return sorted(str(s) for s in xs)
