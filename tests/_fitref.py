"""Shared helpers of the Task 10 tests: rebuild R's fits and results tables from ``test.h5``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spidepy import SpiDEFit


def strs(x):
    return [str(s) for s in np.atleast_1d(x)]


def frame(d: dict) -> pd.DataFrame:
    """A table written by ``as.list(data.frame)`` (column order kept in ``colnames__``)."""
    cols = strs(d["colnames__"])
    return pd.DataFrame({c: np.atleast_1d(d[c]) for c in cols})


def close(got, ref, what, rtol=1e-10, atol=1e-12):
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    assert got.shape == ref.shape, f"{what}: shape {got.shape} vs {ref.shape}"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref), err_msg=f"{what}: NA pattern")
    ok = ~np.isnan(ref)
    np.testing.assert_allclose(got[ok], ref[ok], rtol=rtol, atol=atol, err_msg=what)


def pvals(got, ref, what, rtol=1e-8):
    """p-values: NA pattern exact, values close, both below 1e-300 counts as equal."""
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref), err_msg=what)
    tiny = (got < 1e-300) & (ref < 1e-300)
    ok = ~np.isnan(ref) & ~tiny
    np.testing.assert_allclose(got[ok], ref[ok], rtol=rtol, atol=1e-300, err_msg=what)


def fit_from_ref(d: dict, sigma: float = 30.0) -> SpiDEFit:
    """A ``SpiDEFit`` holding exactly R's slopes (or R's sandwich coefficients)."""
    pt = frame(d["patients"])
    index = {}
    for k in strs(d["index_names"]):
        r = d["index"][k]
        genes, pats = strs(r["genes"]), strs(r["patients"])
        x = {
            "genes": genes,
            "niches": strs(r["niches"]),
            "patients": pats,
            "mean_expr": pd.Series(np.atleast_1d(r["mean_expr"]).astype(float), index=genes),
        }
        if d["engine"] == "slopes":
            nt = len(x["niches"])
            x["beta"] = np.asarray(r["beta"], float).reshape(len(genes), len(pats), nt)
            x["v_model"] = np.asarray(r["v_model"], float).reshape(len(genes), len(pats), nt)
            x["factor"] = pd.DataFrame(
                np.asarray(r["factor"], float).reshape(len(pats), nt),
                index=pats,
                columns=x["niches"],
            )
        else:
            x["coef"] = frame(r["coef"])
        index[k] = x
    return SpiDEFit(
        engine=d["engine"],
        sigma=sigma,
        condition=d["condition"] or None,
        patients=pt,
        index=index,
        params={"strata": d["strata"] or None},
    )


def table_equal(tab: pd.DataFrame, ref: dict, what: str, rtol=1e-8):
    """A results table against R's: labels, ``in_family`` and NA patterns exact."""
    ref = dict(ref)
    cols = [c.replace("p.heterogeneity", "p_heterogeneity") for c in strs(ref.pop("colnames__"))]
    ref = {k.replace("p.heterogeneity", "p_heterogeneity"): v for k, v in ref.items()}
    assert list(tab.columns) == cols, what
    for c in cols:
        v = np.atleast_1d(ref[c])
        assert len(tab) == len(v), f"{what}.{c}: length"
        if c in ("gene", "index", "niche", "test"):
            assert [str(s) for s in tab[c]] == strs(v), f"{what}.{c}"
        elif c == "in_family":
            np.testing.assert_array_equal(tab[c].to_numpy(), v.astype(bool), err_msg=f"{what}.{c}")
        elif c in ("p", "q"):
            pvals(tab[c], v, f"{what}.{c}", rtol)
        else:
            close(tab[c], v, f"{what}.{c}", rtol=rtol, atol=1e-10)
