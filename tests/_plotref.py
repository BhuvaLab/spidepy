"""Shared helpers of the plot tests (Tasks 14-15): R's spe, fits and results from ``plot_helpers.h5``.

The fits hold R's own slopes, intercepts and tables (not Python's refit), so the data builders
are checked on R's inputs. Per-patient arrays are matched to patients by name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _fitref import frame, strs
from _ref import load_ref, load_spe
from spidepy import SpiDEFit, SpiDEResults
from spidepy._io import GROUPS_KEY


def plot_ref() -> dict:
    return load_ref("plot_helpers")


def plot_spe(topic: str = "plot_spe"):
    """R's toy spe with its niche matrix (``Niche30``, R's own values and column names), as an
    AnnData; merged variants (``plot_spe_bc``, ``plot_spe_b``) also carry the merge groups."""
    a = load_spe(topic)
    r = load_ref(topic)
    a.obsm["Niche30"] = pd.DataFrame(
        np.asarray(a.obsm["Niche30"], dtype=float), index=a.obs_names, columns=strs(r["niche_cols"])
    )
    if "groups" in r:
        a.uns[GROUPS_KEY] = {"Niche30": {k: strs(v) for k, v in r["groups"].items()}}
    return a


def fit_from_dump(d: dict) -> SpiDEFit:
    """A ``SpiDEFit`` holding exactly what R's fit holds (see ``make_plot_helpers.R``)."""
    pt = frame(d["patients"])
    index = {}
    for k in strs(d["index_names"]):
        r = d["index"][k]
        genes, pats, nich = strs(r["genes"]), strs(r["patients"]), strs(r["niches"])
        x = {
            "genes": genes,
            "niches": nich,
            "patients": pats,
            "mean_expr": pd.Series(np.atleast_1d(r["mean_expr"]).astype(float), index=genes),
            "intercept": pd.DataFrame(
                np.asarray(r["intercept"], float).reshape(len(genes), len(pats)),
                index=genes,
                columns=pats,
            ),
            "loglib_mean": pd.Series(np.atleast_1d(r["loglib_mean"]).astype(float), index=pats),
            "ncells": pd.Series(np.atleast_1d(r["ncells"]).astype(int), index=pats),
        }
        if "niche_mean" in r:
            x["niche_mean"] = pd.DataFrame(
                np.asarray(r["niche_mean"], float).reshape(len(pats), len(nich)),
                index=pats,
                columns=nich,
            )
        if d["engine"] == "slopes":
            shp = (len(genes), len(pats), len(nich))
            x["beta"] = np.asarray(r["beta"], float).reshape(shp)
            x["v_model"] = np.asarray(r["v_model"], float).reshape(shp)
            x["factor"] = pd.DataFrame(
                np.asarray(r["factor"], float).reshape(len(pats), len(nich)),
                index=pats,
                columns=nich,
            )
        else:
            x["coef"] = frame(r["coef"])
        index[k] = x
    p = d["params"]
    return SpiDEFit(
        engine=d["engine"],
        sigma=float(d["sigma"]),
        condition=d["condition"] or None,
        patients=pt,
        index=index,
        params={
            "cell_type": p["cell_type"],
            "sample_id": p["sample_id"],
            "depth": p["depth"],
            "offset": p["offset"] or None,
            "covariates": [c for c in strs(p["covariates"]) if c],
        },
    )


def res_from_dump(d: dict) -> SpiDEResults:
    tab = frame(d["table"])
    tab["in_family"] = tab["in_family"].astype(bool)
    return SpiDEResults(
        table=tab,
        condition=d["condition"] or None,
        contrast=d["contrast"],
        procedure=d["procedure"],
        fdr=float(d["fdr"]),
        fit=fit_from_dump(d["fit"]),
    )


def same_frame(got: pd.DataFrame, ref: dict, what: str, rtol=1e-10, atol=1e-12):
    """A DataFrame against R's data frame: column names, row order, labels, numbers at 1e-10."""
    ref = frame(ref)
    # an R integer NA arrives as the integer minimum
    for c in ref.columns:
        if ref[c].dtype.kind == "i" and (ref[c] == -2147483648).any():
            ref[c] = ref[c].astype(float).where(ref[c] != -2147483648)
    assert list(got.columns) == list(ref.columns), what
    assert len(got) == len(ref), f"{what}: {len(got)} rows vs {len(ref)}"
    for c in ref.columns:
        a, b = got[c].to_numpy(), ref[c].to_numpy()
        if b.dtype.kind in "fiu":
            np.testing.assert_array_equal(np.isnan(a.astype(float)), np.isnan(b.astype(float)), c)
            ok = ~np.isnan(b.astype(float))
            np.testing.assert_allclose(
                a.astype(float)[ok],
                b.astype(float)[ok],
                rtol=rtol,
                atol=atol,
                err_msg=f"{what}.{c}",
            )
        elif b.dtype.kind == "b" or a.dtype.kind == "b":
            np.testing.assert_array_equal(a.astype(bool), b.astype(bool), err_msg=f"{what}.{c}")
        else:
            assert [str(x) for x in a] == [str(x) for x in b], f"{what}.{c}"
