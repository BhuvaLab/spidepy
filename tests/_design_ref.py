"""Rebuild R's per-index design objects from ``tests/ref/data/design.h5`` (R's own inputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from spanormpy._block_design import nb_block_design

from _ref import load_ref

CONFIGS = ["c1", "c2", "c3", "c4", "c5", "c6"]


def a1(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x))


def strs(x) -> list[str]:
    return [str(s) for s in a1(x)]


def idx0(x) -> np.ndarray:
    """R's 1-based integer index vector as 0-based."""
    return a1(x).astype(int) - 1


def ref():
    return load_ref("design")


def r_design(cfg: dict) -> dict:
    """The design dict of one config exactly as R built it (indices converted to 0-based)."""
    d = cfg["des"]
    out = {
        "patient": idx0(d["patient"]),
        "patients": strs(d["patients"]),
        "npat": int(d["npat"]),
        "start": a1(d["start"]).astype(bool),
        "pen": a1(d["pen"]).astype(float),
        "dense": idx0(d["dense"]),
        "intercept_cols": idx0(d["intercept_cols"]),
        "tested": idx0(d["tested"]),
        "niche_cols": idx0(d["niche_cols"]),
        "tested_niche": strs(d["tested_niche"]),
        "colnames": strs(cfg["colnames"]),
        "absorb": a1(d["absorb"]).astype(bool) if int(cfg["has_absorb"]) else None,
    }
    if int(cfg["block"]):
        out["Zs"] = [np.asarray(cfg["Zs"][f"s{i + 1}"]) for i in range(out["npat"])]
        names = strs(cfg["colnames"])
        px = np.asarray(cfg["X"]).shape[1]
        cat = pd.Categorical.from_codes(out["patient"], categories=out["patients"])
        zc = pd.DataFrame(np.asarray(cfg["Zc"]))
        zc.columns = ["intercept"] + [f"depth{i}" for i in range(1, zc.shape[1])]
        out["W"] = nb_block_design(pd.DataFrame(np.asarray(cfg["X"]), columns=names[:px]), zc, cat)
    else:
        out["Zs"] = None
        out["W"] = np.asarray(cfg["W"])
    return out
