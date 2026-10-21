"""Parity of density_at with spatstat's densityfun (tests/ref/make_niches.R)."""

import numpy as np
import pytest

from _ref import load_ref
from spidepy._kde import density_at

WIN = (0.0, 10.0, 0.0, 8.0)
CASES = [(s, e, d) for s in ("0.7", "2.5", "6") for e, d in ("TT", "TF", "FF", "FT")]


@pytest.fixture(scope="module")
def raw():
    r = load_ref("niches")["raw"]
    return r, np.c_[r["px"], r["py"]], np.c_[r["qx"], r["qy"]]


@pytest.mark.parametrize(("s", "e", "d"), CASES)
def test_density_matches_densityfun(raw, s, e, d):
    r, pts, qry = raw
    got = density_at(pts, qry, float(s), WIN, edge=e == "T", diggle=d == "T")
    np.testing.assert_allclose(got, r[f"s{s}_{e}{d}"], rtol=1e-10, atol=1e-12)


def test_edge_false_ignores_diggle(raw):
    _, pts, qry = raw
    a = density_at(pts, qry, 2.5, WIN, edge=False, diggle=True)
    b = density_at(pts, qry, 2.5, WIN, edge=False, diggle=False)
    np.testing.assert_array_equal(a, b)


def test_single_point_includes_self():
    r = load_ref("niches")["raw"]
    got = density_at([[5, 4]], [[5, 4], [6, 4]], 2.0, WIN)
    np.testing.assert_allclose(got, r["one"], rtol=1e-10)
    assert got[0] > got[1] > 0  # K(0) is included for the coincident query


def test_outside_window_is_nan():
    got = density_at([[1, 1]], [[1, 1], [20, 1]], 1.0, WIN)
    assert np.isfinite(got[0]) and np.isnan(got[1])


def test_blocked_evaluation_matches_single_block(monkeypatch):
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 50, (300, 2))
    win = (0.0, 50.0, 0.0, 50.0)
    full = density_at(pts, pts, 4.0, win)
    monkeypatch.setattr("spidepy._kde._BLOCK", 37)
    np.testing.assert_allclose(density_at(pts, pts, 4.0, win), full, rtol=1e-14)
