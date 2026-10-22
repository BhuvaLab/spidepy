"""Per-patient slopes and the per-patient factor against R (``tests/ref/make_slopes.R``).

R's own shared fits (``design.h5``) feed ``_patient_slopes`` for the closed-form checks; the
end-to-end check refits with ``_fit_index_glm``. Also ports the one-step assertions of R's
``test-engine.R`` / ``test-depth.R``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from _design_ref import r_design, strs
from _ref import load_ref
from spidepy._glm import _fit_index_glm, _working_wr
from spidepy._slopes import _patient_factor, _patient_slopes

CFGS = ["c1", "c3", "c6"]
SCENARIOS = ["plain", "mod", "mintiles"]


@pytest.fixture(scope="module")
def S():
    return load_ref("slopes")


@pytest.fixture(scope="module")
def D():
    return load_ref("design")


def _inputs(S, D, cfg, scen):
    """The arguments R used for one (config, scenario)."""
    i = S["inputs"]
    mod = scen == "mod"
    kw = {
        "min_cells": int(i["min_cells_m"]) if mod else 10,
        "min_tiles": 20 if scen == "mintiles" else 5,
    }
    L = np.asarray(i["Lm"] if mod else i["L"])
    tiles = np.asarray(i["tiles_m"] if mod else i["tiles"]).astype(str)
    off = np.asarray(D["fit_inputs"]["Ok"]) if cfg == "c6" else None
    return L, tiles, off, kw


def _r_fit(D, cfg):
    return {"alpha": np.asarray(D[cfg]["alpha"]), "psi": np.asarray(D[cfg]["psi"])}


def _compare(got, ref, rtol, atol=1e-12, rtol_tile=None):
    for k in ("beta", "v_tile", "v_model"):
        r = np.asarray(ref[k])
        assert got[k].shape == r.shape
        np.testing.assert_array_equal(np.isnan(got[k]), np.isnan(r), err_msg=k)  # NA pattern exact
        rt = rtol_tile if k == "v_tile" and rtol_tile is not None else rtol
        np.testing.assert_allclose(got[k], r, rtol=rt, atol=atol, equal_nan=True, err_msg=k)
    np.testing.assert_allclose(got["pooled"], ref["pooled"], rtol=rtol, atol=atol, equal_nan=True)
    assert got["patients"] == strs(ref["patients"])
    assert got["niches"] == strs(ref["niches"])


@pytest.mark.parametrize("scen", SCENARIOS)
@pytest.mark.parametrize("cfg", CFGS)
def test_patient_slopes_match_r_given_r_fit(S, D, cfg, scen):
    L, tiles, off, kw = _inputs(S, D, cfg, scen)
    des = r_design(D[cfg])
    Yk = np.asarray(D["fit_inputs"]["Yk"])
    ps = _patient_slopes(_r_fit(D, cfg), des, Yk, L, tiles, offset=off, **kw)
    ref = S[scen][cfg]
    # v_tile at 1e-8: in c3 a 1-ulp jitter of alpha moves one tile variance by 3e-10 (centred tile
    # scores that nearly cancel), and a CI runner's BLAS moved it by 1.8e-10. The rest stays 1e-10.
    _compare(ps, ref, rtol=1e-10, rtol_tile=1e-8)
    np.testing.assert_allclose(
        _patient_factor(ps["v_tile"], ps["v_model"]), ref["factor"], rtol=1e-10, atol=1e-12
    )


def test_engineered_patients_hit_every_na_branch(S, D):
    """Absent / constant niche (one slope), collinear niches (patient), too few cells (patient)."""
    L, tiles, off, kw = _inputs(S, D, "c1", "mod")
    des = r_design(D["c1"])
    ps = _patient_slopes(_r_fit(D, "c1"), des, np.asarray(D["fit_inputs"]["Yk"]), L, tiles, **kw)
    fitted = np.isfinite(_r_fit(D, "c1")["psi"])
    b, vm, vt = ps["beta"][fitted], ps["v_model"][fitted], ps["v_tile"][fitted]
    P = ps["patients"]
    s = {p: P.index(p) for p in P}
    # S2 has a zeroed niche column (B, not tested): its tested slope for C survives
    assert np.isfinite(b[:, s["S2"], 0]).all()
    # S4 collinear -> whole patient NA; smallest patient (S7) dropped by min_cells
    assert np.isnan(b[:, s["S4"]]).all() and np.isnan(vm[:, s["S4"]]).all()
    assert np.isnan(b[:, s["S7"]]).all()
    # S5 has < min_tiles tiles: v_tile is filled from v_model x max(1, median ratio)
    assert np.isfinite(vt[:, s["S5"]]).all() and (vt[:, s["S5"]] >= vm[:, s["S5"]]).all()
    # unfitted genes are NA everywhere
    assert np.isnan(ps["beta"][~fitted]).all()


def test_patient_factor_misc(S):
    m = S["factor_misc"]
    f = _patient_factor(np.asarray(m["vt"]), np.asarray(m["vm"]))
    np.testing.assert_allclose(f, m["f"], rtol=1e-12)
    assert f.shape == (4, 2) and (f >= 1).all()


@pytest.mark.parametrize("cfg", CFGS)
def test_patient_slopes_end_to_end_on_python_fit(S, D, cfg):
    """Refit in Python, then slopes: iterative-fit tolerance (rtol 1e-6)."""
    L, tiles, off, kw = _inputs(S, D, cfg, "plain")
    des = r_design(D[cfg])
    Yk = np.asarray(D["fit_inputs"]["Yk"])
    fit = _fit_index_glm(Yk, des, offset=off)
    ps = _patient_slopes(fit, des, Yk, L, tiles, offset=off, **kw)
    # rtol 1e-6: the shared alpha / psi are polish_nb outputs (as in test_glm); the sandwich
    # variances amplify them slightly, hence a small atol on near-zero entries
    _compare(ps, S["plain"][cfg], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        _patient_factor(ps["v_tile"], ps["v_model"]), S["plain"][cfg]["factor"], rtol=1e-6
    )


def test_n_jobs_and_sparse_counts_do_not_change_results(S, D):
    L, tiles, off, kw = _inputs(S, D, "c3", "plain")
    des = r_design(D["c3"])
    Yk = np.asarray(D["fit_inputs"]["Yk"])
    a = _patient_slopes(_r_fit(D, "c3"), des, Yk, L, tiles, **kw)
    b = _patient_slopes(_r_fit(D, "c3"), des, Yk, L, tiles, n_jobs=3, **kw)
    c = _patient_slopes(_r_fit(D, "c3"), des, sp.csr_matrix(Yk), L, tiles, **kw)
    for k in ("beta", "v_tile", "v_model"):
        np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a[k], c[k])


def _step(des, fit, Yk, L, g, s, drop=()):
    """Direct one-Fisher-step slope for gene g, patient s (the R test's closed form)."""
    wr = _working_wr(Yk[g], des["W"], fit["alpha"][g], fit["psi"][g])
    i = np.flatnonzero(des["patient"] == s)
    w, r = wr["w"][i], wr["r"][i]
    cols = [j for j in range(L.shape[1]) if j not in drop]
    Ls = L[i][:, cols]
    if des.get("Zs") is None:
        Lt = Ls - (Ls * w[:, None]).sum(0) / w.sum()
    else:  # lm.wfit residuals on the patient's absorbed block
        Z = des["Zs"][s]
        sw = np.sqrt(w)[:, None]
        cf = np.linalg.lstsq(Z * sw, Ls * sw, rcond=None)[0]
        Lt = Ls - Z @ cf
    step = np.linalg.solve(
        (Lt * np.sqrt(w)[:, None]).T @ (Lt * np.sqrt(w)[:, None]), (Lt * r[:, None]).sum(0)
    )
    return fit["alpha"][g, des["niche_cols"][0]] + step[0]


@pytest.mark.parametrize("cfg", ["c1", "c3"])
def test_patient_slope_is_one_fisher_step(S, D, cfg):
    """Ports test-depth.R 'a patient slope is one Fisher step' (loglib and spatial_spline)."""
    L, tiles, _, kw = _inputs(S, D, cfg, "plain")
    des = r_design(D[cfg])
    Yk = np.asarray(D["fit_inputs"]["Yk"])
    fit = _r_fit(D, cfg)
    ps = _patient_slopes(fit, des, Yk, L, tiles, **kw)
    for g in (0, 2):
        for s in (1, 4):
            np.testing.assert_allclose(ps["beta"][g, s, 0], _step(des, fit, Yk, L, g, s), rtol=1e-8)


def test_absent_niche_drops_that_slope_alone(S, D):
    """Ports test-engine.R: the kept slope is the step with the absent niche's column left out."""
    L, tiles, _, kw = _inputs(S, D, "c1", "mod")
    des = r_design(D["c1"])
    Yk = np.asarray(D["fit_inputs"]["Yk"])
    fit = _r_fit(D, "c1")
    ps = _patient_slopes(fit, des, Yk, L, tiles, **kw)
    s = ps["patients"].index("S2")  # second column (niche B) zeroed for S2; tested niche is C
    assert ps["niches"] == ["C"]
    # S2's tested slope (column C) is solved without the zero column B
    g = 0
    np.testing.assert_allclose(
        ps["beta"][g, s, 0], _step(des, fit, Yk, L, g, s, drop=(1,)), rtol=1e-8
    )
    # all the genes' S2 slope for C finite where the gene is fitted
    fitted = np.isfinite(fit["psi"])
    assert np.isfinite(ps["beta"][fitted, s, 0]).all() and (ps["v_tile"][fitted, s, 0] > 0).all()


def test_rank_check_ignores_a_niche_columns_scale(S, D):
    """Ports test-engine.R: shrinking niche B 1e5-fold keeps every patient and rescales B's slope."""
    L, tiles, _, kw = _inputs(S, D, "c6", "plain")  # c6 tests both niches
    des = r_design(D["c6"])
    Yk = np.asarray(D["fit_inputs"]["Yk"])
    off = np.asarray(D["fit_inputs"]["Ok"])
    fit = _r_fit(D, "c6")
    a = _patient_slopes(fit, des, Yk, L, tiles, offset=off, **kw)
    k = 1e-5
    L2 = L.copy()
    L2[:, 1] *= k
    # the design (so mu) must be unchanged: scale the slope by 1/k, the column by k
    des2 = dict(des)
    W2 = np.array(des["W"], copy=True)
    j = des["niche_cols"][1]
    W2[:, j] *= k
    des2["W"] = W2
    fit2 = {"alpha": fit["alpha"].copy(), "psi": fit["psi"]}
    with np.errstate(all="ignore"):
        fit2["alpha"][:, j] /= k
    if des.get("Zs") is not None:
        pytest.skip("dense design expected")
    b = _patient_slopes(fit2, des2, Yk, L2, tiles, offset=off, **kw)
    np.testing.assert_array_equal(np.isfinite(b["beta"]), np.isfinite(a["beta"]))
    np.testing.assert_allclose(b["beta"][..., 0], a["beta"][..., 0], rtol=1e-8, equal_nan=True)
    np.testing.assert_allclose(b["beta"][..., 1] * k, a["beta"][..., 1], rtol=1e-8, equal_nan=True)
    np.testing.assert_allclose(
        b["v_model"][..., 1] * k**2, a["v_model"][..., 1], rtol=1e-8, equal_nan=True
    )
