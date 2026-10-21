"""Task 15: plot_patient_slopes and plot_patient_effects (R/plot-patients.R; tests
``test-plots-patients.R``).

The data builders are compared with R's own on R's fits (``plot_data.h5``): strip labels, the
slopes' data frames, the patient-effect matrix, covariates, adjusted R^2 and the plots' data.
The figures are checked for structure and R's error messages.
"""

from __future__ import annotations

import copy
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from _fitref import frame, strs  # noqa: E402
from _plotref import fit_from_dump, res_from_dump, same_frame  # noqa: E402
from _ref import load_ref  # noqa: E402
from spidepy import pl  # noqa: E402
from spidepy.pl import _patients as P  # noqa: E402


@pytest.fixture(scope="module")
def R():
    return load_ref("plot_data")


@pytest.fixture(scope="module")
def res(R):
    return res_from_dump(R["res"])


@pytest.fixture(scope="module")
def fit(res):
    return res.fit


@pytest.fixture(scope="module")
def res_sand(R):
    return res_from_dump(R["fs"])


@pytest.fixture(scope="module")
def res_nocond(R):
    return res_from_dump(R["r0"])


def render(fig):
    """Draw the figure with warnings as errors (a PDF and a PNG through Agg)."""
    assert isinstance(fig, Figure)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig.canvas.draw()
    return fig


def close(got, ref, what, rtol=1e-10, atol=1e-12):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    assert got.shape == ref.shape, f"{what}: {got.shape} vs {ref.shape}"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref), err_msg=what)
    ok = ~np.isnan(ref)
    np.testing.assert_allclose(got[ok], ref[ok], rtol=rtol, atol=atol, err_msg=what)


# --- plotPatientSlopes: data parity ----------------------------------------------------------


def test_strip_labels_match_r(R, res, res_nocond):
    got = P._slope_strip_labels(res, ["G1", "G2", "G7"], "A", "B")
    assert list(got.values()) == list(strs(R["slopes"]["strip_cond"]))
    assert list(got) == ["G1", "G2", "G7"]
    got = P._slope_strip_labels(res_nocond, ["G1", "G2"], "A", "B")
    assert list(got.values()) == list(strs(R["slopes"]["strip_pooled"]))


@pytest.mark.parametrize(
    "key, args",
    [
        ("a", dict(gene=["G1", "G2"], index="A", niche="B")),
        ("b", dict(gene=["G3", "G1"], index="B", niche="A", ci=0.9)),
    ],
)
def test_patient_slopes_data_match_r(R, res, key, args):
    D = P._patient_slopes_data(res, **args)
    ref = R["slopes"][key]
    same_frame(D["data"], ref["data"], f"{key}.data")
    same_frame(D["band"], ref["band"], f"{key}.band")
    same_frame(D["gm"], ref["gm"], f"{key}.gm")
    assert list(D["data"]["patient"].cat.categories) == strs(ref["patient_levels"])
    assert list(D["data"]["gene_lab"].cat.categories) == list(D["labels"].values())


def test_patient_slopes_without_condition_and_missing_slope(R, res_nocond):
    D = P._patient_slopes_data(res_nocond, "G1", "A", "B")
    ref = R["slopes"]["none"]
    same_frame(D["data"], ref["data"], "none.data")
    same_frame(D["band"], ref["band"], "none.band")
    same_frame(D["gm"], ref["gm"], "none.gm")
    assert "S3" not in set(D["data"]["patient"])
    assert D["levels"] == ["all patients"]
    assert list(D["data"]["patient"].cat.categories) == strs(ref["patient_levels"])


def test_patient_slopes_weights_sum_to_one(res):
    D = P._patient_slopes_data(res, ["G1", "G2"], "A", "B")
    s = D["data"].groupby("gene")["weight"].sum()
    np.testing.assert_allclose(s.to_numpy(), [1, 1], rtol=1e-12)
    assert set(D["data"]["condition"].cat.categories) == {"Non-responder", "Responder"}


def test_patient_slopes_join_by_name(res):
    """Shuffling the fit's patient table (its rows) does not change which condition a patient has."""
    r2 = copy.copy(res)
    r2.fit = copy.copy(res.fit)
    r2.fit.patients = res.fit.patients.sample(frac=1, random_state=3).reset_index(drop=True)
    a = P._patient_slopes_data(res, "G1", "A", "B")["data"]
    b = P._patient_slopes_data(r2, "G1", "A", "B")["data"]
    pd.testing.assert_frame_equal(a, b)


# --- plotPatientSlopes: figure ---------------------------------------------------------------


def test_patient_slopes_figure(res):
    fig = pl.plot_patient_slopes(res, gene=["G1", "G2"], index="A", niche="B")
    render(fig)
    axs = fig.axes
    assert len(axs) == 4  # two conditions x two genes
    titles = [a.get_title() for a in axs if a.get_title()]
    assert len(titles) == 2 and all("q =" in t for t in titles)
    assert any(t.get_text() == "Responder" for a in axs for t in a.texts)
    ylabs = {t.get_text() for a in axs for t in a.get_yticklabels()}
    assert ylabs == {f"S{i}" for i in range(1, 17)}
    assert fig.texts and "slope on B density (log1p)" in [t.get_text() for t in fig.texts]


def test_patient_slopes_without_condition_figure(res_nocond):
    fig = pl.plot_patient_slopes(res_nocond, gene="G1", index="A", niche="B")
    render(fig)
    assert len(fig.axes) == 1
    assert "slope =" in fig.axes[0].get_title()
    assert not any("slope difference" in t.get_text() for t in fig.texts)


def test_patient_slopes_errors(res, res_sand):
    with pytest.raises(ValueError, match="slopes engine"):
        pl.plot_patient_slopes(res_sand, "G1", "A", "B")
    with pytest.raises(ValueError, match="niches tested in A"):
        pl.plot_patient_slopes(res, "G1", "A", "Z")
    with pytest.raises(ValueError, match="not tested in A: nope"):
        pl.plot_patient_slopes(res, "nope", "A", "B")
    with pytest.raises(ValueError, match="'ci' must be"):
        pl.plot_patient_slopes(res, "G1", "A", "B", ci=1.5)
    with pytest.raises(TypeError, match="SpiDEResults"):
        pl.plot_patient_slopes(res.fit, "G1", "A", "B")


# --- plotPatientEffects: data parity ---------------------------------------------------------


def test_effect_matrix_matches_r(R, fit):
    E, xi = R["effects"], fit.index["A"]
    for key, niche, ntop in (("Mi", None, 500), ("Ms", "B", 500)):
        M = P._patient_effect_matrix(xi, niche, ntop)
        assert list(M.index) == strs(E[f"{key}_genes"]), key
        assert list(M.columns) == strs(E[f"{key}_pats"]), key
        close(M.to_numpy(), E[key], key)
    assert list(P._patient_effect_matrix(xi, None, 6).index) == strs(E["Mt_genes"])


def test_covariates_match_r(R, fit):
    E, xi = R["effects"], fit.index["A"]
    cv = P._patient_covariates(fit, xi)
    same_frame(cv["cov"].reset_index(drop=True), E["cov"], "cov")
    assert list(cv["cov"].index) == strs(E["cov_rows"])
    assert list(cv["group"]) == strs(E["cov_group"]) and list(cv["group"].index) == strs(
        E["cov_names"]
    )
    cv2 = P._patient_covariates(fit, xi, ["condition", "B density"])
    same_frame(cv2["cov"].reset_index(drop=True), E["cov2"], "cov2")
    assert list(cv2["group"]) == strs(E["cov2_group"])
    with pytest.raises(ValueError, match="nope"):
        P._patient_covariates(fit, xi, ["nope"])


def test_covariates_keep_varying_patient_columns(R):
    """A character covariate and a numeric one enter; identifiers (as many values as patients) do not."""
    fp = fit_from_dump(R["fp"])
    cv = P._patient_covariates(fp, fp.index["A"])
    E = R["effects"]
    assert list(cv["cov"].columns) == strs(E["covp_names"])
    assert list(cv["group"]) == strs(E["covp_group"])
    same_frame(cv["cov"].reset_index(drop=True), E["covp"], "covp")
    assert "patient" not in cv["cov"].columns


def test_adj_r2_matches_r(R, fit):
    a = R["effects"]["adj"]
    y = np.asarray(a["y"], float)
    n = len(y)
    assert P._adj_r2(y, np.arange(1, n + 1) ** 2.0) == pytest.approx(a["num"], abs=1e-12)
    assert P._adj_r2(y, 2 * y + np.sin(np.arange(1, n + 1))) == pytest.approx(a["num2"], rel=1e-10)
    lv = np.array(["a", "b", "c"])[np.arange(n) % 3]
    assert P._adj_r2(y, pd.Categorical(lv)) == pytest.approx(a["fac"], abs=1e-12)
    assert P._adj_r2(y, pd.Categorical(np.where(y > np.median(y), "hi", "lo"))) == pytest.approx(
        a["fac2"], rel=1e-10
    )
    assert np.isnan(P._adj_r2(y, np.ones(n))) and np.isnan(a["const"])
    assert np.isnan(P._adj_r2(y[:3], np.arange(3.0))) and np.isnan(a["few"])
    v = pd.Categorical(np.array(["a", "b"])[np.arange(n) % 2])
    y2 = y.copy()
    y2[0] = np.nan
    assert P._adj_r2(y2, v) == pytest.approx(a["na"], abs=1e-12)


@pytest.mark.parametrize(
    "key, kw",
    [
        ("assoc", {}),
        ("assoc_s", dict(niche="B", n_pcs=3)),
        ("pca", dict(type="pca", colour_by="condition")),
        ("pca_s", dict(niche="B", type="pca", colour_by="B density")),
    ],
)
def test_patient_effects_data_match_r(R, res, key, kw):
    D = P._patient_effects_data(res, "A", **kw)
    E = R["effects"]
    d = D["data"]
    if key.startswith("pca"):
        # component signs are arbitrary: compare magnitudes, the rest exactly
        ref = frame(E[key])
        assert list(d.columns) == list(ref.columns)
        assert list(d["patient"]) == list(ref["patient"])
        for c in ("PC1", "PC2"):
            close(np.abs(d[c]), np.abs(ref[c]), f"{key}.{c}")
        other = d.columns[3]
        a, b = d[other].to_numpy(), ref[other].to_numpy()
        if b.dtype.kind in "fi":
            close(a.astype(float), b, other)
        else:
            assert [str(v) for v in a] == [str(v) for v in b]
    else:
        same_frame(d, E[key], key)
    if key == "assoc":
        assert D["pc_levels"] == strs(E["assoc_pc_levels"])
        assert D["covariate_levels"] == strs(E["assoc_cov_levels"])
        assert D["xlab"] == E["assoc_x"]
    if key == "assoc_s":
        assert D["xlab"] == E["assoc_s_x"]
    if key == "pca":
        assert (D["xlab"], D["ylab"]) == (E["pca_x"], E["pca_y"])
        assert d["condition"].dtype.name == "category"
        assert list(d["condition"].cat.categories) == ["Non-responder", "Responder"]


def test_patient_effects_pcs_match_r(R, fit):
    M = P._patient_effect_matrix(fit.index["A"], None, 500)
    s = np.linalg.svd((M.to_numpy().T), compute_uv=False)
    close(s / np.sqrt(M.shape[1] - 1), R["effects"]["pc_sdev"], "sdev")


def test_patient_effects_sandwich_engine_intercepts(R, res_sand):
    D = P._patient_effects_data(res_sand, "A")
    same_frame(D["data"], R["effects"]["sand"], "sand")
    with pytest.raises(ValueError, match="slopes engine"):
        pl.plot_patient_effects(res_sand, index="A", niche="B")


# --- plotPatientEffects: figure --------------------------------------------------------------


def test_patient_effects_association_figure(res):
    fig = pl.plot_patient_effects(res, index="A")
    render(fig)
    D = P._patient_effects_data(res, "A")
    r2 = D["data"]["r2"].to_numpy()
    assert np.all(np.isnan(r2) | ((r2 >= 0) & (r2 <= 1)))
    cov = set(D["data"]["covariate"].astype(str))
    assert {"condition", "cells (log10)", "depth (mean log)", "B density"} <= cov
    assert not {"patient", "sample_id"} & cov
    assert all(__import__("re").fullmatch(r"[0-9]+\n[0-9]+%", s) for s in D["pc_levels"])
    assert "principal component" in D["xlab"]
    ylab = [t.get_text() for a in fig.axes for t in a.get_yticklabels()]
    assert "B density" in ylab and "condition" in ylab


def test_patient_effects_pca_figure(res, fit):
    fig = pl.plot_patient_effects(res, index="A", type="pca", colour_by="condition")
    render(fig)
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("PC1 (") and ax.get_ylabel().startswith("PC2 (")
    assert sum(len(c.get_offsets()) for c in ax.collections) == 16
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Non-responder", "Responder"]
    # numeric colouring gives a colour bar, a categorical covariate a legend
    render(pl.plot_patient_effects(res, index="A", niche="B", type="pca", colour_by="B density"))
    render(pl.plot_patient_effects(fit, index="A", type="pca", colour_by="Age"))


def test_patient_effects_fit_and_covariates(fit, res):
    D = P._patient_effects_data(fit, "A", covariates=["condition", "B density"])
    assert set(D["data"]["covariate"].astype(str)) == {"condition", "B density"}
    with pytest.raises(ValueError, match="nope"):
        pl.plot_patient_effects(res, index="A", covariates="nope")
    with pytest.raises(ValueError, match="colour_by"):
        pl.plot_patient_effects(res, index="A", type="pca", colour_by="nope")
    render(pl.plot_patient_effects(fit, index="A", covariates=["condition", "B density"]))


def test_patient_effects_needs_intercepts(fit):
    f = copy.deepcopy(fit)
    f.index["A"]["intercept"] = None
    with pytest.raises(ValueError, match="refit"):
        pl.plot_patient_effects(f, index="A")
    with pytest.raises(TypeError, match="SpiDEResults"):
        pl.plot_patient_effects(pd.DataFrame(), index="A")


def test_patient_effects_keeps_patients_when_genes_unfitted(fit):
    f = copy.deepcopy(fit)
    xi = f.index["A"]
    xi["intercept"].loc[list(xi["genes"][:6]), :] = np.nan
    D = P._patient_effects_data(f, "A", type="pca")
    assert len(D["data"]) == len(xi["patients"])


def test_patient_effects_finds_planted_condition(fit):
    def r2(f):
        d = P._patient_effects_data(f, "A")["data"]
        return d.loc[
            (d["covariate"].astype(str) == "condition") & d["pc"].astype(str).str.startswith("1\n"),
            "r2",
        ].iloc[0]

    assert r2(fit) > 0.5
    f = copy.deepcopy(fit)
    f.index["A"]["intercept"].loc["G2", "S3"] = -8.8
    assert r2(f) > 0.5


def test_patient_effects_numeric_condition_is_two_groups(fit):
    f = copy.deepcopy(fit)
    f.patients = f.patients.copy()
    f.patients["condition"] = (f.patients["condition"].astype(str) == "Responder").astype(float)
    D = P._patient_effects_data(f, "A", type="pca")
    assert D["data"]["condition"].dtype.name == "category"
    assert D["data"]["condition"].nunique() == 2
    render(pl.plot_patient_effects(f, index="A", type="pca"))


def test_patient_effects_too_few_genes(fit):
    with pytest.raises(ValueError, match="too few genes or patients"):
        pl.plot_patient_effects(fit, index="A", ntop=1)
