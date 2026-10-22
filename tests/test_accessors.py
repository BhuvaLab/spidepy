"""``patient_slopes`` / ``patient_intercepts`` (R ``patientSlopes`` / ``patientIntercepts``):
row and column order, subsetting and dispatch. The values against R are in ``test_fit.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _ref import load_spe
from spidepy import SpiDEResults, build_niches, fit_spide, patient_intercepts, patient_slopes


@pytest.fixture(scope="module")
def fit():
    a = build_niches(load_spe("toySpiDE"), sigma=30)
    return fit_spide(a, sigma=30, condition="condition", verbose=False)


def test_slopes_columns_and_row_order(fit):
    d = patient_slopes(fit)
    assert list(d.columns[:8]) == [
        "gene", "index", "niche", "patient", "slope", "var_spatial", "var_model", "ncells",
    ]  # fmt: skip
    assert {"sample_id", "condition", "Age"} <= set(d.columns)
    assert len(d) == 3 * 20 * 16 * 2  # index types x genes x patients x niches
    # ordered by index, gene, niche, patient (code-point order: G1, G10, ..., S1, S10, S2)
    assert d.equals(
        d.sort_values(["index", "gene", "niche", "patient"], kind="stable").reset_index(drop=True)
    )
    first = d[(d["index"] == "A") & (d.gene == "G1") & (d.niche == "B")]
    assert first.patient.tolist()[:4] == ["S1", "S10", "S11", "S12"]


def test_slopes_values_come_from_the_index_arrays(fit):
    x = fit.index["A"]
    d = patient_slopes(fit, gene="G3", index="A", niche="C")
    g, j = x["genes"].index("G3"), x["niches"].index("C")
    assert len(d) == len(x["patients"])
    for _, r in d.iterrows():
        s = x["patients"].index(r.patient)
        assert r.slope == x["beta"][g, s, j] or (np.isnan(r.slope) and np.isnan(x["beta"][g, s, j]))
        assert r.var_spatial == x["v_tile"][g, s, j] or np.isnan(r.var_spatial)
        assert r.ncells == x["ncells"][r.patient]
        assert r.condition == fit.patients.set_index("patient").condition[r.patient]


def test_subsets_and_empty_results(fit):
    assert set(patient_slopes(fit, index=["C", "A"])["index"]) == {"A", "C"}
    assert patient_slopes(fit, gene="nope").empty
    assert patient_slopes(fit, niche="nope").empty
    assert patient_intercepts(fit, gene=["G2", "G1"], index="B").gene.unique().tolist() == [
        "G1",
        "G2",
    ]
    assert patient_intercepts(fit, gene="nope").empty


def test_intercept_columns_and_values(fit):
    d = patient_intercepts(fit, index="A")
    assert list(d.columns[:5]) == ["gene", "index", "patient", "intercept", "ncells"]
    assert len(d) == 20 * 16
    x = fit.index["A"]
    row = d[(d.gene == "G4") & (d.patient == "S12")].iloc[0]
    assert row.intercept == x["intercept"].loc["G4", "S12"]
    assert row.ncells == x["ncells"]["S12"]


def test_results_objects_dispatch_to_their_fit(fit):
    res = SpiDEResults(pd.DataFrame(), "condition", "", "all", 0.05, fit)
    pd.testing.assert_frame_equal(patient_slopes(res, gene="G1"), patient_slopes(fit, gene="G1"))
    pd.testing.assert_frame_equal(patient_intercepts(res), patient_intercepts(fit))
    with pytest.raises(TypeError):
        patient_slopes("not a fit")
