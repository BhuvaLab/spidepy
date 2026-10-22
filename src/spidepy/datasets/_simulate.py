"""Toy-data simulators: numpy ports of spiDE's `.toySPE()`, `.toyNiche()`, `.toyClustered()`.

All three share one negative-binomial count model adapted from splatter (Zappia, Phipson &
Oshlack, Genome Biology 2017). The generative model matches R, but the random draws do not:
R's Mersenne-Twister stream cannot be reproduced from `numpy.random.default_rng`, so a given
`seed` gives a different (equally valid) data set from R's. Every function takes its own
`numpy.random.Generator` built from `seed`; the global numpy RNG is never read or reseeded
(R's `.localSeed` guarantee).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def _sim_gene_params(
    rng,
    gene_names,
    mean_shape=0.35,
    mean_rate=0.9,
    bcv_common=0.3,
    bcv_disp=1.0,
    out_prob=0.05,
    out_fac_loc=2.0,
    out_fac_scale=0.5,
    mean_min=0.0,
    boost=(),
    boost_gmean=6.0,
):
    """Per-gene mean, BCV and NB size (ports R `.simGeneParams()`, `total.count`/`gmean` unused)."""
    ng = len(gene_names)
    gmean = np.maximum(rng.gamma(shape=mean_shape, scale=1.0 / mean_rate, size=ng), 1e-3)
    is_out = rng.uniform(size=ng) < out_prob
    if is_out.any():
        gmean[is_out] *= rng.lognormal(out_fac_loc, out_fac_scale, size=int(is_out.sum()))
    gmean = np.maximum(gmean, mean_min)
    for g in boost:  # planted genes get a high fixed abundance so effects stay estimable
        gmean[list(gene_names).index(g)] = boost_gmean
    bcv = bcv_common + bcv_disp / np.sqrt(gmean)  # BCV falls with abundance
    return gmean, 1.0 / bcv**2


def _sim_counts(rng, gmean, size, cell_type, log_effect=None, lib_size=None, sd_ct=0.5):
    """Genes x cells NB counts (ports R `.simCounts()`): mu = lib * gmean * exp(delta_ct + effect)."""
    ng = len(gmean)
    cts = sorted(set(cell_type))
    delta = rng.normal(0.0, sd_ct, size=(ng, len(cts)))  # per-(gene, cell type) log-fold offsets
    col = np.searchsorted(cts, cell_type)
    lmu = np.log(gmean)[:, None] + delta[:, col]
    if log_effect is not None:
        lmu = lmu + log_effect
    if lib_size is not None:
        lmu = lmu + np.log(lib_size)[None, :]
    mu = np.exp(lmu)
    sz = np.broadcast_to(size[:, None], mu.shape)
    return rng.negative_binomial(sz, sz / (sz + mu)).astype(float)


def _sim_lib_size(rng, sample_id, cell_type, sd_sample=0.15, sd_celltype=0.35):
    """Per-cell library size from a sample and a cell-type log-scale term (ports `.simLibSize()`)."""
    smp, ct = np.asarray(sample_id), np.asarray(cell_type)
    su, cu = sorted(set(smp)), sorted(set(ct))
    s_lib = rng.normal(0.0, sd_sample, size=len(su))
    c_lib = rng.normal(0.0, sd_celltype, size=len(cu))
    return np.exp(s_lib[np.searchsorted(su, smp)] + c_lib[np.searchsorted(cu, ct)])


def _sample_cells(rng, n_per, field, p_b, a_shift, age):
    """One sample's coordinates, cell types and optional Age draw (the R per-sample `lapply`).

    B cells concentrate in the right 40% of the field. Draw order follows R: x, y, B/non-B
    uniforms, the A/C choice, the A-cell shift (only when `a_shift > 0`), then Age.
    """
    x = rng.uniform(0, field, n_per)
    y = rng.uniform(0, field, n_per)
    is_b = (x > 0.6 * field) & (rng.uniform(size=n_per) < p_b)
    ac = rng.choice(np.array(["A", "C"]), size=n_per, replace=True)
    ct = np.where(is_b, "B", ac)
    if a_shift > 0:
        is_a = ct == "A"
        x[is_a] = rng.uniform(a_shift * field, field, int(is_a.sum()))
    a = rng.normal() if age else None
    return x, y, ct, a


def _assemble(counts, cd, gene_names) -> AnnData:
    """AnnData in the layout `load_spe` builds from R's SpatialExperiment fixtures."""
    cd = cd.copy()
    cd["nCount"] = counts.sum(axis=0)
    adata = AnnData(
        X=counts.T.copy(),
        obs=cd,
        var=pd.DataFrame(index=pd.Index(list(gene_names))),
        layers={"counts": counts.T.copy()},
    )
    adata.obsm["spatial"] = cd[["x", "y"]].to_numpy(dtype=float)
    return adata


def _frame(sample_ids, cond, cells, n_per, age_col):
    """Concatenate per-sample draws into the colData frame (columns in R's order)."""
    sid = np.repeat(sample_ids, n_per)
    cols = {"sample_id": sid}
    if cond is not None:
        cols["condition"] = np.repeat([cond[s] for s in sample_ids], n_per)
    cols["x"] = np.concatenate([c[0] for c in cells])
    cols["y"] = np.concatenate([c[1] for c in cells])
    cols["cell_type"] = np.concatenate([c[2] for c in cells])
    if age_col:  # sample-level nuisance, constant within sample
        cols["Age"] = np.repeat([c[3] for c in cells], n_per)
    n = len(sid)
    cols["cell_id"] = np.array([f"cell{i}" for i in range(1, n + 1)])
    return pd.DataFrame(cols, index=pd.Index(cols["cell_id"]))


def _finish(rng, cd, counts, gene_names):
    """Add Area (independent per-cell lognormal) and nCount, then build the AnnData."""
    cd["Area"] = rng.lognormal(0.0, 0.25, size=len(cd))
    return _assemble(counts, cd, gene_names)


def simulate_toy(
    n_samples=6,
    n_per=80,
    n_genes=20,
    field=500,
    sd_lib_sample=0.15,
    sd_lib_celltype=0.35,
    beta=2,
    composition=0,
    seed=1,
) -> AnnData:
    """Small synthetic spatial data set with a planted neighbourhood-dependent effect.

    Port of spiDE's internal `.toySPE()`. In the index cell type "A", gene `G1` is
    up-regulated in the "Responder" condition in proportion to the local density of the niche
    cell type "B" (B cells sit at high x, so B-niche density rises with x). The negative
    binomial count model follows splatter's Splat model.

    The generative model is R's, but the random draws are not: numpy's generator differs from
    R's Mersenne-Twister, so the same `seed` gives a different data set from R's. The
    global numpy RNG is not touched. Power to detect the planted effect at the default design
    is low (about 15% of seeds reach q < 0.05, as in R); use more samples and cells for a
    reliable detection.

    Parameters
    ----------
    n_samples : int
        Number of samples (alternately Responder, Non-responder; ids `S1`, `S2`, ...).
    n_per : int
        Cells per sample.
    n_genes : int
        Number of genes (`G1`, `G2`, ...).
    field : float
        Side length of the square spatial field.
    sd_lib_sample, sd_lib_celltype : float
        Log-scale SDs of the sample-level and cell-type-level library-size terms.
    beta : float
        Maximum log-fold change of the planted G1 effect, planted as `beta * x / field`.
        The response is non-monotonic in `beta`; the default is the peak.
    composition : float
        Size of a planted between-sample composition confound on G2 in index type A
        (constant within sample, so the true within-sample niche slope is zero). `0` plants
        none and draws no extra random numbers.
    seed : int
        Seed of the `numpy.random.default_rng` used for every draw.

    Returns
    -------
    anndata.AnnData
        Cells x genes counts in `X` and `layers["counts"]`, `obs` columns `sample_id`,
        `condition`, `x`, `y`, `cell_type`, `Age`, `cell_id`, `Area`, `nCount`
        and `obsm["spatial"]`.
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(1, n_genes + 1)]
    sids = [f"S{i}" for i in range(1, n_samples + 1)]
    cond = {s: ("Responder" if i % 2 == 0 else "Non-responder") for i, s in enumerate(sids)}

    p_b = {s: 0.7 for s in sids}
    a_shift = {s: 0.0 for s in sids}
    if composition != 0:  # wide spread of B prevalence; A cells shifted toward/away from B region
        p_b = dict(zip(sids, rng.uniform(0.3, 0.9, n_samples)))
        a_shift = dict(zip(sids, rng.uniform(0.0, 0.85, n_samples)))

    cells = [_sample_cells(rng, n_per, field, p_b[s], a_shift[s], age=True) for s in sids]
    cd = _frame(sids, cond, cells, n_per, True)
    n = len(cd)

    gmean, size = _sim_gene_params(
        rng, gene_names, bcv_disp=0.6, mean_min=0.5, boost=["G1"], boost_gmean=8
    )
    is_a = (cd["cell_type"] == "A").to_numpy()
    is_resp = (cd["condition"] == "Responder").to_numpy()
    log_effect = np.zeros((n_genes, n))
    log_effect[0] = beta * is_a * is_resp * (cd["x"].to_numpy() / field)
    if composition != 0 and n_genes >= 2:
        shift = np.array([a_shift[s] for s in cd["sample_id"]])
        log_effect[1] = composition * is_a * is_resp * (shift - 0.3)

    lib = _sim_lib_size(rng, cd["sample_id"], cd["cell_type"], sd_lib_sample, sd_lib_celltype)
    counts = _sim_counts(rng, gmean, size, cd["cell_type"].to_numpy(), log_effect, lib)
    return _finish(rng, cd, counts, gene_names)


def simulate_niche(
    n_samples=6,
    n_per=80,
    n_genes=20,
    field=500,
    sd_lib_sample=0.15,
    sd_lib_celltype=0.35,
    beta=2.5,
    seed=1,
) -> AnnData:
    """Synthetic spatial data set with a condition-free neighbourhood effect.

    Port of spiDE's internal `.toyNiche()` (dropped from R in c8db60a; kept here). Like `simulate_toy`, but gene `G1` in
    index cell type "A" rises with the local density of niche cell type "B" in every cell, and
    there is no `condition` column, so it can only be analysed in the condition-free mode.
    The generative model is R's; the random draws are not (numpy vs Mersenne-Twister), and the
    global numpy RNG is not touched.

    Parameters
    ----------
    n_samples, n_per, n_genes, field, sd_lib_sample, sd_lib_celltype, seed
        As in `simulate_toy`.
    beta : float
        Effect size of the planted B-niche association on G1 in A cells.

    Returns
    -------
    anndata.AnnData
        As `simulate_toy` but `obs` has no `condition` column.
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(1, n_genes + 1)]
    sids = [f"S{i}" for i in range(1, n_samples + 1)]
    cells = [_sample_cells(rng, n_per, field, 0.7, 0.0, age=True) for s in sids]
    cd = _frame(sids, None, cells, n_per, True)
    n = len(cd)
    gmean, size = _sim_gene_params(
        rng, gene_names, bcv_disp=0.6, mean_min=0.5, boost=["G1"], boost_gmean=8
    )
    log_effect = np.zeros((n_genes, n))
    log_effect[0] = beta * (cd["cell_type"] == "A").to_numpy() * (cd["x"].to_numpy() / field)
    lib = _sim_lib_size(rng, cd["sample_id"], cd["cell_type"], sd_lib_sample, sd_lib_celltype)
    counts = _sim_counts(rng, gmean, size, cd["cell_type"].to_numpy(), log_effect, lib)
    return _finish(rng, cd, counts, gene_names)


def simulate_clustered(
    n_samples=8,
    n_per=80,
    n_genes=30,
    field=500,
    sd_lib_sample=0.15,
    sd_lib_celltype=0.35,
    sd_patient=0.7,
    sd_nested=0,
    seed=1,
) -> AnnData:
    """Synthetic null with patient-level clustering.

    Port of spiDE's internal `.toyClustered()` (dropped from R in c8db60a; kept here). A per-(gene, sample) intercept is shared by
    every cell of a sample and there is no niche or condition effect; condition is a
    sample-level label, so a cell-level test is anti-conservative while a patient-level test
    stays calibrated. The generative model is R's; the random draws are not (numpy vs
    Mersenne-Twister), and the global numpy RNG is not touched. `obs` has no `Age`.

    Parameters
    ----------
    n_samples, n_per, n_genes, field, sd_lib_sample, sd_lib_celltype, seed
        As in `simulate_toy`.
    sd_patient : float
        Planted between-sample intercept SD.
    sd_nested : float
        SD of an optional per-(gene, sample, cell type) intercept on top (drawn only if > 0).

    Returns
    -------
    anndata.AnnData
        As `simulate_toy` but `obs` has no `Age` column.
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(1, n_genes + 1)]
    sids = [f"S{i}" for i in range(1, n_samples + 1)]
    cond = {s: ("Responder" if i % 2 == 0 else "Non-responder") for i, s in enumerate(sids)}
    cells = [_sample_cells(rng, n_per, field, 0.7, 0.0, age=False) for s in sids]
    cd = _frame(sids, cond, cells, n_per, False)
    gmean, size = _sim_gene_params(
        rng, gene_names, mean_shape=8, mean_rate=1.5, bcv_disp=0.3, out_prob=0, mean_min=1
    )
    u = rng.normal(0.0, sd_patient, size=(n_genes, n_samples))
    log_effect = u[:, np.searchsorted(sorted(sids), cd["sample_id"].to_numpy())]
    if sd_nested > 0:  # per-(gene, sample, cell type) intercept; absorbed by per-patient fits
        key = (cd["sample_id"] + ":" + cd["cell_type"]).to_numpy()
        keys = sorted(set(key))
        v = rng.normal(0.0, sd_nested, size=(n_genes, len(keys)))
        log_effect = log_effect + v[:, np.searchsorted(keys, key)]
    lib = _sim_lib_size(rng, cd["sample_id"], cd["cell_type"], sd_lib_sample, sd_lib_celltype)
    counts = _sim_counts(rng, gmean, size, cd["cell_type"].to_numpy(), log_effect, lib)
    return _finish(rng, cd, counts, gene_names)
