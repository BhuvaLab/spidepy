# spidepy

[![CI](https://github.com/BhuvaLab/spidepy/actions/workflows/ci.yaml/badge.svg)](https://github.com/BhuvaLab/spidepy/actions/workflows/ci.yaml)

A Python port of the R package [spiDE](https://github.com/BhuvaLab/spiDE): context-specific,
neighbourhood-dependent differential expression for spatial transcriptomics. Within an index cell type,
spidepy asks whether a gene's expression changes with the local density (the niche) of another cell type,
and whether that dependence differs between two conditions. The patient is the unit of replication. It
operates on `AnnData`.

Version 0.1.0 (alpha). Ports spiDE 0.99.37 (R commit `e31e744`); see [CHANGELOG.md](CHANGELOG.md).

## Install

spidepy is not on PyPI or conda yet. It needs [spanormpy](https://github.com/BhuvaLab/spanormpy) (the
port of SpaNorm, whose negative binomial polishing it reuses):

```bash
pip install git+https://github.com/BhuvaLab/spanormpy
pip install git+https://github.com/BhuvaLab/spidepy
```

Python 3.11 or newer. For the tests and tutorial: `pip install "spidepy[dev]"` (add `jupyter` to run the
notebook).

## Quickstart

```python
import spidepy as sp

adata = sp.toy_spide()             # 16 patients, 3 cell types, 20 genes; raw counts in layers["counts"]

# one call: niches -> fit -> pooled test and condition test
res = sp.spide(adata, condition="condition", sigma=30)
sp.results(res, test="pooled").head()
sp.results(res, test="condition").query("in_family")

# or step by step
sp.build_niches(adata, sigma=30)                                   # adata.obsm["Niche30"]
fit = sp.fit_spide(adata, condition="condition", sigma=30)         # SpiDEFit (slopes engine)
res = sp.test_spide(fit)                                           # SpiDEResults
sp.patient_slopes(fit, gene="G1", index="A", niche="B")            # each patient's own slope

sp.pl.plot_patient_slopes(res, gene="G1", index="A", niche="B")
sp.spi_gsea(res, {"planted": ["G1", "G2", "G3"]}, min_size=3)    # gene sets
sp.test_niche_abundance(adata, condition="condition", sigma=30)    # between-patient association
```

Your data needs raw integer counts (in `layers["counts"]`), a `cell_type` and a `sample_id` column in `obs`,
and coordinates in `obsm["spatial"]`. The condition must be a two-level patient-level column. A full
walk-through, with outputs and figures, is in [docs/tutorial.ipynb](docs/tutorial.ipynb) (the R vignette
on the same example). Every R export and its Python name are listed in
[docs/r-to-python.md](docs/r-to-python.md); the API is in [docs/api.md](docs/api.md).

## Differences from R

The full list, with reasons, is in [docs/r-discrepancies.md](docs/r-discrepancies.md). The ones you will notice:

- **AnnData, not SpatialExperiment.** Counts in `layers["counts"]`, niche matrices in
  `obsm["Niche<sigma>"]`. `build_niches` and `merge_niches` work in place (`copy=True` to keep the input).
- **`n_jobs` replaces `BPPARAM`**; results are identical for any value.
- **Plots are matplotlib figures**, not ggplot objects. The data behind each plot is the same, but the
  layout is not pixel-identical.
- **Toy data.** `toy_spide()` is R's own example data. `simulate_toy()` and friends use numpy's generator, so
  a given seed gives a different data set from R's.
- **Label sorting.** Patients, conditions and cell types are sorted in code-point order, not the session
  locale, so labels that mix case or punctuation can order differently from R.
- **Column names** of the result tables keep R's spelling (`p.heterogeneity`, `q.global`).

Numbers agree with R to numerical closeness in the test suite; where a limma prior is decided by rounding
noise on very small gene families the page says so.

## Citation

Cite the R package, which defines the method:

Bhuva DD, Tan CW. spiDE: context-specific, neighbourhood-dependent differential expression. R package,
<https://github.com/bhuvad/spiDE>.

## Licence

GPL-3.0-or-later, as for spiDE.
