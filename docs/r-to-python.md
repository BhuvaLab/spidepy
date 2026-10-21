# R to Python mapping

Every export of R spiDE 0.99.37 (commit `e31e744`, `NAMESPACE`) and its spidepy counterpart. The
`NAMESPACE` also exports the `$` method of the result classes; in Python that is plain attribute access
(`res.contrast`, `fit.engine`).

Input is an `AnnData` instead of a `SpatialExperiment`: counts live in `layers["counts"]` (`assay=`
names another layer, `"X"` uses `adata.X`), cell coordinates in `obsm["spatial"]`, and niche matrices in
`obsm["Niche<sigma>"]` instead of `reducedDim`. The first argument `spe` is called `adata`. Functions
that R runs on a copy (`buildNiches`, `mergeNiches`) work in place and return the object; pass
`copy=True` to leave the input alone.

## Argument names

Arguments are snake_case with R's defaults.

| R | Python |
|---|---|
| `spe` | `adata` |
| `object` | `res` or `fit` (the result or fit object) |
| `min.cells`, `min.patients`, `min.detect`, `min.tiles` | `min_cells`, `min_patients`, `min_detect`, `min_tiles` |
| `min.size`, `max.size` | `min_size`, `max_size` |
| `pooled.df` | `pooled_df` |
| `prior.count`, `max.leverage` | `prior_count`, `max_leverage` |
| `min.calls`, `colour.by`, `n.pcs`, `point.size`, `scale.bar` | `min_calls`, `colour_by`, `n_pcs`, `point_size`, `scale_bar` |
| `BPPARAM` | `n_jobs` (an integer; results are identical for any value) |
| `verbose` | `verbose` (messages go to stderr, only when `True`) |

Column names of the result tables keep R's spelling (`p.heterogeneity`, `q.global`), so a table from
Python reads like one from R. The differences between the two packages, with reasons, are listed in
[r-discrepancies](r-discrepancies.md) and summarised in the README.

## Exports

| R export | Python | Notes |
|---|---|---|
| `buildNiches` | `spidepy.build_niches` | writes `obsm["Niche<sigma>"]`; in place unless `copy=True` |
| `fitSpiDE` | `spidepy.fit_spide` | returns a `SpiDEFit`; `engine="slopes"` or `"sandwich"` |
| `mergeNiches` | `spidepy.merge_niches` | `groups` is a dict of name to list of cell types; group map in `uns["spiDE_niche_groups"]` |
| `patientIntercepts` | `spidepy.patient_intercepts` | returns a `DataFrame` |
| `patientSlopes` | `spidepy.patient_slopes` | returns a `DataFrame`; slopes engine only |
| `plotCallMap` | `spidepy.pl.plot_call_map` | matplotlib; returns a `Figure` |
| `plotNicheMap` | `spidepy.pl.plot_niche_map` | matplotlib; returns a `Figure` |
| `plotNicheResponse` | `spidepy.pl.plot_niche_response` | matplotlib; returns a `Figure` |
| `plotPValues` | `spidepy.pl.plot_p_values` | matplotlib; returns a `Figure` |
| `plotPatientEffects` | `spidepy.pl.plot_patient_effects` | matplotlib; returns a `Figure` |
| `plotPatientSlopes` | `spidepy.pl.plot_patient_slopes` | matplotlib; returns a `Figure` |
| `plotSpillover` | `spidepy.pl.plot_spillover` | matplotlib; returns a `Figure` |
| `plotTopTriplets` | `spidepy.pl.plot_top_triplets` | matplotlib; returns a `Figure` |
| `plotTripletHeatmap` | `spidepy.pl.plot_triplet_heatmap` | matplotlib; returns a `Figure` |
| `results` | `spidepy.results` | returns a `DataFrame` |
| `spiDE` | `spidepy.spide` | extra fit options pass through `**fit_kwargs`; the niche matrix it builds is removed again |
| `spiDEColours` | `spidepy.pl.spide_colours` | the same colour-blind-safe palettes |
| `spiGSEA` | `spidepy.spi_gsea` | `genesets` is a dict of set name to gene list; exact argument values only |
| `testNicheAbundance` | `spidepy.test_niche_abundance` | returns a `DataFrame` |
| `testSpiDE` | `spidepy.test_spide` | returns a `SpiDEResults` |
| `theme_spiDE` | `spidepy.pl.style` | a context manager (`with spidepy.pl.style(): ...`) instead of a ggplot theme |
| `SpiDEFit` | `spidepy.SpiDEFit` | exported class (a dataclass); `$` is attribute access |
| `SpiDEResults` | `spidepy.SpiDEResults` | exported class (a dataclass); `$` is attribute access |
| `toySpiDE` | `spidepy.toy_spide` | R's own example data as an `AnnData`; `spidepy.datasets.simulate_toy`, `simulate_niche` and `simulate_clustered` are numpy-seeded simulators |
