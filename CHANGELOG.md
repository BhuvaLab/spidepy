# Changelog

## 0.1.0 (unreleased)

First release. Ports R spiDE 0.99.37 (commit `e31e744`).

- Workflow: `build_niches`, `merge_niches`, `fit_spide` (slopes and sandwich engines), `test_spide`
  (pooled test and condition test), `spide` (all three steps).
- Results and accessors: `results`, `patient_slopes`, `patient_intercepts`, `SpiDEFit`, `SpiDEResults`.
- Related tests: `spi_gsea` (gene sets) and `test_niche_abundance` (between-patient association).
- Plots in `spidepy.pl` (matplotlib): niche map, niche response, patient slopes, patient effects, call map,
  top triplets, spillover, triplet heatmap, p-value bands.
- Data: `toy_spide()` (R's own example data) and numpy-seeded simulators in `spidepy.datasets`.
- Numerical parity with R is tested against reference outputs from the R package (limma moderation,
  clubSandwich-style CR2 sandwich, kernel density, GLM fits). Intended differences are in
  `docs/r-discrepancies.md`.
- Documentation: `docs/tutorial.ipynb` (executed walk-through of the R vignette), `docs/r-to-python.md`,
  `docs/api.md`.
