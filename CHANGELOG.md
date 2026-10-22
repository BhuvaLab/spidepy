# Changelog

## Unreleased

Syncs spidepy to R spiDE commit `c8db60a` (still version 0.99.37; the previous pin was `e31e744`).
The R side is the 2026-10-22 simplification pass plus the palette "V" plots.

- Removed `pooled_df` (`pooled.df = "capped"` is gone from R): the argument of `test_spide`, `spide` and
  `spi_gsea`, the `SpiDEResults.pooled_df` field and the "(... df)" suffix of its `repr`. The pooled test always
  uses limma's df scaled by the Kish effective share of the patients; `spi_gsea` follows it.
- `spi_gsea` rows are ordered by `round(p, 12)`, then index, niche and set name (code-point order), so ties in
  p (complementary sets) no longer depend on floating-point noise.
- `test_niche_abundance` no longer tests an index type against its own niche under merged niches (the same
  self-niche rule as `fit_spide`), and its `index` and `niche` columns keep the original labels (`"T cell"`, not
  `"T.cell"`).
- `spide(name="Niche")`: the `obsm` prefix the niches are read from, or built into, and passed to `fit_spide`.
- Palette V in `spidepy.pl`: gold / violet direction, dark violet / pink conditions, magma magnitude ramp
  (starting near white), a new `"spatial"` role of `spide_colours`, the dark panel of `plot_niche_map`, and the
  small layout changes of the patient-slopes, spillover and p-value plots.
- Internal: `fit.index[k]["ncells"]` comes from the shared patient levels for both engines; `_is_self_niche` takes
  a single index type; `_working_wr` and `_patient_slopes` no longer return `mu` and `ncells`.
- Kept: `datasets.simulate_niche` and `simulate_clustered` (R dropped `.toyNiche()` and `.toyClustered()`; see
  `docs/r-discrepancies.md`).
- Reference fixtures regenerated against `c8db60a`; `tests/ref/data/NAMESPACE` updated.
- `docs/tutorial.ipynb` follows the order of R's new "Getting started" article and is re-executed.

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
