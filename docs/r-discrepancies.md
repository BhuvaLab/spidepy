# Differences from R spiDE

One bullet per item: what R does, what spidepy does, and why. Reference: spiDE 0.99.37 (`c8db60a`). Where R has a bug or an R-specific accident, spidepy matches R and logs it here. Items are grouped by area.

## Interface

- **AnnData instead of SpatialExperiment.** Counts are read from `layers[assay]` (or `X`), coordinates from `obsm["spatial"]`, niche matrices live in `obsm["Niche<sigma>"]` instead of `reducedDim`, and only `_io.py` transposes to R's genes x cells. `build_niches` and `merge_niches` work in place unless `copy=True` (R returns a modified copy).
- **`n_jobs` replaces `BPPARAM`.** An integer; the result is identical for any value.
- **Result tables keep R's column names.** `p.heterogeneity` and `q.global` are not renamed to snake_case, so tables from the two packages line up column for column.
- **`spide` does not keep the niches it builds.** R's `spiDE()` returns the results and leaves the
  caller's object untouched; `spide` builds the niche matrix in place (to avoid copying the counts)
  and removes it from `adata.obsm` once the fit is done. A niche matrix already present is used
  and kept.
- **`spiGSEA` argument matching.** R's `match.arg` accepts unambiguous prefixes (`type = "self"`,
  `test = "pool"`). `spi_gsea` takes exact values only, as `test_spide`
  does; an unknown value raises `ValueError`. Also `spi_gsea` refuses anything but a
  `SpiDEResults` with a `TypeError` (R dispatches and errors on the missing method).
- **`spiGSEA` direction of a missing statistic.** R writes `NA_character_`; `spi_gsea` uses
  `None` in the `direction` column (object dtype).
- **Result and fit objects.** R stores `character(0)` where Python has `None`: `SpiDEFit.condition` when no
  condition was given; `SpiDEResults.condition` is `None` when no condition was tested, and `contrast` is `""` then.
- **Legacy-object guard.** R's `checkResults` also rejects pre-0.99.23 mixed-model objects
  (`.assertCurrent`). Not ported: spidepy has no such objects.
- **Strata that vary within a patient.** R's `fitSpiDE` reads `as.character(patients[[strata]])`;
  a strata column that is not constant within patients is absent from the patient table, so R
  fails with the obscure "'names' attribute [n] must be the same length as the vector [0]".
  `fit_spide` raises `ValueError("strata column '...' is not a patient-level column")` (same for
  a condition, as R's `.conditionCoding`). Same inputs fail in both; only the message differs.

## Ordering and sorting

- **Sorting of character labels.** R sorts character labels (patients, cell types, conditions)
  in the session locale (`en_US.UTF-8`, which interleaves `a A b B` and ignores punctuation at
  first pass). `spidepy._rcompat.r_sort_str` (and Python `sorted`) uses code-point order
  (R's `method = "radix"` / `LC_COLLATE=C`), so mixed-case or punctuated labels can order
  differently (for example `b, C` in R but `C, b` here). Labels that share case and punctuation
  style, such as `S1, S10, S2` or `Non-responder, Responder`, order identically. Code-point
  order is deterministic and independent of the user's locale, and it matches the reference
  fixtures; it is the documented choice rather than emulating glibc collation.
  Likewise the rows of `patient_slopes` / `patient_intercepts`: R sorts rows with `order()` on
  character keys (locale collation); Python sorts in code-point order (see "Sorting of
  character labels"). Identical for labels such as `G1, G10, G2`, `S1, S10, S2`.
- **Niche bandwidth lookup.** R finds niche reducedDims with the regex `^Niche[0-9.]+$`, which
  misses a bandwidth that `paste0` writes in exponent form (`Niche1e+05`). `_io.niche_keys`
  uses the same pattern, so such a niche is not found either. Matched to R because bandwidths
  of 1e5 or more are not meaningful in spatial units; `niche_key` still writes the R string.

## Niches

- **Niche kernel density.** R `densityfun` returns the values of in-window queries only
  (`drop = TRUE`) and has `diggle = FALSE` as its own default. `_kde.density_at` keeps the
  query order, returns `nan` for a query outside the window and defaults to `diggle = True`
  (the value `buildNiches` passes). The numbers for in-window queries are identical: spatstat
  uses the exact `pnorm` edge weight for a rectangular window, not a pixel image.
- **`merge_niches` group named like a leftover column.** If a group name equals a column that
  is not in any group (for example `{"B": ["A", "C"]}` with a column `B`), R produces two
  columns named `B` and its `[[` lookup of the group map sees the first. We keep both columns
  and record the first in `uns["spiDE_niche_groups"]` (a dict cannot hold duplicate keys).
- **`merge_niches` key prefix.** With `sigma = None`, R greps `^Niche` over `reducedDimNames`,
  so any key beginning with `name` (for example `NicheFoo`) is merged and given a group map,
  not only `Niche<number>` keys. `merge_niches` does the same over `obsm` keys.

## Fit: design and slopes

- **Constant-column test in `_niche_columns`.** R's `var` accumulates in long double, so a niche
  column of identical values has variance exactly 0 and is dropped. `_design._colvar` does the
  same (long double, two-pass mean); a plain double mean can leave about 1e-33 and keep the column.
- **`_partial_block` singularity.** R's `solve` fails when the reciprocal condition number of
  `Z'WZ` (LAPACK `dgecon` estimate) is below machine epsilon, and the block is then skipped
  (`NULL`). We compute the exact 1-norm reciprocal condition number from the inverse and apply
  the same threshold; the two can disagree only for blocks within a few percent of that threshold.
- **`r2` of the depth blocks.** R uses `lm.fit` (pivoted QR, tolerance 1e-7) for the R^2 of a
  niche column on a patient's block; we use `numpy.linalg.lstsq`. They agree whenever the block
  has full column rank (it does for the sections whose df rule applies); a numerically
  rank-deficient block could drop different directions.

## limma and empirical-Bayes moderation

- **`fitFDistUnequalDF1` with exactly two informative genes.** R sets `prior.weights <- NULL`
  after having set its `PriorWeights` flag, so when NA or tiny-df genes are also present the
  weighted branch runs on NULL weights: the prior scale is `0/0 = NaN` and the objective is
  identically 0, so `optimize` returns the value of a flat search. Matched (`scale = nan`, same
  `df2`); the posterior variances are then all `nan`, as in R.
- **limma `weightedLowess`.** `_limma._weighted_lowess` is a port of `weighted_lowess.c` from
  limma 3.66.0 (Bioconductor RELEASE_3_22). The window search is vectorised (other points sorted
  by distance, right side first on ties, then a cumulative sum in the same order of additions)
  instead of growing the window point by point; results agree with R to rounding, including on
  tied x. `rsort_with_index` ties among equal residuals are broken stably (only affects the
  robustness-iteration median when residuals tie exactly). The equal-weights branch of
  `loessFit` (plain `lowess`) is not ported: it cannot occur from `fitFDistUnequalDF1`.
- **`eBayes` outputs not computed.** `var.prior`, `lods` and the `F` statistics
  (`tmixture.matrix`, `classifyTestsF`) are not read by spiDE and are not returned; `lmFit`'s
  `cat("Coefficients not estimable")` message is not printed (aliased coefficients are `nan`).
- **Unequal-df prior to ~1e-9.** `fitFDistUnequalDF1` finds `df2` by Brent's method (R `optimize`)
  on a sum that R accumulates in long double and numpy in double, so `df.prior` agrees with R to
  about 1e-9 relative, not to machine precision.
- **Robust `fitFDistUnequalDF1` drops `span` in its refit.** R refits with prior weights through
  `Recall(x, df1, covariate, robust = FALSE, prior.weights = FDR)`, which omits `span`, so the
  refit trend always uses the default `chooseLowessSpan(n, small.n = 500)`. We do the same.
- **Robust fits with tied F statistics.** `fitFDistRobustly` ranks the F statistics for
  `prob.outlier`. When a trend interpolates several genes exactly their statistics tie up to
  rounding noise (about 1e-16) and the ranks, hence `prob.outlier` and `df2.shrunk` of those
  genes, depend on the last bits (R and numpy can differ). Not matched beyond rounding.
- **`qf` with `log.p`.** `fitFDistRobustly` with unequal `df1` calls `qf(..., log.p = TRUE)`;
  scipy has no log-scale quantile, so we use `exp(logp)` first. Identical unless the tail
  probability underflows (about 1e-308).
- **`fitFDistRobustly` with two variances** returns `fitFDist`'s result without `df2.shrunk`, as
  R does; reaching that after set-aside genes fails in R (`replacement has length zero`) and
  raises `ValueError` here.
- **`.pooledColumnTest` robust-to-non-robust fallback.** R wraps `eBayes(robust = TRUE)` in
  `tryCatch` and, on failure, retries with `robust = FALSE`. The helper `eb()` reads the free
  variable `fit`, and `fit <- eb(TRUE)` has just overwritten it with `NULL`, so the retry runs
  `eBayes(NULL)`, fails, and the whole column is NA. `_pooled_column_test` reproduces this: a
  failed robust prior leaves every row of the column NA (no retry). Unit-tested with a failing
  `e_bayes`; no reference fixture reaches it.
- **Tie noise in limma's robust prior on small gene families.** With a handful of genes whose
  residuals from the expression trend are exactly zero, their F statistics tie up to rounding
  (1e-16) and `fitFDistRobustly` ranks, hence the shrunk prior df, depend on that noise: a
  1e-15 relative perturbation of the residual variances moves `df.prior` of the tied genes by
  up to 1e-2 in Python, and R's own value is one such draw. On the end-to-end slopes fixture
  (`toyspe_s16_n150_g12_seed7`, index A, niche B, 12 genes) the pooled se/t/df/p differ from R
  by up to 1e-5 / 1e-3 / 6e-4 although every input to the test is identical to 1e-14. Not fixable
  (R's draw depends on its BLAS rounding); tests compare those rows at 2e-2 (0.53% was seen between numpy / BLAS builds).

## Toy data

- **Toy-data simulators.** R's `.toySPE()`, `.toyNiche()` and `.toyClustered()` seed R's
  Mersenne-Twister; `simulate_toy()`, `simulate_niche()` and `simulate_clustered()` use the same
  generative model with `numpy.random.default_rng(seed)`, so a given `seed` gives a different
  data set from R's (R's draws cannot be reproduced). The planted effect is the same but its
  statistical power is draw-dependent (R's seed 7 is a favourable draw; here seed 4 is for the
  same design). Neither touches the global RNG. They return AnnData (the layout `load_spe` builds
  from R's SpatialExperiment) with plain-string `sample_id`, `condition` and `cell_type` and float
  counts. `toy_spide()` is R's own `toySpiDE` object, not a simulation; its text columns are
  returned as strings rather than the categories stored in the bundled file.
- **`.toyNiche()` and `.toyClustered()` are gone from R.** R dropped the originals of these two
  internal simulators in c8db60a (the simplification pass). spidepy keeps their ports,
  `simulate_niche()` and `simulate_clustered()`, as public toy-data helpers; they no longer have an
  R original to be compared with, and `simulate_toy()` (R's `.toySPE()`) is still the one with a
  counterpart.

## Plots

- **Plots return matplotlib figures, not ggplot objects.** `plot_niche_map` and
  `plot_niche_response` return a `Figure`; R's `p$data` and layer data are available from the
  private builders `_niche_map_data` and `_niche_response_data` (a dict: `data`, and `summary` and
  `lines` or `centroids`), which keep R's column names, row order and plotmath `gene_lab` strings.
  Panels are boxed titles (not full-width strips), legends are two stacked legends placed by hand,
  and `scale_y_log10` labels use a comma for thousands via a formatter; pixel identity is not a goal.
- **R's `message()` in `plotNicheResponse`.** "G2 not fitted in A, left out" is an unconditional
  `message()` in R; `plot_niche_response` emits it as a `UserWarning` so callers can catch it.
- **`.fmtq` pads `q = 1`.** `formatC(1, digits = 2, format = "g")` returns `"  1"` (padded to three
  characters); `_fmt_q` reproduces it.
- **Plotted counts are dense.** `_index_cell_data` returns the plotted genes' counts as a dense
  genes x cells array (R keeps a sparse subset); only the requested genes are ever extracted.
- **Italic gene labels.** R plotmath (`italic("G1")`) becomes mathtext (`$\mathit{G1}$`) with each
  non-alphanumeric character set in `\text{}` (a bare `-` would render as a spaced minus sign), so
  names with `_ - " \ $ { } % # & ^` print as written.
- **`plotPValues` row order follows `merge()`'s unstable tie order.** R's `merge(tab, bands, by =
  c("gene", "index"))` sorts the matches of each key with an unstable shell sort (`isort_with_index`),
  so the rows of one (gene, index) are not in table order. `_isort_with_index` reproduces that
  sort, so `p$data` has R's row order; the histograms do not depend on it.
- **Clustered heatmap rows.** `hclust(dist(m))$order` is reproduced from scipy's complete-linkage
  merges (singleton before cluster, lower-numbered cluster first, leaves read left to right);
  only exact distance ties could differ.
- **Spillover labels.** R places them with `ggrepel` (`seed = 1`); matplotlib has no equivalent and
  `ggrepel` is optional in R. Labels are right-aligned at `log2(fold) - 0.4`, pushed apart
  vertically, with a segment to the point. Which markers are labelled (largest `|z|`) is R's.
- **`marker` with a missing fold.** R's `marker` is `NA` where the fold is `NA`; here it is `False`
  (the fold of a gene is only `NA` when the gene is missing from the object, which is an error).
- **Principal-component signs.** `prcomp` and numpy's SVD may flip a component; only magnitudes
  (and adjusted R^2, which is sign invariant) are compared with R.
- **The call graph's edges.** `geom_curve(curvature = 0.22)` becomes a matplotlib `arc3` arrow;
  the drawn curvature and arrow heads are approximate, the edge data are R's.
