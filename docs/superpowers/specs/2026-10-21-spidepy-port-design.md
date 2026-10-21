# spidepy: a Python / scverse port of spiDE — design

Date: 2026-10-21. Status: approved by the user (scope, dependency, repo and model split chosen 2026-10-21).

## Goal

A faithful Python port of the R package **spiDE 0.99.37** (repo `/scratch/project_mnt/S0249/R_projects/spiDE`,
commit `e31e744`; the package files are unchanged since `7fc6979`). Same statistics, same defaults, same
numbers as R to tight tolerance, on `AnnData` input. Modelled on the earlier spanormpy port of SpaNorm
(`/scratch/project_mnt/S0249/R_projects/spanormpy`), whose conventions, reference harness and process we reuse.

A per-function inventory of spiDE (every signature, external call and porting hazard) is in
`docs/superpowers/specs/spide-0.99.37-inventory.md`. It is a reading aid; the R source is the authority.

## Scope (full port)

Everything spiDE exports:

| R | Python |
|---|---|
| `buildNiches` | `spidepy.build_niches` |
| `mergeNiches` | `spidepy.merge_niches` |
| `fitSpiDE` | `spidepy.fit_spide` -> `SpiDEFit` |
| `testSpiDE` | `spidepy.test_spide` -> `SpiDEResults` (also `SpiDEFit.test(...)`) |
| `spiDE` | `spidepy.spide` |
| `results` | `SpiDEResults.results(test=None, fdr=None)` and `spidepy.results(res, ...)` |
| `patientSlopes`, `patientIntercepts` | `spidepy.patient_slopes`, `spidepy.patient_intercepts` (accept fit or results) |
| `testNicheAbundance` | `spidepy.test_niche_abundance` |
| `spiGSEA` | `spidepy.spi_gsea` |
| `toySpiDE` (data) | `spidepy.datasets.toy_spide()` (bundled `.h5ad` exported from R) |
| internal simulators `.toySPE`, `.toyNiche`, `.toyClustered` | `spidepy.datasets.simulate_toy`, `simulate_niche`, `simulate_clustered` (numpy RNG; same model, not the same draws) |
| `plotNicheMap`, `plotNicheResponse`, `plotSpillover`, `plotPatientSlopes`, `plotPatientEffects`, `plotPValues`, `plotTopTriplets`, `plotTripletHeatmap`, `plotCallMap` | `spidepy.pl.plot_niche_map`, ... (matplotlib, return a `Figure`) |
| `theme_spiDE`, `spiDEColours` | `spidepy.pl.style` (rcParams context) and `spidepy.pl.spide_colours()` |

Not ported: the legacy-object refusal (`.isLegacy`/`.assertCurrent`, only meaningful for R `.rds`
files), S4 `show` beyond a readable `__repr__`, BiocParallel specifics.

## Dependencies

- `spanormpy` (sibling repo, installed editable) supplies the NB solver and the bases spiDE takes from
  SpaNorm: `spanormpy.polish_nb` (R `polishNB`), `spanormpy._block_design.nb_block_design`
  (`nbBlockDesign`), `spanormpy._tps.tps_basis` (`tpsBasis`) and its `splines::ns` port. If something
  spiDE needs is missing or private there, the task adds a thin public wrapper in spanormpy on its own
  branch and reports it; it never forks solver code into spidepy.
- numpy, scipy, pandas, anndata, matplotlib. No rpy2 at runtime. R is used only to generate test
  references.
- limma (lmFit with weights, eBayes with `trend` and `robust`, squeezeVar, fitFDist, fitFDistRobustly)
  and spatstat's `densityfun(edge = TRUE, diggle = TRUE)` have no faithful Python equivalent and are
  ported (`_limma.py`, `_kde.py`).

## Data model

- Input: `AnnData`, cells x genes. Counts in `X` or `layers[assay]` (`assay="counts"` mirrors R: if
  `"counts"` is not a layer, `X` is used; document this). Coordinates `obsm["spatial"]` (first two
  columns, as float). Cell metadata columns in `obs` (`cell_type="cell_type"`, `sample_id="sample_id"`).
- Niches: `obsm[f"{name}{sigma_str}"]` as a `pandas.DataFrame` (index = `obs_names`, columns = cell
  types), where `sigma_str` reproduces R's `paste0` number formatting (`10 -> "10"`, `2.5 -> "2.5"`,
  `1e5 -> "1e+05"`). Merge groups: `uns["spiDE_niche_groups"][rd_name] = {merged: [members]}`.
- Internal orientation follows R: genes x cells for counts inside the fit code (as spanormpy).
- `SpiDEFit` and `SpiDEResults` are dataclasses holding the same fields as the R slots (`engine`,
  `sigma`, `condition`, `patients` DataFrame, `index` dict of per-index dicts with numpy arrays and their
  label vectors, `params`; results add `table` DataFrame, `contrast`, `procedure`, `fdr`, `pooled_df`,
  `fit`). They pickle; `SpiDEResults.table` has R's columns in R's order.
- R argument names become snake_case with R's defaults (`min.cells -> min_cells`,
  `pooled.df -> pooled_df`). `BPPARAM -> n_jobs: int = 1`. Patient and condition ordering follow R's
  `sort()`/`factor()` (lexicographic, so `S1, S10, S2`); the condition contrast is second level minus first.

## Accuracy contract

- References come from R: spiDE `e31e744` and SpaNorm `f233b7b` (1.7.16, what spanormpy ports) installed
  into `tests/ref/Rlib` (git-ignored), written as HDF5 by `tests/ref/make_<topic>.R`, read by
  `tests/_ref.py::load_ref` (same harness as spanormpy).
- Tolerances: closed-form steps `rtol=1e-10, atol=1e-12`; anything downstream of an iterative fit given
  identical inputs `rtol=1e-6`; p-values compared on the log scale where tiny. Any looser tolerance carries
  a comment justifying it. Discrete outputs (which genes are fitted/dropped, `in_family`, status counts)
  must match exactly.
- Each component is tested against R on its own inputs (so an upstream difference cannot hide a
  downstream one), and end to end on `toySpiDE` for both engines and every depth mode.
- Where R has a bug or an R-specific accident (e.g. `.pickSigma` not recognising `"Niche1e+05"`), match R
  and log it in `docs/r-discrepancies.md`; deliberate deviations are also logged there.
- The ported R testthat assertions (behaviour, errors, warnings) come along with each task, skipping
  S4-, SPE-class- and ggplot-object-specific ones.

## Repository

`/scratch/project_mnt/S0249/R_projects/spidepy`, package `src/spidepy`, hatchling build, ruff config and
CI (`ruff` + `pytest` on 3.11/3.12) as spanormpy. GPL-3.0-or-later. Branch `port/spide`; GitHub remote
`BhuvaLab/spidepy` (push the branch after each task once the remote exists). Commits authored as
`Dharmesh D Bhuva <30132854+bhuvad@users.noreply.github.com>` (repo-local config) with a Claude
`Co-Authored-By` trailer.

## Process

Opus wrote this spec and the plan. Each task is run by a Sonnet (medium effort) implementer and checked
by a Sonnet reviewer; Opus only triages, rules on conflicts and steps in when a Sonnet agent is stuck.
