# spidepy Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A faithful Python / scverse port of spiDE 0.99.37 (`e31e744`) that matches R numerically.

**Architecture:** Internal functions are line-by-line ports of spiDE's R helpers (same names in snake_case with a leading underscore: `.indexDesign` -> `_index_design`), working on numpy/scipy arrays in R's orientation (counts genes x cells). The NB solver, block design and TPS basis come from `spanormpy`. limma's moderated statistics and spatstat's edge-corrected Gaussian KDE are ported into spidepy. A thin glue layer reads `AnnData`. Every numerical task is gated by a parity test against references that R writes from spiDE `e31e744` + SpaNorm `f233b7b`.

**Tech Stack:** Python >= 3.11, numpy, scipy, pandas, anndata, matplotlib, spanormpy (editable sibling), h5py + pytest (tests). R 4.5 (`latest-r` conda env) with rhdf5 for references.

**Spec:** `docs/superpowers/specs/2026-10-21-spidepy-port-design.md` (binding: scope, data model, tolerances). **Inventory:** `docs/superpowers/specs/spide-0.99.37-inventory.md` — read the sections for your task's R files before reading the R source.

## Global Constraints

- Source of truth: `git -C /scratch/project_mnt/S0249/R_projects/spiDE show e31e744:<path>` (files under `R/`, `tests/testthat/`, `vignettes/`, `data-raw/`). Never modify the spiDE, SpaNorm or user R libraries. Only Task 1 may touch spanormpy, and only if a needed function is missing (see Task 1).
- SpaNorm R source: `git -C /scratch/project_mnt/S0249/R_projects/SpaNorm show f233b7b:<path>` (1.7.16). spanormpy (`/scratch/project_mnt/S0249/R_projects/spanormpy`, branch `main`) ports it.
- R references: spiDE `e31e744` and SpaNorm `f233b7b` installed into `tests/ref/Rlib/` (git-ignored) with `~/miniconda3/envs/latest-r/bin/R`; scripts `tests/ref/make_<topic>.R` begin `source("tests/ref/_common.R")`, write `tests/ref/data/<topic>.h5` with `write_ref(list, topic)`, and are run from the repo root. Internal R functions via `spiDE:::name`. `tests/_ref.py::load_ref(topic)` returns R shapes (length-1 vectors as scalars; `NA` as `nan`). Reference `.h5` files are committed (keep each under ~5 MB).
- Python env: `/scratch/user/uqdbhuva/envs/spidepy/bin/python` (venv over the spanormpy env with spanormpy and spidepy installed editable; created in Task 1). Tests: `.../bin/python -m pytest`.
- Orientation: counts genes x cells inside the fit code; only `_io.py` transposes from AnnData.
- Ordering: patients, conditions, cell types and genes are ordered exactly as R orders them (`sort()`, `factor()` levels, `unique()` first appearance, `order()`), each as the specific R line does. R `sort` on character uses the C-like collation of the reference strings used in tests; use Python `sorted` and note any locale-dependent case in `docs/r-discrepancies.md`.
- Tolerances: closed-form `rtol=1e-10, atol=1e-12`; downstream of an iterative fit (polish_nb, KDE edge correction if iterative) `rtol=1e-6`; p-values below 1e-300 compared as equal-or-both-tiny. Discrete outputs (genes dropped, `status` counts, `in_family`, NA patterns) match exactly. A looser tolerance needs a comment saying why.
- Naming: R argument names to snake_case with R defaults (`min.cells` -> `min_cells`, `pooled.df` -> `pooled_df`, `max.leverage` -> `max_leverage`); `BPPARAM` -> `n_jobs: int = 1` (joblib or plain loop; results identical for any `n_jobs`). Engine, arm and test names in code, docs and commit messages are the human names (slopes engine, sandwich engine, pooled test, condition test), never study codes.
- Errors: R `stop()` -> `ValueError` (or `TypeError` for wrong types) with the same message text where tests assert it; R `warning()` -> `warnings.warn(..., UserWarning)`; R `message()` under `verbose` -> `print` to stderr only when `verbose=True`.
- Where R has a bug or R-specific accident, match R and log it in `docs/r-discrepancies.md` (create it if absent; one bullet per item: what R does, what we do, why).
- Each task ports the assertions of the R testthat file(s) named on its **R tests** line (behaviour, errors, warnings), skipping S4-class, `.rds`-legacy and ggplot-object-internals checks. The fixture `toySpiDE` and `.toySPE(...)` calls in R tests are replaced by the h5ad fixtures Task 1 exports (never by Python simulations, whose draws differ).
- Commits: author/committer come from repo-local git config (already set: Dharmesh D Bhuva <30132854+bhuvad@users.noreply.github.com>); each message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Stage only your task's files (`git add <paths>`); never `git add -A`; never commit `.superpowers/`. Do not push (the controller pushes).
- Style: ruff (line length 100, rules `E4,E7,E9,F,I`, ignore `E731,E741`), numpy-style docstrings on public functions, comment density like spanormpy. Each private helper's docstring names the R function it ports.

## Review Focus

1. **Parity is per component and end to end.** Every numerical helper has a test feeding it R's own inputs (from the `.h5`), not upstream Python output, plus the end-to-end tests of Tasks 9-10.
2. **NA / non-finite propagation** in the slopes arrays (genes not fitted, patients with too few cells or tiles, constant niche columns per patient) must produce NA in exactly the cells R does.
3. **Ordering** of patients (`S1, S10, S2`), conditions (contrast = second level minus first), niche columns (tested first) and table rows.
4. **Sparse input**: `X`/layer as CSR/CSC gives identical results to dense (owner: Task 9).
5. **n_jobs** never changes results (owner: Tasks 6, 9).

---

## File Structure

```
pyproject.toml, README.md, CHANGELOG.md, LICENSE, .gitignore, .pre-commit-config.yaml, .github/workflows/ci.yaml
docs/ r-to-python.md, r-discrepancies.md, api.md, tutorial.ipynb
src/spidepy/
  __init__.py          public API re-exports, __version__ = "0.1.0"
  _rcompat.py          R semantics helpers: r_num_str (paste0 of a number), sanitise, p_adjust_bh, r_order helpers
  _io.py               counts/coords/obs access on AnnData; niche obsm read/write; group map in uns
  _checks.py           checkers.R
  _kde.py              spatstat densityfun(edge, diggle) for a rectangular window
  niches.py            buildNiches.R, mergeNiches.R, utils.R (.isSelfNiche)
  _limma.py            lmFit (weights), squeezeVar/fitFDist/fitFDistRobustly, eBayes(trend, robust)
  _design.py           fit-design.R (+ .cellCovariates/.indexCovariates/.indexOffset/.patientTable/.pickSigma from fitSpiDE.R)
  _glm.py              fit-glm.R
  _slopes.py           patient-slopes.R
  _sandwich.py         sandwich.R
  _classes.py          SpiDEFit, SpiDEResults dataclasses (AllClasses.R) + accessors.R
  fit.py               fitSpiDE.R
  _test_slopes.py      test-slopes.R
  test.py              testSpiDE.R, spiDE.R
  abundance.py         niche-abundance.R
  gsea.py              spiGSEA.R
  datasets/            __init__.py (toy_spide loader, simulate_*), toySpiDE.h5ad, _simulate.py (toydata.R)
  pl/                  __init__.py, _theme.py, _helpers.py, _niche.py, _patients.py, _results.py
tests/
  ref/install_refs.sh, ref/_common.R, ref/make_<topic>.R, ref/make_all.sh, ref/data/*.h5, ref/data/*.h5ad
  _ref.py, conftest.py, test_<topic>.py
```

---

### Task 1: Scaffold, environment and R reference harness

**R sources:** `DESCRIPTION`, `data-raw/make_toySpiDE.R`, `R/toydata.R` (`.toySPE` signature only). spanormpy's `pyproject.toml`, `.github/workflows/ci.yaml`, `.pre-commit-config.yaml`, `tests/_ref.py`, `tests/ref/_common.R`, `tests/ref/install_spanorm.sh` as templates. **R tests:** none.

**Files:** Create `pyproject.toml`, `README.md` (stub), `CHANGELOG.md` (stub), `.pre-commit-config.yaml`, `.github/workflows/ci.yaml`, `src/spidepy/__init__.py`, `src/spidepy/_rcompat.py`, `tests/ref/install_refs.sh`, `tests/ref/_common.R`, `tests/ref/make_all.sh`, `tests/ref/make_fixtures.R`, `tests/_ref.py`, `tests/conftest.py`, `tests/test_rcompat.py`, `tests/test_harness.py`, `src/spidepy/datasets/__init__.py`, `src/spidepy/datasets/toySpiDE.h5ad`.

**Interfaces:**
- Env: `/scratch/user/uqdbhuva/envs/spanormpy/bin/python -m venv --system-site-packages /scratch/user/uqdbhuva/envs/spidepy`, then `pip install -e /scratch/project_mnt/S0249/R_projects/spanormpy -e ".[dev]"` (no new downloads should be needed; if a dependency is missing and pip cannot fetch it, report it).
- `pyproject.toml`: name `spidepy`, hatchling, version from `src/spidepy/__init__.py`, deps `numpy>=1.26, scipy>=1.11, pandas, anndata>=0.10, matplotlib>=3.8, spanormpy>=0.1`, extras `dev = [pytest, h5py, ruff, pre-commit]`, ruff config copied from spanormpy, pytest `testpaths=["tests"]`, `pythonpath=["."]`. Authors: Dharmesh D. Bhuva (d.bhuva@uq.edu.au), Chin Wee Tan (cwtan@wehi.edu.au) as in spiDE's DESCRIPTION. CI as spanormpy's, installing spanormpy from `git+https://github.com/BhuvaLab/spanormpy`.
- `install_refs.sh`: installs SpaNorm `f233b7b` (from `/scratch/project_mnt/S0249/R_projects/SpaNorm` via `git archive`) and then spiDE `e31e744` (from `/scratch/project_mnt/S0249/R_projects/spiDE` via `git archive`, package files only — exclude `research` if `git archive` would include a submodule stub) into `tests/ref/Rlib`; tries `install.packages("clubSandwich", lib = "tests/ref/Rlib")` and continues with a printed note if that fails (no network). Prints both versions.
- `_common.R`: as spanormpy's (same `write_ref`, `.write_elem`), loading `SpaNorm` and `spiDE` from `tests/ref/Rlib`, plus `write_spe_h5ad(spe, path)` writing a SpatialExperiment as AnnData-compatible HDF5 **without zellkonverter**: simplest robust route is to write counts (cells x genes dense), `colData` columns, `spatialCoords`, gene names and cell names via `write_ref` into an `.h5`, and have `tests/_ref.py::load_spe(topic)` build the `AnnData` (`X` = counts as float64? no: keep integer dtype as R; `layers["counts"]` = same matrix; `obs` with R's column types — factors become pandas Categorical with R's levels; `obsm["spatial"]`). The bundled `toySpiDE.h5ad` is then written from Python by `make_fixtures` support code (`tests/_ref.py::load_spe("toySpiDE").write_h5ad(...)`).
- `make_fixtures.R` writes spe fixtures: `toySpiDE` (`data(toySpiDE, package = "spiDE")`), and each distinct `.toySPE(...)` / `.toyNiche(...)` / `.toyClustered(...)` call used in `tests/testthat/*.R` (grep them; name each `toyspe_<args>` e.g. `toyspe_default`, `toyspe_s8_n120`), so later tasks never need Python simulations for parity. Also writes any `reducedDim` already on `toySpiDE`.
- `_rcompat.py`: `r_num_str(x: float) -> str` reproducing `as.character()`/`paste0` of a double (15 significant digits, `1e+05` style exponent when R uses it — R uses scientific when it is shorter, `scipen=0`); `sanitise(x)` (`gsub(" |-", ".", x)`); `p_adjust_bh(p)` (NA-aware `p.adjust(method="BH")`: NAs stay NA and do not count in n — check R: `p.adjust` with NA excludes them from n by default); `r_sort_str(xs)` (document the collation choice).
- `conftest.py` fixtures: `toy_spide` (fresh `AnnData` of toySpiDE each test), `ref` (`load_ref`), `load_spe` helper.
- spanormpy check: confirm `spanormpy.polish_nb` returns the per-gene `polished` and `psi_bound` flags and accepts `absorb`, `start_cols`, `lambda_a` vector, `offset` matrix and an `nb_block_design`; confirm `nb_block_design` and `tps_basis(x, y, df, ref_x, ref_y, center)` are importable. If anything is missing, stop and report NEEDS_CONTEXT (do not edit spanormpy).

- [ ] **Step 1:** Create the env, scaffold, install refs (`bash tests/ref/install_refs.sh`), write `make_fixtures.R` and run it.
- [ ] **Step 2:** Tests: `test_harness.py` (load_ref round-trips scalars, vectors, matrices with R shape, NA, factors with levels; `toy_spide` has the expected cells/genes/obs columns/coords against R-reported dims written in the fixture file), `test_rcompat.py` (`r_num_str` against R's `paste0` for a vector of values written by R: `10, 30, 2.5, 0.1, 1e5, 123456789, 1e-5, 1/3, 100000, 1e15`; `p_adjust_bh` against R with NAs; `sanitise`).
- [ ] **Step 3:** Run tests, ruff. Commit.

### Task 2: Input glue and checkers

**R sources:** `R/checkers.R`, `R/utils.R`, the input handling at the top of `R/fitSpiDE.R` and `R/buildNiches.R`. **R tests:** the error/warning assertions about inputs scattered in `test-fitSpiDE.R`, `test-buildNiches.R`, `test-small-n-defaults.R` (port only the input-validation ones; the rest belong to later tasks).

**Files:** Create `src/spidepy/_io.py`, `src/spidepy/_checks.py`, `tests/test_checks.py`.

**Interfaces:**
- `_io.py`: `get_counts(adata, assay="counts") -> scipy.sparse.csc_matrix | np.ndarray` returned **genes x cells** (layer if present, else `X`; never densify a sparse input), `get_coords(adata) -> np.ndarray (n, 2) float` (first two columns of `obsm["spatial"]`, DataFrame or int accepted; error for < 2 columns or non-finite), `get_obs(adata, column) -> np.ndarray` (object/str, NaN preserved), `niche_key(name, sigma) -> str` (`name + r_num_str(sigma)`), `get_niche(adata, key) -> pd.DataFrame`, `set_niche(adata, key, df)`, `niche_keys(adata, name)`, `get_group_map(adata, key) -> dict | None`, `set_group_map(adata, key, map)`; views are converted with `adata._init_as_actual(adata.copy())` style as spanormpy does (copy spanormpy's helper).
- `_checks.py`: one function per R checker with R's name in snake_case (`check_spe` -> `check_adata(adata, assay, cell_type, sample_id)`, `check_counts(Y, integer_only=False)` (first 20 genes rule), `check_condition`, `check_covariates(finite_only)`, `check_sample` (constant-within-sample errors, `< 3` samples warning), `check_column`, `check_fdr`, `check_procedure`, `check_niche`, `check_intercepts`, `check_results`, `check_slopes_engine`, `check_triplet`, `check_fit_adata` (R `checkFitSPE`), `check_genesets`). Messages copied from R.

- [ ] **Step 1:** Tests for every checker: the passing case and each error path with R's message (regex), including integer-count first-20-rows rule and sparse input.
- [ ] **Step 2:** Implement, run, ruff, commit.

### Task 3: Niche densities (KDE, buildNiches, mergeNiches)

**R sources:** `R/buildNiches.R`, `R/mergeNiches.R`, `R/utils.R`; spatstat: `Rscript -e 'library(spatstat.explore); print(spatstat.explore:::densityfun.ppp)'` and the functions it calls (installed spatstat.explore 3.5.3 in the latest-r env; read them, do not guess). **R tests:** `test-buildNiches.R`, `test-mergeNiches.R`.

**Files:** Create `src/spidepy/_kde.py`, `src/spidepy/niches.py`, `tests/ref/make_niches.R`, `tests/test_kde.py`, `tests/test_niches.py`.

**Interfaces:**
- `_kde.py`: `density_at(points_xy, query_xy, sigma, window=(xmin, xmax, ymin, ymax), edge=True, diggle=True) -> np.ndarray` reproducing `densityfun(ppp, sigma, edge, diggle)(qx, qy)` exactly. First establish from the spatstat source how the Diggle edge correction is computed for a **rectangular** window in `densityfun` (exact Gaussian mass via `pnorm` products, or a pixel-image/FFT approximation looked up at the points). Implement whichever spatstat actually does, including any pixel grid defaults (`dimyx`, `eps`) and interpolation; record the finding in the module docstring. Self-inclusion: a query point that is also a data point contributes `K(0)` (as R). Use a KD-tree or blocked distance computation for memory (cut-off at 8 sigma is acceptable only if it stays inside `rtol=1e-10`; otherwise exact).
- `niches.py`: `build_niches(adata, sigma=(10, 30, 50, 70), cell_type="cell_type", sample_id="sample_id", edge=True, diggle=True, name="Niche", n_jobs=1, copy=False)`; `merge_niches(adata, groups: dict[str, list[str]], sigma=None, name="Niche", copy=False)`; `_is_self_niche(index, niche, group_map=None)`; `_resolve_members`. Column order, NA->0 padding, cell order restoration, `uns["spiDE_niche_groups"]` composition and the `^Niche` regex quirk of R `mergeNiches` (matches any key starting with `name`) as in R.

- [ ] **Step 1:** `make_niches.R`: on toySpiDE, `buildNiches(sigma = c(10, 30, 2.5))` matrices; a raw `densityfun` call on a small hand-made ppp (points near all four edges, duplicate points, one type absent from a sample) for `edge/diggle` in `{TT, TF, FF}`; `mergeNiches` twice (composition of groups) and its metadata; `.isSelfNiche` cases.
- [ ] **Step 2:** Tests: KDE parity (`rtol=1e-10` if spatstat's computation is exact, else the tolerance the pixel approximation allows once replicated — target 1e-10 on identical grids), `build_niches` parity per sigma, column/row order, `merge_niches` values and metadata, self-niche, the testthat cases (incl. serial vs `n_jobs=2` identical).
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 4: limma core (lmFit with weights, squeezeVar, fitFDist with trend, eBayes)

**R sources:** limma 3.66.0 in the latest-r env (`Rscript -e 'print(limma::lmFit)'`, `limma:::lm.series`, `limma::eBayes`, `limma:::.ebayes`, `limma::squeezeVar`, `limma:::.squeezeVar`, `limma::fitFDist`, `limma::tmixture.matrix`, `limma::trigammaInverse`; read them). What spiDE uses: `R/test-slopes.R::.pooledColumnTest` (`lmFit(b, matrix(1, ncol(b), 1), weights = w)` with NA entries, `eBayes(fit, robust, trend)` with `fit$Amean` replaced — check the exact line) and `R/niche-abundance.R` (`eBayes(lmFit(Y, X, weights = w), robust = nrow(X) >= 6)`). **R tests:** none directly.

**Files:** Create `src/spidepy/_limma.py`, `tests/ref/make_limma.R`, `tests/test_limma.py`.

**Interfaces:** `lm_fit(Y: (G, n), design: (n, p), weights=None | (G, n) | (n,)) -> dict` (coefficients, stdev_unscaled, sigma, df_residual, Amean, with limma's per-row handling of NA observations and zero weights via `lm.series`), `fit_f_dist(x, df1, covariate=None) -> (scale, df2)` (with the natural-spline trend limma uses when `covariate` is given: limma chooses `splinedf` and calls `splines::ns` — reuse spanormpy's `ns` port, check knots/boundary handling against R), `squeeze_var(var, df, covariate=None, robust=False, winsor_tail_p=(0.05, 0.1))` (robust branch raises `NotImplementedError` until Task 5), `e_bayes(fit, proportion=0.01, stdev_coef_lim=(0.1, 4), trend=False, robust=False, winsor_tail_p=(0.05, 0.1)) -> dict` (t, df_total, p_value, s2_prior, df_prior, s2_post; `df.total` capped at the pooled residual df as limma does). Only compute what spiDE reads, but compute it exactly.

- [ ] **Step 1:** `make_limma.R`: several random weighted problems (G=200 genes; n in {4, 7, 16}; NA observations in some rows; some rows with < 2 observations; a covariate for trend) with outputs of `lmFit`, `squeezeVar` (robust FALSE, trend and no-trend), `eBayes(trend = TRUE/FALSE, robust = FALSE)`; a two-column-design case like niche-abundance's.
- [ ] **Step 2:** Tests at `rtol=1e-10` (trigamma inversion is Newton to limma's tolerance: if it is iterative, `rtol=1e-8` with a comment).
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 5: limma robust (fitFDistRobustly)

**R sources:** `limma::fitFDistRobustly`, `limma:::.squeezeVar` robust branch, everything they call (winsorising, `qf`/`pf` tails, `trigammaInverse`, `linkfun`/`linkinv`, `optimize` on the df). **R tests:** none directly.

**Files:** Modify `src/spidepy/_limma.py`, `tests/ref/make_limma.R`, `tests/test_limma.py`.

**Interfaces:** `fit_f_dist_robustly(x, df1, covariate=None, winsor_tail_p=(0.05, 0.1)) -> dict` (scale, df2, df2_shrunk — per-gene `df2.shrunk`, which `squeezeVar` then uses); `squeeze_var(..., robust=True)` and `e_bayes(..., robust=True)` now work. R's `optimize` is Brent's golden-section with R's tolerance: port it exactly (spanormpy has a `brent_fmin` port in `spanormpy._polish_engine`; reuse it if it matches R's `optimize` semantics, else port `optimize`).

- [ ] **Step 1:** Extend `make_limma.R` with robust cases (with and without trend; data with outlying variances; small n).
- [ ] **Step 2:** Tests at `rtol=1e-8` (comment: Brent optimisation), including df2_shrunk and the final t/p.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 6: Design matrices and the per-index NB fit

**R sources:** `R/fit-design.R`, `R/fit-glm.R`, and from `R/fitSpiDE.R`: `.pickSigma`, `.cellCovariates`, `.indexCovariates`, `.indexOffset`, `.patientTable`. **R tests:** `test-depth.R` (design/blocks parts).

**Files:** Create `src/spidepy/_design.py`, `src/spidepy/_glm.py`, `tests/ref/make_design.R`, `tests/test_design.py`, `tests/test_glm.py`.

**Interfaces:** `_index_cells`, `_niche_columns`, `_index_design(L, cov, patient_codes, patient_levels, trt=None, strata=None, tested=None, blocks=None) -> dict` (same fields as R's list; column names as R's `colnames(W)`, `model.matrix(~ st)[, -1]` treatment coding, `patient:<level>` names), `_lin_pred`, `_dense_x`, `_patient_blocks`, `_partial_block`, `_depth_blocks` (calls spanormpy `tps_basis(..., df=(df, df), ref_x, ref_y)`; `r2` via least squares like `lm.fit`), `_pick_sigma`, `_cell_covariates`, `_index_covariates` (centring; `nonlinear` uses the `ns` port with R's knots: `splines::ns(x, df = 3)`), `_index_offset`, `_patient_table`; `_glm_start`, `_fit_index_glm(Yk, des, offset=None, n_jobs=1) -> dict(alpha, psi, polish, status)` calling `spanormpy.polish_nb(Yk, des["W"], a0, np.ones(G), lambda_a=des["pen"], offset=offset, absorb=des["absorb"], start_cols=des["start"], psi_method="profile", n_jobs=n_jobs)`, `_fit_status` (bound handling: compare NB log-likelihood at both bounds with `scipy.stats.nbinom.logpmf` matching R `dnbinom(size = 1/p, mu)`; refit with `psi_method="fixed"`), `_working_wr`. `n_jobs` only schedules genes.

- [ ] **Step 1:** `make_design.R`: on toySpiDE niches (sigma 30) for one index type, with depth `loglib`, `nonlinear`, `spatial_spline` (use a fixture with >= 100 index cells per section so df 2/3 branches occur, and a `section` column so two sections per patient occur; create the section column in R and write it), `none`, plus `offset` and `strata`: every helper's output (`W` dense, or the block design's `X`, `Zc` and per-patient `Zs`; `pen`, `absorb`, `start`, `tested`, `niche_cols`, `intercept_cols`, `r2`), `.glmStart`, and `.fitIndexGLM` output (alpha, psi, status) — including a gene engineered to hit the psi bound and a runaway gene if one can be produced (else note it).
- [ ] **Step 2:** Tests: helpers at `rtol=1e-10`; `_fit_index_glm` alpha/psi at `rtol=1e-6`, status counts exact; `n_jobs=2` identical; the testthat cases.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 7: The slopes engine's per-patient slopes

**R sources:** `R/patient-slopes.R`. **R tests:** the per-patient-slope assertions of `test-engine.R`.

**Files:** Create `src/spidepy/_slopes.py`, `tests/ref/make_slopes.R`, `tests/test_slopes.py`.

**Interfaces:** `_patient_slopes(fit, des, Yk, L, tiles, min_cells=10, min_tiles=5, offset=None, n_jobs=1) -> dict(beta, v_tile, v_model, pooled, ncells)` (G x S x nt arrays, NaN for R NA; R's rank check on the correlation scale, per-patient dropped constant niche columns, tile sandwich with `tiles` keys), `_patient_factor(v_tile, v_model)`.

- [ ] **Step 1:** `make_slopes.R`: feed `.patientSlopes` R's own `fit0`/`des0` (write them) for depth `loglib` and `spatial_spline`; include a patient with a constant niche column and one with fewer than `min.tiles` tiles if a fixture produces them (construct one by editing coordinates in R if needed).
- [ ] **Step 2:** Tests: Python `_patient_slopes` on R's inputs at `rtol=1e-10` (closed form given the fit), NA pattern exact, `_patient_factor` exact; then on Python's own `_fit_index_glm` output at `rtol=1e-6`.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 8: The sandwich engine (CR2 + Bell-McCaffrey)

**R sources:** `R/sandwich.R`. **R tests:** `test-sandwich.R` (clubSandwich comparisons: if clubSandwich installed in Task 1, write its values into the reference; else port the assertion against R's `.sandwichCR2` only and note it).

**Files:** Create `src/spidepy/_sandwich.py`, `tests/ref/make_sandwich.R`, `tests/test_sandwich.py`.

**Interfaces:** `_sandwich_cr2(fit, des, Yk, offset=None, n_jobs=1) -> pd.DataFrame` with R's columns (`gene, niche, estimate, se, df, se_model, psi, t, p`) and row order.

- [ ] **Step 1:** `make_sandwich.R`: R's inputs and outputs for the pooled design and the condition design (with strata), dense and block (`spatial_spline`) designs.
- [ ] **Step 2:** Tests at `rtol=1e-10` on R's inputs; the testthat cases.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 9: fit_spide and the SpiDEFit class

**R sources:** `R/fitSpiDE.R`, `R/AllClasses.R` (SpiDEFit), `R/accessors.R` (`patientSlopes`, `patientIntercepts`). **R tests:** `test-fitSpiDE.R`, `test-depth.R` (fit-level), `test-intercepts.R`.

**Files:** Create `src/spidepy/_classes.py` (both dataclasses; `SpiDEResults` fields only, its methods come in Task 10), `src/spidepy/fit.py`, `tests/ref/make_fit.R`, `tests/test_fit.py`, `tests/test_accessors.py`; modify `src/spidepy/__init__.py`.

**Interfaces:** `fit_spide(adata, condition=None, engine="slopes", index=None, niche=None, sigma=None, cell_type="cell_type", sample_id="sample_id", section=None, covariates=(), depth="loglib", offset=None, strata=None, assay="counts", name="Niche", min_cells=10, min_patients=6, min_detect=0.1, genes=None, tile=None, min_tiles=5, n_jobs=1, verbose=True) -> SpiDEFit`. `SpiDEFit(engine, sigma, condition, patients: pd.DataFrame, index: dict[str, dict], params: dict)` with `__repr__` mirroring R `show`. `patient_slopes(obj, gene=None, index=None, niche=None) -> pd.DataFrame`, `patient_intercepts(obj, gene=None, index=None)` (accept a fit or results), R column order and row order.

- [ ] **Step 1:** `make_fit.R`: `fitSpiDE` on toySpiDE (sigma 30) with slopes and sandwich engines, with and without condition, depths `loglib`/`nonlinear`/`spatial_spline`/`none`, an `offset` assay, `strata`, `genes` subset, `index`/`niche` subsets; write every `@index[[k]]` field, `@patients`, and `patientSlopes`/`patientIntercepts` tables.
- [ ] **Step 2:** Tests: per-field parity (`rtol=1e-6` for fitted quantities, exact for labels/status/NA patterns), sparse CSR/CSC `X` identical to dense, `n_jobs=2` identical, the testthat cases.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 10: Tests across patients, testSpiDE, spiDE and results

**R sources:** `R/test-slopes.R`, `R/testSpiDE.R`, `R/spiDE.R`, `R/AllClasses.R` (SpiDEResults, `.pooledDfOf`), `R/accessors.R` (`results`). **R tests:** `test-engine.R`, `test-small-n-defaults.R`, `test-spiDE-e2e.R`.

**Files:** Create `src/spidepy/_test_slopes.py`, `src/spidepy/test.py`, `tests/ref/make_test.R`, `tests/test_test_slopes.py`, `tests/test_testspide.py`; modify `src/spidepy/_classes.py`, `src/spidepy/__init__.py`.

**Interfaces:** `_dl_tau2`, `_heterogeneity_p`, `_kish_df`, `_pooled_column_test` (uses `_limma.lm_fit`/`e_bayes(trend, robust)` exactly as R, including the `Amean` substitution and R's `tryCatch` fallback from robust to non-robust), `_robust_condition_test`, `_slopes_tests`, `_bh_families`; `test_spide(fit, condition=None, procedure=None, fdr=0.05, strata=None, pooled_df="proportional", n_jobs=1) -> SpiDEResults`; `spide(adata, condition=None, sigma=..., engine="slopes", procedure=None, fdr=0.05, pooled_df="proportional", cell_type=..., sample_id=..., n_jobs=1, verbose=True, **fit_kwargs)`; `SpiDEResults.results(test=None, fdr=None)` and module-level `results(res, ...)`; `SpiDEResults.__repr__` mirrors R `show`. Name the module function `test_spide` but make sure pytest does not collect it (set `test_spide.__test__ = False`).

- [ ] **Step 1:** `make_test.R`: each helper on R's own inputs (from R fits written in `make_fit.R` or recomputed), `testSpiDE` tables for every procedure and both `pooled.df` rules, both engines, with strata; `spiDE()` end-to-end table; `results()` for each `test`/`fdr`.
- [ ] **Step 2:** Tests: helpers on R inputs `rtol=1e-10` (limma parts at Task 4/5 tolerance), full table on R's fit `rtol=1e-8`, end to end from AnnData `rtol=1e-6` with `q`/`in_family` exact, the testthat cases.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 11: testNicheAbundance

**R sources:** `R/niche-abundance.R`. **R tests:** `test-niche-abundance.R`.

**Files:** Create `src/spidepy/abundance.py`, `tests/ref/make_abundance.R`, `tests/test_abundance.py`; modify `__init__.py`.

**Interfaces:** `test_niche_abundance(adata, condition=None, sigma=..., index=None, niche=None, covariates=(), assay="counts", cell_type=..., sample_id=..., name="Niche", min_cells=10, prior_count=1, max_leverage=3, genesets=None, type="competitive", min_size=5, max_size=500, verbose=True) -> pd.DataFrame` (R's columns and order); helpers `_pseudobulk_by_index`, `_abundance_pair` (with the `niche * condition` -> `niche` fallback), `_abundance_set_scores`, `_cap_leverage` (`stats::hat` = diagonal of the hat matrix via QR), `_patient_value`. `__test__ = False` on the public function.

- [ ] **Step 1:** `make_abundance.R`: every helper and the public function with and without condition, covariates, genesets (both types), a high-leverage patient.
- [ ] **Step 2:** Tests at `rtol=1e-10` (limma robust parts at Task 5 tolerance); the testthat cases.
- [ ] **Step 3:** Implement, run, ruff, commit.

### Task 12: spiGSEA

**R sources:** `R/spiGSEA.R`. **R tests:** `test-spiGSEA.R`.

**Files:** Create `src/spidepy/gsea.py`, `tests/ref/make_gsea.R`, `tests/test_gsea.py`; modify `__init__.py`.

**Interfaces:** `spi_gsea(res, genesets: dict[str, list[str]], test=None, type="competitive", min_size=5, max_size=500, pooled_df=None, n_jobs=1) -> pd.DataFrame`; `_set_slopes`.

- [ ] **Step 1-3:** R references on R's own results object (written by `make_test.R` or recomputed), parity at `rtol=1e-8`, testthat cases, commit.

### Task 13: Datasets and simulators

**R sources:** `R/toydata.R`, `data-raw/make_toySpiDE.R`, `R/data.R`. **R tests:** `test-toydata.R`.

**Files:** Create `src/spidepy/datasets/_simulate.py`, `tests/test_datasets.py`; modify `src/spidepy/datasets/__init__.py`, `__init__.py`.

**Interfaces:** `toy_spide() -> AnnData` (bundled file from Task 1; docstring from `R/data.R`); `simulate_toy(n_samples=6, n_per=80, n_genes=20, field=500, sd_lib_sample=0.15, sd_lib_celltype=0.35, beta=2, composition=0, seed=1)`, `simulate_niche(...)`, `simulate_clustered(...)` with R's arguments and the same generative model using `numpy.random.default_rng(seed)` (draws differ from R; say so in docstrings and `docs/r-discrepancies.md`). The global RNG is never touched (R's `.localSeed` guarantee).

- [ ] **Step 1:** Tests: shapes, obs columns and dtypes match the R fixture of the same arguments; statistical properties asserted by `test-toydata.R`; seeds reproducible; an end-to-end `spide()` on `simulate_toy()` finds the planted effect (the property R's e2e test asserts).
- [ ] **Step 2:** Implement, run, ruff, commit.

### Task 14: Plot theme, data helpers and niche plots

**R sources:** `R/plot-theme.R`, `R/plot-helpers.R`, `R/plot-niche.R`, `design/specs/2026-10-01-plots.md` (in the spiDE repo: the plot design spec), `inst/spide_palette.R`. **R tests:** `test-plot-helpers.R`, `test-plots-niche.R`, `test-plots-devices.R`.

**Files:** Create `src/spidepy/pl/__init__.py`, `pl/_theme.py`, `pl/_helpers.py`, `pl/_niche.py`, `tests/ref/make_plot_helpers.R`, `tests/test_pl_helpers.py`, `tests/test_pl_niche.py`.

**Interfaces:** `pl.spide_colours(role=None, n=9)` (R's hex values exactly), `pl.style()` (context manager of rcParams approximating `theme_spiDE`), every `plot-helpers.R` data builder as `_snake_case` returning DataFrames (parity with R's data frames at `rtol=1e-10`), `pl.plot_niche_map(adata, niche, sample, index=None, sigma=None, ...)`, `pl.plot_niche_response(adata, res, gene, index, niche, adjust=True, bins=8, ...)` (R arguments in snake_case; return `matplotlib.figure.Figure`). Use `matplotlib.use("Agg")` in tests.

- [ ] **Step 1:** Helpers parity tests on R's data frames; plot smoke tests (renders, axis labels/titles/legend entries as R's, panels count, errors as R); one PNG per plot written to a temp dir in tests.
- [ ] **Step 2:** Implement, run, ruff, commit.

### Task 15: Patient and results plots

**R sources:** `R/plot-patients.R`, `R/plot-results.R`. **R tests:** `test-plots-patients.R`, `test-plots-results.R`.

**Files:** Create `pl/_patients.py`, `pl/_results.py`, `tests/ref/make_plot_data.R` (only if the plots' internal data builders need R parity: `.slopeStripLabels`, `.patientEffectMatrix`, `.patientCovariates`, `.callHeatmap`/`.callGraph` counts, top-triplet selection, heatmap matrix), `tests/test_pl_patients.py`, `tests/test_pl_results.py`; modify `pl/__init__.py`.

**Interfaces:** `pl.plot_patient_slopes`, `pl.plot_patient_effects`, `pl.plot_call_map(style="heatmap"|"graph")`, `pl.plot_top_triplets`, `pl.plot_spillover`, `pl.plot_triplet_heatmap`, `pl.plot_p_values` with R's arguments in snake_case, each returning a `Figure`; the call graph is drawn with matplotlib only (no networkx dependency unless already installed in the env — check; if needed, compute a circular layout by hand).

- [ ] **Step 1:** Data parity for the internal builders; smoke tests and the testthat cases.
- [ ] **Step 2:** Implement, run, ruff, commit.

### Task 16: Documentation, API coverage and tutorial

**R sources:** `vignettes/spiDE.Rmd`, `vignettes/spiDE-model.Rmd`, `vignettes/spiDE-calibration.Rmd`, `README.md`, `NEWS.md`, `NAMESPACE`. spanormpy's `README.md`, `docs/r-to-python.md`, `docs/api.md`, `tests/test_api_coverage.py` as templates. **R tests:** none.

**Files:** Create/complete `README.md`, `CHANGELOG.md` (0.1.0: ports spiDE 0.99.37 `e31e744`), `docs/r-to-python.md` (every NAMESPACE export -> Python name, and argument renames), `docs/api.md`, `docs/r-discrepancies.md` (consolidate entries from earlier tasks), `docs/tutorial.ipynb` (the `spiDE.Rmd` walk-through on `toy_spide()`, executed, outputs stored), `tests/test_api_coverage.py` (parses spiDE's `NAMESPACE` at `e31e744` and asserts every `export()` has a mapped Python object).

- [ ] **Step 1:** Write docs and notebook; execute it with `jupyter nbconvert --execute --inplace` (install `nbconvert ipykernel` into the env if missing; if they cannot be installed, store an unexecuted notebook and report).
- [ ] **Step 2:** Run the full suite and ruff. Commit.
