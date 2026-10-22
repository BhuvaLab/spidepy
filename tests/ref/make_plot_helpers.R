# Task 14 references: the plotting layer's colours, labels and shared data builders
# (R/plot-theme.R, R/plot-helpers.R) and the data behind plotNicheMap / plotNicheResponse
# (R/plot-niche.R). Every data frame is dumped column by column (order kept in colnames__).
# Fits and results are dumped with the arrays the builders read, so Python is fed R's own
# slopes, intercepts and tables. Run from the repo root.
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(ggplot2))

dump_df <- function(d) {
  l <- lapply(as.list(d), function(v) if (is.factor(v)) as.character(v) else v)
  l$colnames__ <- colnames(d)
  l
}
dump_fit <- function(fit) {
  p <- fit@params
  idx <- lapply(fit@index, function(x) {
    o <- list(genes = x$genes, niches = x$niches, patients = x$patients, mean_expr = unname(x$mean_expr),
              intercept = unname(x$intercept), loglib_mean = unname(x$loglib_mean[x$patients]),
              ncells = unname(x$ncells[x$patients]))
    if (fit@engine == "slopes") {
      o$beta <- unname(x$beta); o$v_model <- unname(x$v_model); o$factor <- unname(x$factor)
    } else o$coef <- dump_df(x$coef)
    o
  })
  list(engine = fit@engine, sigma = fit@sigma, condition = if (length(fit@condition)) fit@condition else "",
       patients = dump_df(fit@patients), index_names = names(fit@index), index = idx,
       params = list(cell_type = p$cell_type, sample_id = p$sample_id, depth = p$depth,
                     offset = if (is.null(p$offset)) "" else p$offset,
                     covariates = if (length(p$covariates)) p$covariates else ""))
}
dump_res <- function(r) {
  list(table = dump_df(r@table), condition = if (length(r@condition)) r@condition else "",
       contrast = if (length(r@contrast)) r@contrast else "", procedure = r@procedure, fdr = r@fdr,
       fit = dump_fit(r@fit))
}

spe <- buildNiches(spiDE:::.toySPE(n_samples = 16, n_per = 150, seed = 7), sigma = 30, verbose = FALSE)
write_spe_h5ad(spe, file.path(REF_DIR, "plot_spe.h5"),
               extra = list(niche_cols = colnames(reducedDim(spe, "Niche30"))))
fit <- fitSpiDE(spe, condition = "condition", sigma = 30, index = "A", verbose = FALSE)
res <- testSpiDE(fit, procedure = "all")
fs <- fitSpiDE(spe, condition = "condition", sigma = 30, index = "A", engine = "sandwich", verbose = FALSE)
rs <- testSpiDE(fs)
fsp <- fitSpiDE(spe, sigma = 30, index = "A", depth = "spatial_spline", verbose = FALSE)
fnc <- fitSpiDE(spe, sigma = 30, index = "A", verbose = FALSE)
rnc <- testSpiDE(fnc)

# merged niches: BC (members B, C) and a group named after one of its members
merge_out <- function(groups, file) {
  sm <- mergeNiches(spe, groups = groups, sigma = 30)
  write_spe_h5ad(sm, file.path(REF_DIR, file),
                 extra = list(niche_cols = colnames(reducedDim(sm, "Niche30")),
                              groups = metadata(sm)[["spiDE_niche_groups"]][["Niche30"]]))
  fm <- fitSpiDE(sm, sigma = 30, index = "A", verbose = FALSE)
  rm_ <- testSpiDE(fm)
  list(res = dump_res(rm_), fold = spiDE:::.nicheFold(rm_@table, sm, fm),
       fold_g1 = spiDE:::.nicheFold(rm_@table[rm_@table$gene == "G1", ], sm, fm))
}
mBC <- merge_out(list(BC = c("B", "C")), "plot_spe_bc.h5")
mB <- merge_out(list(B = c("B", "C")), "plot_spe_b.h5")

# --- colours and labels
cols <- list(fixed = spiDE::spiDEColours(), direction = spiDE::spiDEColours("direction"),
             condition = spiDE::spiDEColours("condition"))
for (n in c(1, 2, 5, 9, 11, 20)) {
  cols[[paste0("magnitude", n)]] <- spiDE::spiDEColours("magnitude", n)
  cols[[paste0("spatial", n)]] <- spiDE::spiDEColours("spatial", n)
  cols[[paste0("diverging", n)]] <- spiDE::spiDEColours("diverging", n)
}
for (n in c(1, 5, 8, 9, 10, 12, 20)) cols[[paste0("nuisance", n)]] <- spiDE::spiDEColours("nuisance", n)
cols$ramp256 <- spiDE:::.magnitudeRamp(256L)
cols$ramp256_spatial <- spiDE:::.magnitudeRamp(256L, spatial = TRUE)
cols$ramp12 <- spiDE:::.magnitudeRamp(12L)
cols$ramp7_spatial <- spiDE:::.magnitudeRamp(7L, spatial = TRUE)
cols$break_in <- c("Non-responder", "Responder", "Late stage group", "Plain", "Averyveryverylongword", "a b c d e f g h i")
cols$break_out <- spiDE:::.breakLabel(cols$break_in)
lab <- list(
  trip = spiDE:::.itTriplet('a"b\\c', "T cell", "B|x"), gene = spiDE:::.itGene('a"b\\c'),
  fmtq_in = c(0.0234, 2e-8, NA, 0.5, 1, 0.001, 0.00099, Inf),
  fmtq = spiDE:::.fmtq(c(0.0234, 2e-8, NA, 0.5, 1, 0.001, 0.00099, Inf)))

# --- helpers of R/plot-helpers.R
xi <- fit@index$A
sw <- spiDE:::.slopeWeights(xi)
M <- spiDE:::.cellTypeMeans(spe, c("G1", "G2"), "counts", "cell_type")
bands <- spiDE:::.expressionBands(fit)
bands2 <- spiDE:::.expressionBands(fit, k = 3L)
icd <- spiDE:::.indexCellData(spe, fit, "A", "B", c("G2", "G1"))
grp <- spiDE:::.groupLevels(fit, "condition")
tbf <- res@table[res@table$gene == "G1", ]
helpers <- list(
  slope_v = unname(sw$v), slope_tau = unname(sw$tau),
  ctm = unname(M), ctm_genes = rownames(M), ctm_types = colnames(M), ctm_n = unname(attr(M, "ncells")),
  bands = dump_df(bands), bands_levels = levels(bands$band), bands3 = dump_df(bands2),
  bands3_levels = levels(bands2$band),
  icd_counts = unname(as.matrix(icd$counts)), icd_lib = unname(icd$lib), icd_patient = icd$patient,
  icd_density = icd$density,
  grp = as.character(grp), grp_names = names(grp), grp_levels = levels(grp),
  grp_none = levels(spiDE:::.groupLevels(fnc, character())),
  dir_cond = unname(spiDE:::.directionLabels(res, "condition")),
  dir_pooled = unname(spiDE:::.directionLabels(res, "pooled")),
  fold = spiDE:::.nicheFold(res@table, spe, fit), fold_g1 = spiDE:::.nicheFold(tbf, spe, fit),
  test_cond = spiDE:::.resolveTest(res, NULL), test_pooled = spiDE:::.resolveTest(rnc, NULL),
  trip_rows = dump_df(spiDE:::.tripletRows(res@table, "pooled", c("G3", "G1"), "A", "B"))
)

# --- plotNicheMap data
sec <- ifelse(spatialCoords(spe)[, 1] > 250, "right", "left")
s2 <- spe
s2$section <- sec
bar_of <- function(p) p$layers[[length(p$layers)]]$data
map_out <- function(p) list(data = dump_df(p$data), bar = dump_df(bar_of(p)))
maps <- list(
  plain = map_out(plotNicheMap(spe, niche = "B", sample = "S1")),
  index = map_out(plotNicheMap(spe, niche = "B", sample = "S1", index = "A", unit = "µm")),
  section = map_out(plotNicheMap(s2, niche = "B", sample = "S1", section = "section", index = "A")),
  small = map_out(plotNicheMap(spe, niche = "C", sample = "S3", point.size = 0.5)),
  section_vec = sec)

# --- plotNicheResponse data
resp_out <- function(p, adjust) {
  o <- list(data = dump_df(p$data))
  if (adjust) {
    o$lines <- dump_df(p$layers[[2]]$data)
    o$summary <- dump_df(p$layers[[3]]$data)
  } else o$centroids <- dump_df(p$layers[[2]]$data)
  o
}
resp <- list(
  adj = resp_out(plotNicheResponse(spe, res, gene = c("G1", "G2"), index = "A", niche = "B"), TRUE),
  adj4 = resp_out(plotNicheResponse(spe, res, gene = c("G3", "G1", "G2", "G1"), index = "A", niche = "C", bins = 4L), TRUE),
  raw = resp_out(plotNicheResponse(spe, res, gene = c("G1", "G2"), index = "A", niche = "B", adjust = FALSE), FALSE),
  sand = resp_out(plotNicheResponse(spe, rs, gene = "G1", index = "A", niche = "B"), TRUE),
  sand_raw = resp_out(plotNicheResponse(spe, rs, gene = "G1", index = "A", niche = "B", adjust = FALSE), FALSE),
  spline = resp_out(plotNicheResponse(spe, fsp, gene = "G1", index = "A", niche = "B"), TRUE),
  nocond = resp_out(plotNicheResponse(spe, fnc, gene = "G1", index = "A", niche = "B"), TRUE)
)
f2 <- fit
f2@index$A$intercept["G2", ] <- NA
f2@index$A$beta["G2", , ] <- NA
resp$dropped <- resp_out(suppressMessages(plotNicheResponse(spe, f2, c("G1", "G2"), "A", "B")), TRUE)
resp$dropped_raw <- resp_out(plotNicheResponse(spe, f2, c("G1", "G2"), "A", "B", adjust = FALSE), FALSE)
f3 <- fit
f3@index$A$intercept["G1", "S2"] <- -50
resp$extreme <- resp_out(plotNicheResponse(spe, f3, "G1", "A", "B"), TRUE)
f2d <- dump_fit(f2); f3d <- dump_fit(f3)

write_ref(list(cols = cols, lab = lab, helpers = helpers, maps = maps, resp = resp,
               fit = dump_fit(fit), res = dump_res(res), rs = dump_res(rs), fsp = dump_fit(fsp),
               fnc = dump_fit(fnc), rnc = dump_res(rnc), f2 = f2d, f3 = f3d,
               bc = mBC, b = mB), "plot_helpers")
