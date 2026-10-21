# Task 15 references: the data behind the patient and results plots (R/plot-patients.R,
# R/plot-results.R): strip labels, the patient-effect matrix and covariates, call counts and the
# call graph's edges and nodes, the top triplets, spillover, triplet heatmap and p-value bands.
# Fits and results are the ones of the R tests (all index types), dumped with the arrays the
# builders read. Run from the repo root.
source("tests/ref/_common.R")
suppressPackageStartupMessages(library(ggplot2))

dump_df <- function(d) {
  l <- lapply(as.list(d), function(v) if (is.factor(v)) as.character(v) else v)
  l$colnames__ <- colnames(d)
  l
}
lv <- function(d, col) levels(d[[col]])
dump_fit <- function(fit) {
  p <- fit@params
  idx <- lapply(fit@index, function(x) {
    o <- list(genes = x$genes, niches = x$niches, patients = x$patients, mean_expr = unname(x$mean_expr),
              intercept = unname(x$intercept), loglib_mean = unname(x$loglib_mean[x$patients]),
              ncells = unname(x$ncells[x$patients]), niche_mean = unname(x$niche_mean[x$patients, , drop = FALSE]))
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
       pooled_df = if (length(r@pooled.df)) r@pooled.df else "", fit = dump_fit(r@fit))
}
# the data of the layer whose data frame has all of these columns (and none of `not`)
layer_data <- function(p, has, not = character()) {
  for (l in p$layers) {
    d <- l$data
    if (is.data.frame(d) && all(has %in% names(d)) && !any(not %in% names(d))) return(d)
  }
  NULL
}

spe <- buildNiches(spiDE:::.toySPE(n_samples = 16, n_per = 150, seed = 7), sigma = 30, verbose = FALSE)
fit <- fitSpiDE(spe, condition = "condition", sigma = 30, verbose = FALSE)
res <- testSpiDE(fit, procedure = "all")
fs <- fitSpiDE(spe, condition = "condition", sigma = 30, index = "A", engine = "sandwich", verbose = FALSE)
f0 <- fitSpiDE(spe, sigma = 30, index = "A", verbose = FALSE)
f0@index$A$beta["G1", "S3", ] <- NA
r0 <- testSpiDE(f0)

# --- patient slopes
ps <- plotPatientSlopes(res, gene = c("G1", "G2"), index = "A", niche = "B")
ps3 <- plotPatientSlopes(res, gene = c("G3", "G1"), index = "B", niche = "A", ci = 0.9)
psn <- plotPatientSlopes(r0, gene = "G1", index = "A", niche = "B")
slopes <- list(
  strip_cond = spiDE:::.slopeStripLabels(res, c("G1", "G2", "G7"), "A", "B"),
  strip_pooled = spiDE:::.slopeStripLabels(r0, c("G1", "G2"), "A", "B"),
  a = list(data = dump_df(ps$data), band = dump_df(ps$layers[[1]]$data), gm = dump_df(ps$layers[[4]]$data),
           patient_levels = levels(ps$data$patient)),
  b = list(data = dump_df(ps3$data), band = dump_df(ps3$layers[[1]]$data), gm = dump_df(ps3$layers[[4]]$data),
           patient_levels = levels(ps3$data$patient)),
  none = list(data = dump_df(psn$data), band = dump_df(psn$layers[[1]]$data), gm = dump_df(psn$layers[[4]]$data),
              patient_levels = levels(psn$data$patient))
)

# --- patient effects
xi <- fit@index$A
Mi <- spiDE:::.patientEffectMatrix(xi, NULL, 500L)
Ms <- spiDE:::.patientEffectMatrix(xi, "B", 500L)
Mt <- spiDE:::.patientEffectMatrix(xi, NULL, 6L)
cv <- spiDE:::.patientCovariates(fit, xi)
cv2 <- spiDE:::.patientCovariates(fit, xi, c("condition", "B density"))
fp <- fit
fp@patients$batch <- rep(c("x", "y", "z", "x"), length.out = nrow(fp@patients))
fp@patients$score <- seq_len(nrow(fp@patients))^1.5 %% 7
cvp <- spiDE:::.patientCovariates(fp, fp@index$A)
y <- xi$intercept["G1", ]
adj <- list(
  num = spiDE:::.adjR2(y, seq_along(y)^2), num2 = spiDE:::.adjR2(y, 2 * y + sin(seq_along(y))),
  fac2 = spiDE:::.adjR2(y, factor(ifelse(y > median(y), "hi", "lo"))), fac = spiDE:::.adjR2(y, factor(rep(c("a", "b", "c"), length.out = length(y)))),
  const = spiDE:::.adjR2(y, rep(1, length(y))), few = spiDE:::.adjR2(y[1:3], 1:3),
  na = spiDE:::.adjR2(c(NA, y[-1]), factor(rep(c("a", "b"), length.out = length(y)))),
  y = unname(y))
pe_assoc <- plotPatientEffects(res, index = "A")
pe_assoc_s <- plotPatientEffects(res, index = "A", niche = "B", n.pcs = 3L)
pe_pca <- plotPatientEffects(res, index = "A", type = "pca", colour.by = "condition")
pe_pca_s <- plotPatientEffects(res, index = "A", niche = "B", type = "pca", colour.by = "B density")
pe_sand <- plotPatientEffects(fs, index = "A")
pc <- prcomp(t(Mi), center = TRUE, scale. = FALSE)
effects <- list(
  Mi = unname(Mi), Mi_genes = rownames(Mi), Mi_pats = colnames(Mi),
  Ms = unname(Ms), Ms_genes = rownames(Ms), Ms_pats = colnames(Ms),
  Mt_genes = rownames(Mt),
  cov = dump_df(cv$cov), cov_group = unname(cv$group), cov_names = names(cv$group),
  cov_rows = rownames(cv$cov),
  cov2 = dump_df(cv2$cov), cov2_group = unname(cv2$group),
  covp = dump_df(cvp$cov), covp_names = colnames(cvp$cov), covp_group = unname(cvp$group),
  adj = adj,
  assoc = dump_df(pe_assoc$data), assoc_pc_levels = levels(pe_assoc$data$pc),
  assoc_cov_levels = levels(pe_assoc$data$covariate), assoc_x = pe_assoc$labels$x,
  assoc_s = dump_df(pe_assoc_s$data), assoc_s_x = pe_assoc_s$labels$x,
  pca = dump_df(pe_pca$data), pca_x = pe_pca$labels$x, pca_y = pe_pca$labels$y,
  pca_s = dump_df(pe_pca_s$data),
  sand = dump_df(pe_sand$data),
  pc_sdev = pc$sdev, pc_x_abs = unname(abs(pc$x))
)

# --- call map
cm_pool <- plotCallMap(res, test = "pooled", fdr = 0.05)
cm_cond <- plotCallMap(res, test = "condition")
cm_none <- plotCallMap(res, test = "pooled", fdr = 1e-12)
g_cond <- plotCallMap(res, test = "condition", style = "graph", fdr = 0.9)
g_pool <- plotCallMap(res, test = "pooled", style = "graph")
g_pool2 <- plotCallMap(res, test = "pooled", style = "graph", min.calls = 2L, fdr = 0.2)
g_none <- plotCallMap(res, test = "pooled", fdr = 1e-12, style = "graph")
r3 <- res
r3@table$niche[r3@table$niche == "B"] <- "Airway epithelium B"
r3@table$index[r3@table$index == "B"] <- "Airway epithelium B"
r3@table$niche[r3@table$niche == "C"] <- "Tumour-associated macrophages"
r3@table$index[r3@table$index == "C"] <- "Tumour-associated macrophages"
g_wrap <- plotCallMap(r3, test = "pooled", style = "graph")
graph_out <- function(g) list(edges = dump_df(g$layers[[1]]$data), nodes = dump_df(g$layers[[2]]$data))
calls <- list(
  pool = list(data = dump_df(cm_pool$data), dir_levels = levels(cm_pool$data$direction)),
  cond = list(data = dump_df(cm_cond$data), dir_levels = levels(cm_cond$data$direction)),
  none = list(data = dump_df(cm_none$data)),
  g_cond = graph_out(g_cond), g_pool = graph_out(g_pool), g_pool2 = graph_out(g_pool2), g_none = graph_out(g_none),
  g_wrap = graph_out(g_wrap), g_pool_levels = levels(g_pool$layers[[1]]$data$panel),
  r3_table = dump_df(r3@table)
)

# --- top triplets, spillover
tt <- plotTopTriplets(res, test = "pooled", n = 10)
tt_spe <- plotTopTriplets(res, gene = c("G1", "G3"), spe = spe)
tt_cond <- plotTopTriplets(res, n = 12, spe = spe, fold = 2)
tt_idx <- plotTopTriplets(res, test = "pooled", index = "B", niche = "A", n = 5, fdr = 0.2)
top <- list(
  pooled = dump_df(tt$data), pooled_levels = levels(tt$data$label),
  genes = dump_df(tt_spe$data), genes_levels = levels(tt_spe$data$label),
  cond = dump_df(tt_cond$data), cond_levels = levels(tt_cond$data$label),
  idx = dump_df(tt_idx$data))
sp <- plotSpillover(res, spe, fold = 0.4, label = 4L, fdr = 0.2)
sp_c <- plotSpillover(res, spe, test = "condition", fold = 1.2, label = 3L, fdr = 0.9)
spill <- list(
  pooled = dump_df(sp$data), pooled_status_levels = levels(sp$data$status),
  pooled_labels = dump_df(sp$layers[[7]]$data),
  cond = dump_df(sp_c$data), cond_labels = dump_df(sp_c$layers[[7]]$data))

# --- triplet heatmap
th <- plotTripletHeatmap(res, test = "pooled", n = 8)
th_c <- plotTripletHeatmap(res, n = 12, cluster = FALSE)
r2 <- res
r2@table <- r2@table[!(r2@table$gene == "G2" & r2@table$index == "B"), ]
th_u <- plotTripletHeatmap(r2, features = c("G1", "G2"), test = "pooled")
th_u3 <- plotTripletHeatmap(r2, features = c("G2", "G1", "G5", "G9"), test = "pooled", limit = 3)
gs <- spiGSEA(res, list(one = paste0("G", 1:5), two = paste0("G", 6:12), three = paste0("G", 13:20)), min.size = 3)
th_g <- plotTripletHeatmap(gs, test = "pooled")
heat <- list(
  main = dump_df(th$data), main_levels = levels(th$data$feature),
  noclust = dump_df(th_c$data), noclust_levels = levels(th_c$data$feature),
  untested = dump_df(th_u$data), untested_nt = dump_df(layer_data(th_u, c("feature", "index", "niche"), "t")),
  u3 = dump_df(th_u3$data), u3_levels = levels(th_u3$data$feature),
  u3_nt = dump_df(layer_data(th_u3, c("feature", "index", "niche"), "t")),
  gsea_in = dump_df(gs), gsea = dump_df(th_g$data), gsea_levels = levels(th_g$data$feature),
  lim_main = max(stats::quantile(abs(th$data$t), 0.98), 1))

# --- p-value bands
pv <- plotPValues(res)
pv3 <- plotPValues(res, bands = 3L)
pvals <- list(data = dump_df(pv$data), band_levels = levels(pv$data$band), test_levels = levels(pv$data$test),
              uni = dump_df(pv$layers[[2]]$data),
              data3 = dump_df(pv3$data), uni3 = dump_df(pv3$layers[[2]]$data))

strw_in <- c("Tumour-associated macrophages", "T cell", "Supercalifragilisticexpialidocious cell",
             "a  b   c d", "End. Next sentence here", "x-y-z-w-v-u-t-s", "  lead and trail  ", "A.B C", "One.  Two three", "(Quote.\" next one")
strw <- lapply(c(8L, 10L, 12L), function(w) vapply(strw_in, function(x) paste(strwrap(x, w), collapse = "\n"), ""))
names(strw) <- c("w8", "w10", "w12")
strw$input <- strw_in

# hclust's leaf order (the triplet heatmap's row order) on matrices of several sizes
set.seed(11)
hc <- lapply(c(3L, 4L, 7L, 12L, 25L), function(n) {
  m <- matrix(rnorm(n * 6), n, 6)
  list(m = unname(m), order = hclust(dist(m))$order)
})
names(hc) <- paste0("n", c(3, 4, 7, 12, 25))

write_ref(list(strw = lapply(strw, unname), hc = hc, res = dump_res(res), fs = dump_res(testSpiDE(fs)), r0 = dump_res(r0),
               slopes = slopes, effects = effects, calls = calls, top = top, spill = spill,
               heat = heat, pvals = pvals, fp = dump_fit(fp)), "plot_data")
