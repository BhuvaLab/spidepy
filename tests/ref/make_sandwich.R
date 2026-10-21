# Task 8 references: .sandwichCR2 on R's own shared fits (the alpha / psi of tests/ref/data/design.h5,
# written by make_design.R) for the six designs of that file -- c1 pooled loglib, c2 condition
# (trt + strata) with nonlinear depth, c3 pooled spatial_spline blocks, c4 condition + strata with
# blocks, c5 no depth + offset + one-level strata, c6 every column tested with offset -- plus the
# clubSandwich oracles of test-sandwich.R (16-patient fixture, pooled and condition designs, intercept-only
# absorption) and test-depth.R (spatial_spline, full block design). clubSandwich vcovCR(type = "CR2",
# inverse_var = TRUE) and coef_test(test = "Satterthwaite") run on the working weighted lm of each
# converged fit. Run from the repo root after make_design.R.
source("tests/ref/_common.R")
.libPaths(c("tests/ref/Rlib", .libPaths()))
library(clubSandwich)

D <- h5read(file.path(REF_DIR, "design.h5"), "/")   # R's own fits, rebuilt below as design objects
spe <- spiDE:::.toySPE(n_samples = 8, n_per = 650, n_genes = 4, seed = 11)
spe <- buildNiches(spe, sigma = 30, verbose = FALSE)
Y <- assay(spe, "counts"); smp <- as.character(spe$sample_id)
sec <- smp
sec[smp == "S1"] <- ifelse(spe$x[smp == "S1"] < 250, "S1a", "S1b")
sec[smp == "S3"] <- ifelse(spe$x[smp == "S3"] < 100, "S3a", "S3b")
cd <- colData(spe); xy <- spatialCoords(spe); NM <- reducedDim(spe, "Niche30")
lib <- colSums(Y)
ct_u <- ifelse(lib > 0, as.character(spe$cell_type), NA_character_)
ik <- spiDE:::.indexCells(ct_u, smp, "A", 10L)
nc2 <- spiDE:::.nicheColumns(NM, "A", ik, niche = "C")
cols <- c(nc2$tested, setdiff(nc2$cols, nc2$tested))
L <- log1p(NM[ik, cols, drop = FALSE]); pat <- factor(smp[ik])
Yk <- D$fit_inputs$Yk; dimnames(Yk) <- list(D$fit_inputs$genes, NULL)
Ok <- D$fit_inputs$Ok
trt_cell <- as.numeric(cd$condition[ik] == "Responder")
slide <- ifelse(smp %in% paste0("S", 1:3), "X", ifelse(smp %in% paste0("S", 4:6), "Y", "Z"))[ik]
n <- length(ik)

mkdes <- function(depth, covariates, trt = NULL, strata = NULL, tested = nc2$tested) {
  cov <- spiDE:::.cellCovariates(cd, covariates, depth, Y)
  covk <- spiDE:::.indexCovariates(cov[ik, , drop = FALSE], depth)
  blk <- if (depth == "spatial_spline") spiDE:::.depthBlocks(cov[ik, "loglib"], xy, sec, ik, pat, L) else NULL
  spiDE:::.indexDesign(L, covk, pat, trt = trt, strata = if (is.null(strata)) NULL else factor(strata),
                       tested = tested, blocks = blk)
}
cfgs <- list(
  c1 = list(des = mkdes("loglib", "Area"), off = NULL),
  c2 = list(des = mkdes("nonlinear", character(), trt_cell, slide), off = NULL),
  c3 = list(des = mkdes("spatial_spline", "Area"), off = NULL),
  c4 = list(des = mkdes("spatial_spline", character(), trt_cell, slide), off = NULL),
  c5 = list(des = mkdes("none", character(), strata = rep("X", n)), off = Ok),
  c6 = list(des = mkdes("loglib", "Area", tested = cols), off = Ok))

tab <- function(d) {
  if (is.null(d) || !nrow(d)) return(list(n = 0L))
  c(list(n = nrow(d)), lapply(as.list(d), function(v) if (is.factor(v)) as.character(v) else v))
}
out <- list(inputs = list(genes = rownames(Yk)))
for (cn in names(cfgs)) {
  c0 <- cfgs[[cn]]
  fit <- list(alpha = D[[cn]]$alpha, psi = D[[cn]]$psi)
  got <- spiDE:::.sandwichCR2(fit, c0$des, Yk, offset = c0$off)
  out[[cn]] <- tab(got)
}

# --- all genes unfitted: the typed empty table ---
fit_na <- list(alpha = matrix(NA_real_, nrow(Yk), ncol(D$c1$alpha)), psi = rep(NA_real_, nrow(Yk)))
e <- spiDE:::.sandwichCR2(fit_na, cfgs$c1$des, Yk)
out$empty <- list(n = nrow(e), cols = colnames(e))
# --- clubSandwich oracles on the working weighted lm ---
cs_compare <- function(des, fit, Yg, got, spline = FALSE) {
  pid <- des$patient
  if (spline) {
    Zf <- do.call(cbind, lapply(seq_along(des$Zs), function(s) {
      z <- matrix(0, length(pid), ncol(des$Zs[[s]])); z[pid == s, ] <- des$Zs[[s]]; z }))
    Xd <- des$W$X
  } else {
    Xd <- des$W[, -seq_len(des$npat), drop = FALSE]
  }
  colnames(Xd) <- make.names(colnames(Xd))
  cl <- factor(pid)
  res <- list()
  for (g in rownames(Yg)) {
    eta <- spiDE:::.linPred(des$W, fit$alpha[g, ]); mu <- exp(eta)
    w <- mu / (1 + fit$psi[g] * mu); z <- eta + (as.numeric(Yg[g, ]) - mu) / mu
    m <- if (spline) stats::lm(z ~ 0 + Zf + Xd, weights = w) else stats::lm(z ~ 0 + factor(pid) + Xd, weights = w)
    V <- clubSandwich::vcovCR(m, cluster = cl, type = "CR2", inverse_var = TRUE)
    cs <- clubSandwich::coef_test(m, vcov = V, test = "Satterthwaite")
    o <- cs[match(paste0("Xd", make.names(colnames(des$W)[des$tested])), cs$Coef), ]
    res[[g]] <- list(beta = unname(o$beta), SE = unname(o$SE), df = unname(o$df_Satt))
  }
  res
}
# test-sandwich.R: 16 patients x 150 cells, genes G1-G3, pooled and condition designs
spe5 <- buildNiches(spiDE:::.toySPE(n_samples = 16, n_per = 150, n_genes = 6, seed = 5), sigma = 30, verbose = FALSE)
ct5 <- as.character(spe5$cell_type); smp5 <- as.character(spe5$sample_id)
ik5 <- spiDE:::.indexCells(ct5, smp5, "A", 5L)
NM5 <- reducedDim(spe5, "Niche30")
nc5 <- spiDE:::.nicheColumns(NM5, "A", ik5)
L5 <- log1p(NM5[ik5, nc5$cols, drop = FALSE])
Y5 <- assay(spe5, "counts")[c("G1", "G2", "G3"), ik5, drop = FALSE]
pat5 <- factor(smp5[ik5]); trt5 <- as.numeric(spe5$condition[ik5] == "Responder")
out$cs16 <- list(inputs = list(L = unname(L5), cols = nc5$cols, tested = nc5$tested, patient = as.character(pat5),
                               trt = trt5, Y = unname(as.matrix(Y5)), genes = rownames(Y5)))
for (nm in c("pooled", "cond")) {
  des5 <- spiDE:::.indexDesign(L5, NULL, pat5, trt = if (nm == "cond") trt5 else NULL, tested = nc5$tested)
  fit5 <- spiDE:::.fitIndexGLM(Y5, des5)
  got5 <- spiDE:::.sandwichCR2(fit5, des5, Y5)
  out$cs16[[nm]] <- list(alpha = unname(fit5$alpha), psi = fit5$psi, got = tab(got5),
                         cs = cs_compare(des5, fit5, Y5, got5))
}
# test-depth.R: spatial_spline on the 8 x 650 fixture (every cell is a section: section = patient), the
# condition design without strata, genes G1 and G2, against the full block design
ll <- log(colSums(Y))
nc8 <- spiDE:::.nicheColumns(NM, "A", ik)
L8 <- log1p(NM[ik, nc8$cols, drop = FALSE])
blk8 <- spiDE:::.depthBlocks(ll[ik], xy, smp, ik, pat, L8)
trt8 <- as.numeric(cd$condition[ik] == "Responder")
des8 <- spiDE:::.indexDesign(L8, NULL, pat, trt = trt8, tested = nc8$tested, blocks = blk8)
Y8 <- Y[c("G1", "G2"), ik, drop = FALSE]
fit8 <- spiDE:::.fitIndexGLM(Y8, des8)
got8 <- spiDE:::.sandwichCR2(fit8, des8, Y8)
out$cs_spline <- list(inputs = list(L = unname(L8), cols = nc8$cols, tested = nc8$tested, ell = ll[ik],
                        xy = unname(xy), smp = smp, ik = ik, patient = as.character(pat), trt = trt8,
                        Y = unname(as.matrix(Y8)), genes = rownames(Y8)),
                      alpha = unname(fit8$alpha), psi = fit8$psi, got = tab(got8),
                      cs = cs_compare(des8, fit8, Y8, got8, spline = TRUE))
write_ref(out, "sandwich")
for (cn in names(cfgs)) cat(cn, "rows", out[[cn]]$n, "\n")
cat("empty", out$empty$n, "\n")
