# Task 6 references: the per-index design (.indexCells, .nicheColumns, .indexDesign, .depthBlocks and the
# small helpers), the covariate / offset / patient-table helpers of fitSpiDE.R, and the per-gene NB fit
# (.glmStart, .fitIndexGLM with its status handling) on the 8-patient x 650-cell fixture. A `section`
# column is created here so that some patients have two sections (and some a section too small for a
# spline), which exercises every df branch of .depthBlocks. Run from the repo root.
source("tests/ref/_common.R")

spe <- spiDE:::.toySPE(n_samples = 8, n_per = 650, n_genes = 4, seed = 11)  # == toyspe_s8_n650_g4_seed11
spe <- buildNiches(spe, sigma = 30, verbose = FALSE)
Y <- assay(spe, "counts")
smp <- as.character(spe$sample_id)
# sections: S1 split at x = 250 (two sections of ~130 cells: df 2), S3 split at x = 100 (a ~45-cell
# section: df 0, and a ~190-cell one: df 2), every other patient one section (~240 cells: df 3)
sec <- smp
sec[smp == "S1"] <- ifelse(spe$x[smp == "S1"] < 250, "S1a", "S1b")
sec[smp == "S3"] <- ifelse(spe$x[smp == "S3"] < 100, "S3a", "S3b")
spe$section <- sec
spe$slide <- ifelse(smp %in% paste0("S", 1:3), "X", ifelse(smp %in% paste0("S", 4:6), "Y", "Z"))
qc <- as.numeric(spe$Age) * 2
qc[which(smp == "S3")[1]] <- NA_real_   # constant within patient apart from an NA on its first cell
spe$qc <- qc
set.seed(5)
off_all <- matrix(rnorm(nrow(Y) * ncol(Y), 0, 0.3), nrow(Y), ncol(Y), dimnames = dimnames(Y)) +
  rep(log(pmax(colSums(Y), 1)) - 2.5, each = nrow(Y))
assay(spe, "off") <- off_all
cd <- colData(spe)
ct <- as.character(spe$cell_type)
xy <- spatialCoords(spe)
NM <- reducedDim(spe, "Niche30")
lib <- colSums(Y)
usable <- lib > 0
ct_u <- ifelse(usable, ct, NA_character_)

out <- list(
  inputs = list(NM = unname(NM), NM_cols = colnames(NM), section = sec, slide = as.character(spe$slide),
                qc = qc, off_all = unname(off_all), obs_columns = colnames(cd))
)

# --- .indexCells ---
ik <- spiDE:::.indexCells(ct_u, smp, "A", 10L)
out$cells <- list(ik = ik, ik_b = spiDE:::.indexCells(ct_u, smp, "B", 10L),
                  ik_hi = spiDE:::.indexCells(ct_u, smp, "A", 245L),
                  ik_none = spiDE:::.indexCells(ct_u, smp, "A", 100000L),
                  n_unusable = sum(!usable))
# NA cell types and a patient absent from a type
ct_na <- ct_u; ct_na[smp == "S2" & ct_na %in% "A"] <- NA
out$cells$ik_na <- spiDE:::.indexCells(ct_na, smp, "A", 10L)

# --- .nicheColumns ---
nc <- spiDE:::.nicheColumns(NM, "A", ik)
out$niche_cols <- list(
  all = nc, only_C = spiDE:::.nicheColumns(NM, "A", ik, niche = "C"),
  b_all = spiDE:::.nicheColumns(NM, "B", spiDE:::.indexCells(ct_u, smp, "B", 10L)),
  grp = spiDE:::.nicheColumns(NM, "A", ik, group_map = list(B = c("A", "B"))),
  grp_niche = spiDE:::.nicheColumns(NM, "C", spiDE:::.indexCells(ct_u, smp, "C", 10L), niche = c("B", "A"),
                                    group_map = list(A = c("A", "B")))
)
NMc <- NM; NMc[, "C"] <- 0.5                      # a constant niche column drops out
out$niche_cols$const <- spiDE:::.nicheColumns(NMc, "A", ik)
out$inputs$NM_const_C <- unname(NMc)
NMz <- NM; NMz[ik[1], "B"] <- NA                  # NA density: var() is NA, so not finite, dropped
out$niche_cols$na <- tryCatch(spiDE:::.nicheColumns(NMz, "A", ik), error = function(e) list(cols = "error", tested = "error"))
out$inputs$NM_na_B <- unname(NMz)

# --- .pickSigma ---
mk <- function(...) { s <- spe; for (r in c(...)) reducedDim(s, r) <- reducedDim(spe, "Niche30"); s }
s1 <- mk("Niche10"); s1 <- s1[, ]; reducedDim(s1, "Niche30") <- NULL
s2 <- mk("Niche10", "Niche2.5")
msg <- function(e) conditionMessage(e)
out$pick_sigma <- list(
  only = spiDE:::.pickSigma(s1, NULL, "Niche"),
  given = spiDE:::.pickSigma(s2, 2.5, "Niche"),
  multi_msg = tryCatch(spiDE:::.pickSigma(s2, NULL, "Niche"), error = msg),
  none_msg = tryCatch(spiDE:::.pickSigma(spe[, ], NULL, "Foo"), error = msg),
  len_msg = tryCatch(spiDE:::.pickSigma(s2, c(1, 2), "Niche"), error = msg),
  names_s1 = reducedDimNames(s1), names_s2 = reducedDimNames(s2)
)

# --- .patientTable ---
pt <- spiDE:::.patientTable(cd, smp)
ptl <- list(names = colnames(pt))
for (cn in colnames(pt)) ptl[[cn]] <- pt[[cn]]
out$patient_table <- ptl

# --- .cellCovariates / .indexCovariates ---
covs <- list(loglib = spiDE:::.cellCovariates(cd, "Area", "loglib", Y),
             none = spiDE:::.cellCovariates(cd, "Area", "none", Y),
             none_empty = spiDE:::.cellCovariates(cd, character(), "none", Y),
             nonlinear = spiDE:::.cellCovariates(cd, character(), "nonlinear", Y),
             two = spiDE:::.cellCovariates(cd, c("Area", "Age"), "loglib", Y))
out$cell_cov <- lapply(covs, function(m) list(m = unname(m), cols = colnames(m) %||% character()))
out$cell_cov$none_empty$dim <- dim(covs$none_empty)
ixc <- function(depth, cv) {
  r <- spiDE:::.indexCovariates(cv[ik, , drop = FALSE], depth)
  if (is.null(r)) return(list(null = 1L))
  list(null = 0L, m = unname(r), cols = colnames(r), centre = attr(r, "scaled:center"))
}
out$index_cov <- list(loglib = ixc("loglib", covs$loglib), nonlinear = ixc("nonlinear", covs$two),
                      spatial = ixc("spatial_spline", covs$two), spatial_only = ixc("spatial_spline", covs$nonlinear),
                      none = ixc("none", covs$none), none_empty = ixc("none", covs$none_empty))
# a log library size with a tied boundary (many cells share the minimum): knots shoved inside, R warns
gk4 <- rownames(Y)
out$index_offset <- list(m = unname(spiDE:::.indexOffset(spe, "off", gk4[c(3, 1)], ik)),
                         genes = gk4[c(3, 1)])
sbad <- spe; o2 <- off_all; o2[2, ik[7]] <- Inf; assay(sbad, "off") <- o2
out$index_offset$bad_msg <- tryCatch(spiDE:::.indexOffset(sbad, "off", gk4, ik), error = msg)

# --- .depthBlocks ---
L2 <- log1p(NM[ik, c("B", "C")])
ell <- cov_ll <- covs$loglib[, "loglib"]
patf <- factor(smp[ik])
blk <- spiDE:::.depthBlocks(ell[ik], xy, sec, ik, patf, L2)
out$depth <- list(Zc = unname(blk$Zc), Zc_cols = colnames(blk$Zc), r2 = unname(blk$r2),
                  Zs = setNames(lapply(blk$Zs, unname), paste0("s", seq_along(blk$Zs))),
                  L = unname(L2), ncols = vapply(blk$Zs, ncol, integer(1)))
# a constant niche column: tss == 0, r2 stays NA
L2c <- L2; L2c[, 1] <- 1
out$depth$r2_const <- unname(spiDE:::.depthBlocks(ell[ik], xy, sec, ik, patf, L2c)$r2)
out$depth$bad_msg <- tryCatch(spiDE:::.depthBlocks(c(ell[ik][-1], NA), xy, sec, ik, patf, L2), error = msg)

# --- engineered genes: the 4 fixture genes plus ones that hit the bound / runaway branches ---
n <- length(ik)
cols <- c(nc$tested, setdiff(nc$cols, nc$tested))
pick_niche <- "C"
nc2 <- spiDE:::.nicheColumns(NM, "A", ik, niche = pick_niche)
cols <- c(nc2$tested, setdiff(nc2$cols, nc2$tested))   # tested first: C, then B
L <- log1p(NM[ik, cols, drop = FALSE])
pat <- factor(smp[ik])
set.seed(1)
Yk <- rbind(as.matrix(Y[, ik]),
  pois = rpois(n, exp(1 + 0.2 * scale(L[, 1]))),
  over = rnbinom(n, size = 1e-4, mu = 3),
  over2 = rnbinom(n, size = 0.02, mu = 3),
  one = { v <- numeric(n); v[5] <- 1; v },
  one40 = { v <- numeric(n); v[5] <- 40; v },
  few = { v <- numeric(n); v[sample(n, 3)] <- c(1, 2, 30); v },
  sep = ifelse(L[, 1] > median(L[, 1]), rpois(n, 5), 0),
  allzero = numeric(n))
set.seed(6)
Ok <- rbind(spiDE:::.indexOffset(spe, "off", rownames(Y), ik),
            matrix(rnorm(8 * n, 0, 0.3), 8, n) + rep(ell[ik] - 2.5, each = 8))
out$fit_inputs <- list(Yk = unname(Yk), genes = rownames(Yk), L = unname(L), L_cols = cols, Ok = unname(Ok),
                       ik = ik)

trt_cell <- as.numeric(cd$condition[ik] == "Responder")
slide_cell <- as.character(spe$slide)[ik]

run_cfg <- function(depth, covariates, trt = NULL, strata = NULL, offset = FALSE, tested = nc2$tested) {
  cov <- spiDE:::.cellCovariates(cd, covariates, depth, Y)
  covk <- spiDE:::.indexCovariates(cov[ik, , drop = FALSE], depth)
  blk <- if (depth == "spatial_spline") spiDE:::.depthBlocks(cov[ik, "loglib"], xy, sec, ik, pat, L) else NULL
  des <- spiDE:::.indexDesign(L, covk, pat, trt = trt, strata = if (is.null(strata)) NULL else factor(strata),
                              tested = tested, blocks = blk)
  O <- if (offset) Ok else NULL
  a0 <- spiDE:::.glmStart(Yk, des, O)
  fit <- spiDE:::.fitIndexGLM(Yk, des, offset = O)
  res <- list(depth = depth,
              covk = if (is.null(covk)) list(null = 1L) else list(null = 0L, m = unname(covk), cols = colnames(covk)),
              des = des[setdiff(names(des), c("W", "Zs", "absorb"))],
              has_absorb = as.integer(!is.null(des$absorb)),
              colnames = colnames(des$W), a0 = unname(a0),
              alpha = unname(fit$alpha), psi = unname(fit$psi), status = fit$status,
              polished = fit$polish$polished, psi_bound = fit$polish$psi_bound)
  if (!is.null(des$absorb)) res$des$absorb <- des$absorb
  if (inherits(des$W, "nbBlockDesign")) {
    res$X <- unname(des$W$X); res$Zc <- unname(des$W$Z); res$block <- 1
    res$Zs <- setNames(lapply(des$Zs, unname), paste0("s", seq_along(des$Zs)))
  } else {
    res$W <- unname(des$W); res$block <- 0
  }
  # the helpers on the fitted gene 1 (G1) and 3 (G3)
  for (g in c(1L, 3L)) {
    o <- if (offset) Ok[g, ] else 0
    wr <- spiDE:::.workingWR(Yk[g, ], des$W, fit$alpha[g, ], fit$psi[g], o)
    res[[paste0("wr", g)]] <- list(w = wr$w, r = wr$r, mu = wr$mu)
    res[[paste0("lp", g)]] <- spiDE:::.linPred(des$W, fit$alpha[g, ])
  }
  # partialBlock on patient 2's rows, weights from G1
  i <- which(des$patient == 2L)
  w <- res$wr1$w[i]
  X <- spiDE:::.denseX(des)
  res$dense_x <- unname(X)
  Z <- spiDE:::.patientBlocks(des, split(seq_along(des$patient), des$patient))[[2]]
  res$partial <- unname(spiDE:::.partialBlock(X[i, , drop = FALSE], w, Z))
  res$z2_ncol <- ncol(Z)
  res
}

out$c1 <- run_cfg("loglib", "Area")
out$c2 <- run_cfg("nonlinear", character(), trt = trt_cell, strata = slide_cell)
out$c3 <- run_cfg("spatial_spline", "Area")
out$c4 <- run_cfg("spatial_spline", character(), trt = trt_cell, strata = slide_cell)
out$c5 <- run_cfg("none", character(), offset = TRUE, strata = rep("X", n))   # a one-level strata is a no-op
out$c6 <- run_cfg("loglib", "Area", offset = TRUE, tested = cols)               # every column tested, with offset
out$inputs$trt_cell <- trt_cell
out$inputs$slide_cell <- slide_cell

# partialBlock: a singular block returns NULL, the intercept-only case is a weighted centring
Xs <- matrix(rnorm(30), 10, 3); Zs <- cbind(1, rnorm(10)); Zs <- cbind(Zs, Zs[, 2]); ws <- runif(10, 0.5, 2)
out$partial_misc <- list(X = Xs, Z = Zs, w = ws,
                         singular_null = as.integer(is.null(spiDE:::.partialBlock(Xs, ws, Zs))),
                         ok = spiDE:::.partialBlock(Xs, ws, Zs[, 1:2]),
                         centre = spiDE:::.partialBlock(Xs, ws, matrix(1, 10, 1)))

# a design with strata whose niche products vanish in one level (a zero column is dropped)
Lz <- L; Lz[slide_cell == "Z", "B"] <- 0
dz <- spiDE:::.indexDesign(Lz, NULL, pat, strata = factor(slide_cell), tested = nc2$tested)
out$strata_zero <- list(L = unname(Lz), colnames = colnames(dz$W), W = unname(dz$W), tested = dz$tested)

# a design without any covariate column and without niche variation beyond one column
d0 <- spiDE:::.indexDesign(L[, 1, drop = FALSE], NULL, pat, tested = cols[1])
out$d_min <- list(L = unname(L[, 1, drop = FALSE]), colnames = colnames(d0$W), W = unname(d0$W),
                  dense = d0$dense, tested = d0$tested, niche_cols = d0$niche_cols, pen = d0$pen)

# the status fields of a fit with no bound / runaway gene
fit_ok <- spiDE:::.fitIndexGLM(Yk[1:4, ], spiDE:::.indexDesign(L, NULL, pat, tested = nc2$tested))
out$fit_ok <- list(status = fit_ok$status, alpha = unname(fit_ok$alpha), psi = unname(fit_ok$psi))

write_ref(out, "design")
cat("status c1..c6\n"); for (k in paste0("c", 1:6)) print(out[[k]]$status)
