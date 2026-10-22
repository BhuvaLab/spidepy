# Task 7 references: .patientSlopes and .patientFactor fed R's own shared fits (the alpha / psi of
# tests/ref/data/design.h5, written by make_design.R) on the same 8-patient x 650-cell fixture.
# Configs: c1 (loglib), c3 (spatial_spline blocks), c6 (loglib + per-cell offset, every column
# tested). Each runs plain and with engineered patients: one with a niche column absent from its
# tissue (constant), one with collinear niche columns, one with fewer than min.tiles tiles, and a
# min.cells that drops the smallest patient. Run from the repo root after make_design.R.
source("tests/ref/_common.R")

spe <- spiDE:::.toySPE(n_samples = 8, n_per = 650, n_genes = 4, seed = 11)
spe <- buildNiches(spe, sigma = 30, verbose = FALSE)
Y <- assay(spe, "counts")
smp <- as.character(spe$sample_id)
sec <- smp
sec[smp == "S1"] <- ifelse(spe$x[smp == "S1"] < 250, "S1a", "S1b")
sec[smp == "S3"] <- ifelse(spe$x[smp == "S3"] < 100, "S3a", "S3b")
cd <- colData(spe)
xy <- spatialCoords(spe)
NM <- reducedDim(spe, "Niche30")
lib <- colSums(Y)
ct_u <- ifelse(lib > 0, as.character(spe$cell_type), NA_character_)
ik <- spiDE:::.indexCells(ct_u, smp, "A", 10L)
D <- h5read(file.path(REF_DIR, "design.h5"), "/")   # R's own shared fits
nc2 <- spiDE:::.nicheColumns(NM, "A", ik, niche = "C")
cols <- c(nc2$tested, setdiff(nc2$cols, nc2$tested))
L <- log1p(NM[ik, cols, drop = FALSE])
pat <- factor(smp[ik])
Yk <- D$fit_inputs$Yk; dimnames(Yk) <- list(D$fit_inputs$genes, NULL)
Ok <- D$fit_inputs$Ok
tile <- 90   # 3 * sigma, fitSpiDE's default
tiles <- paste(sec, floor(xy[, 1] / tile), floor(xy[, 2] / tile), sep = "\r")[ik]

mkdes <- function(depth, covariates, tested) {
  cov <- spiDE:::.cellCovariates(cd, covariates, depth, Y)
  covk <- spiDE:::.indexCovariates(cov[ik, , drop = FALSE], depth)
  blk <- if (depth == "spatial_spline") spiDE:::.depthBlocks(cov[ik, "loglib"], xy, sec, ik, pat, L) else NULL
  spiDE:::.indexDesign(L, covk, pat, tested = tested, blocks = blk)
}
cfgs <- list(c1 = list(des = mkdes("loglib", "Area", nc2$tested), off = NULL),
             c3 = list(des = mkdes("spatial_spline", "Area", nc2$tested), off = NULL),
             c6 = list(des = mkdes("loglib", "Area", cols), off = Ok))

# engineered patients (patient codes are 1-based in des$patient)
Lm <- L; tm <- tiles
Lm[pat == "S2", cols[2]] <- 0                                  # niche absent from S2: one slope drops
Lm[pat == "S4", cols[2]] <- 2 * Lm[pat == "S4", cols[1]]       # collinear within S4: patient drops
set.seed(3)
tm[pat == "S5"] <- sample(paste0("T", 1:3), sum(pat == "S5"), TRUE)   # 3 tiles < min.tiles
Lm[pat == "S7", cols[2]] <- 1                                  # constant (nonzero) column in S7
nper <- table(pat)
mc <- as.integer(min(nper)) + 1L                               # drops the smallest patient

out <- list(inputs = list(L = unname(L), Lm = unname(Lm), tiles = tiles, tiles_m = tm, genes = rownames(Yk),
                          min_cells_m = mc, patient_n = as.integer(nper)))
run <- function(nm, L_, tl, min.cells = 10L, min.tiles = 5L) {
  r <- list()
  for (cn in names(cfgs)) {
    c0 <- cfgs[[cn]]
    fit <- list(alpha = D[[cn]]$alpha, psi = D[[cn]]$psi)
    ps <- spiDE:::.patientSlopes(fit, c0$des, Yk, L_, tl, min.cells = min.cells, min.tiles = min.tiles,
                                 offset = c0$off)
    r[[cn]] <- list(beta = unname(ps$beta), v_tile = unname(ps$v_tile), v_model = unname(ps$v_model),
                    pooled = unname(ps$pooled),
                    factor = unname(spiDE:::.patientFactor(ps$v_tile, ps$v_model)),
                    patients = dimnames(ps$beta)[[2]], niches = dimnames(ps$beta)[[3]])
  }
  r
}
out$plain <- run("plain", L, tiles)
out$mod <- run("mod", Lm, tm, min.cells = mc)
out$mintiles <- run("mintiles", L, tiles, min.tiles = 20L)   # most patients below the tile threshold
# .patientFactor on an array with non-finite ratios and an all-NA (patient, niche) cell
set.seed(9)
vt <- array(rexp(6 * 4 * 2), c(6, 4, 2)); vm <- array(rexp(6 * 4 * 2), c(6, 4, 2))
vt[, 2, 1] <- NA; vm[1:3, 3, 2] <- 0; vm[, 4, 1] <- NA; vt[2, 1, 1] <- Inf
out$factor_misc <- list(vt = vt, vm = vm, f = spiDE:::.patientFactor(vt, vm))
write_ref(out, "slopes")
cat("patients per cfg\n"); print(nper)
for (k in c("plain", "mod", "mintiles")) for (cn in names(cfgs))
  cat(k, cn, "finite beta:", mean(is.finite(out[[k]][[cn]]$beta)), " finite v_tile:", mean(is.finite(out[[k]][[cn]]$v_tile)), "\n")
