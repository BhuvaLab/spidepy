# Task 9 references: fitSpiDE end to end, both engines, on the shipped toySpiDE (sigma 30) and on the
# 8-patient x 650-cell fixture (toyspe_s8_n650_g4_seed11, with sections and zero-count cells). Every
# @index[[k]] field, @patients, the recorded params and the patientSlopes / patientIntercepts tables
# are written. Run from the repo root.
source("tests/ref/_common.R")

data("toySpiDE", package = "spiDE", envir = environment())
smp_toy <- as.character(toySpiDE$sample_id)
num <- as.integer(sub("S", "", smp_toy))
# two slides, each holding both conditions (parity of the patient number is the condition)
toySpiDE$slide <- ifelse(num %% 4 < 2, "slide1", "slide2")
Yt <- assay(toySpiDE, "counts")
# a smooth per-gene, per-cell log offset (no random draws: the Python test rebuilds it)
off_fun <- function(G, N) {
  g <- matrix(seq_len(G), G, N)
  cc <- matrix(seq_len(N), G, N, byrow = TRUE)
  0.3 * sin(1.3 * g + cc / 37)
}
O <- off_fun(nrow(Yt), ncol(Yt)); dimnames(O) <- dimnames(Yt)
assay(toySpiDE, "off") <- O
spe <- buildNiches(toySpiDE, sigma = 30, verbose = FALSE)

dump_df <- function(d) {
  l <- as.list(d)
  l[vapply(l, is.null, NA)] <- NULL
  l$colnames__ <- colnames(d)   # HDF5 groups iterate alphabetically: keep the column order
  l
}
dump_index <- function(x) {
  x <- x[!vapply(x, is.null, NA)]
  x <- x[!duplicated(names(x))]
  for (n in names(x)) {
    v <- x[[n]]
    if (is.data.frame(v)) x[[n]] <- dump_df(v)
    else if (n == "status") x[[n]] <- as.list(v)
    else if (is.matrix(v) || is.array(v)) x[[n]] <- unname(v)
    else if (!is.null(names(v))) x[[n]] <- unname(v)
  }
  if (!is.null(x$genes)) x$genes <- as.character(x$genes)
  x
}
dump_fit <- function(fit) {
  p <- fit@params
  p <- p[!vapply(p, is.null, NA)]
  p <- p[names(p) != "version"]
  p <- lapply(p, function(v) if (length(v) == 0) "" else v)
  list(engine = fit@engine, sigma = fit@sigma, condition = if (length(fit@condition)) fit@condition else "",
       patients = dump_df(fit@patients), params = p, index_names = names(fit@index),
       index = lapply(fit@index, dump_index))
}
run <- function(s, tables = FALSE, ...) {
  fit <- fitSpiDE(s, sigma = 30, verbose = FALSE, ...)
  out <- dump_fit(fit)
  if (!tables) return(out)
  if (fit@engine == "slopes") {
    out$slopes <- dump_df(patientSlopes(fit))
    out$slopes_g1 <- dump_df(patientSlopes(fit, gene = c("G2", "G1"), niche = "C"))
  }
  out$intercepts <- dump_df(patientIntercepts(fit))
  out$intercepts_g <- dump_df(patientIntercepts(fit, gene = "G3", index = "A"))
  out
}

cfg <- list(
  slopes_all = run(spe, tables = TRUE),
  slopes_cond = run(spe, condition = "condition", index = "A"),
  sand_pooled = run(spe, engine = "sandwich", index = "A"),
  sand_cond_all = run(spe, tables = TRUE, engine = "sandwich", condition = "condition"),
  sand_cov = run(spe, engine = "sandwich", condition = "condition", index = "A", covariates = "Area"),
  slopes_nonlinear = run(spe, index = "A", depth = "nonlinear"),
  slopes_none = run(spe, index = "A", depth = "none"),
  slopes_spline = run(spe, index = "A", depth = "spatial_spline"),
  slopes_offset = run(spe, tables = TRUE, index = "A", offset = "off"),
  sand_none_offset = run(spe, engine = "sandwich", condition = "condition", index = "A",
                         depth = "none", offset = "off"),
  slopes_genes = run(spe, index = "A", genes = c("G5", "G1", "G2")),
  slopes_subset = run(spe, index = c("B", "A"), niche = "C"),
  sand_subset = run(spe, engine = "sandwich", condition = "condition", index = c("C", "A"), niche = "B"),
  sand_strata = run(spe, engine = "sandwich", condition = "condition", index = "A", strata = "slide"),
  sand_strata_pooled_only = run(spe, engine = "sandwich", index = "A", strata = "slide"),
  slopes_strata = run(spe, condition = "condition", index = "A", strata = "slide"),
  slopes_detect = run(spe, index = "A", min.detect = 0.5, min.cells = 40L, min.patients = 8L, tile = 50,
                      min.tiles = 3L)
)

# the 650-cell fixture: two sections for S1 and S3, zero-count cells, spatial_spline depth
s2 <- spiDE:::.toySPE(n_samples = 8, n_per = 650, n_genes = 4, seed = 11)
s2 <- buildNiches(s2, sigma = 30, verbose = FALSE)
smp2 <- as.character(s2$sample_id)
sec <- smp2
sec[smp2 == "S1"] <- ifelse(s2$x[smp2 == "S1"] < 250, "S1a", "S1b")
sec[smp2 == "S3"] <- ifelse(s2$x[smp2 == "S3"] < 100, "S3a", "S3b")
s2$section <- sec
s2$slide <- ifelse(smp2 %in% c("S1", "S2", "S4", "S5"), "X", "Y")
cfg2 <- list(
  big_slopes_spline = run(s2, tables = TRUE, index = "A", depth = "spatial_spline", section = "section"),
  big_sand_spline = run(s2, engine = "sandwich", condition = "condition", index = "A",
                        depth = "spatial_spline", section = "section", strata = "slide"),
  big_slopes_loglib = run(s2, section = "section", min.patients = 6L),
  big_slopes_nonlinear = run(s2, index = "A", depth = "nonlinear")
)

write_ref(c(cfg, cfg2), "fit")
