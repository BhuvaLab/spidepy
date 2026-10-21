# Task 10 references: the tests across patients. Every helper of test-slopes.R on R's own inputs
# (the fits written below, so Python feeds them R's slopes, not its own), testSpiDE tables for every
# procedure and both pooled.df rules, both engines, with strata and with a patient dropped for one
# index type, spiDE() end to end, results() and show(). Run from the repo root.
source("tests/ref/_common.R")

data("toySpiDE", package = "spiDE", envir = environment())
num <- as.integer(sub("S", "", as.character(toySpiDE$sample_id)))
toySpiDE$slide <- ifelse(num %% 4 < 2, "slide1", "slide2")
spe <- buildNiches(toySpiDE, sigma = 30, verbose = FALSE)
spe16 <- buildNiches(spiDE:::.toySPE(n_samples = 16, n_per = 120, n_genes = 12, seed = 2), sigma = 30,
                     verbose = FALSE)

dump_df <- function(d) {
  l <- as.list(d)
  l$colnames__ <- colnames(d)
  l
}
dump_in <- function(fit) {
  idx <- lapply(fit@index, function(x) {
    o <- list(genes = x$genes, niches = x$niches, patients = x$patients, mean_expr = unname(x$mean_expr))
    if (fit@engine == "slopes") {
      o$beta <- unname(x$beta); o$v_model <- unname(x$v_model); o$factor <- unname(x$factor)
    } else o$coef <- dump_df(x$coef)
    o
  })
  list(engine = fit@engine, condition = if (length(fit@condition)) fit@condition else "",
       strata = if (is.null(fit@params$strata)) "" else fit@params$strata,
       patients = dump_df(fit@patients), index_names = names(fit@index), index = idx)
}
dump_res <- function(r) {
  list(table = dump_df(r@table), condition = if (length(r@condition)) r@condition else "",
       contrast = if (length(r@contrast)) r@contrast else "", procedure = r@procedure, fdr = r@fdr,
       pooled_df = if (length(r@pooled.df)) r@pooled.df else "")
}

# helpers on a slopes fit's own arrays
dump_helpers <- function(fit, condition = NULL, strata = NULL) {
  pt <- fit@patients
  trt_of <- if (!is.null(condition)) spiDE:::.conditionCoding(pt, condition) else NULL
  out <- list()
  for (k in names(fit@index)) {
    x <- fit@index[[k]]
    G <- length(x$genes)
    vp <- sweep(x$v_model, c(2, 3), x$factor, "*")
    tau <- spiDE:::.dlTau2(x$beta, vp)
    het <- spiDE:::.heterogeneityP(x$beta, vp)
    h <- list(vpool = unname(vp), tau = unname(tau), het = unname(het))
    trt <- if (!is.null(trt_of)) unname(trt_of[x$patients])
    st <- if (!is.null(strata)) setNames(as.character(pt[[strata]]), pt$patient)[x$patients]
    for (j in seq_along(x$niches)) {
      b <- matrix(x$beta[, , j], nrow = G); v <- matrix(vp[, , j], nrow = G)
      cj <- list(
        pooled_prop = dump_df(spiDE:::.pooledColumnTest(b, v, tau[, j], x$mean_expr)),
        pooled_cap = dump_df(spiDE:::.pooledColumnTest(b, v, tau[, j], x$mean_expr, df.rule = "capped")),
        pooled_flat = dump_df(spiDE:::.pooledColumnTest(b, v, tau[, j], x$mean_expr, trend = FALSE)),
        kish = spiDE:::.kishDf(1 / (v + tau[, j]), b, 1L))
      if (!is.null(trt)) {
        cj$cond <- dump_df(spiDE:::.robustConditionTest(b, v, tau[, j], trt = trt))
        if (!is.null(st) && length(unique(st)) > 1L)
          cj$cond_strata <- dump_df(spiDE:::.robustConditionTest(b, v, tau[, j], trt = trt, strata = st))
      }
      h[[paste0("col", j)]] <- cj
    }
    out[[k]] <- h
  }
  out
}

variants <- function(fit, condition, procs = c("pooled_or_heterogeneity", "heterogeneity", "filtered", "all"),
                     dfs = c("proportional", "capped"), ...) {
  out <- list()
  for (p in procs) for (d in dfs)
    out[[paste(p, d, sep = "__")]] <- dump_res(testSpiDE(fit, condition = condition, procedure = p,
                                                         pooled.df = d, ...))
  out
}

cfg <- list()

# 16 patients x 12 genes, index A: all procedures x both df rules; no recorded condition
f16 <- fitSpiDE(spe16, index = "A", sigma = 30, verbose = FALSE)
cfg$sl16 <- list(fit = dump_in(f16), helpers = dump_helpers(f16, "condition"),
                 default = dump_res(testSpiDE(f16, condition = "condition")),
                 nocond = dump_res(testSpiDE(f16)),
                 variants = variants(f16, "condition"))
# the same fit's condition-free calls and results()
r16 <- testSpiDE(f16, condition = "condition")
cfg$sl16$results <- list(
  default = dump_df(results(r16)), pooled = dump_df(results(r16, test = "pooled")),
  both = dump_df(results(r16, test = "both")), cond_fdr = dump_df(results(r16, test = "condition", fdr = 0.1)),
  pooled_fdr = dump_df(results(r16, test = "pooled", fdr = 0.05)), fdr1 = dump_df(results(r16, fdr = 1)),
  nocond = dump_df(results(testSpiDE(f16))))
cfg$sl16$show <- capture.output(show(r16))
cfg$sl16$show_capped <- capture.output(show(testSpiDE(f16, condition = "condition", procedure = "filtered",
                                                     pooled.df = "capped")))

# the toy fixture, all index types, the condition recorded and slide strata recorded
ft <- fitSpiDE(spe, condition = "condition", strata = "slide", sigma = 30, verbose = FALSE)
cfg$toy_strata <- list(fit = dump_in(ft), helpers = dump_helpers(ft, "condition", "slide"),
                       default = dump_res(testSpiDE(ft)),
                       variants = variants(ft, NULL, procs = c("heterogeneity", "all"), dfs = "proportional"))
# strata named at test time on a fit without them
ft0 <- fitSpiDE(spe, sigma = 30, verbose = FALSE)
cfg$toy_testtime <- list(fit = dump_in(ft0),
                         all_slide = dump_res(testSpiDE(ft0, condition = "condition", procedure = "all",
                                                        strata = "slide")))

# a patient dropped for one index type: S1 keeps 3 index-A cells, so A is fitted without it
cdrop <- spe
ct <- as.character(cdrop$cell_type)
a1 <- which(cdrop$sample_id == "S1" & ct == "A")
ct[a1[-(1:3)]] <- "B"
cdrop$cell_type <- ct
cdrop <- buildNiches(cdrop, sigma = 30, verbose = FALSE)
fd <- fitSpiDE(cdrop, condition = "condition", strata = "slide", sigma = 30, min.patients = 3L, verbose = FALSE)
cfg$dropped_slopes <- list(fit = dump_in(fd), helpers = dump_helpers(fd, "condition", "slide"),
                           default = dump_res(testSpiDE(fd)),
                           all = dump_res(testSpiDE(fd, procedure = "all")))

# sandwich engine
fs <- fitSpiDE(spe, engine = "sandwich", condition = "condition", sigma = 30, verbose = FALSE)
fs0 <- fitSpiDE(spe, engine = "sandwich", index = "A", sigma = 30, verbose = FALSE)
cfg$sand <- list(fit = dump_in(fs), default = dump_res(testSpiDE(fs)),
                 all = dump_res(testSpiDE(fs, procedure = "all")),
                 pooled_only = dump_res(testSpiDE(fs0)), fit0 = dump_in(fs0),
                 show = capture.output(show(testSpiDE(fs))))

# the 8-patient x 650-cell fixture
s2 <- buildNiches(spiDE:::.toySPE(n_samples = 8, n_per = 650, n_genes = 4, seed = 11), sigma = 30,
                  verbose = FALSE)
fb <- fitSpiDE(s2, sigma = 30, verbose = FALSE)
cfg$big <- list(fit = dump_in(fb), helpers = dump_helpers(fb, "condition"),
                all = dump_res(testSpiDE(fb, condition = "condition", procedure = "all")))

# a synthetic column family (test-small-n-defaults.R)
set.seed(21)
G <- 150; S <- 10
mean_expr <- exp(seq(-3, 3, length.out = G))
v <- matrix(stats::rgamma(G * S, 2, 2 / 0.05), G, S)
b <- matrix(stats::rnorm(G * S, 0.05, sqrt(v + 0.01)), G, S)
b[1, 1:2] <- NA
tau <- rep(0.01, G)
cfg$family <- list(b = b, v = v, tau = tau, mean_expr = mean_expr,
                   prop = dump_df(spiDE:::.pooledColumnTest(b, v, tau, mean_expr)),
                   cap = dump_df(spiDE:::.pooledColumnTest(b, v, tau, mean_expr, df.rule = "capped")),
                   flat = dump_df(spiDE:::.pooledColumnTest(b, v, tau, mean_expr, trend = FALSE)),
                   kish = spiDE:::.kishDf(1 / (v + tau), b, 1L),
                   tau_fam = spiDE:::.dlTau2(array(b, c(G, S, 1)), array(v, c(G, S, 1)))[, 1],
                   het_fam = spiDE:::.heterogeneityP(array(b, c(G, S, 1)), array(v, c(G, S, 1)))[, 1])
# unusable patients and heavy weights in the condition test (test-engine.R)
set.seed(3)
S <- 12; trt <- rep(0:1, each = 6)
bh <- matrix(stats::rnorm(2 * S), 2, S); vh <- matrix(1, 2, S); vh[2, c(1, 7)] <- 1e-6
cfg$heavy <- list(b = bh, v = vh, trt = trt,
                  out = dump_df(spiDE:::.robustConditionTest(bh, vh, c(0, 0), trt)))
# NaN / Inf weights, a single patient, overflow (heterogeneityP)
bo <- array(c(0, 3, 1), c(1, 3, 1)); vo <- array(c(2e-308, 2e-308, 1), c(1, 3, 1))
cfg$het_edge <- list(bo = bo, vo = vo, p = spiDE:::.heterogeneityP(bo, vo)[1, 1])

# spiDE() end to end
sp7 <- spiDE:::.toySPE(n_samples = 16, n_per = 150, n_genes = 12, seed = 7)
cfg$e2e <- list(
  slopes = dump_res(spiDE(sp7, condition = "condition", sigma = 30, index = "A", procedure = "all", verbose = FALSE)),
  sandwich = dump_res(spiDE(sp7, condition = "condition", sigma = 30, index = "A", engine = "sandwich",
                            procedure = "all", verbose = FALSE)),
  toy_default = dump_res(spiDE(toySpiDE, condition = "condition", sigma = 30, verbose = FALSE)),
  toy_capped = dump_res(spiDE(toySpiDE, condition = "condition", sigma = 30, index = "A",
                              procedure = "heterogeneity", pooled.df = "capped", fdr = 0.1, verbose = FALSE)))

# the fit behind the end-to-end slopes table, for the helpers on R's own inputs
f7 <- fitSpiDE(buildNiches(sp7, sigma = 30, verbose = FALSE), condition = "condition", sigma = 30, index = "A",
               verbose = FALSE)
cfg$e2e_fit <- list(fit = dump_in(f7), helpers = dump_helpers(f7, "condition"))

# the testthat case of test-fitSpiDE.R: a fit without a condition names one at test time
spd <- buildNiches(spiDE:::.toySPE(), sigma = 20, verbose = FALSE)
fdef <- fitSpiDE(spd, index = "A", sigma = 20, min.patients = 6, verbose = FALSE)
cfg$default_toy <- list(fit = dump_in(fdef), all = dump_res(testSpiDE(fdef, condition = "condition", procedure = "all")))

write_ref(cfg, "test")
