# Task 11 references: testNicheAbundance and its helpers (.pseudobulkByIndex, .patientValue,
# .abundancePair, .abundanceSetScores, .capLeverage, stats::hat), each on its own inputs, then the
# public function end to end (with / without condition, covariates, gene sets of both types, index and
# niche restrictions, a sample dropped by min.cells for one index type, a high-leverage sample).
# Also writes the fixture of test-niche-abundance.R (12 samples x 200 cells), absent from
# make_fixtures.R. Run from the repo root.
source("tests/ref/_common.R")

spe0 <- spiDE:::.toySPE(n_samples = 12, n_per = 200, n_genes = 12, composition = 3, seed = 3)
write_spe_h5ad(spe0, file.path(REF_DIR, "toyspe_s12_n200_g12_comp3_seed3.h5"),
               extra = list(call = ".toySPE(n_samples = 12, n_per = 200, n_genes = 12, composition = 3, seed = 3)"))
spe <- buildNiches(spe0, sigma = 50, verbose = FALSE)

dump_df <- function(d) {
  l <- as.list(d)
  l$colnames__ <- colnames(d)
  l
}
mat <- function(m) unname(as.matrix(m))

cd <- colData(spe)
ct <- spiDE:::.sanitise(as.character(cd$cell_type))
smp <- as.character(cd$sample_id)
Y <- assay(spe, "counts")
NM <- as.matrix(reducedDim(spe, "Niche50"))
colnames(NM) <- spiDE:::.sanitise(colnames(NM))

out <- list(niche50 = unname(NM), niche_cols = colnames(NM))

# ---- .pseudobulkByIndex: all types, a min.cells that drops samples, an index vector in odd order ----
pbdump <- function(pb) lapply(pb, function(b) list(Y = unname(b$Y), niche = unname(b$niche),
                                                  niche_rows = rownames(b$niche), ncells = b$ncells,
                                                  samples = b$samples))
pb_args <- list(
  all = list(index = NULL, min.cells = 10L, prior.count = 1),
  strict = list(index = NULL, min.cells = 70L, prior.count = 1),
  idx = list(index = c("C", "A", "zz"), min.cells = 10L, prior.count = 0.5),
  huge = list(index = NULL, min.cells = 1000L, prior.count = 1))
pbs <- list()
for (nm in names(pb_args)) {
  a <- pb_args[[nm]]
  pb <- spiDE:::.pseudobulkByIndex(Y, NM, ct, smp, a$index, a$min.cells, a$prior.count)
  pbs[[nm]] <- if (length(pb)) c(list(types = names(pb)), pbdump(pb)) else list(types = "")
}
out$pb <- pbs

# ---- .patientValue ----
pat <- c("S2", "S1", "S2", "S1", "S3", "S3", "S3")
pats <- unique(pat)
out$pv <- list(
  pat = pat, pats = pats,
  num_x = c(NA, 1.5, 2, NA, NA, NA, 7),
  num = unname(spiDE:::.patientValue(c(NA, 1.5, 2, NA, NA, NA, 7), pat, pats)),
  chr_x = c("a", "b", NA, "c", NA, "d", "e"),
  chr = unname(spiDE:::.patientValue(c("a", "b", NA, "c", NA, "d", "e"), pat, pats)))

# ---- stats::hat and .capLeverage ----
set.seed(5)
X1 <- cbind(1, c(rnorm(19), 12))
X2 <- cbind(1, c(rnorm(19), 6))
X3 <- cbind(1, seq(-1, 1, length.out = 20))
set.seed(8)
n <- 16
cond <- rep(c(-0.5, 0.5), each = 8); nv <- c(rnorm(15), 9)
X4 <- cbind(1, nv, cond, nv * cond)
X5 <- cbind(1, rnorm(10), 0)                       # an all-zero column: rank deficient
X5[, 3] <- X5[, 2] * 2
X6 <- cbind(1, c(rnorm(12), 0, 25), c(rep(-0.5, 12), 0.5, 0.5))
caps <- list(a = list(X = X1, m = 3), b = list(X = X2, m = 3), c = list(X = X3, m = 3),
             d = list(X = X4, m = 2), e = list(X = X1, m = Inf), f = list(X = X6, m = 2),
             g = list(X = X4, m = 1.5))
out$cap <- lapply(caps, function(cc) {
  r <- spiDE:::.capLeverage(cc$X, cc$m)
  list(X = unname(cc$X), m = cc$m, w = r$w, leverage = r$leverage, downweighted = r$downweighted,
       hat = stats::hat(cc$X, intercept = FALSE),
       hat_w = stats::hat(cc$X * sqrt(r$w), intercept = FALSE))
})
out$hat_rankdef <- list(X = unname(X5), hat = stats::hat(X5, intercept = FALSE))

# ---- .abundancePair ----
set.seed(4)
Yp <- matrix(rnorm(5 * 12), 5, 12, dimnames = list(paste0("g", 1:5), NULL))
dfp <- data.frame(niche = rnorm(12), condition = factor(rep(c("a", "b"), each = 6), levels = c("a", "b")))
set.seed(11)
Y16 <- matrix(rnorm(8 * 16, 5), 8, 16, dimnames = list(paste0("g", 1:8), NULL))
df16 <- data.frame(niche = c(rnorm(15), 8), condition = factor(rep(c("x", "y"), each = 8), levels = c("x", "y")),
                   age = rnorm(16, 50, 10), sex = rep(c("f", "m", "f", "m"), 4),
                   cst = rep(3, 16), stringsAsFactors = FALSE)
set.seed(12)
Ysq <- matrix(rnorm(14 * 14), 14, 14, dimnames = list(paste0("g", 1:14), NULL))      # genes == samples
dfsq <- data.frame(niche = c(rnorm(12), 0, 25),
                   condition = factor(c(rep("a", 12), "b", "b"), levels = c("a", "b")))
df_na <- df16; df_na$age[3] <- NA
df_nacond <- df16; df_nacond$condition[2] <- NA
set.seed(13)
Y5 <- matrix(rnorm(6 * 5), 6, 5, dimnames = list(paste0("g", 1:6), NULL))
df5 <- data.frame(niche = rnorm(5))
pair_cases <- list(
  plain = list(Y = Yp, df = dfp, cov = character(), m = Inf),
  one_cond = list(Y = Yp, df = transform(dfp, condition = factor(c(rep("a", 11), "b"), levels = c("a", "b"))),
                  cov = character(), m = Inf),
  cov = list(Y = Y16, df = df16, cov = c("age", "sex", "cst"), m = 3),
  cov_cond = list(Y = Y16, df = df16, cov = c("age"), m = 2),
  nocond_cap = list(Y = Y16, df = df16[c("niche", "age")], cov = "age", m = 3),
  square = list(Y = Ysq, df = dfsq, cov = character(), m = 2),
  square_pooled = list(Y = Ysq, df = dfsq["niche"], cov = character(), m = 2),
  small = list(Y = Y5, df = df5, cov = character(), m = 3),
  na_cov = list(Y = Y16, df = df_na, cov = "age", m = 3),
  na_cond = list(Y = Y16, df = df_nacond, cov = character(), m = 3),
  too_few = list(Y = Y5[, 1:3], df = df5[1:3, , drop = FALSE], cov = character(), m = 3))
out$pair <- lapply(pair_cases, function(cs) {
  r <- spiDE:::.abundancePair(cs$Y, cs$df, cs$cov, cs$m)
  d <- list(Y = unname(cs$Y), niche = cs$df$niche, cov = if (length(cs$cov)) cs$cov else "",
            m = cs$m, has_cond = as.integer("condition" %in% names(cs$df)),
            null = as.integer(is.null(r)))
  if ("condition" %in% names(cs$df)) {
    d$cond <- as.character(cs$df$condition); d$cond_levels <- levels(cs$df$condition)
  }
  for (cv in setdiff(names(cs$df), c("niche", "condition"))) {
    d[[paste0("cv_", cv)]] <- cs$df[[cv]]
  }
  if (!is.null(r)) {
    d$terms <- names(r$terms)
    for (tm in names(r$terms)) d[[paste0("t_", tm)]] <- lapply(r$terms[[tm]], unname)
    d$n <- r$n; d$leverage <- r$leverage; d$downweighted <- r$downweighted
  }
  d
})

# ---- .abundanceSetScores ----
set.seed(7)
Ys <- matrix(rnorm(30 * 10, 5), 30, 10, dimnames = list(paste0("g", 1:30), paste0("s", 1:10)))
Ys[7, ] <- 2                                                       # a constant gene is dropped
sets <- list(a = paste0("g", 1:6), b = paste0("g", 4:12), small = c("g1", "g2"),
             absent = c("x", "y", "z", "w", "v"), all = paste0("g", 1:30), big = paste0("g", 1:20))
sc_args <- list(comp = list("competitive", 5L, 500L), self = list("self-contained", 5L, 500L),
                capped = list("competitive", 5L, 10L), none = list("competitive", 50L, 500L))
out$sets <- lapply(sc_args, function(a) {
  r <- spiDE:::.abundanceSetScores(Ys, sets, a[[1]], a[[2]], a[[3]])
  if (is.null(r)) list(null = 1L) else list(null = 0L, S = unname(r$S), sets = rownames(r$S), size = unname(r$size))
})
out$sets_in <- list(Y = unname(Ys), genes = rownames(Ys))

# ---- public function ----
dump_na <- function(tab) dump_df(tab)
run <- function(s = spe, ...) testNicheAbundance(s, sigma = 50, verbose = FALSE, ...)
gs <- list(with_G2 = c("G2", "G3", "G4", "G5", "G6"), other = c("G7", "G8", "G9", "G10", "G11"),
           tiny = c("G1", "G2"), wide = paste0("G", 1:12))
pubs <- list(
  cond = run(condition = "condition"),
  nocond = run(condition = NULL),
  cov = run(condition = "condition", covariates = "Age"),
  cov_nocond = run(covariates = "Age"),
  strict = run(condition = "condition", min.cells = 70L),
  strict_nocond = run(min.cells = 70L),
  idx = run(condition = "condition", index = "A", niche = c("C", "B")),
  nocap = run(condition = "condition", max.leverage = Inf),
  cap15 = run(condition = "condition", max.leverage = 1.5),
  prior = run(condition = "condition", prior.count = 0.25),
  gs_comp = run(condition = "condition", genesets = gs),
  gs_self = run(condition = "condition", genesets = gs, type = "self-contained"),
  gs_nocond = run(genesets = gs, min.size = 2L),
  gs_strict = run(condition = "condition", genesets = gs, min.cells = 70L))
out$pub <- lapply(pubs, dump_na)

# a sample far out on the niche axis for every index type
NMhl <- reducedDim(spe, "Niche50")
NMhl[smp == "S1", ] <- NMhl[smp == "S1", ] * 25
spe_hl <- spe
reducedDim(spe_hl, "Niche50") <- NMhl
out$niche50_hl <- unname(as.matrix(NMhl))
out$pub_hl <- list(cond = dump_na(run(spe_hl, condition = "condition", max.leverage = 2)),
                   nocond = dump_na(run(spe_hl)),
                   gs = dump_na(run(spe_hl, condition = "condition", genesets = gs, max.leverage = 2)))

# errors / messages
msg <- function(expr) tryCatch({ force(expr); "" }, error = function(e) conditionMessage(e))
out$err <- list(
  no_pb = msg(run(min.cells = 1000L)),
  no_niche = msg(run(niche = "zz")),
  leverage = msg(run(max.leverage = 1)),
  genesets = msg(run(genesets = c("G1", "G2"))),
  min_size = msg(run(genesets = gs, min.size = 50L)))

write_ref(out, "abundance")
