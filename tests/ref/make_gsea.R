# Task 12 references: spiGSEA on R's own results objects. The fits are the ones make_test.R writes
# (rebuilt here with the same calls; Python loads them from test.h5); only the gene-set outputs and
# .setSlopes arrays are written. Run from the repo root.
source("tests/ref/_common.R")

dump_df <- function(d) {
  l <- as.list(d)
  l$colnames__ <- colnames(d)
  l
}
gs_dump <- function(...) {
  d <- spiGSEA(...)
  if (!nrow(d)) return(list(empty = 1L))
  dump_df(d)
}

data("toySpiDE", package = "spiDE", envir = environment())
num <- as.integer(sub("S", "", as.character(toySpiDE$sample_id)))
toySpiDE$slide <- ifelse(num %% 4 < 2, "slide1", "slide2")
spe <- buildNiches(toySpiDE, sigma = 30, verbose = FALSE)
spe16 <- buildNiches(spiDE:::.toySPE(n_samples = 16, n_per = 120, n_genes = 12, seed = 2), sigma = 30,
                     verbose = FALSE)

cfg <- list()

# the testthat fixture: 16 patients x 12 genes, index A (seed 7), planted effect in niche B
sp7 <- spiDE:::.toySPE(n_samples = 16, n_per = 150, n_genes = 12, seed = 7)
res <- spiDE(sp7, condition = "condition", sigma = 30, index = "A", procedure = "all", verbose = FALSE)
sets <- list(with_g1 = paste0("G", 1:3), without = paste0("G", 4:8), rest = paste0("G", 9:12), tiny = "G1")
sets_l <- lapply(sets, as.character)
cfg$e2e <- list(
  sets = sets_l,
  default = gs_dump(res, sets, min.size = 3),
  self = gs_dump(res, sets, type = "self-contained", min.size = 3),
  pooled = gs_dump(res, sets, test = "pooled", min.size = 3),
  cond = gs_dump(res, sets, test = "condition", min.size = 3),
  capped = gs_dump(res, sets, pooled.df = "capped", min.size = 3),
  proportional = gs_dump(testSpiDE(res@fit, condition = "condition", procedure = "all", pooled.df = "capped"),
                         sets, pooled.df = "proportional", min.size = 3),
  inherit_capped = gs_dump(testSpiDE(res@fit, condition = "condition", procedure = "all", pooled.df = "capped"),
                           sets, min.size = 3),
  size_default = gs_dump(res, sets),
  max2 = gs_dump(res, sets, min.size = 1, max.size = 3),
  none = gs_dump(res, sets, min.size = 50))
x <- res@fit@index$A
mem <- list(a = match(sets$with_g1, x$genes), b = match(sets$without, x$genes))
sl <- function(type) {
  s <- spiDE:::.setSlopes(x, mem, type)
  list(b = unname(s$b), v = unname(s$v), mean_expr = unname(s$mean_expr))
}
cfg$setslopes <- list(comp = sl("competitive"), self = sl("self-contained"))

# a family of >= 20 sets, so the pooled prior is trended (random subsets of the 12 genes)
set.seed(11)
big <- setNames(lapply(1:24, function(i) sort(sample(x$genes, sample(3:6, 1)))), paste0("set", 1:24))
cfg$big <- list(sets = lapply(big, as.character), out = gs_dump(res, big, min.size = 3))

# a set with no usable patients drops out (test-spiGSEA.R)
r2 <- res
x2 <- r2@fit@index$A
jB <- match("B", x2$niches)
x2$beta[1:2, , jB] <- NA
r2@fit@index$A <- x2
s3 <- list(dead = x2$genes[1:2], setA = x2$genes[4:6], setB = x2$genes[7:9])
cfg$dead <- list(sets = s3, out = gs_dump(r2, s3, min.size = 2, test = "pooled"),
                 both = gs_dump(r2, s3, min.size = 2))

# 16 patients, recorded condition absent: the condition is named at test time
f16 <- fitSpiDE(spe16, index = "A", sigma = 30, verbose = FALSE)
r16 <- testSpiDE(f16, condition = "condition", procedure = "all")
cfg$sl16 <- list(sets = sets_l, out = gs_dump(r16, sets, min.size = 3),
                 nocond = gs_dump(testSpiDE(f16), sets, min.size = 3))

# the toy fixture: three index types, slide strata recorded; and a dropped patient
ft <- fitSpiDE(spe, condition = "condition", strata = "slide", sigma = 30, verbose = FALSE)
rt <- testSpiDE(ft, procedure = "all")
toy_genes <- rownames(spe)
toy_sets <- list(s1 = toy_genes[1:5], s2 = toy_genes[6:12], s3 = toy_genes[c(3, 9, 15, 18)],
                 s4 = toy_genes[13:20])
cfg$toy <- list(sets = lapply(toy_sets, as.character), out = gs_dump(rt, toy_sets),
                self = gs_dump(rt, toy_sets, type = "self-contained", test = "condition"))
cdrop <- spe
ct <- as.character(cdrop$cell_type)
a1 <- which(cdrop$sample_id == "S1" & ct == "A")
ct[a1[-(1:3)]] <- "B"
cdrop$cell_type <- ct
cdrop <- buildNiches(cdrop, sigma = 30, verbose = FALSE)
fd <- fitSpiDE(cdrop, condition = "condition", strata = "slide", sigma = 30, min.patients = 3L, verbose = FALSE)
cfg$dropped <- list(sets = lapply(toy_sets, as.character),
                    out = gs_dump(testSpiDE(fd, procedure = "all"), toy_sets))

write_ref(cfg, "gsea")
