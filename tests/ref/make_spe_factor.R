# A tiny SpatialExperiment whose colData carries factor columns (one with an unused level and a
# non-alphabetical level order), to check that load_spe keeps R factor levels.
# Output: tests/ref/data/spe_factor.h5, read by tests/_ref.py::load_spe("spe_factor").
source("tests/ref/_common.R")

spe <- spiDE:::.toySPE(n_samples = 2, n_per = 20, n_genes = 2, seed = 1)
n <- ncol(spe)
colData(spe)$grade <- factor(rep(c("mid", "low", "high"), length.out = n),
                             levels = c("low", "mid", "high", "unused"))
colData(spe)$arm <- factor(rep(c("b", "a"), length.out = n), levels = c("b", "a"))
write_spe_h5ad(spe, file.path(REF_DIR, "spe_factor.h5"), extra = list(call = "factor columns"))
