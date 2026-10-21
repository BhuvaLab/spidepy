# SpatialExperiment fixtures as HDF5, one file per distinct call in the spiDE testthat suite
# (grep `.toySPE(` over tests/testthat at e31e744; there are no .toyNiche()/.toyClustered()
# calls and `toySpiDE` is only used in examples) plus the shipped `toySpiDE`.
# Output: tests/ref/data/<name>.h5, read by tests/_ref.py::load_spe(name).
source("tests/ref/_common.R")

toy <- spiDE:::.toySPE
calls <- list(
  toyspe_default = list(),
  toyspe_comp3 = list(composition = 3),
  toyspe_s12_comp3 = list(n_samples = 12, composition = 3),
  toyspe_s16_n120_g12_seed2 = list(n_samples = 16, n_per = 120, n_genes = 12, seed = 2),
  toyspe_s16_n150_g12_seed7 = list(n_samples = 16, n_per = 150, n_genes = 12, seed = 7),
  toyspe_s16_n150_g6_seed5 = list(n_samples = 16, n_per = 150, n_genes = 6, seed = 5),
  toyspe_s16_n150_seed7 = list(n_samples = 16, n_per = 150, seed = 7),
  toyspe_s16_n150_seed99 = list(n_samples = 16, n_per = 150, seed = 99),
  toyspe_s2_n20_g2_seed1 = list(n_samples = 2, n_per = 20, n_genes = 2, seed = 1),
  toyspe_s8_n120_g6_seed3 = list(n_samples = 8, n_per = 120, n_genes = 6, seed = 3),
  toyspe_s8_n120_g8_seed11 = list(n_samples = 8, n_per = 120, n_genes = 8, seed = 11),
  toyspe_s8_n650_g4_seed11 = list(n_samples = 8, n_per = 650, n_genes = 4, seed = 11)
)

for (nm in names(calls)) {
  spe <- do.call(toy, calls[[nm]])
  call_txt <- paste0(".toySPE(", paste(names(calls[[nm]]), unlist(calls[[nm]]), sep = " = ",
                                       collapse = ", "), ")")
  write_spe_h5ad(spe, file.path(REF_DIR, paste0(nm, ".h5")), extra = list(call = call_txt))
}

data("toySpiDE", package = "spiDE", envir = environment())
write_spe_h5ad(toySpiDE, file.path(REF_DIR, "toySpiDE.h5"), extra = list(call = "data(toySpiDE)"))

# `composition = 0` is the default call, so it is the same object as toyspe_default.
stopifnot(identical(unname(assay(toy(composition = 0))), unname(assay(toy()))))
# the shipped dataset is .toySPE(n_samples = 16, n_per = 150, n_genes = 20, seed = 7)
stopifnot(identical(unname(assay(toySpiDE)), unname(assay(toy(n_samples = 16, n_per = 150, seed = 7)))))
