# Shared helpers for reference generation. Scripts are run from the repo root.
suppressPackageStartupMessages({
  library(SpaNorm, lib.loc = "tests/ref/Rlib")
  library(spiDE, lib.loc = "tests/ref/Rlib")
  library(SummarizedExperiment)
  library(SingleCellExperiment)
  library(SpatialExperiment)
  library(rhdf5)
})

REF_DIR <- "tests/ref/data"
dir.create(REF_DIR, showWarnings = FALSE, recursive = TRUE)

`%||%` <- function(a, b) if (is.null(a)) b else a

.write_elem <- function(file, name, x) {
  if (is.list(x)) {
    h5createGroup(file, name)
    for (n in names(x)) .write_elem(file, paste0(name, "/", n), x[[n]])
    return(invisible())
  }
  lev <- NULL
  d <- dim(x) %||% length(x)
  if (is.factor(x)) {
    lev <- levels(x)
    x <- as.character(x)
  }
  if (is.logical(x)) {
    # logical -> int8
    h5createDataset(file, name, dims = d, storage.mode = "integer", size = 1)
    h5write(array(as.integer(x), dim = d), file, name)
  } else {
    h5write(array(x, dim = d), file, name)
  }
  if (!is.null(lev)) {
    fid <- H5Fopen(file)
    did <- H5Dopen(fid, name)
    h5writeAttribute(lev, did, "levels")
    H5Dclose(did)
    H5Fclose(fid)
  }
}

.write_file <- function(x, file) {
  if (file.exists(file)) file.remove(file)
  h5createFile(file)
  for (n in names(x)) .write_elem(file, n, x[[n]])
  invisible(file)
}

# Write a named list to tests/ref/data/<topic>.h5 (nested lists become groups,
# logical becomes int8, factors become character + a `levels` attribute).
write_ref <- function(x, topic) .write_file(x, file.path(REF_DIR, paste0(topic, ".h5")))

# Write a SpatialExperiment as plain HDF5 (no zellkonverter); tests/_ref.py::load_spe turns it
# into an AnnData. Layout (R shapes are kept; load_ref transposes back):
#   counts        cells x genes, R's storage type (double for the toy data)
#   cell_names, gene_names
#   obs/<col>     one dataset per colData column (factors carry a `levels` attribute)
#   obs_columns   colData column order (HDF5 groups iterate alphabetically)
#   spatial       cells x D spatialCoords, spatial_names its column names
#   reduced/<nm>  cells x k reducedDims, reduced_names their order (omitted when none)
#   dim           c(n_genes, n_cells) as R reports them
write_spe_h5ad <- function(spe, path, extra = list()) {
  cd <- as.data.frame(colData(spe))
  sc <- spatialCoords(spe)
  x <- list(
    counts = t(as.matrix(assay(spe, "counts"))),
    cell_names = colnames(spe),
    gene_names = rownames(spe),
    obs = as.list(cd),
    obs_columns = colnames(cd),
    spatial = unname(sc),
    spatial_names = colnames(sc),
    dim = dim(spe)
  )
  rd <- reducedDims(spe)
  if (length(rd)) {
    x$reduced <- lapply(as.list(rd), function(m) unname(as.matrix(m)))
    x$reduced_names <- names(rd)
  }
  .write_file(c(x, extra), path)
}
