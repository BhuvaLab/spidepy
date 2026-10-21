# Task 3 references: buildNiches / mergeNiches / .isSelfNiche on toySpiDE, plus a raw
# spatstat densityfun() on a small hand-made ppp. Run from the repo root.
source("tests/ref/_common.R")
suppressPackageStartupMessages({
  library(spatstat.geom)
  library(spatstat.explore)
})

data("toySpiDE", package = "spiDE", envir = environment())
spe <- toySpiDE

# --- buildNiches on toySpiDE, sigma c(10, 30, 2.5) (non-default, fractional) ---
sg <- c(10, 30, 2.5)
b <- buildNiches(spe, sigma = sg)
nm <- paste0("Niche", sg)
build <- lapply(setNames(nm, nm), function(k) unname(reducedDim(b, k)))
build_cols <- colnames(reducedDim(b, nm[1]))
build_names <- reducedDimNames(b)

# --- interleaved samples, one type absent from a sample, one sample with a new type order ---
set.seed(99)
sh <- spe[, sample(ncol(spe))]
sh$cell_type <- as.character(sh$cell_type)
sh$cell_type[sh$sample_id == "S2" & sh$cell_type == "B"] <- "A"  # B absent from S2
sh$cell_type[sh$sample_id == "S3"] <- ifelse(sh$cell_type[sh$sample_id == "S3"] == "A", "C", "A")
sh$cell_type <- factor(sh$cell_type)
write_spe_h5ad(sh, file.path(REF_DIR, "niches_shuffled.h5"))
shb <- buildNiches(sh, sigma = c(15, 20))
shuf <- list(
  n15 = unname(reducedDim(shb, "Niche15")), n20 = unname(reducedDim(shb, "Niche20")),
  cols = colnames(reducedDim(shb, "Niche15"))
)
# no edge correction / non-Diggle edge correction through buildNiches
shuf$n20_ff <- unname(reducedDim(buildNiches(sh, sigma = 20, edge = FALSE, diggle = FALSE), "Niche20"))
shuf$n20_tf <- unname(reducedDim(buildNiches(sh, sigma = 20, edge = TRUE, diggle = FALSE), "Niche20"))
# custom name
shuf$named <- unname(reducedDim(buildNiches(sh, sigma = 20, name = "Nb"), "Nb20"))

# --- raw densityfun on a hand-made ppp: points near all four edges, duplicates ---
px <- c(0, 10, 0, 10, 5, 5, 0.3, 9.7, 5, 5, 2.2, 2.2, 7.1, 3.3, 8.8, 0.05, 4.4, 6.6)
py <- c(0, 0, 8, 8, 0, 8, 4, 4, 4, 4, 1.1, 1.1, 6.5, 7.7, 0.4, 3, 2.2, 5.9)
W <- owin(c(0, 10), c(0, 8))
P <- ppp(px, py, window = W, check = FALSE)
qx <- c(px, 1, 9.99, 5.5, 0, 10)
qy <- c(py, 7.9, 0.01, 4.2, 4, 4)
raw <- list(px = px, py = py, qx = qx, qy = qy)
for (s in c(0.7, 2.5, 6)) {
  for (cfg in list(c(TRUE, TRUE), c(TRUE, FALSE), c(FALSE, FALSE), c(FALSE, TRUE))) {
    f <- densityfun(P, sigma = s, edge = cfg[1], diggle = cfg[2])
    key <- sprintf("s%s_%s%s", s, substr(cfg[1], 1, 1), substr(cfg[2], 1, 1))
    raw[[key]] <- as.numeric(f(qx, qy))
  }
}
# a window that is larger than the points' bounding box and a single-point pattern
P1 <- ppp(5, 4, window = W)
raw$one <- as.numeric(densityfun(P1, sigma = 2, edge = TRUE, diggle = TRUE)(c(5, 6), c(4, 4)))

# --- mergeNiches ---
m0 <- buildNiches(spe, sigma = c(10, 20))
m1 <- mergeNiches(m0, groups = list(AC = c("A", "C")), sigma = 20)
m2 <- mergeNiches(m1, groups = list(ACB = c("AC", "B")), sigma = 20)
mall <- mergeNiches(m0, groups = list(AC = c("A", "C")))
mz <- mergeNiches(m0, groups = list(AC = c("A", "C", "Zzz"), Q = c("Nope")), sigma = 10)
gm <- function(x, k) S4Vectors::metadata(x)$spiDE_niche_groups[[k]]
merge <- list(
  before10 = unname(reducedDim(m0, "Niche10")), before20 = unname(reducedDim(m0, "Niche20")),
  before_cols = colnames(reducedDim(m0, "Niche20")),
  m1_20 = unname(reducedDim(m1, "Niche20")), m1_cols = colnames(reducedDim(m1, "Niche20")),
  m1_10_cols = colnames(reducedDim(m1, "Niche10")),
  m1_gm = gm(m1, "Niche20"),
  m1_md_names = names(S4Vectors::metadata(m1)$spiDE_niche_groups),
  m2_20 = unname(reducedDim(m2, "Niche20")), m2_cols = colnames(reducedDim(m2, "Niche20")),
  m2_gm = gm(m2, "Niche20"),
  all_10 = unname(reducedDim(mall, "Niche10")), all_cols = colnames(reducedDim(mall, "Niche10")),
  all_gm10 = gm(mall, "Niche10"), all_gm20 = gm(mall, "Niche20"),
  z_10 = unname(reducedDim(mz, "Niche10")), z_cols = colnames(reducedDim(mz, "Niche10")),
  z_gm = gm(mz, "Niche10")
)
# the ^Niche quirk: an unrelated reducedDim whose name starts with "Niche" is also merged
m3 <- m0
reducedDim(m3, "NicheFoo") <- reducedDim(m0, "Niche10")
m3e <- try(mergeNiches(m3, groups = list(AC = c("A", "C"))), silent = TRUE)
merge$quirk_ok <- as.integer(!inherits(m3e, "try-error"))
merge$quirk_foo_cols <- colnames(reducedDim(m3e, "NicheFoo"))
merge$quirk_gm_names <- names(S4Vectors::metadata(m3e)$spiDE_niche_groups)

# --- .isSelfNiche ---
isn <- spiDE:::.isSelfNiche
selfn <- list(
  c1 = list(index = c("A", "B", NA, "A", "C"), niche = c("A", "A", "A", NA, "C")),
  c2 = list(index = c("A", "B", "C", "T.cell", "x"), niche = c("AC", "AC", "AC", "T.m", "T.m"))
)
selfn$c1$res <- isn(selfn$c1$index, selfn$c1$niche)
gmap <- list(AC = c("A", "C"), B = "B", `T m` = c("T-cell", "x y"))
selfn$c2$res <- isn(selfn$c2$index, selfn$c2$niche, gmap)
selfn$c2$res_null <- isn(selfn$c2$index, selfn$c2$niche)
# merged label already used as the index label; sanitised names
selfn$c3 <- list(index = c("AC", "A", "T.m", "T.cell", "x.y", "B"),
                 niche = c("AC", "AC", "T.m", "T.m", "T.m", "AC"))
selfn$c3$res <- isn(selfn$c3$index, selfn$c3$niche, gmap)
selfn$gmap_names <- names(gmap)

write_ref(list(build = build, build_cols = build_cols, build_names = build_names,
               shuf = shuf, raw = raw, merge = merge, selfn = selfn), "niches")
