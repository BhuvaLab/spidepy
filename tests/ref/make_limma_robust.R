source("tests/ref/_common.R")
suppressPackageStartupMessages(library(limma))
# Reference values for the robust branch of _limma.py: fitFDistRobustly (equal and unequal df1,
# trend / no trend, outliers in either tail, small n, ties in the covariate), the robust half
# of fitFDistUnequalDF1, squeezeVar(robust = TRUE) and eBayes(robust = TRUE) on a pooled-test-
# shaped weighted lmFit with NA observations.
set.seed(20261022)

keep <- function(o) o[intersect(names(o), c("scale", "df2", "df2.shrunk", "df2.outlier",
                                           "tail.p.value", "prob.outlier"))]
nm <- function(o) setNames(o, gsub("\\.", "_", names(o)))

# scaled F variances with a mean-variance trend and outliers on chosen tails
make_x <- function(G, df1 = 6, df2 = 8, cov = NULL, hi = 0, lo = 0, hi_f = 25, lo_f = 0.02) {
  tr <- if (is.null(cov)) 0 else 0.3 * cov
  x <- exp(tr) * rchisq(G, df1) / df1 / (rchisq(G, df2) / df2)
  if (hi > 0) x[sample(G, hi)] <- x[sample(G, hi)] * hi_f
  if (lo > 0) x[sample(G, lo)] <- x[sample(G, lo)] * lo_f
  x
}

rob <- function(x, df1, cov = NULL, wtp = c(0.05, 0.1)) {
  o <- suppressWarnings(fitFDistRobustly(x, df1, covariate = cov, winsor.tail.p = wtp))
  c(list(x = x, df1 = df1, wtp = wtp), if (!is.null(cov)) list(cov = cov), nm(keep(o)))
}

fr <- list()
cv <- rnorm(300)
fr$flat_clean <- rob(make_x(300), 6)
fr$flat_hi <- rob(make_x(300, hi = 8), 6)
fr$flat_lo <- rob(make_x(300, lo = 8), 6)
fr$flat_both <- rob(make_x(300, hi = 6, lo = 6), 6)
fr$trend_clean <- rob(make_x(300, cov = cv), 6, cv)
fr$trend_hi <- rob(make_x(300, cov = cv, hi = 8), 6, cv)
fr$trend_lo <- rob(make_x(300, cov = cv, lo = 8), 6, cv)
fr$trend_both <- rob(make_x(300, cov = cv, hi = 6, lo = 6), 6, cv)
fr$big_trend <- rob(make_x(3000, df1 = 4, df2 = 12, cov = rnorm(3000), hi = 40, lo = 40), 4,
                    {c3 <- rnorm(3000); c3})
# small n
for (n in c(6, 8, 10, 12, 16, 30)) {
  c1 <- rnorm(n)
  fr[[paste0("small_flat_", n)]] <- rob(make_x(n, hi = 1), 5)
  fr[[paste0("small_trend_", n)]] <- rob(make_x(n, cov = c1, hi = 1), 5, c1)
}
fr$n3 <- rob(make_x(3), 5)
fr$n4_trend <- rob(make_x(4), 5, rnorm(4))
# df2 = Inf branch: variances from one chi-square, no prior spread
xi <- rchisq(400, 4) / 4
fr$inf_flat <- rob(xi, 4)
xi2 <- xi; xi2[1:3] <- xi2[1:3] * 12
fr$inf_flat_out <- rob(xi2, 4)
ci <- rnorm(400)
fr$inf_trend <- rob(xi * exp(0.2 * ci), 4, ci)
# df2 estimated but the non-robust is already above the lower bound
fr$df2_big <- rob(make_x(500, df1 = 3, df2 = 60, hi = 5), 3)
# tie-heavy covariate (integer, a zero spike) and a constant covariate
ct <- c(rep(0, 120), sample(1:6, 180, TRUE))
fr$ties_trend <- rob(make_x(300, cov = ct, hi = 6, lo = 3), 6, ct)
fr$ties_trend_small <- rob(make_x(14, cov = ct[c(1:7, 200:206)], hi = 1), 6, ct[c(1:7, 200:206)])
fr$const_cov <- rob(make_x(100, hi = 3), 6, rep(2, 100))
# winsor tails below 1/n: non-robust result returned
fr$wtp_small <- rob(make_x(40, hi = 2), 6, wtp = c(0.01, 0.02))
fr$wtp_other <- rob(make_x(300, cov = cv, hi = 8), 6, cv, wtp = c(0.1, 0.2))
# NA and zero-df genes set aside
xn <- make_x(200, cov = cv[1:200], hi = 4); dfn <- rep(6, 200)
xn[c(3, 9)] <- NA; dfn[c(5, 17)] <- 0
fr$notallok_trend <- rob(xn, dfn, cv[1:200])
fr$notallok_flat <- rob(xn, dfn)
cv_tie <- ct[1:200]
fr$notallok_ties <- rob(xn, dfn, cv_tie)
# very small and zero variances (offset by median * 1e-12)
xz <- make_x(150, hi = 3); xz[c(2, 8)] <- 0
fr$zero_var <- rob(xz, 6)
# unequal df1 (fitFDistRobustly maps x to the largest df)
d1 <- sample(3:9, 300, TRUE)
fr$unequal_flat <- rob(make_x(300, df1 = 6, hi = 8), d1)
fr$unequal_trend <- rob(make_x(300, df1 = 6, cov = cv, hi = 8, lo = 5), d1, cv)
fr$unequal_lo <- rob(make_x(300, df1 = 6, lo = 10), d1)
d1b <- sample(c(2, 12), 200, TRUE)
fr$unequal_wide <- rob(make_x(200, df1 = 6, hi = 5), d1b)

# fitFDistUnequalDF1 robust (direct)
ru <- function(x, df1, cov = NULL, span = NULL) {
  o <- suppressWarnings(limma:::fitFDistUnequalDF1(x, df1, covariate = cov, span = span, robust = TRUE))
  c(list(x = x, df1 = df1), if (!is.null(cov)) list(cov = cov), if (!is.null(span)) list(span = span),
    nm(keep(o)))
}
fu <- list()
du <- sample(2:9, 300, TRUE)
fu$flat_clean <- ru(make_x(300), du)
fu$flat_hi <- ru(make_x(300, hi = 8), du)
fu$flat_lo <- ru(make_x(300, lo = 8), du)
fu$flat_both <- ru(make_x(300, hi = 6, lo = 6), du)
fu$trend_hi <- ru(make_x(300, cov = cv, hi = 8), du, cv)
fu$trend_both <- ru(make_x(300, cov = cv, hi = 6, lo = 6), du, cv)
fu$trend_span <- ru(make_x(300, cov = cv, hi = 8, lo = 4), du, cv, span = 0.5)
for (n in c(6, 8, 12, 16, 40)) {
  c1 <- rnorm(n); dn <- sample(2:8, n, TRUE)
  fu[[paste0("small_flat_", n)]] <- ru(make_x(n, hi = 1), dn)
  fu[[paste0("small_trend_", n)]] <- ru(make_x(n, cov = c1, hi = 1), dn, c1)
}
fu$ties_trend <- ru(make_x(300, cov = ct, hi = 6, lo = 3), du, ct)
fu$ties_small <- ru(make_x(14, cov = ct[c(1:7, 200:206)], hi = 1, lo = 1), du[1:14], ct[c(1:7, 200:206)])
xn2 <- make_x(300, hi = 6); xn2[c(4, 20)] <- NA
dn2 <- du; dn2[c(7, 31)] <- 0
fu$na_flat <- ru(xn2, dn2)
fu$na_trend <- ru(xn2, dn2, cv)
fu$scalar_df <- ru(make_x(200, hi = 5), 6)
fu$big_trend <- ru(make_x(2500, cov = rnorm(2500), hi = 30, lo = 30), sample(2:9, 2500, TRUE),
                   rnorm(2500))
# the covariate of the last case is not saved above: redo it so x, df1 and cov agree
cb <- rnorm(2500); xb <- make_x(2500, cov = cb, hi = 30, lo = 30); db <- sample(2:9, 2500, TRUE)
fu$big_trend <- ru(xb, db, cb)
# a gene with x == 0 and another with a huge variance (right tail underflow in RightP)
xe <- make_x(100, hi = 0); xe[1] <- xe[1] * 1e6; xe[2] <- 0
fu$extreme <- ru(xe, sample(3:8, 100, TRUE))

# squeezeVar(robust = TRUE): equal df (legacy) and unequal df, trend / flat
sq <- list()
sv <- function(var, df, cov = NULL, tag = NULL) {
  o <- suppressWarnings(squeezeVar(var, df, covariate = cov, robust = TRUE))
  c(list(var = var, df = df), if (!is.null(cov)) list(cov = cov),
    list(df_prior = o$df.prior, var_prior = o$var.prior, var_post = o$var.post))
}
sq$equal_flat <- sv(make_x(300, hi = 8, lo = 4), rep(6, 300))
sq$equal_trend <- sv(make_x(300, cov = cv, hi = 8, lo = 4), rep(6, 300), cv)
sq$unequal_flat <- sv(make_x(300, hi = 8, lo = 4), du)
sq$unequal_trend <- sv(make_x(300, cov = cv, hi = 8, lo = 4), du, cv)
vz <- make_x(300, cov = cv, hi = 5); dz <- rep(6, 300); dz[c(5, 50)] <- 0; vz[c(5, 50)] <- NA
sq$df0_equal_trend <- sv(vz, dz, cv)
sq$df0_equal_flat <- sv(vz, dz)
dz2 <- du; dz2[c(5, 50)] <- 0
sq$df0_unequal_trend <- sv(vz, dz2, cv)
sq$small_equal <- sv(make_x(8, hi = 1), rep(5, 8))
sq$small_unequal <- sv(make_x(8, hi = 1), c(3, 4, 5, 6, 3, 4, 5, 6))
vs <- make_x(300, cov = cv, hi = 8)
o <- suppressWarnings(squeezeVar(vs, du, covariate = cv, span = 0.5, robust = TRUE))
sq$span_unequal <- list(var = vs, df = du, cov = cv, span = 0.5, df_prior = o$df.prior,
                        var_prior = o$var.prior, var_post = o$var.post)

# pooled-test-shaped eBayes(robust = TRUE): weighted intercept-only lmFit with NA observations,
# Amean replaced by a log-mean-expression stand-in; also equal-df and tiny gene counts
make_Y <- function(G, n, na = 0.1, zero_w = 0.05, short_rows = 0, out = 0) {
  mu <- rnorm(G, 2, 1.5)
  Y <- matrix(rnorm(G * n, mu, exp(-0.3 * mu) * 0.8 + 0.1), G, n)
  if (out > 0) {
    r <- sample(G, out)
    Y[r, ] <- Y[r, ] + matrix(rnorm(out * n, 0, 4), out, n)
  }
  W <- matrix(rexp(G * n) + 0.05, G, n)
  W[matrix(runif(G * n) < zero_w, G, n)] <- 0
  Y[matrix(runif(G * n) < na, G, n)] <- NA
  if (short_rows > 0) for (i in seq_len(short_rows)) {
    k <- if (i %% 2) integer(0) else sample(n, 1)
    Y[i, setdiff(seq_len(n), k)] <- NA
  }
  list(Y = Y, W = W, cov = mu + rnorm(G, 0, .1))
}
run_eb <- function(Y, X, W, cov) {
  fit <- suppressWarnings(lmFit(Y, X, weights = W))
  eb <- function(trend) {
    f <- fit; f$Amean <- cov
    e <- suppressWarnings(eBayes(f, robust = TRUE, trend = trend))
    list(t = unname(e$t), p_value = unname(e$p.value), df_total = e$df.total,
         s2_prior = e$s2.prior, df_prior = e$df.prior, s2_post = e$s2.post)
  }
  out <- list(input = list(Y = Y, X = X, cov = cov), eb_trend = eb(TRUE), eb_flat = eb(FALSE))
  if (!is.null(W)) out$input$W <- W
  out
}
eb <- list()
for (n in c(4, 7, 16)) {
  d <- make_Y(300, n, short_rows = 6, out = 12)
  eb[[paste0("pooled_n", n)]] <- run_eb(d$Y, matrix(1, n, 1), d$W, d$cov)
}
d <- make_Y(200, 7, na = 0, zero_w = 0, out = 10)
eb$equal_df <- run_eb(d$Y, matrix(1, 7, 1), d$W, d$cov)
eb$no_w <- run_eb(d$Y, matrix(1, 7, 1), NULL, d$cov)
# few genes (6 to 16), equal df and with NA observations
for (G in c(6, 9, 16)) {
  d <- make_Y(G, 8, na = 0, zero_w = 0)
  eb[[paste0("few_equal_", G)]] <- run_eb(d$Y, matrix(1, 8, 1), d$W, d$cov)
  d <- make_Y(G, 8, na = 0.2, zero_w = 0.05)
  eb[[paste0("few_na_", G)]] <- run_eb(d$Y, matrix(1, 8, 1), d$W, d$cov)
}
# niche-abundance-shaped: 3-column design, per-patient weights, no NA
n <- 16; cond <- rep(c(0, 1), each = n / 2); niche <- rnorm(n)
X <- cbind(`(Intercept)` = 1, niche = niche, `niche:condition` = niche * cond)
d <- make_Y(200, n, na = 0, zero_w = 0, out = 8)
eb$niche_w <- run_eb(d$Y, X, runif(n) + 0.3, d$cov)
eb$niche_nw <- run_eb(d$Y, X, NULL, d$cov)
# tie-heavy covariate in a pooled-shaped fit
d <- make_Y(300, 7, short_rows = 6, out = 10)
d$cov <- c(rep(0, 100), sample(1:6, 200, TRUE)) + 0
eb$ties_cov <- run_eb(d$Y, matrix(1, 7, 1), d$W, d$cov)

write_ref(list(fr = fr, fu = fu, sq = sq, eb = eb), "limma_robust")
