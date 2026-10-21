source("tests/ref/_common.R")
suppressPackageStartupMessages(library(limma))
# Reference values for _limma.py: lmFit (weights, NA observations, zero weights, rank
# deficiency), fitFDist (trend / no trend), squeezeVar (equal-df "legacy" and unequal-df paths,
# trend / no trend) and eBayes(robust = FALSE), plus weightedLowess, trigammaInverse and
# statmod's logmdigamma.
set.seed(20261021)

# One weighted problem: lmFit, then eBayes with and without the trend (Amean replaced by `cov`,
# as spiDE does). `W` may be a matrix, a vector (array or gene weights) or NULL.
run_case <- function(Y, X, W, cov, trend_only = FALSE) {
  fit <- suppressWarnings(lmFit(Y, X, weights = W))
  keep <- list(coefficients = unname(fit$coefficients),
               stdev_unscaled = unname(fit$stdev.unscaled),
               sigma = fit$sigma, df_residual = fit$df.residual, Amean = fit$Amean)
  eb <- function(trend) {
    f <- fit
    f$Amean <- cov
    e <- suppressWarnings(eBayes(f, robust = FALSE, trend = trend))
    list(t = unname(e$t), p_value = unname(e$p.value), df_total = e$df.total,
         s2_prior = e$s2.prior, df_prior = e$df.prior, s2_post = e$s2.post)
  }
  out <- list(input = list(Y = Y, X = X, cov = cov), lm = keep,
              eb_trend = eb(TRUE), eb_flat = eb(FALSE))
  if (!is.null(W)) out$input$W <- W
  out
}

# random expression-like data with a mean-variance trend, NA observations and zero weights
make_Y <- function(G, n, na = 0.1, zero_w = 0.05, short_rows = 0) {
  mu <- rnorm(G, 2, 1.5)
  Y <- matrix(rnorm(G * n, mu, exp(-0.3 * mu) * 0.8 + 0.1), G, n)
  W <- matrix(rexp(G * n) + 0.05, G, n)
  W[matrix(runif(G * n) < zero_w, G, n)] <- 0
  Y[matrix(runif(G * n) < na, G, n)] <- NA
  if (short_rows > 0) {
    r <- seq_len(short_rows)
    for (i in r) {
      keep <- if (i %% 2) integer(0) else sample(n, 1)
      Y[i, setdiff(seq_len(n), keep)] <- NA
    }
  }
  list(Y = Y, W = W, cov = mu + rnorm(G, 0, .1))
}

cases <- list()
for (n in c(4, 7, 16)) {
  d <- make_Y(200, n, short_rows = 6)
  cases[[paste0("pooled_n", n)]] <- run_case(d$Y, matrix(1, n, 1), d$W, d$cov)
}
# no NA, positive weights: every row has the same df (the equal-df "legacy" path)
d <- make_Y(200, 7, na = 0, zero_w = 0)
cases$equal_df <- run_case(d$Y, matrix(1, 7, 1), d$W, d$cov)
# array weights (length n) and gene weights (length G) as plain vectors
cases$array_w <- run_case(d$Y, matrix(1, 7, 1), runif(7) + 0.2, d$cov)
cases$gene_w <- run_case(d$Y, matrix(1, 7, 1), runif(200) + 0.2, d$cov)
cases$no_w <- run_case(d$Y, matrix(1, 7, 1), NULL, d$cov)
# niche-abundance-like design: intercept, niche, niche:condition, with per-patient weights
n <- 16
cond <- rep(c(0, 1), each = n / 2)
niche <- rnorm(n)
X <- cbind(`(Intercept)` = 1, niche = niche, `niche:condition` = niche * cond)
d <- make_Y(200, n, na = 0, zero_w = 0)
cases$niche_w <- run_case(d$Y, X, runif(n) + 0.3, d$cov)
cases$niche_nw <- run_case(d$Y, X, NULL, d$cov)
# the same design with NA observations (some genes lose their niche variation: aliased column)
d2 <- make_Y(200, n, na = 0.25, zero_w = 0.05)
d2$Y[1, ] <- NA
d2$Y[1, 1:3] <- rnorm(3)          # 3 observations, rank 3 -> df 0
d2$Y[2, ] <- NA
d2$Y[2, c(1, 2, 9, 10)] <- rnorm(4)
X2 <- X
X2[, "niche"] <- c(rep(0.5, 8), rnorm(8))   # constant niche over patients 1:8 ...
d2$Y[3:12, 9:16] <- NA                      # ... and genes 3:12 observed only there
cases$niche_na <- run_case(d2$Y, X2, d2$W, d2$cov)
# a larger family, so weightedLowess interpolates between fitted points (delta > 0)
d <- make_Y(1500, 7, short_rows = 10)
cases$big_unequal <- run_case(d$Y, matrix(1, 7, 1), d$W, d$cov)

# fitFDist directly
fd <- function(x, df1, covariate = NULL) {
  o <- suppressWarnings(if (is.null(covariate)) fitFDist(x, df1) else fitFDist(x, df1, covariate = covariate))
  list(scale = o$scale, df2 = o$df2)
}
G <- 120
s2 <- rchisq(G, 5) / 5 * exp(rnorm(G, 0, .3))
cv <- rnorm(G)
fdist <- list(
  plain = c(list(x = s2, df1 = 5), fd(s2, 5)),
  trend = c(list(x = s2, df1 = 5, cov = cv), fd(s2, 5, cv))
)
x2 <- s2; x2[c(3, 9)] <- NA; x2[5] <- Inf
df2v <- rep(5, G); df2v[c(7, 11)] <- 0; df2v[20] <- 1
cv2 <- cv; cv2[c(2, 4)] <- c(Inf, -Inf)
fdist$notallok <- c(list(x = x2, df1 = df2v, cov = cv2), fd(x2, df2v, cv2))
fdist$notallok_flat <- c(list(x = x2, df1 = df2v), fd(x2, df2v))
dfv <- sample(2:9, G, TRUE)
fdist$unequal_df1 <- c(list(x = s2, df1 = dfv, cov = cv), fd(s2, dfv, cv))
fdist$small_trend <- c(list(x = s2[1:5], df1 = 5, cov = cv[1:5]), fd(s2[1:5], 5, cv[1:5]))
fdist$mid_trend <- c(list(x = s2[1:20], df1 = 5, cov = cv[1:20]), fd(s2[1:20], 5, cv[1:20]))
fdist$const_cov <- c(list(x = s2, df1 = 5, cov = rep(1, G)), fd(s2, 5, rep(1, G)))

# squeezeVar directly (robust = FALSE): equal / unequal df, trend / flat, df = 0 rows
sv <- function(var, df, covariate = NULL) {
  o <- suppressWarnings(squeezeVar(var, df, covariate = covariate))
  list(var = var, df = df, df_prior = o$df.prior, var_prior = o$var.prior, var_post = o$var.post)
}
G <- 300
v <- rchisq(G, 6) / 6 * exp(rnorm(G, 0, .4)); cvv <- rnorm(G)
dfu <- sample(1:8, G, TRUE)
dfz <- dfu; dfz[c(5, 50)] <- 0; vz <- v; vz[c(5, 50)] <- NA
sqz <- list(
  equal_flat = sv(v, rep(6, G)),
  equal_trend = c(sv(v, rep(6, G), cvv), list(cov = cvv)),
  unequal_flat = sv(v, dfu),
  unequal_trend = c(sv(v, dfu, cvv), list(cov = cvv)),
  df0_flat = sv(vz, dfz),
  df0_trend = c(sv(vz, dfz, cvv), list(cov = cvv)),
  df0_equal_trend = c(sv(replace(v, 5, NA), replace(rep(6, G), 5, 0), cvv), list(cov = cvv)),
  tiny = sv(c(1, 2), c(3, 3))
)
sqz$tiny$cov <- 0

# weightedLowess, trigammaInverse, logmdigamma
wl <- list()
for (k in 1:5) {
  n <- c(30, 80, 200, 700, 3000)[k]
  x <- if (k %% 2) rnorm(n) else runif(n, 0, 10)
  y <- sin(x) + rnorm(n, sd = .3)
  w <- if (k < 4) rexp(n) + .05 else pmin(1 / trigamma(sample(1:15, n, TRUE) / 2), 100)
  span <- c(.3, .5, .7, limma::chooseLowessSpan(n, small.n = 500), .4)[k]
  f <- weightedLowess(x, y, weights = w, span = span, iterations = 1, npts = 200)
  wl[[paste0("c", k)]] <- list(x = x, y = y, w = w, span = span, fitted = f$fitted)
}
xs <- c(1e-8, 1e-4, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1e4, 1e6, 5e7)
tgi <- list(x = xs, y = vapply(xs, function(v) limma::trigammaInverse(v), 0))
zs <- c(0.005, 0.1, 0.5, 1, 2.5, 4.9, 5, 7, 20, 100, 1e3, 1e5)
lmd <- list(z = zs, y = statmod::logmdigamma(zs))

write_ref(c(cases, list(fdist = fdist, sqz = sqz, wl = wl, tgi = tgi, lmd = lmd)), "limma")
