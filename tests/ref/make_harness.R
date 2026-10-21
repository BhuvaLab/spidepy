source("tests/ref/_common.R")
write_ref(list(
  m = matrix(c(1, 2, 3, 4, 5, 6.5), 2, 3),
  v = c(1.5, NA, 3),
  s = c("a", "b"),
  x = 3.25,
  f = factor(c("u", "v", "u"), levels = c("v", "u", "w")),
  s1 = "only",
  m11 = matrix(7.5, 1, 1),
  i = c(1L, 2L, 3L),
  l = c(TRUE, FALSE, TRUE)
), "harness")
