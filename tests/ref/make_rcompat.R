source("tests/ref/_common.R")
vals <- c(10, 30, 2.5, 0.1, 1e5, 123456789, 1e-5, 1 / 3, 100000, 1e15,
          0, -2.5, 1e-4, 1234567, 100000.5, 3e-10, 1e22, 0.30000000000000004, 2 / 3,
          1e5 + 1, 150000, -1e5, 12345678901234567, 1e15 + 2, 2^60, 0.1 + 0.2, 12345678901234.5, 3333333333.33333, 0.0001234, 1e100, 1e-100, 5e-324,
          NaN, Inf, -Inf)
p <- c(0.01, 0.04, NA, 0.03, 0.005, 0.5, NA, 0.5, 1e-300, 0.2)
strs <- c("S1", "S10", "S2", "b", "A", "a", "B", "Non-responder", "Responder", "_x", "1", "s")
write_ref(list(
  num = vals,
  num_str = vapply(vals, function(v) paste0(v), ""),
  p = p,
  p_bh = p.adjust(p, method = "BH"),
  p_all = p[!is.na(p)],
  p_all_bh = p.adjust(p[!is.na(p)], method = "BH"),
  san_in = c("T cell", "Non-responder", "a-b c", "plain", "x  y--z"),
  san_out = gsub(" |-", ".", c("T cell", "Non-responder", "a-b c", "plain", "x  y--z")),
  strs = strs,
  sort_default = sort(strs),
  sort_radix = sort(strs, method = "radix"),
  locale_collate = Sys.getlocale("LC_COLLATE")
), "rcompat")
