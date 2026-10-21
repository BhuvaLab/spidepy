# Differences from R spiDE

One bullet per item: what R does, what spidepy does, and why.

- **Sorting of character labels.** R sorts character labels (patients, cell types, conditions)
  in the session locale (`en_US.UTF-8`, which interleaves `a A b B` and ignores punctuation at
  first pass). `spidepy._rcompat.r_sort_str` (and Python `sorted`) uses code-point order
  (R's `method = "radix"` / `LC_COLLATE=C`), so mixed-case or punctuated labels can order
  differently (for example `b, C` in R but `C, b` here). Labels that share case and punctuation
  style, such as `S1, S10, S2` or `Non-responder, Responder`, order identically. Code-point
  order is deterministic and independent of the user's locale, and it matches the reference
  fixtures; it is the documented choice rather than emulating glibc collation.
- **Niche bandwidth lookup.** R finds niche reducedDims with the regex `^Niche[0-9.]+$`, which
  misses a bandwidth that `paste0` writes in exponent form (`Niche1e+05`). `_io.niche_keys`
  uses the same pattern, so such a niche is not found either. Matched to R because bandwidths
  of 1e5 or more are not meaningful in spatial units; `niche_key` still writes the R string.
- **Legacy-object guard.** R's `checkResults` also rejects pre-0.99.23 mixed-model objects
  (`.assertCurrent`). Not ported: spidepy has no such objects.
- **Niche kernel density.** R `densityfun` returns the values of in-window queries only
  (`drop = TRUE`) and has `diggle = FALSE` as its own default. `_kde.density_at` keeps the
  query order, returns `nan` for a query outside the window and defaults to `diggle = True`
  (the value `buildNiches` passes). The numbers for in-window queries are identical: spatstat
  uses the exact `pnorm` edge weight for a rectangular window, not a pixel image.
- **`merge_niches` group named like a leftover column.** If a group name equals a column that
  is not in any group (for example `{"B": ["A", "C"]}` with a column `B`), R produces two
  columns named `B` and its `[[` lookup of the group map sees the first. We keep both columns
  and record the first in `uns["spiDE_niche_groups"]` (a dict cannot hold duplicate keys).
- **`merge_niches` key prefix.** With `sigma = None`, R greps `^Niche` over `reducedDimNames`,
  so any key beginning with `name` (for example `NicheFoo`) is merged and given a group map,
  not only `Niche<number>` keys. `merge_niches` does the same over `obsm` keys.
