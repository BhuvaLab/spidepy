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
