# Tables regenerated from paper_results/ (scripts/make_tables.py)

## Mean-reversion headline: shrinkage gain over identity at 10 ps (Section 4.1)

| step | improvement over identity |
|---|---|
| own mean over all replicas, noisy latents (as first measured) | 15.2% [13.0, 17.0] |
| mean from the other replicas, noisy latents | 12.5% [10.4, 14.3] |
| noise removed, measured per protein (headline) | 10.3% [8.5, 11.7] |

## MSE(identity) / MSE(mean baseline) by lag (Section 3.3)

| lag | 10 ps | 100 ps | 1 ns | 10 ns |
|---|---|---|---|---|
| own mean (all replicas) | 0.669 | 0.825 | 1.039 | 1.418 |
| predicted from the autocorrelation, 2(1 - rho) | 0.694 | 0.810 | 1.019 | 1.388 |
| mean of the other replicas | 0.570 | 0.702 | 0.883 | 1.192 |
| pooled mean of the training proteins | 0.006 | 0.007 | 0.009 | 0.013 |

## In-sample improvement over identity, end of the matched budget (Section 7.1)

| arm | padding | centering | in-sample improvement over identity |
|---|---|---|---|
| 3c | masked | centered | 18.91% |
| T3c | masked | uncentered | 10.43% |
| T3a | counted | uncentered | 6.82% |
| T3b (seed 0) | counted | centered | -0.0005% |
| T3b (seed 1) | counted | centered | -0.0006% |

## Per-protein extraction noise (Appendix A.1)

| protein | residues | noise variance (1e-4 per element) | two-extraction difference, % of the 10 ps step | noise share of the latent variance (%) |
|---|---|---|---|---|
| 1ab1_A | 46 | 5.56 | 14.6 | 5.6 |
| 1bq8_A | 54 | 6.59 | 29.6 | 9.2 |
| 1bx7_A | 55 | 6.23 | 11.3 | 2.8 |
| 1bxy_A | 60 | 8.26 | 33.2 | 15.8 |
| 1bxy_B | 60 | 8.13 | 34.2 | 15.2 |
| 1ail_A | 73 | 5.88 | 13.4 | 5.1 |
| 1b2s_E | 90 | 6.64 | 28.6 | 13.0 |
| 1b2s_F | 90 | 6.44 | 27.2 | 11.9 |
| 1b0n_A | 111 | 7.97 | 41.3 | 8.2 |
| 1bgf_A | 124 | 7.12 | 15.8 | 4.2 |
| 1a62_A | 130 | 5.26 | 18.9 | 6.0 |
| 1b8d_K | 164 | 6.06 | 7.9 | 1.7 |
| 1btk_A | 169 | 5.40 | 17.4 | 5.7 |
| 1aol_A | 228 | 3.55 | 7.2 | 1.8 |
| 1ah7_A | 245 | 5.78 | 13.1 | 4.3 |
| 1af7_A | 274 | 6.70 | 19.8 | 6.7 |
| 1bkp_A | 278 | 3.70 | 8.3 | 3.1 |
| 16pk_A | 415 | 2.87 | 8.6 | 4.0 |
| 1b25_D | 619 | 3.19 | 6.8 | 2.5 |

## Headline against a common factor on every noise variance (Appendix A.3)

| factor | noise-free gain, 10 ps |
|---|---|
| 0.5 | 11.5% [9.5, 13.0] |
| 0.75 | 10.9% [9.0, 12.4] |
| 1 (measured) | 10.3% [8.5, 11.7] |
| 1.25 | 9.6% [7.9, 11.1] |
| 1.5 | 9.0% [7.3, 10.6] |
| 2.0 | 7.5% [5.6, 9.4] |

## Held-out R(t) against the two reference curves, % of the squared displacement explained (Appendix A.7)

| t | f(t), 19 proteins | f(t), matched | uniform EMA | uniform raw | Beta EMA | Beta raw | 3c EMA |
|---|---|---|---|---|---|---|---|
| 0.01 | 14.71 | 15.94 | 14.74 | 14.64 | 14.86 | 12.49 | 12.99 |
| 0.23 | 4.92 | 5.42 | 5.09 | 4.89 | 5.02 | 3.83 | 3.24 |
| 0.50 | -0.00 | -0.00 | -0.01 | -0.05 | -0.06 | -0.22 | -0.00 |
| 0.79 | 6.11 | 6.34 | 5.89 | 6.00 | 5.75 | 4.34 | 3.74 |
| 0.99 | 15.14 | 15.89 | 14.64 | 14.99 | 14.62 | 13.54 | 12.59 |
