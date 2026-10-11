# E3 gradient SNR: full table

Checkpoint step 192325 of the ORIGINAL run (raw weights, 39.86M parameters, bf16 autocast, train mode), 4096 microbatches of 4 samples, training loss, random t.
Original run's effective batch CONFIRMED from its config (scripts/train_full_run.sh: torchrun --nproc_per_node=4, --batch-size 4, DDP-averaged gradients => 16; LR 1e-4, gate LR 1e-3).

## Scalars

- ||E g|| = 1.2331e-04;  mu2 = ||E g||^2 = 1.5205e-08 (jackknife SE 6.49e-10)
- tr Cov(g) per microbatch = 3.9186e-07 (jackknife SE 1.39e-08);  per-microbatch std = 6.2599e-04
- k* = trCov/||Eg||^2 = 25.77 microbatches = 103 samples.  Leave-one-chunk-out k* (8 chunks): 24.4, 25.5, 25.8, 25.8, 26.4, 26.4, 25.7, 26.3  (range 24.4-26.4); jackknife SE of k* = 1.65, i.e. approx. 95% interval [22.5, 29.0] microbatches = [90, 116] samples
- gate/adaLN-zero group (11,088,000 params): k* = 25.8 microbatches (jackknife range 24.5-26.5)
- pre-stated verdict (log, E3): 4 < k* <= 64 microbatches => AMBIGUOUS

## SNR versus batch size (SNR = 1 at k*)

| samples per step | microbatches k | SNR | note |
|---|---|---|---|
| 4 | 1 | 0.197 |  |
| 8 | 2 | 0.279 | NOT used: below noise floor |
| 16 | 4 | 0.394 | the ORIGINAL run (effective batch 16) |
| 64 | 16 | 0.788 |  |
| 103 | 25.75 | 1.000 | SNR = 1 (k*) |
| 256 | 64 | 1.576 | Phase 3a design batch |
| 1024 | 256 | 3.152 | (measured group) |

Measured directly (empirical mean squared norm of k-averaged gradients vs ||Eg||^2 + trCov/k):

| k | n groups | empirical | predicted | ratio |
|---|---|---|---|---|
| 1 | 4096 | 4.0707e-07 | 4.0707e-07 | 1.000 |
| 4 | 1024 | 1.1242e-07 | 1.1317e-07 | 0.993 |
| 16 | 256 | 3.9256e-08 | 3.9697e-08 | 0.989 |
| 64 | 64 | 2.0516e-08 | 2.1328e-08 | 0.962 |
| 256 | 16 | 1.6563e-08 | 1.6736e-08 | 0.990 |

Fraction of coordinates with per-coordinate SNR > 1 at k = 1, 4, 16, 64, 256: 0.12%, 1.62%, 10.95%, 28.51%, 47.97%

Caveats: measured at the ORIGINAL run's final (collapsed) checkpoint; SNR at initialization or for the 9M Phase 3a models can differ. The original 192k run is not a data point for any Phase 3a comparison.
