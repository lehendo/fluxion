# Practitioner notes (each states its evidence level)

## N1. Peak learning rate (with a x10 gate LR) was the instability lever in the Phase 3 CFM runs; a clamp-specific mechanism is NOT supported (8 runs, 2 seeds per condition, 600 steps)

Setting: 9M FluxionDiT, batch 256, peak LR 5.657e-4 (square-root scaled), gate-parameter LR x10, warmup 350 then cosine; t uniform or Beta(0.25, 0.25) clamped (or truncated) to [0.01, 0.99].
Observation: at peak LR, excursions (loss/v=0 loss > 1.5 over 10 steps, grad-norm spikes up to 0.5 vs a normal 0.001-0.002) occurred in at least one seed of every condition (uniform A_s1, clamp B_s0, truncate C_s0/C_s1); two beta runs (B_s0, C_s1) then COLLAPSED to ~zero velocity (the learned solution was lost). Halving the peak LR (D) removed every excursion in both seeds, and learning began
at 180-200 steps with a stable held-out monitor. The registered definition (windowed loss > 2x v=0 after step 350) flags only B_s0; it excludes C_s1's collapse at step ~300 (on the LR ramp) and so under-counts.
Not supported: that clamping (17% of draws at exactly t = 0.01 / 0.99) is the cause. Truncation did not avoid the collapse (C_s1), and uniform sampling also had an excursion (A_s1). Not tested: uniform t at half LR; more seeds; longer horizons.
Practical recommendation that does not depend on the cause: use an in-run guard that halts or records when the windowed training loss exceeds ~2x the v = 0 baseline, apply it from the first step (not only after warmup), and be wary of a x10 gate LR at a square-root-scaled peak LR.
