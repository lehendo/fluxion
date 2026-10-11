# Phase 3 report (DRAFT, in progress): what the controlled arms show

Status 2026-10-11. Results marked **[75%]** are from the milestone-75% (step 4,800) pre-registered evaluation; the 100% and 50% evaluations, the integration-step sweep, arm 3c and the stability diagnostic are pending. Nothing here is final.
Numbers trace to `experiments/EXPERIMENT_LOG.md` and `results/phase3/`. The original 192k-step run is **not** a data point in any comparison below (40M vs 9M parameters, padding counted in the loss, uncentered coordinates, constant LR, no held-out split,
and the differences listed in section 6): it must not be read as the low-batch end of a trend.

## 1. Design

Small FluxionDiT (9.15M parameters), per-position-centered latents, real-residue mask, 13 training proteins (12 PDB entries) and 5 held-out proteins (4 entries: 16pk, 1aol, 1bq8, 1bxy), batch 256 by accumulation, peak LR 5.657e-4 with warmup and cosine,
6,400 steps. Two arms differing only in t: uniform, and Beta(0.25, 0.25) with t clamped to [0.01, 0.99] (17.1% of treatment draws sit exactly at each bound). Pre-registered predictions of the attainable flow-loss reduction (fraction of E||delta||^2 explained
vs v = 0): uniform 5.41%, treatment 10.25%, ratio 1.89.

## 2. Both models learn the linear conditional mean, pointwise **[75%]**

On held-out proteins the flow-loss reduction R(t) of both arms lies on the measured per-channel linear-optimal curve f(t) of Phase 1 (which was measured on 17 folds and was not fitted to these arms). Raw weights, held-out, selected t:

| t | f(t) (linear-optimal, Phase 1) | uniform arm | beta arm |
|---|---|---|---|
| 0.01 | 14.71% | 14.05% | 13.91% |
| 0.21 | 5.79% | 5.69% | 5.35% |
| 0.50 | 0.00% | −0.04% | −0.14% |
| 0.79 | 6.11% | 5.87% | 5.87% |
| 0.99 | 15.14% | 14.73% | 14.70% |

The attained reduction under each arm's own sampling distribution is therefore 0.89–1.03 of the prediction (uniform raw 5.32%, beta raw 9.93% against 5.41% and 10.25%). The flow models recover the linear estimator's curve and, in this evaluation, do not exceed it
— a nonlinear network trained to this point finds no structure beyond the per-channel linear conditional mean, which is the same conclusion as Phase 1C reached from the probes.

**What this does not show.** The treatment/control ratio (1.87 [1.76, 2.06] raw; predicted 1.89) is not an independent confirmation. It is the ratio of two integrals of (nearly) the same curve against two different weightings: with R_arm(t) = f(t) for both arms it equals 1.89 exactly, and
evaluating the uniform arm's own curve under both weights gives 1.891. A reviewer would be right to say so. There is also no "decoupling" between the flow-matching objective's loss and one-step predictive quality: the one-step readout is R(t) at t ≈ 0, and the arms agree there
(14.05% vs 13.91%), which is the same fact seen at a different weight. The EMA ratio (2.18 [2.00, 2.60]) differs only because the uniform arm's EMA still lagged at t ≈ 0 at step 4,800.

## 3. Where endpoint weighting does matter: speed **[monitor, not the pre-registered eval]**

On the trainer's held-out monitor (raw weights; 16 pairs per held-out protein; one-step improvement over identity), the beta arm was above 10% by step 200 (the first monitor point above 1%; step 100 was −0.03%), the uniform arm first exceeded 1% at step 1,400 and 10% at step 2,500:
at least 7× sooner to 1% and 12× sooner to 10%, before the beta arm's instability at step ~350. That is 4–6× larger than the 1.89× difference in attainable signal and is **unexplained**. The held-out evaluations are the only numbers that count for levels; this is a training-dynamics observation
from a monitor, with one seed per arm.

## 4. The model is below the shrinkage ceiling, not above it **[75%]**

Shrinkage α·z0 with α = 0.717 fitted on the training entries, scored on the same four held-out entries: +16.47% over identity (per entry 19.20, 11.17, 15.72, 19.80). One-step model results, model minus shrinkage in percentage points:
uniform raw −1.84 [−4.65, +0.02] (1 of 4 entries positive), uniform EMA −4.33 [−6.86, −2.15] (0/4), beta raw −1.99 [−4.02, −0.57] (0/4), beta EMA −1.23 [−2.08, −0.40] (0/4). Of 32 entry-by-variant comparisons across both step counts and both weight types, one is positive.
The gap is dominated by 1bq8_A (raw −6.1 and −5.1 points), and the CIs are over four entries. State it plainly: a 9M-parameter flow model trained for 6,400 steps does not match a one-parameter mean-reversion baseline on held-out proteins at milestone 75%. The 100% evaluation decides the final level; the earlier impressions
of "equal to" and "a few tenths above" shrinkage were wrong (the first came from a 16-pair monitor, the second from three proteins of a crashed evaluation).

## 5. The integration-step gap: a closed-form prediction (inference-only sweep pending)

Eight-step Euler gives +3% to +5% against +12% to +15% for one step **[75%]**. For a stationary reversible pair the optimal field is odd about t = ½, so its integral over [0, 1] vanishes: the exact marginal flow map is the **identity**, and only the first Euler step from t = 0 yields the conditional-mean forecast.
For the Gaussian reversible model at ρ_eff = 0.695 the predicted improvement over identity is +15.2% (1 step), +11.3% (2), +6.7% (4), +3.65% (8), +1.9% (16), → 0 as steps grow. The 8-step measurement is already near this curve. The 1/2/4/8 sweep on the existing checkpoints is pre-registered in the log; a monotone decrease toward 0 *from above* is the closed-form prediction,
whereas "accumulated integration error" would also decrease monotonically but sit below that curve. The practical reading: for equal-marginal pairs the ODE sampler is not a forecaster, and one Euler step from the observed frame is the learned model's forecast.

## 6. Differences between the original run and the new setup (why 3c cannot resolve the original null)

Nine: (1) batch 16 → 256; (2) padding masked from loss and attention; (3) per-position centering; (4) 40M → 9M parameters; (5) LR schedule (constant 1e-4 → warmup + cosine, peak 5.66e-4); (6) gate LR (1e-3 → 5.66e-3); (7) EMA decay (0.9999 → 0.999);
(8) t range (affine [1e-5, 1 − 1e-5] → clamp [0.01, 0.99]); (9) held-out split and exclusion of the 768-bucket protein (13 training proteins vs 19). Arm 3c varies (1) with its tied LR/warmup/EMA scalings and holds the other eight at their new values, so a null supports the gradient-SNR account *conditional on the other fixes*
and does not establish that batch size alone explains the original result.

## 7. Treatment-arm instability, and the guard

The beta arm's training loss reached 2.19× the v = 0 loss at step ~350 (peak LR), collapsed to ~zero velocity, and re-learned. The explanation is open: clamping puts 17.1% of draws on near-clean inputs (a hypothesis), but the peak LR/gate LR (shared with the uniform arm, which had a smaller transient) is an equally supported cause. A 4-condition × 2-seed diagnostic is
queued. An in-run guard (halt on windowed loss > 2× the v = 0 loss, sentinel file, no requeue) is now in the trainer; replayed on the logged runs it fires on the beta arm at step 360 and never on the uniform arm. `docs/practitioner_notes.md` records the evidence level.
