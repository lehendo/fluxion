# Phase 3 report (DRAFT, in progress): what the controlled arms show

Status 2026-10-11. Results marked **[75%]** / **[100%]** are from the pre-registered held-out evaluations at milestones 75% (step 4,800) and 100% (step 6,400); the 50% evaluation, the in-sample columns, the sweep at 25%/75%, arm 3c and the stability diagnostic are pending. Nothing here is final.
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

**[100%]** The same holds at the end of training: attained held-out flow-loss reduction uniform 5.46% (EMA) / 5.44% (raw) vs 5.41% predicted (1.01); beta 10.27% (EMA, 1.00) / 8.81% (raw, 0.86) vs 10.25%; ratio 1.88 [1.78, 2.10] (EMA), 1.62 [1.49, 1.69] (raw). The pre-registered pass condition (both arms > 0, ratio in [1.5, 2.3]) is met on point estimates for both weight types; the raw-weights ratio is below the predicted 1.89 with a CI that excludes it, because the beta arm's raw weights degraded between 75% and 100% (9.93% → 8.81%).

**What this does not show.** The treatment/control ratio (1.87 [1.76, 2.06] raw; predicted 1.89) is not an independent confirmation. It is the ratio of two integrals of (nearly) the same curve against two different weightings: with R_arm(t) = f(t) for both arms it equals 1.89 exactly, and
evaluating the uniform arm's own curve under both weights gives 1.891. A reviewer would be right to say so. There is also no "decoupling" between the flow-matching objective's loss and one-step predictive quality: the one-step readout is R(t) at t ≈ 0, and the arms agree there
(14.05% vs 13.91%), which is the same fact seen at a different weight. The EMA ratio (2.18 [2.00, 2.60]) differs only because the uniform arm's EMA still lagged at t ≈ 0 at step 4,800.

## 3. Where endpoint weighting does matter: speed **[monitor, not the pre-registered eval]**

On the trainer's held-out monitor (raw weights; 16 pairs per held-out protein; one-step improvement over identity), the beta arm was above 10% by step 200 (the first monitor point above 1%; step 100 was −0.03%), the uniform arm first exceeded 1% at step 1,400 and 10% at step 2,500:
at least 7× sooner to 1% and 12× sooner to 10%, before the beta arm's instability at step ~350. That is 4–6× larger than the 1.89× difference in attainable signal and is **unexplained**. The held-out evaluations are the only numbers that count for levels; this is a training-dynamics observation
from a monitor, with one seed per arm.

## 4. Against the shrinkage ceiling: about one point below at the end, with one protein driving the gap

Shrinkage α·z0 (α = 0.717, fitted on the training entries), scored on the same four held-out entries: +16.47% over identity (per entry 19.20, 11.17, 15.72, 19.80). One-step model results at **[100%]**, model minus shrinkage in points [CI over 4 entries], entries positive:
uniform EMA −1.16 [−3.80, +0.48] (2/4), uniform raw −1.26 [−3.77, +0.29] (2/4), beta EMA −1.05 [−2.79, +0.03] (1/4), beta raw −3.33 [−6.17, −1.56] (0/4). At **[75%]** the same deltas were −1 to −4 points with 1 of 32 comparisons positive; training from 75% to 100% closed most of the EMA gap.
The deficit is concentrated in **1bq8_A** (−3.6 to −7.6 points across variants); on 16pk_A, 1aol_A and 1bxy the one-step model sits within about ±1 point of shrinkage (EMA weights: +0.57, +0.40, −0.42 for uniform). The plain statement: a 9M-parameter flow model after 6,400 steps matches the one-parameter mean-reversion baseline to
within roughly a point on three of four held-out entries and is clearly below it on one; the pooled gap is about one point with a CI that includes 0 for the EMA weights. It is not above shrinkage, and "recovers the shrinkage solution plus a small increment" is not supported. Four entries and the dominance of one protein limit what can be said; the earlier impressions
("equal to", "a few tenths above", "1–4 points below") each reflected a smaller or earlier slice of the data. With more steps the gap might close; the milestone trend (75% to 100%) points that way but is two points.

## 5. The integration-step gap: closed-form direction confirmed, magnitudes not **[100%]**

Held-out improvement over identity for k-step Euler from t = 0.01 (uniform EMA): 15.31, 13.22, 8.17, 4.52% for k = 1, 2, 4, 8 (raw: 15.21, 13.42, 8.73, 5.38; beta EMA 15.42, 12.86, 7.59, 3.79; beta raw 13.15, 13.81, 8.45, 4.28). The pre-registered closed form for the exact optimal field (Gaussian reversible, ρ_eff = 0.695) predicts 15.23, 11.32, 6.71, 3.65, going to 0 as k grows because the exact marginal flow map is the identity.
Observed: a monotone decrease in 3 of 4 variants (beta raw is non-monotone between k = 1 and 2, where its one-step value is itself degraded), staying positive at k = 8, and sitting about 1–2 points above the Gaussian curve at k = 2, 4, 8. "Accumulated integration error on a near-zero field" would predict values below that curve and eventually below zero; none is observed. So the direction is what the identity-map argument predicts and the specific
magnitudes are not reproduced (measured decay is slower than the single-ρ Gaussian curve; per-channel heterogeneity in ρ is the obvious untested reason). The practical reading stands: for equal-marginal pairs the ODE is not a forecaster, and one Euler step from the observed frame is the learned model's forecast; at k = 8 the model is 11–13 points below shrinkage.

## 6. Differences between the original run and the new setup (why 3c cannot resolve the original null)

Nine: (1) batch 16 → 256; (2) padding masked from loss and attention; (3) per-position centering; (4) 40M → 9M parameters; (5) LR schedule (constant 1e-4 → warmup + cosine, peak 5.66e-4); (6) gate LR (1e-3 → 5.66e-3); (7) EMA decay (0.9999 → 0.999);
(8) t range (affine [1e-5, 1 − 1e-5] → clamp [0.01, 0.99]); (9) held-out split and exclusion of the 768-bucket protein (13 training proteins vs 19). Arm 3c varies (1) with its tied LR/warmup/EMA scalings and holds the other eight at their new values, so a null supports the gradient-SNR account *conditional on the other fixes*
and does not establish that batch size alone explains the original result.

## 7. Treatment-arm instability, and the guard

The beta arm's training loss reached 2.19× the v = 0 loss at step ~350 (peak LR), collapsed to ~zero velocity, and re-learned. The explanation is open: clamping puts 17.1% of draws on near-clean inputs (a hypothesis), but the peak LR/gate LR (shared with the uniform arm, which had a smaller transient) is an equally supported cause. A 4-condition × 2-seed diagnostic is
queued. An in-run guard (halt on windowed loss > 2× the v = 0 loss, sentinel file, no requeue) is now in the trainer; replayed on the logged runs it fires on the beta arm at step 360 and never on the uniform arm. `docs/practitioner_notes.md` records the evidence level.
