# Experiment log (TMLR conversion)

Rules (from the project brief): raw numbers before interpretation; every experiment's
KILL CONDITION is written here and committed BEFORE the experiment runs; if a kill
condition triggers it is reported as a finding, not reinterpreted.

## Pre-registered facts established by reading code (2026-10-09, before any new compute)

- F1. Old probe baseline = predict-the-mean. `scripts/compute_nonlinear_probe.py` uses
  `z1_train.mean()` of position-centered data as "trivial". 0.001701/(1-0.9125)=0.01944
  is that variance. Identity MSE = E||z1-z0||^2 is a different, much lower baseline.
- F2. The old MLP early-stops on the held-out validation proteins' own MSE (selection on
  the test split). The new pipeline early-stops on training-side groups only.
- F3. The 72h DiT run trained on all 19 proteins (`scripts/train_full_run.sh`
  PILOT_PROTEINS), including 1bxy_A, 1bxy_B, 1btk_A. The DiT's "held-out" number
  (0.002080 vs 0.002079) is therefore an IN-SAMPLE result. It must be labelled that way.
  The DiT cannot be re-evaluated held-out without retraining, which is out of scope.
- F4. `LatentDataset` self-pairs z1->z0 within `lag` frames of each replica end. Harmless at
  lag=1, but at lag=5000 it would make ~50% of pairs zero-displacement. The lag sweep
  must exclude self-pairs.
- F5. 1bxy_A/1bxy_B (chains of 1bxy) and 1b2s_E/1b2s_F (chains of 1b2s) are siblings.
  Plain leave-one-protein-out would leak a sibling chain into training. Folds are therefore
  leave-one-PDB-entry-out (17 groups); per-protein numbers are still reported for all 19.
  This is a deliberate refinement of the brief's "leave-one-protein-out", made for the
  reason the brief itself gives.
- F6. Probe centering uses the held-out protein's own per-position mean (transductive). It
  hands the probes information identity does not have, so any "no gain over identity"
  result is conservative; a "gain over identity" result may partly be mean-reversion
  (that is exactly what Phase 1C tests).

## Phase 1A — identity-relative LOGO rescoring across lags

Design: lags {1,10,100,1000,5000} frames = {10ps,100ps,1ns,10ns,50ns}. Per (protein,lag):
600 uniformly sampled valid start frames (self-pairs excluded), 48 random positions per
frame. Predictors: mean, identity, linear-diag (per-channel OLS, matches old probe),
linear-full (A z0 + b), MLP (with linear skip), DiT (in-sample, EMA/raw, Euler 1 and 8
steps). Metric: % improvement over IDENTITY only. CIs: bootstrap over PDB-entry groups
(primary, 17 groups); per-protein block bootstrap over time blocks (block = max(500, lag)).

Step 0 (verification, before the sweep): reproduce on the OLD split (train 16 / val
1bxy_A,1bxy_B,1btk_A, lag 1) the old numbers (mean 0.01944, linear 0.001638, MLP 0.001701)
and the identity MSE (expect ~0.002079). Verify probe MSE and flow-loss MSE share a
normalization (same norm_stats, same per-token-channel averaging).

KILL CONDITION 1A: if improvement-over-identity is within its bootstrap 95% CI of zero at
EVERY lag for linear and MLP, there is no exploitable frame-to-frame signal beyond
persistence in this latent space and that is the paper's central empirical finding.
Step-0 kill: if identity MSE in the probe pipeline differs from 0.002079 by more than
sampling error, the pipelines are NOT on a common scale and every comparison so far is
suspect; stop and report before running the sweep.
