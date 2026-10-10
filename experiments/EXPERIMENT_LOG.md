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

- F7. Chai-1 bucket padding. Stored latents are [N, L_bucket, 384] with L_bucket in
  {256,384,512,768}; positions beyond the real residue count are CONSTANT (frame-std
  exactly 0, checked on 1b25_D: positions 619-767, i.e. 19% of its tokens). The DiT's
  training loss, its v_pred=0 baseline (0.002079) and the old probes all count these as
  ordinary zero-error tokens, diluting absolute MSEs (ratios between predictors are only
  approximately preserved). New pipeline: real-residue tokens only (positions with nonzero
  frame variance), plus a stored padded-fraction so Step 0 can reproduce the old weighting.

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

## Pre-registration addendum, 2026-10-09 (before the 19-protein extraction and Step 0 run)

Status of Correction 1 (brief's 89.3% / 21.2% / 18.2%): PROVISIONAL. It was backed out of
reported figures that were padding-diluted and (for the MLP) selected on the test split.
The masked Step 0 numbers, not the reconstruction, decide the framing. The MLP's true
improvement over identity is expected to be BELOW 18.2%.

S0-TOL. Step 0 "reproduces" the old numbers iff, under the DiT/old-probe weighting (bucket
padding counted as zero-error tokens), each of identity (vs 0.002079), mean (vs 0.01944) and
linear(z0,zprev) (vs 0.001638) lies within +/-10% relative of the old value. Rationale: the old
numbers each carry sampling error of a few percent (the DiT baseline used 400 samples) and
ours carries its own, so two noisy estimates should agree to ~10%. The MLP is excluded from the
pass/fail because the old value was selected on the test split. Actual deltas are reported
either way. KILL: any FAIL is a sixth finding; stop and understand before the sweep.

MASK. The padding mask is derived from sequence length (atlas_sequences.json, from the ATLAS
topology via the extractor's own residue rule). Zero frame-variance is only an assertion: every
masked position must be constant AND masked count == L_bucket - n_residues. A violation aborts.
(ATLAS raw files had been purged from scratch; re-fetched by scripts/fetch_atlas_sequences.py.)

BLOCKS / n_eff. Block length per protein per lag = ceil(lag + 2*tau_int), tau_int measured from the
latent structure function (scripts/phase1a_acf.py): rho(tau) = 1 - S(tau)/(2V), integrated until
rho < 0.05. n_eff = sum over replicas of (n_frames - lag) / block_len. The nominal pair count is
held constant at 600 starts per (protein, lag); n_eff is NOT constant across lags and is
reported next to it. The lag range (cap at 10 ns vs keep 50 ns marked uninterpretable) is decided
by the user after seeing the n_eff table, before the sweep.

LABELS. The DiT is in-sample (trained on all 19 proteins) and is keyed "DiT[in-sample]_*" in every
output; every table/figure/caption must carry that label. Probes are leave-one-PDB-entry-out.

MLP SELECTION. Early stopping uses two held-in entries only; never the held-out fold.

## Pre-registration addendum 2, 2026-10-10 (after review; supersedes S0-TOL above)

CHANGE OF CRITERION. S0-TOL (+/-10% relative) is REPLACED. Reason: S0.0 recomputes the same
quantity on the same data, so the only legitimate disagreement is sampling error; a 10% band
would hide a real discrepancy and is not defensible. New criterion: the OLD PUBLISHED value
must lie inside the bootstrap 95% CI of the new sampled estimate (blocks per measured tau_int).
KILL: if S0.0 fails this criterion, stop at S0.0 and report; that is a sixth finding. The
old +/-10% code path is removed from scripts/phase1a_fit.py.

STEP 0 IS A LADDER (one change per rung; old split {1bxy_A,1bxy_B,1btk_A} until S0.3):
  S0.0  old protocol verbatim: padding counted, probe-style pooling (equal weight per protein,
        all L positions), (z0,zprev) inputs, old MLP (hidden 256, lr 1e-3, patience 4) early-
        stopping on the TEST proteins. Targets: mean 0.01944, linear 0.001638, MLP 0.001701;
        identity 0.002079 (a DiT-eval number: token-weighted, 400 samples; a different protocol,
        reported with its own weighting).
  S0.1  + MLP early-stops on training-side entries only.
  S0.2  + padding masked (real residues only).
  S0.3  + leave-one-PDB-entry-out (17 folds), lag 1.
For S0.1-S0.3 the deltas ARE the result (no tolerance). Output: a table attributing the 91.25%
figure to baseline choice, padding dilution, selection leak and split structure.
Dense lag-1 cache for the ladder (3000 starts x 48 rows ~ 144k rows/protein, matching the old
~150k) is separate from the 600-start sweep cache. Weighting note: the old probe pooled ~150k
rows PER PROTEIN (equal weight per protein); the DiT eval was token-weighted (~L_bucket). The
two weightings are applied to their own quantities and never mixed.
Expectation recorded in advance: S0.2 moves a lot (1btk_A is 34% padding, not ~19%).

v_theta ANALYSIS (Phase 1, inference on the existing checkpoint, in-sample like all DiT numbers):
  - distribution of ||v_theta(z_t,t)|| vs ||z1-z0|| over real tokens, pooled and per protein, ratio.
  - ||v_theta|| vs t on a fixed grid of 21 values in [0.01,0.99], using true (z0,z1) lag-1 pairs.
  - overlay: empirical E[delta|z_t] per t (diagonal linear estimator fit on sufficient statistics
    over all 19 proteins, in-sample to match the DiT), plus corr(v_theta, delta).
  Pre-stated outcomes: (a) ||v|| ~ 0 at every t incl. endpoints = total collapse, learned nothing;
  (b) ||v|| small but structured (dip at t=1/2, rising to the endpoints) = learned exactly the
  signal the detailed-balance argument says is available. Either is reported; (b) would confirm the
  Phase 2 proposition from the trained model. Phase 1C (shrinkage C1-C4) is NOT run on the DiT: a
  six-digit collapse to identity means it learned no contraction; stated explicitly in the paper.

ATLAS RE-FETCH VERIFICATION (before the re-fetched data is used for anything but the mask):
  (i) sequence/residue count vs the stored L_bucket (Chai-1 bucket for n_res) and the extractor's
  constant-token count; (ii) frames per replica and stride vs the shard manifests and frame_idx;
  (iii) Mantel rho (latent vs Calpha-RMSD) recomputed on re-fetched coordinates for 3 proteins and
  compared to the rho logged when the latents were validated; a frame-misaligned re-fetch would
  collapse it. Record ATLAS release/version and fetch date. If any check fails or cannot be done:
  Phase 1B is CONDITIONAL on unverified data, stated plainly.

HOUSEKEEPING. Commits on branch tmlr up to and including the series starting at 877f8d3 and ending
at the commit that precedes this addendum's first commit carry a Co-Authored-By trailer; they predate
the standing no-trailer rule (2026-10-09). Left in place deliberately (history is part of the record,
not force-pushed). All later commits have none.

## RESULTS, 2026-10-10 (raw numbers first; interpretation separated; files in results/phase1/)

### Step 0 ladder (lag 1 = 10 ps, old split {1bxy_A,1bxy_B,1btk_A}, (z0,z_prev) inputs, old MLP)
S0.0 reproduction check (old published value inside new bootstrap 95% CI): PASS on all four.
  mean   old 0.019440  new 0.016131  CI [0.003182, 0.041470]  inside, delta -17.0%  (CI is very wide, see padding below)
  linear old 0.001638  new 0.001627  CI [0.001583, 0.001675]  inside, delta  -0.7%
  MLP    old 0.001701  new 0.001681  CI [0.001638, 0.001731]  inside, delta  -1.1%  (3 seeds: 0.001683/0.001684/0.001677)
  identity (DiT token weighting) old 0.002079  new 0.002096  CI [0.002044, 0.002153]  inside, delta +0.8%
Ladder (mean-baseline MSE | identity | linear | MLP || id/mean | lin/mean | MLP/mean || lin/id | MLP/id):
  S0.0 0.016131 0.002096 0.001627 0.001681 || 87.01% 89.92% 89.58% || 22.38% 19.78%
  S0.1 0.016131 0.002096 0.001627 0.001771 || 87.01% 89.92% 89.02% || 22.38% 15.51%   (MLP early-stops on held-in 1b8d,1b0n)
  S0.2 0.006656 0.005228 0.003771 0.004048 || 21.46% 43.35% 39.19% || 27.87% 22.57%   (padding masked)
  S0.3 0.006656 0.005228 0.003769 0.004033 || 21.46% 43.37% 39.41% || 27.91% 22.85%   (leave-one-entry-out, 3 old proteins)
  S0.3 all 19 proteins, mean % with 95% CI over PDB entries:
     linear vs identity +26.73% [+25.59,+27.65]; MLP vs identity +16.01% [+11.10,+19.99];
     identity vs mean +33.28% [+25.27,+42.45]; linear vs mean +51.45% [+46.11,+57.55]; MLP vs mean +45.48% [+41.15,+50.42]
  S0.3 per-fold MLP/identity MSE ratio: 0.74-1.05; MLP does NOT beat identity on 1b0n_A (1.005), 1b8d_K (1.047), 1bgf_A 0.967, 1bx7_A 0.978.
Padding decomposition of the S0.0 mean baseline (direct, from the dense cache, old val proteins, lag 1):
  1bxy_A: 76.5% padding rows; mean-baseline MSE real 0.00537, PADDED 0.05195; padded rows = 96.9% of its mean-baseline MSE.
  1bxy_B and 1btk_A: padded rows contribute exactly 0 (padding constant in the sampled frames).
  => ~82% of the S0.0 pooled mean baseline (0.0132 of 0.0161) is the padding constant of ONE protein (1bxy_A), an
     extraction artifact (run-dependent padding value mixed in the per-position mean), not protein motion.
Attribution of the old 91.25% (MLP vs mean): S0.0 89.58 -> S0.1 89.02 (selection leak: -0.6 pts vs mean; -4.3 pts vs identity)
  -> S0.2 39.19 (padding masked: -49.8 pts) -> S0.3 39.41 (split structure: +0.2).

### Autocorrelation / effective sample size (results/phase1/acf.json)
  lag | nominal pairs | independent windows | n_eff total | n_eff/protein median [min,max]
  10 ps 570000 | 570057 | 596.0 | 15.8 [8.1,285.7];  100 ps 569487 | 57000 | 571.0 | 15.7 [8.1,262.9]
  1 ns 564357 | 5700 | 432.9 | 14.9 [7.8,145.6];  10 ns 513057 | 570 | 193.7 | 9.3 [5.8,24.5];  50 ns 285057 | 114 | 41.5 | 2.2 [1.7,2.9]
  tau_int median 947 frames (9.5 ns), range 52 (1af7_A) to 1844 (1b8d_K). Block length lag+2*tau_int ~2000 frames even at 10 ps.

### v_theta analysis (checkpoint step 192325, IN-SAMPLE, 19 proteins, real tokens, 21 t values; results/phase1/vtheta_summary.json)
  ||v||/||u||: raw 0.01261 overall, EMA 0.00160 overall; per-protein EMA 0.00118-0.00232.
  Ratio is flat in t to 3-4 digits (EMA 0.00159-0.00162 at every t). mid(0.4-0.6)/endpoints = 0.995. No dip at t=0.5.
  corr(v,u) EMA = 0.00036-0.00037 at every t; |v-u|/|u| = 1.000001.
  Empirical linear E[delta|z_t] (fit on all 19 proteins, in-sample): RMS 0.00049 at t=0.5 vs 0.0332 at t=0.01 and 0.0338 at t=0.99;
  mean slope -0.298 at t=0.01, +0.0034 at t=0.5, +0.302 at t=0.99 (antisymmetric, crosses zero at 1/2);
  fraction of |u|^2 it explains vs the v=0 predictor: 0.144 (t=0.01), 0.0001 (t=0.5), 0.149 (t=0.99); ~0.058 averaged over the t grid.
  OUTCOME: pre-stated (a), total collapse, not (b). The model output does not track the empirical conditional mean.

### ATLAS re-fetch
  ATLAS API unreachable since 2026-10-10 ~00:35 (api/* and database/* endpoints hang, HTTP status 000 after 40 s; home page serves);
  fetch 11260924 gave up after 3h05m; 11263373 running. verify_atlas_refetch.py not run. Phase 1B (B1 Cartesian control, B2 random-init
  trunk re-extraction, B3) is BLOCKED: all three need the raw trajectories. Residue/atom/frame fingerprints from the extraction logs are
  stored (results/phase1/extraction_metadata.json) for verification if the data returns.

## Correction to the brief's Correction 1, recorded 2026-10-10 (at Arjun's request)

The brief's reconstruction of the old numbers was wrong in BOTH directions, not just one:
  - identity beats the predict-the-mean baseline by ~21% (masked, original 3 proteins: id/mean 21.46%), NOT 89%;
  - the probes beat identity by ~27% (linear 27.9% on the original 3; 26.7% [25.6,27.7] across 19 under
    leave-one-entry-out), NOT ~21%.
The structural point survives: improvement-over-mean is the wrong metric. The numbers that motivated it do not.
Both of its inputs were padding artifacts: the old mean baseline (0.01944) was ~82% the padding constant of a
single protein (1bxy_A), and the DiT's identity baseline (0.002079) was padding-diluted.

## Decisions 2026-10-10 and pre-registration for the next experiments (before any of them runs)

LAG RANGE. Analysis range 10 ps - 10 ns. 50 ns is computed but rendered greyed and labelled n_eff ~ 2/protein, not
interpretable (beyond tau_int, median 9.5 ns, one is no longer measuring dynamics). Cross-protein claims are
bootstrapped over the 17 PDB-entry folds, not rows.

E1 HEADLINE (C1): fraction of ||u||^2 explained, on identical masked rows and identical S0.3 folds, for
  (a) direct regression z0 -> z1 (per-channel linear), and (b) the optimal per-channel linear E[delta|z_t] at each of 21 t
  values (fit on 16 entries, scored on the held-out entry), uniform-t average marked; ratio (a)/(b) with CI over entries.
  No kill condition (measurement). Mechanism to be stated in the caption: t is sampled uniformly, the recoverable signal is
  odd about t=1/2 and vanishes there, so most draws land where there is little to learn. Wording rule: "the objective offers
  almost no learnable signal", never "the model found the optimum".
E2 THEORY: theory.md revised: v=0 is NOT the exact minimizer; the optimal field is nonzero, odd about t=1/2, zero at the midpoint;
  degeneracy = severe attenuation (here ~6% of the loss vs ~27% for direct regression), not exact nullity.
E3 GRADIENT SNR at the final checkpoint (raw weights, bf16 autocast as in training): per-microbatch (4 samples) gradient g of the
  training loss, unbiased estimates of ||E[g]||^2 and tr(Cov g) from M microbatches via S=sum g and sum||g||^2; SNR(k) =
  ||E g|| / sqrt(tr Cov / k) for k = 1,4,16,64,256 (k=4 microbatches = the effective batch 16 actually used), plus empirical group
  averages for validation; crossover k* = tr Cov / ||E g||^2.
  PRE-STATED INTERPRETATION: k* > 64 (>=16x the batch used) => the signal sits below the gradient noise floor at feasible batch sizes
  (a quantitative mechanism for "learned nothing"); k* <= 4 => the failure is optimization/architecture and is reported as such;
  4 < k* <= 64 => ambiguous, reported as ambiguous. Also reported for the gate/adaLN group and per-coordinate (Adam-relevant).
E4 ASYMPTOTE CHECK (lag sweep): at lags >> tau_int, z0 and z1 decorrelate: E||z1-z0||^2 -> 2 sigma^2, E||z1-mu||^2 -> sigma^2, so
  identity/mean -> 2, probe-vs-identity -> 50%, probe-vs-mean -> 0. With per-channel rho_c, direct linear regression predicts
  improvement-over-identity = (1 - rho_eff)/2 and improvement-over-mean = rho_eff'^2-type decay, both -> (50%, 0) as rho -> 0.
  If the curves do not bend toward these asymptotes as lag approaches and exceeds tau_int, the pipeline is wrong and writing stops.
  Both baselines are reported across the whole sweep, with where each dominates (identity right at short lags, wrong at long).
E5 PHASE 1C on the probes (brief's kill conditions apply): C1 contraction ratio ||zhat1-mu||/||z1-mu||; C2 eigenvalue spectrum of A in
  zhat1 = A z0 + b; C3 one-parameter shrinkage alpha*z0 (+(1-alpha)mu; mu=0 in centered coordinates), alpha by least squares; C4 cosine of
  predicted vs true displacement. KILL: if full linear does not beat shrinkage by more than the CI, or cosine ~0 while MSE improves, the
  gain is mean reversion with no directional information and "dynamics" must not appear near it.
ATLAS SALVAGE: check mirrors (Zenodo / institutional / bulk archive); check whether pairwise C-alpha RMSD matrices from the original Mantel
  validation were logged (a distance-matrix-level B1 would then be possible); keep the patient retry running; do not block on it.
PHASE 1 REPORTING: Phase 1 = 1A + 1C, with 1B explicitly marked BLOCKED and PENDING.

## RETRACTION and ATLAS plan, 2026-10-10 (before the verification runs)

RETRACTION. The earlier report "the ATLAS API is down" was WRONG. With HTTP/1.1 and a normal User-Agent the API answers from the
cluster in 0.5 s, and it answered from outside the cluster too. Two separate things were misread as an outage: (1) the cluster's
default HTTP/2 requests to API endpoints hang; (2) /ATLAS/protein/{id} (3 x 10,000 frames) is zipped on the fly and hits the gateway
timeout (HTTP 502 after exactly 5 min), and my fetcher's 'server up' probe used exactly that slow endpoint. /ATLAS/analysis/{id}
(3 replicas x 1000 frames, saved every 100 ps) downloads in ~10 s. Independent confirmation from the live metadata endpoint:
1btk_A has 169 residues (extraction log: 169). ATLAS release v2 = 2024-11-18; 1btk_A_analysis.zip last-modified 2025-11-21.

FRAME MAP. Analysis frame k (100 ps) <-> latent frame_idx 10k (our 10 ps frames). Cartesian controls therefore reach lags >= 100 ps only.

VERIFICATION (scripts/verify_atlas_refetch.py, rewritten for the analysis layout) PASS iff, for all 19 proteins:
  (i) re-fetched PDB residue count == extraction-log n_residues AND protein atom count == extraction-log real_atoms (exact);
  (ii) n_frames == 1001 per replica at dt == 100 ps (== every 10th of 10001 latent frames);
  (iii) alignment: Spearman(pairwise C-alpha RMSD, pairwise latent distance) on 250 frames of replica 1, latent frame_idx = 10k:
        aligned rho >= 0.20 AND aligned rho exceeds every cross-replica NEGATIVE-CONTROL rho (R1 coordinates vs R2 / R3 latents) by >= 0.15.
        (a frame-misaligned or re-versioned trajectory would collapse the aligned rho)
  Any failure => B1 is run only on the proteins that pass and the rest are reported as unverified.

B1 CARTESIAN CONTROL (needs the verified coordinates). Per protein, leave-one-replica-out (3 folds), frames at 100 ps stride, lags
  100 ps, 1 ns, 10 ns (50 ns computed, marked uninterpretable). Representations, both reduced to the SAME per-protein dimension
  k = min(3*n_res, 384): (a) Chai-1 latents (real tokens, flattened, PCA to k), (b) aligned C-alpha coordinates (flattened, PCA to k).
  Identical probe suite (identity, mean, linear_diag, linear_full, shrink, MLP), reported as % over identity. Statistic: paired
  difference (Chai-1 minus Cartesian) per protein, 95% CI over PDB entries.
  KILL: if Cartesian PCA matches (CI of the paired difference includes 0) or beats (CI < 0) Chai-1 at lags <= 10 ns, the premise
  'pretrained trunk latents encode dynamics beyond coordinates' is falsified and becomes the paper's headline, not an appendix.
B2 (random-init trunk) is now feasible in principle (coordinates available, 1000 frames/replica) but needs a Chai-1 re-extraction
  with re-initialized weights; NOT started. B3 (sequence-only floor): the pre-fix latents ARE that condition; partial result
  from scripts/b3_prefix_variance.py (frame variance corrected vs pre-fix).

B3 STATUS, 2026-10-10. The pre-fix latents (latents_prefix_broken_20260813) were also purged from scratch (232 KB of empty directories
remain), so the planned frame-variance comparison cannot be run and the script was removed. B3's expected result (sequence-only,
no template => zero frame variance) is documented in docs/code_review_2026-08.md from the original investigation: pre-fix, a protein's
latents were bit-identical across frames, and the coordinate-consuming features (TokenDistanceRestraint, TokenPairPocketRestraint,
MissingChainContact) were IDENTICAL_ACROSS_CONFORMATIONS. Re-extracting a sequence-only condition would reproduce one latent N times by
construction and is not informative. B3 therefore rests on the earlier documentation, not on new data.
