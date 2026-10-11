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

=====================================================================================================================
2026-10-10 (later). RESULTS: ATLAS verification (job 11272956) and the lag sweep + Phase 1C (job 11271942).
Raw files: results/phase1/atlas/{atlas_verification.json,verification_table.txt}, results/phase1/sweep/{across_proteins,asymptote,onec}.json,
sweep_stdout.txt. Numbers first; interpretation after.

ATLAS VERIFICATION, as pre-registered. Criteria (i) residue/atom counts and (ii) frame count/dt/atoms: PASS for 19/19 (exact matches).
Criterion (iii) (aligned Mantel rho >= 0.20 AND exceeds every negative control by >= 0.15): PASS for 8/19, FAIL for 11/19.
  aligned rho / max cross-replica control rho / shift control (full table in verification_table.txt)
  PASS: 1ab1_A .648/.002/.040   1ah7_A .674/.365/.281   1aol_A .609/.446/.232   1b2s_F .600/.165/-.080
        1bq8_A .536/.252/-.018  1btk_A .603/.363/.071   1bxy_A .555/.058/.251   1bxy_B .580/.105/.019
  FAIL: 16pk_A .223/.191   1a62_A .328/.310   1af7_A .618/.541   1ail_A .071/.005 (aligned < .20)   1b0n_A .548/.402 (shift .459)
        1b25_D .685/.627   1b2s_E .664/.558   1b8d_K .704/.571   1bgf_A .314/.185 (shift .352)   1bkp_A .488/.410   1bx7_A .260/.126
Two honesty notes on the verdict. (1) The script's control set includes the half-window circular-shift control, which the text of this
pre-registration (line ~246) did not list (it listed cross-replica only). It is MORE stringent; I checked every verdict by hand and none
changes if the shift control is dropped (closest case: 1b0n_A margin .146 vs .15 threshold). (2) Reading of the failures: cross-replica
negative-control rho is high (up to .63) for most failing proteins. All three replicas start from one structure and relax along similar
paths, so pairwise-distance matrices on 250 frames share a slow trend even across replicas; the control is therefore not a clean null for
those proteins. That is a hypothesis about WHY the criterion fails; it has NOT been tested. It does not turn a FAIL into a PASS.
CONSEQUENCE (pre-registered): B1 runs only on the 8 passing proteins (7 PDB entries: 1bxy_A/B share an entry); the other 11 are
reported as "unverified for criterion (iii); (i) and (ii) pass". A detrended/short-lag alignment check on the 11 would be a POST HOC
control; it is NOT being run without Arjun's decision.

B1 KILL CONDITION, restated before launch: on the verified proteins, at lags <= 10 ns, paired difference (Chai-1 PCA minus Cartesian PCA,
matched dimension k = min(3 n_res, 384), leave-one-replica-out, % over identity), 95% CI over PDB entries. If the CI includes 0 or lies
below 0, 'trunk latents encode dynamics beyond coordinates' is falsified and becomes the paper's headline. Power note: 7 entries, so wide CIs.

SWEEP (19 proteins, 17 PDB-entry folds, masked real tokens, S0.3 held-out folds; 95% CI over entries; 50 ns column NOT interpretable, n_eff~2):
  improvement over IDENTITY (%)   10 ps        100 ps       1 ns         10 ns        50 ns*
    linear_diag                   +15.23       +19.10       +24.84       +34.69       +49.62
    shrink (alpha*z0)             +15.22       +19.09       +24.84       +34.70       +49.66
    linear_full                   +14.81       +18.58       +24.11       +33.86       +48.62
    MLP                           +13.39       +17.60       +23.32       +33.17       +48.34
    mean                          -60.28       -28.76       -0.49 [-12.6,+9.3]  +27.79  +49.62
    DiT (in-sample) raw/EMA, 1 or 8 Euler steps: between -0.02 and +0.01 at every lag (CIs at or around 0)
  improvement over MEAN (%)       10 ps        100 ps       1 ns         10 ns        50 ns*
    identity (= DiT row)          +33.1        +17.5        -3.9         -41.8        -104.7
    linear_diag                   +43.97       +34.14       +22.89       +8.50        -0.14
  ASYMPTOTE (E4): id/mean error ratio measured 0.669, 0.825, 1.039, 1.418, 2.047 [1.89,2.23] at 10 ps..50 ns vs 2(1-rho_ACF) 0.694, 0.810,
  1.019, 1.388, 1.926; probe-vs-identity -> 49.6% (50% predicted); probe-vs-mean -> -0.14% (0 predicted). Differences between measured and
  ACF-predicted id/mean are small but CI excludes 0 at 1 ns (+.020) and 10 ns (+.030): the ACF-based prediction is not exact.
  Crossover: mean baseline beats identity from ~1 ns on (id/mean = 1.04 there); identity beats mean below that. Probe gain over identity
  at 10 ps (15.23%) equals the headline direct-regression f(0) = 15.23% from phase1_headline.json (independent code path, same rows).
  Lag-1 contradiction check: the log line ~191 says identity beats mean by ~21% (original 3 proteins, id/mean = 21.46%); on 19 proteins
  with fold-held-out evaluation the same quantity at 10 ps is 33.1%. Different protein sets and row sampling; both are 'identity beats mean
  at short lag'. The paper must quote the 19-protein number and define it (mean over proteins of per-protein ratios).

PHASE 1C (probes only; kill condition pre-registered at E5): alpha = 0.704, 0.632, 0.525, 0.334, 0.037 (10 ps ... 50 ns). Eigenvalues of A at
10 ps: max|lambda| = 0.911, median Re = 0.640, fraction |lambda|>0.9 = 0.002, fraction complex = 0.000.
  full linear over shrinkage: -0.44% [-1.04,+0.06] (10 ps), -0.58% [-1.19,-0.08]* (100 ps), -0.91% [-1.50,-0.45]* (1 ns), -1.25% [-1.56,-1.00]* (10 ns)
  linear_diag over shrinkage: +0.01% [-0.01,+0.03] (10 ps); MLP over shrinkage -2.04% [-3.38,-0.91]* at 10 ps.
  cos(predicted, true displacement): shrinkage alone .404; linear_full .398; 'cosine beyond shrinkage' +.05.
KILL CONDITION TRIGGERED (first clause): full linear does not beat one-parameter shrinkage beyond the CI at any lag (it is worse at
100 ps, 1 ns, 10 ns). The per-residue linear predictability of z1 from z0 in these latents is mean reversion (contraction by alpha toward
the per-position mean), with no direction-specific structure the 384x384 map can add. Scope: this is per-residue-token (384-d) rows; no
cross-residue coupling was tested. 'Dynamics' must not be claimed on the strength of the probe gains.
The in-sample DiT is indistinguishable from identity at every lag: it learned a velocity field whose Euler integral does nothing to z0.
GRADIENT SNR (11271943) still running (1024/4096 microbatches at last look); result pending.

=====================================================================================================================
2026-10-10 (later still). B1 RESULT (job 11273095; scripts/phase1b_cartesian.py, scripts/phase1b_report.py; results/phase1/b1/).
Verified proteins only: 1ab1_A 1ah7_A 1aol_A 1b2s_F 1bq8_A 1btk_A 1bxy_A 1bxy_B = 7 PDB entries (1bxy_A/B averaged to one value).
Frames at 100 ps stride, 1001/replica, leave-one-replica-out, both representations PCA-reduced to k = min(3 n_res, 384) (PCA fit on training replicas only).
Paired difference = Chai-1 % over identity MINUS Cartesian % over identity; 95% percentile-bootstrap CI over the 7 entries; no best-of selection (all predictors shown).
  linear_diag   100 ps +3.69 [-1.94,+10.89]   1 ns -0.41 [-4.52,+4.36]   10 ns -3.35 [-7.61,+2.34]   (50 ns -2.87 [-8.02,+5.12], uninterpretable)
  shrink        100 ps -2.79 [-6.62,+1.04]    1 ns -4.32 [-8.73,+0.17]   10 ns -5.43 [-10.25,-0.05]  (Cartesian better, CI < 0)
  linear_full   +13.04 [-0.51,+28.59] / +10.98 [-3.37,+29.13] / +9.37 [-7.37,+32.34]; MLP +12.04 / -7.74 / -10.04 (all CIs include 0)
  id/mean ratio (Chai - Cart)  100 ps -0.03 [-0.17,+0.08]   1 ns -0.09 [-0.24,+0.03]   10 ns -0.14 [-0.28,-0.02]
  entries with Chai > Cartesian: between 1/7 and 4/7 for every cell; exact one-sided sign-flip p >= 0.117 everywhere for 'Chai better'.
  Absolute levels, linear_diag % over identity: Chai 9.4 / 11.1 / 15.8 vs Cartesian 5.7 / 11.5 / 19.2 at 100 ps / 1 ns / 10 ns.
KILL CONDITION (restated before launch) MET: no CI of the paired difference lies above 0 at any lag <= 10 ns for any predictor; for several
cells (10 ns shrink, 10 ns id/mean) it lies below 0. Under the pre-registered rule the premise 'frozen trunk latents encode dynamics beyond
coordinates' is NOT supported by this control, and the finding moves from appendix to headline-level.
Caveats, all of which weaken what 'not supported' can mean: (a) only 7 entries / 8 proteins passed verification (11 unverified, see above), so
CIs are wide and the test has little power to detect a modest Chai-1 advantage; absence of a CI above 0 is not evidence of equality;
(b) only lags >= 100 ps are reachable (coordinates are every 100 ps); the lag-1 regime where the DiT was trained is untested here;
(c) per-protein training sets are ~2000 pairs, so the 384x384 full linear map and the MLP overfit in BOTH representations (negative % over identity
in the table); linear_diag and shrink are the trustworthy rows, and neither shows a Chai-1 advantage; (d) Chai-1 latents are themselves computed
from the coordinates (template/restraint features), so a Cartesian representation matching them is the expected null, not a surprise;
(e) 6 predictors x 4 lags are shown, two CIs excluding 0 (both favouring Cartesian) are not multiplicity-corrected.
B2 (random-init trunk) not started. GRADIENT SNR (11271943) still running at last look.

=====================================================================================================================
2026-10-10 (evening). GRADIENT SNR RESULT (E3; job 11271943, 81 min; results/phase1/gradsnr.json, gradsnr_stdout.txt).
Final checkpoint (step 192,325, raw weights, 39.86M params), 4096 microbatches of 4 samples, training loss, random t.
  ||E g|| = 1.233e-4;  per-microbatch std sqrt(tr Cov) = 6.260e-4;  k* = tr Cov/||E g||^2 = 25.8 microbatches (= effective batch ~103 samples)
  jackknife over 8 chunks: k* in [24.4, 26.4];  gate/adaLN group k* = 25.8.
  SNR(k) for k = 1, 4, 16, 64, 256 microbatches: 0.197, 0.394, 0.788, 1.576, 3.152.   Real training used k = 4 (effective batch 16): SNR = 0.39.
  Validation: empirical mean ||avg of k grads||^2 matches ||Eg||^2 + trCov/k within 4% at every k (k=64: 2.05e-8 vs 2.13e-8).
  Per-coordinate SNR > 1 at k = 1, 4, 16, 64, 256: 0.12%, 1.6%, 10.9%, 28.5%, 48.0% of coordinates.
PRE-STATED INTERPRETATION APPLIES: 4 < k* <= 64 => AMBIGUOUS. This result neither establishes that the signal is below the gradient noise floor at
feasible batch sizes (that needed k* > 64) nor that the failure is optimization/architecture (k* <= 4). What it does show: at the batch actually used,
the mean gradient is ~2.5x smaller than the per-step noise (SNR 0.39), and about 6.5x more samples per step (batch ~103) would be needed for SNR = 1;
whether Adam's momentum/averaging over ~192k steps compensates is NOT tested here. ||E g|| is nonzero (SNR is well-measured, jackknife SE small),
so the checkpoint is not at a stationary point of the training loss.
Not claimed: that the trained DiT 'could' learn the attenuated signal with a larger batch. The in-sample DiT equals identity at every lag (sweep result above).

=====================================================================================================================
2026-10-10 (night). CORRECTION TO "CORRECTION 2": THE OT-COUPLING LEG DOES NOT APPLY TO THE TRAINED MODEL.
Fact (read from code, training/train.py): the trained temporal DiT pairs z0 = frame i, z1 = frame i+1 and passes the pair to FlowMatcher unchanged;
train.py never imports training/ot_coupling.py. Only training/train_static.py (noise -> equilibrium frame, a different model, not analysed in the TMLR
work) applies minibatch OT. Consequently any statement that the temporal model's collapse is partly due to OT re-coupling is withdrawn. The coupling
argument survives only as a general remark about OT-CFM as commonly configured and about train_static.py (theory.md section 6).
Propagated to: theory.md (header, section 6, wording 'trained CFM model'), README.md (correction notice at top), docs/novelty_assessment.md (notice item 4),
this log. STILL TO PROPAGATE: the research-summary doc's correction block (done in the restructure, step 6 of the order of work) and project memory.
Also recorded: the README title/abstract still say 'OT-CFM'; the paper must describe the temporal model as 'CFM with the observed adjacent-frame coupling'.

=====================================================================================================================
2026-10-10 (night). POST HOC REVISION OF VERIFICATION CRITERION (iii), pre-registered here BEFORE the new numbers are computed.
LABEL: POST HOC. The original criterion (iii) was badly specified by me, not by the data. The three ATLAS replicas of a protein start from the
same structure, so a cross-replica negative control cannot be near zero; its observed 0.15-0.63 range on the failing proteins is expected, not
anomalous. The right negative control for 'are these the same frames' is a time-shifted WITHIN-replica control (same trajectory, wrong time).
REVISED (iii'): aligned Spearman rho >= 0.20 AND aligned - max over within-replica circular time shifts of the latent sequence by 1/4, 1/2 and 3/4
of the 250-frame window (25, 50, 75 ns) >= 0.15. Cross-replica rho (R2, R3) is still computed and REPORTED as a confound, but is not gating.
Already seen before this pre-registration: the half-window shift column of the first run. The 1/4 and 3/4 shifts are new; the revised criterion
takes the max over all three, and the half-window-only variant is also reported. The original and revised verdicts are shown side by side; (i), (ii) unchanged.
Reason not to rescue proteins: criterion thresholds (0.20, 0.15) are unchanged. If the revised criterion passes more proteins, B1 is re-run on the
revised set and reported under BOTH the 8-protein and revised-set versions; B1's kill condition is evaluated on each.

=====================================================================================================================
2026-10-11. RESULT OF THE POST HOC REVISED CRITERION (iii') (job 11275134; results/phase1/atlas/verification_revised_table.txt, atlas_verification_revised.json).
Criteria (i), (ii): PASS 19/19 (unchanged). Criterion (iii): pre-registered original PASS 8/19; REVISED (iii') PASS 13/19; half-window-only variant PASS 15/19.
  aligned rho | within-replica shifts 1/4, 1/2, 3/4 | cross-replica R2, R3 (confound, not gating)
  newly PASS under revised: 1af7_A .618 | .151 -.067 .020 | .541 .512     1b25_D .685 | .172 .017 .191 | .445 .627     1b2s_E .664 | .194 .367 .030 | .558 .350
                            1b8d_K .704 | .018 -.129 -.075 | .571 .199    1bkp_A .488 | .153 .253 .199 | .410 .312
  still FAIL under revised: 16pk_A .223 | .101 .163 .140 (margin .06)   1a62_A .328 | .326 -.005 .212 (1/4 shift .326 ~ aligned)   1ail_A .071 (< .20)
                            1b0n_A .548 | .240 .459 .309 (margin .09)   1bgf_A .314 | .037 .352 .009 (margin -.04)   1bx7_A .260 | .069 .042 .287 (margin -.03)
  (1a62_A and 1bx7_A pass under the half-window-only variant; the max over three shifts rejects them.)
Thresholds (0.20, 0.15) unchanged. Honest reading: six proteins stay unverified under the stricter within-replica max control. For 1b0n_A and 1bgf_A the
half-window shifted latents correlate with the Calpha RMSD matrix as strongly as the aligned ones, i.e. these trajectories revisit similar structures
at 25-75 ns offsets, so the Mantel test cannot discriminate; that is a limit of the test on those proteins, not evidence of a frame mismatch. 1ail_A has
aligned rho .071: Calpha RMSD and the latent distance are only weakly related there, regardless of control. None of the six is called mismatched.
The first failure analysis stands as the explanation of the original criterion's failure (cross-replica controls are confounded by the shared starting
structure; cross-replica rho 0.15-0.63 is expected), and the original verdicts remain reported side by side with the revised ones everywhere.
B1 RE-RUN on the revised set (13 proteins = 11 PDB entries; 1b2s_E/F share an entry, 1bxy_A/B share an entry): the five new proteins are run now
(job below); B1 is then reported under BOTH the 8-protein (7-entry) set and the 13-protein (11-entry) set. KILL CONDITION unchanged and evaluated on each:
the premise 'no learnability advantage' is the null; the finding that Chai-1 offers an advantage requires the paired-difference CI (linear_diag and shrinkage rows)
to lie above 0 at lags <= 10 ns; if the CI includes 0 or lies below 0 the 'no advantage' statement stands.
CLAIM LANGUAGE (fixed here, for every document): Chai-1 latents are a deterministic function of coordinates plus sequence, so B1 cannot show the latents contain
less information. B1 tests learnability only. The claim is: 'frozen Chai-1 trunk latents offer no learnability advantage over matched-dimension aligned-Calpha PCA for
linear or MLP probes at lags >= 100 ps.' Matching Cartesian is the expected null; this is a failure to find an advantage, not a surprise. The interesting counterfactual
(an advantage at 10 ps, where the DiT trained) is unreachable with 100 ps analysis frames. Comparison rests on the identity-relative and shrinkage-relative rows;
the full-linear and MLP rows overfit at ~2000 pairs per protein in both representations and are not used for the claim.

=====================================================================================================================
2026-10-11. B1 RESULT ON THE REVISED SET, side by side with the 8-protein set (job 11275535 for the 5 new proteins; results/phase1/b1/b1_report.{txt,json}).
Paired difference, Chai-1 minus Cartesian PCA, % over identity, 95% bootstrap CI over PDB entries.
                          8 proteins / 7 entries                                   13 proteins / 11 entries
  shrink       100 ps  -2.79 [-6.62,+1.04]   1 ns -4.32 [-8.73,+0.17]   10 ns -5.43 [-10.25,-0.05]      +0.56 [-2.85,+3.89]   -0.17 [-4.28,+4.06]   -0.53 [-5.50,+5.01]
  linear_diag  100 ps  +3.69 [-1.94,+10.89]  1 ns -0.41 [-4.52,+4.36]   10 ns -3.35 [-7.61,+2.34]       +7.10 [+2.10,+12.47]* +4.35 [-0.32,+9.34]   +4.60 [-2.33,+12.90]
  id/mean ratio 100 ps -0.03 [-0.17,+0.08]   1 ns -0.09 [-0.24,+0.03]  10 ns -0.14 [-0.28,-0.02]       +0.06 [-0.06,+0.16]   +0.02 [-0.12,+0.14]   -0.01 [-0.15,+0.14]
  (full linear and MLP rows are in b1_report.txt; they overfit in both spaces and are not used.  * = CI above 0.)
Absolute levels, 13 proteins, 100 ps / 1 ns / 10 ns, % over identity:  shrink Chai 10.9/14.7/21.5 vs Cart 10.4/14.9/22.1;  linear_diag Chai 4.6/5.5/11.2 vs Cart -2.5/1.1/6.6.
READING (stated plainly, including what does not fit the earlier B1 entry):
 (1) The earlier entry's wording 'no CI above 0 at any lag for any predictor' was true for the 8-protein set and is NOT true for the 13-protein set: per-channel linear shows
     Chai-1 ahead by +7.10 [+2.10,+12.47] at 100 ps (1 of 3 lags, 1 of the 2 trusted rows; at 1 ns the CI is [-0.32,+9.34]).
 (2) The only row where the representations are compared at the BEST probe in each space is one-parameter shrinkage (it beats per-channel linear, full linear and the MLP in BOTH spaces at every
     lag). There the difference is within +/-0.6 points of 0 at every lag (13 set) with CIs of about +/-4 points: indistinguishable, with limited power to see a modest difference.
 (3) The linear_diag lead is a comparison of an over-parameterised probe that both representations' own shrinkage baseline beats (Chai 4.6% vs its shrinkage 10.9%; Cartesian -2.5% vs its
     shrinkage 10.4%): per-component slopes fit on ~2000 pairs over up to 384 PCA components. The Cartesian PCA components are noisier to fit, so this is most simply an estimation-noise difference
     between probes that should not be used, not a representation advantage. This is an interpretation I have not tested (e.g. by ridge/cross-validated per-channel fits).
 (4) Multiplicity: 2 trusted rows x 3 lags = 6 cells per set, one CI excluding 0 (favouring Chai-1) on the 13 set; the 8-set had two excluding 0 favouring Cartesian (10 ns shrink, 10 ns id/mean). Neither is corrected.
KILL CONDITION as restated on 2026-10-11 ('a Chai-1 advantage requires the CI of the paired difference (linear_diag, shrinkage rows) to lie above 0 at lags <= 10 ns'): NOT met for an advantage
(shrinkage: no lag; linear_diag: 1 of 3 lags, and only on the larger set). The 'no advantage' statement therefore stands only in this form:
  'On the shrinkage-referenced comparison, frozen Chai-1 trunk latents and matched-dimension aligned-C-alpha PCA are indistinguishable for lags >= 100 ps (paired differences within about +/-0.6 points, CIs about +/-4);
   a per-channel-linear lead for Chai-1 at 100 ps on the 13-protein set (+7.1 [+2.1,+12.5]) concerns a probe both spaces' shrinkage baseline beats.'
This is a learnability statement, not an information statement, and the earlier phrase 'offer no learnability advantage for linear or MLP probes' is withdrawn as too broad.
B1 fig: results/phase1/figs/fig4_b1_paired.png.

=====================================================================================================================
2026-10-11. DOCUMENTS UPDATED: docs/phase1_report.md written (correction-chain table, both verification criteria, B1 on both sets, SNR, prior art); the research-summary doc
now has a 'Current state' section on top (retitled framing, what died/survived, correction chain) and in-place correction annotations on its two OT-CFM passages;
original text and earlier correction blocks are kept. Project memory updated. OT-coupling correction is now propagated to: theory.md, README.md, docs/novelty_assessment.md,
the log, the summary doc, memory.

=====================================================================================================================
2026-10-11. PHASE 3a PRE-REGISTRATION (written BEFORE any 3a arm is trained). Status: pipeline built and CPU-smoke-tested; the treatment arm's t distribution is PENDING
Arjun's decision (see 'OPEN DECISION'); nothing below has been run on the cluster except the per-position centering statistics (job 11275974).
PURPOSE. Not 'fix the model': test whether attenuation under uniform-t sampling is the mechanism behind the trained DiT's collapse to identity.
THE ONE CONTROLLED CHANGE: t ~ Uniform(0,1) vs t drawn toward the endpoints. Everything else identical across the two arms: architecture (FluxionDiT d_model 256, 6 layers,
8 heads, d_ff 1024, ~9M params), data, split, seed, optimizer (AdamW lr 1e-4, gate lr x10, warmup 2000, clip 1, EMA 0.999), batch size 8, step count, loss (plain MSE, no importance
weights). Two arms = the full two-configuration budget; 3b (conditional p(z1|z0)) therefore goes to FUTURE_WORK.md unless the limit is lifted. Budget: <=12 GPU-hours total, one H100 per arm.
CHANGES VS THE ORIGINAL 72 h RUN, shared by both arms and stated so nobody mistakes 3a for a re-run of it: (1) latents CENTERED by the protein's own per-position mean (the paper's
measurement coordinates; transductive for held-out proteins exactly like the probes, F6); (2) loss and attention masked by the REAL-residue mask (the original mask covered only positions
beyond the bucket, so padding counted as zero-error tokens); (3) HELD-OUT PDB entries (the original was in-sample); (4) ~9M params vs 40M; (5) 1b25_D (768 bucket) excluded.
SPLIT (deterministic, seed 0, by entry): TRAIN 13 proteins/12 entries = 1a62_A 1ab1_A 1af7_A 1ah7_A 1ail_A 1b0n_A 1b2s_E 1b2s_F 1b8d_K 1bgf_A 1bkp_A 1btk_A 1bx7_A;
HELD-OUT 5 proteins/4 entries = 16pk_A 1aol_A 1bq8_A 1bxy_A 1bxy_B. The held-out CI is over only 4 entries and is reported as coarse; per-protein values are shown.
EVALUATION (scripts/phase3_eval.py): lag 1 (10 ps), real tokens, 400 sampled valid starts per protein (fixed seed, identical for every method), % improvement over identity of:
shrinkage alpha*z0 and per-channel linear (both fitted on the TRAINING entries only), and the trained model's z1_hat = Euler integral of v from z0 with 1 step and with 8 steps (EMA and raw weights).
The 1-step prediction z0 + v(z0, 0) is the direct forecast and the matched comparison to the 15% per-channel-linear figure. In-sample (training proteins) columns are reported and labelled.
CLOSED-FORM PREDICTION OF WHAT THE ATTENUATION ACCOUNT SAYS (rho_eff = 0.695, analysis/theory.py; E_w[f] = attainable loss reduction under sampling density w):
  uniform 5.42%; Beta(0.5,0.5) 7.93% (1.46x); Beta(0.25,0.25) 10.23% (1.89x); Beta(0.1,0.1) 12.02% (2.2x); f(0) = 15.25%. Mass in t<0.1 or >0.9: 0.20 / 0.41 / 0.62 / 0.81.
  Reading: Beta(0.5,0.5) is a MILD change of the training signal. With the measured k* = 25.8 at the old batch, a 1.46x signal gain moves k* to about 12, still 'ambiguous' by the
  pre-stated thresholds, so a null at Beta(0.5,0.5) would be only weakly informative about the attenuation account.
SUCCESS CRITERION (stated before running; for the pre-specified treatment arm T):
  PASS (attenuation account supported): held-out 1-step improvement over identity of arm T >= (shrinkage improvement on the same held-out frames) - 5 points, AND uniform arm <= 3%.
  Matching shrinkage is a pass; exceeding it is not expected and would be noted, not claimed. Compared against BOTH identity and shrinkage.
  PARTIAL: arm T between 3% and (shrinkage - 5) points: the account explains some but not all.
  FAIL / account incomplete: arm T <= 3% (no recovery): attenuation does not suffice; points toward the gradient-SNR / optimization explanation.
  SURPRISE: uniform arm also recovers (> 3%): attenuation is not the binding mechanism for a small centered/masked model; reported as such.
KILL CONDITION: if the uniform arm's loss does not decrease from its initial value within the first 2000 steps (training broken, not 'no signal'), stop and debug; this is checked from loss_log.csv
before the budget is spent. No arm is re-run to improve a result; if a bug is found the fix is logged and the arm restarted from scratch.
OPEN DECISION (asked of Arjun): treatment distribution Beta(0.5,0.5) as specified, or a heavier endpoint mixture (Beta(0.25,0.25) or Beta(0.1,0.1)). Recommendation: Beta(0.25,0.25)
or heavier, because Beta(0.5,0.5) changes the attainable fraction only 1.46x. Not decided by me; no arm launched.

=====================================================================================================================
2026-10-11. PHASE 3a THROUGHPUT BENCHMARK (timing only; NOT an experimental arm; job 11278346 on IllinoisComputes-GPU node ccc0388, 1 GPU, uniform t, 300 steps).
The first benchmark submission (11276952, secondary, H100-only) sat PENDING with ~800 jobs queued in that partition and was cancelled; the same pipeline was resubmitted on
IllinoisComputes-GPU (3-day cap, --time=3-00:00:00 per the max-time rule) and started immediately.
Result: 9.15M parameters; training set 13 proteins / 390,000 lag-1 pairs; 300 steps at batch 8 in 47 s including ~18 s of startup => ~10.5 steps/s (~9.5 s per 100 steps); no errors.
Loss in centered coordinates ~0.0072-0.0078 (early, warmup). Sizing: 3 GPU-hours/arm = ~113,000 steps; PROPOSED 100,000 steps per arm (~2.6 GPU-h each, ~5.3 GPU-h for both arms, well under the 12 GPU-hour
cap), identical for both arms. 100,000 steps x batch 8 = 800k samples ~ 2 passes over the training pairs (the original run used 192k steps x 16 = 3.1M samples). Headroom exists to go to ~200k steps per arm
within the cap if Arjun prefers a longer run. Batch size is part of the shared optimizer settings (8); a larger batch would raise per-step gradient SNR (k* = 25.8 microbatches of 4 samples at the old checkpoint)
and is a legitimate knob to decide together with the t-distribution, identical across arms.
NO ARM HAS BEEN LAUNCHED. Waiting on the open decision (treatment t-distribution; step count and batch size).

=====================================================================================================================
2026-10-12. PHASE 3a PRE-REGISTRATION, REVISED (supersedes the 2026-10-11 entry's success criterion, batch size, step count and treatment distribution; written BEFORE any arm is launched).
Reason for revision: Arjun's decisions after the benchmark, plus the E3 table's implication for batch size.

(1) FACT CHECKS FROM THIS REVIEW, recorded plainly.
  * Original run's effective batch CONFIRMED from its config: scripts/train_full_run.sh launches torchrun --nproc_per_node=4 with --batch-size 4 (DDP, gradients averaged) => effective batch 16; peak LR 1e-4, gate LR 1e-3
    (visible in the job logs train_full_10279682 / 10642653). This matches what E3 assumed (k = 4 microbatches of 4).
  * Arithmetic correction: in E3 one microbatch = 4 samples, so k* = 25.8 microbatches = ~103 SAMPLES, not 206. SNR(n samples) = 0.0985 * sqrt(n): n = 8: 0.28; n = 16 (the original run): 0.39 (measured); n = 103: 1.00;
    n = 256: 1.58. So the original run sat at 0.39 (not 0.28), and batch 8 would sit at 0.28. The conclusion drawn from this is unchanged: batch 8 would invalidate 3a; batch 256 puts both arms above SNR = 1 on this measure.
  * Caveats on E3 as a design input: it was measured at the ORIGINAL run's final (collapsed) checkpoint with raw 40M weights; gradient signal at initialization or for the new 9M models can differ. The pre-stated verdict of E3 stays 'AMBIGUOUS' (4 < k* <= 64).
    The SNR account of the original run's 0% is ANALYTICAL; Phase 3a is what tests it. The original 192k run is NOT a data point in any Phase 3a comparison (40M vs 9M parameters, different padding handling, uncentered
    coordinates, no held-out split): it must not appear as the low-batch end of a trend, and the write-up must say so.

(2) DESIGN (identical across arms except the t distribution).
  Batch 256 by gradient accumulation (micro-batch 8 x 32); peak LR 1e-4 x sqrt(256/8) = 5.657e-4 (square-root scaling from the batch-8 setting; gate-parameter LR multiplier x10 as in the original, i.e. 5.657e-3);
  AdamW (betas 0.9/0.999, wd 1e-5), linear warmup 350 optimizer steps then cosine decay to 10% of peak at the final step; grad-clip 1.0; EMA 0.999; bf16 autocast.
  STEPS: 6,400 optimizer steps per arm = 1.638M samples (~4.2 passes over the 390k training pairs). At the measured 84 samples/s (10.5 steps/s at micro-batch 8) this is ~5.4 GPU-hours per arm, ~10.8 for both,
  inside the 12 GPU-hour cap with margin for evaluation; 6 GPU-hours per arm was the allowance. One GPU per arm, IllinoisComputes-GPU (3-day cap, --time=3-00:00:00).
  Seed 0 for both arms: identical initialization and identical training-pair order; only the t draws differ.
  Arms: UNIFORM (t ~ U(0,1)) and TREATMENT t ~ Beta(0.25, 0.25); in BOTH arms t is CLAMPED to [0.01, 0.99]. Mass clamped: 1% at each end (uniform), 17.1% at each end (Beta(0.25,0.25)); the clipping is reported.
  CLAMP, NOT TRUNCATION (disclosure): my first implementation truncated (renormalised) instead of clamping. Truncation removes the endpoint mass and cuts the treatment's predicted attainable reduction to 7.94%
  (ratio 1.52x). The clamp is the literal reading of 'clip' and reproduces the 10.2% / 1.9x quoted in the review (10.25% / 1.89x here), so the clamp is what was built, tested (tests/test_flow_matching.py) and will run.
  Model: FluxionDiT d_model 256, 6 layers, 8 heads, d_ff 1024, 9.15M parameters. Data: centered coordinates, real-residue mask, 13 training proteins / 12 entries, 5 held-out proteins / 4 entries
  (split in the 2026-10-11 entry; 1b25_D excluded). The one-step prediction is z0 + v(z0, 0.01) (0.01 = the smallest t the models are trained at).

(3) PRE-REGISTERED PREDICTIONS (results/phase3/predictions.json, scripts/phase3_predict.py; the attainable flow-loss reduction E_w[f], fraction of E||delta||^2 explained relative to v = 0):
  uniform 5.41% (closed form, rho_eff = 0.695) / 5.51% (measured per-channel LINEAR f(t) curve); Beta(0.25,0.25) 10.25% / 10.41%; predicted treatment/control RATIO 1.894 / 1.889.
  The measured-curve version is the attainable value for LINEAR estimators of E[delta | z_t]; a nonlinear model could exceed it, and a 9M model may fall short of it.

(4) METRICS, both reported for every evaluation.
  PRIMARY (the pre-registered test of the attenuation account): flow-loss reduction R(t) = 1 - L_model(t)/L(v=0) on a fixed grid of 41 t values in [0.01, 0.99], pooled over 400 sampled frames per held-out protein; each arm's ATTAINED
  reduction = integral of R against that arm's OWN clamped t distribution (point masses included); measured treatment/control ratio against the predicted 1.89, with a bootstrap CI over the 4 held-out entries (coarse; per-entry values shown).
  DOWNSTREAM readout: % improvement over identity of the 1-step and 8-step Euler z1_hat, against identity AND against the alpha*z0 shrinkage ceiling (alpha and per-channel linear fitted on the training entries only).
  In-sample columns (training proteins) are reported and labelled. EMA and raw weights both reported.
  INTERMEDIATE EVALUATIONS at 25/50/75/100% of the run (milestone checkpoints at steps 1600, 3200, 4800, 6400) so we can see whether the curve is still moving when the budget expires.
  LIVE METRIC logged every 10 optimizer steps: train loss, v=0 loss on the same batches, and flow-loss reduction = 1 - L/L(v=0) (loss_log.csv); plus a held-out one-step-vs-identity monitor every 100 steps.

(5) PASS CONDITION for the attenuation account: both arms' attained flow-loss reduction > 0 AND the measured treatment/control ratio within [1.5, 2.3] (the predicted 1.89 +/- ~20%). A treatment arm that beats the control but misses
  the predicted ratio is still informative; the measured ratio is reported against 1.89 either way, along with each arm's attained/predicted fraction.
  INTERPRETATION TABLE (stated in advance; the SNR mechanism and the attenuation mechanism are separate and both are scored):
    uniform ~ 0, treatment > 0 with ratio ~ 1.9      -> attenuation binding even above the noise floor (SNR >= 1 does not rescue uniform t)
    both > 0, ratio ~ 1.9                             -> both arms learn once the batch is large: batch/SNR was the binding constraint for the original run, and attenuation then costs the predicted factor
    both > 0, ratio ~ 1                               -> SNR/batch explains the original failure; attenuation is not binding at this batch
    both ~ 0                                          -> neither mechanism is sufficient: capacity, optimization or the objective's attainable signal at 9M parameters; the account is incomplete
  Downstream: matching the shrinkage ceiling on held-out frames is a pass; exceeding it is not expected and would be noted, not claimed.

(6) KILL / STOP CONDITIONS (checked from loss_log.csv).
  (a) divergence: NaN, or training loss > 2x the v=0 loss after warmup -> stop that arm, log the fix, restart from scratch (an arm is never re-run to improve a result).
  (b) STEP 500 CHECK: report to Arjun both arms' live flow-loss reduction and held-out monitor. If neither arm's live reduction exceeds 1% by step 500, flag it immediately; continuing or stopping is Arjun's decision
      (500 steps is inside the end of warmup, so this is a flag, not an automatic kill).
  3b (conditional p(z1|z0)) goes to FUTURE_WORK.md unrun; two arms are the whole configuration budget.

=====================================================================================================================
2026-10-12. B1 ANOMALOUS CELL: PRE-REGISTERED RIDGE CHECK AND COMPARISON COUNT (post hoc diagnostic, requested in review; written BEFORE it runs).
Cell: per-channel linear, 100 ps, 13-protein set, Chai-1 minus Cartesian = +7.10 [+2.10, +12.47]. Current reading, UNTESTED: estimation noise from ~384 slopes fit on ~2000 pairs.
TEST: a ridge per-channel linear probe (common absolute penalty lambda = c * mean(Sxx) on the centered slope; c chosen from {0, .01, .03, .1, .3, 1, 3, 10} by validation MSE on a held-in replica, never the test fold;
analysis/probes.py fit_ridge_diag / select_ridge_strength, unit-tested) is added to the identical B1 pipeline and the 13 proteins are re-run into results/phase1/b1_ridge (the original JSONs are not overwritten).
Integrity check first: the re-run must reproduce the original linear_diag, shrink and identity numbers (same seeds); a mismatch > 0.01 percentage points is reported before anything else.
DECISION RULE, stated in advance. If the overfitting reading is right, ridge should pull BOTH spaces' per-channel linear toward the shrinkage level and the paired difference for linear_diag_ridge should shrink toward 0
(95% CI including 0). If the ridge difference remains >= +5 with a 95% CI above 0 at 100 ps, the overfitting reading is NOT supported and the cell stands as stated. Intermediate outcomes are reported as intermediate.
The cell is not dropped under either outcome.
COMPARISON COUNT. B1 grid as reported: lags {100 ps, 1 ns, 10 ns, 50 ns} x predictor rows {linear_diag, linear_full, shrink, mlp, mean, id/mean} x protein sets {8, 13} = 48 displayed cells (36 at interpretable lags).
The claim rests on 12 pre-specified TRUSTED cells: {linear_diag, shrink} x {100 ps, 1 ns, 10 ns} x {8-set, 13-set}; the ridge row adds 6 (m = 18 with it). Bonferroni-corrected CIs use m = 12 (99.58% level) and are printed beside every
cell by scripts/phase1b_report.py. The report states how many of the 36 interpretable cells exclude 0 uncorrected, for both directions.

=====================================================================================================================
2026-10-11. B1 RIDGE CHECK RESULT (job 11290912; results/phase1/b1_ridge/; scripts/phase1b_report.py results/phase1/b1_ridge). Applying the rule pre-registered above.
INTEGRITY: the re-run reproduces the original identity/linear_diag/shrink/linear_full/mean numbers on all 520 compared cells (max |difference| 6.2e-5 percentage points). PASS.
RESULT, 13-protein set, 100 ps, Chai-1 minus Cartesian, % over identity (95% CI over 11 entries):
  per-channel linear (OLS)    +7.10 [+2.10, +12.47]      Chai 4.56, Cartesian -2.53
  per-channel linear (RIDGE)  +7.47 [+2.21, +13.10]      Chai 4.28, Cartesian -3.19
  shrinkage                   +0.56 [-2.85, +3.89]       Chai 10.91, Cartesian 10.36
  (1 ns: OLS +4.35 [-0.32,+9.34], ridge +4.23 [-0.36,+9.20]; 10 ns: OLS +4.60 [-2.33,+12.90], ridge +4.55 [-2.59,+13.05])
DECISION RULE: the ridge difference stayed >= +5 with a 95% CI above 0 at 100 ps (+7.47 [+2.21,+13.10]) => the overfitting reading is NOT SUPPORTED by this check; the cell STANDS as stated. It is not dropped.
COMPARISON COUNT AND CORRECTION. B1 grid = 4 lags x 6 rows x 2 sets = 48 displayed cells; 36 at interpretable lags. Uncorrected, 8 of the 36 exclude 0 (6 favour Chai-1, 2 favour Cartesian; the cells are strongly
correlated, so this is not 8 independent draws). Bonferroni at m = 12 (the pre-specified trusted cells): the 100 ps linear_diag cell CI is [+0.08, +14.88] (survives by 0.08 points: marginal); the ridge cell [+0.11, +15.67].
At m = 18 (adding the 6 post hoc ridge cells): [-0.17, +15.28] and [-0.12, +16.05] (does NOT survive). The shrinkage row, the row the claim rests on, excludes 0 in no cell at either correction.
WEAKNESSES OF THE CHECK ITSELF (stated because the check came out 'the cell stands' and that deserves scrutiny, not celebration):
  (1) Ridge pulls slopes toward ZERO, but the right target for these slopes is the common shrinkage level (alpha ~ 0.7); pulling toward 0 is not the 'toward shrinkage' direction the pre-registration anticipated. The infinite-penalty
      limit toward a common slope IS the shrinkage row, and it shows no difference (+0.56 [-2.85,+3.89]). A ridge toward alpha would be a better-targeted test; not run (a further post hoc variant).
  (2) The validation replica selected c = 0 (plain OLS) in 93/156 Chai folds-cells and 85/156 Cartesian ones, and the grid edge c = 10 in 28 and 31: a bimodal choice, not the smooth interior optimum a pure
      estimation-noise story would predict. The grid's upper end may be too low.
READING (revised): per-channel linear loses less relative to shrinkage in Chai-1 space (10.91 -> 4.56, -6.4 points) than in Cartesian PCA space (10.36 -> -2.53, -12.9 points): Chai-1's per-component slopes transfer across
replicas somewhat better than Cartesian PCA slopes do. Both are beaten by one scalar. This is a descriptive difference in a probe nobody should use; it is not support for 'Chai-1 latents are a better dynamics space', and it
is not robust to multiplicity correction at m = 18. 'Estimation noise' as the explanation is not supported and is withdrawn as the stated reading.

=====================================================================================================================
2026-10-11. PHASE 3a: BOTH ARMS COMPLETED UNATTENDED; TWO PRE-REGISTERED CONDITIONS FIRED AND WERE NOT ACTED ON IN REAL TIME. (results/phase3/loss_log_{uniform,beta}.csv, config_*.json)
WHAT HAPPENED. Both arms were submitted 2026-10-12-ish and started after a queue wait; my scheduled polls could not run (usage-limit pause) and by the time I resumed both arms had finished all 6,400 steps
(uniform 11,720 s = 3.26 GPU-h; beta 14,310 s = 3.98 GPU-h; total 7.24 of 12 GPU-hours; evaluations not yet counted). No step-500 check and no stop-condition check was performed in real time. Reading the logs afterwards:
(1) STOP CONDITION (a) FIRED in the BETA arm. Around step 350 (the end of warmup, LR at its 5.66e-4 peak, gate LR 5.66e-3) the training loss jumped to 2.19x the v=0 loss (max ratio after warmup 2.19; the logged
    10-step window at step 350 shows flow-loss reduction -665.8%, grad norm 0.72 vs ~0.002 normally). Before it, the beta arm had been LEARNING FAST: live flow-loss reduction 2.4% at step 150, 8.4% at step 200, 10.0% at 250,
    10.2% at 300, and the raw-weights held-out one-step monitor 15.3% at step 200 and 17.1% at step 300 (EMA lagging). After the spike the arm sat at ~0% (loss = v=0 loss, i.e. collapsed to ~zero velocity) from step ~400 to ~1000,
    then re-learned: 0.4% at 1000, 3.5% at 1600, 11.8% at 3200, 12.8% at 4800, 14.2% at 6400. The pre-registered action was: stop that arm, log the fix, restart from scratch. That did NOT happen.
(2) STEP-500 FLAG CONDITION HELD: neither arm's live reduction exceeded 1% at step 500 (uniform +0.035%, beta -0.004%; held-out EMA monitors -20.3% and -9.6%, raw -0.05% and -0.05%). Not raised in real time.
(3) The UNIFORM arm had a smaller transient (step 250: -2.3% window, grad norm 0.012) that stayed under the 2x criterion; it learned slowly: live reduction first > 1% at step 1890 and > 5% at step 2860;
    5.3% at 3200, 7.3% at 4800, 8.5% at 6400 (live, in-sample, last window; 9.1% at step 6390).
(4) Final held-out one-step-vs-identity monitor (16 pairs per held-out protein, a monitor not the pre-registered evaluation): uniform EMA 17.5% / raw 17.2%; beta EMA 16.7% / raw 14.9%.
CAVEATS ON THESE NUMBERS. The live flow-loss reductions are on TRAINING batches (in-sample) and exceed the LINEAR-attainable predictions (uniform 5.4-5.5%, beta 10.3-10.4%): either the 9M model beats the linear estimator, or in-sample fitting
inflates them; the pre-registered held-out evaluation decides. The live ratio of final windows is 14.17/8.51 = 1.67 (noisy single windows) vs the predicted 1.89. Both arms are therefore ABOVE ZERO and the interpretation table's rows
'both > 0' apply provisionally; nothing here is the pre-registered test.
CONSEQUENCE FOR VALIDITY. The two arms did not experience the same optimization: the beta arm diverged at the LR peak and re-learned from a collapsed state, the uniform arm did not diverge. The A/B comparison is therefore confounded by an
instability that is not part of the t-distribution effect. Whether the beta arm's recovered final state is comparable to a clean run is NOT known. Options (Arjun's decision, NOT taken by me): (A) report the arms as run, with the divergence
documented as part of the result; (B) restart BOTH arms with a pre-registered fix (e.g. lower peak LR or gate LR multiplier, or longer warmup): ~7.2 GPU-hours, which together with the 7.24 already used exceeds the 12 GPU-hour cap, so
it needs the cap lifted; (C) restart only the beta arm (~4 GPU-h, total ~11.2 + evaluations): the arms would then differ in optimizer settings unless the fix is neutral to the uniform arm. None is launched.
EVALUATIONS (pre-registered protocol: flow-loss R(t) on a 41-point grid, attained reduction under each arm's own clamped distribution, ratio vs 1.89; downstream vs identity and shrinkage; at 25/50/75/100%) are QUEUED
(jobs 11311533 on IllinoisComputes-GPU, 11311553-56 on secondary, one per milestone); they will be run on the checkpoints as they stand regardless of (A)/(B)/(C), so that the data exists either way.
