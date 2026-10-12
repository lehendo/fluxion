# Phase 1 report — measurement practice for latent-space protein dynamics

Branch `tmlr`, report date 2026-10-11. Every number below is traceable to `experiments/EXPERIMENT_LOG.md` and files under `results/phase1/`.
Kill conditions were written in the log before each experiment ran; where one triggered it is reported as a finding.

## 0. What the work is now

The original framing (a frozen-Chai-1-latent DiT that generates MD trajectories) did not survive its own controls and is not recovered here. The paper is a
**measurement-practice paper for latent-space protein dynamics**. Its spine is the correction chain (Section 1): each reported figure, the artifact that moved it,
and the value after the artifact is removed.

| Claim | Status | Evidence |
|---|---|---|
| Bucket padding is a measurement artifact in AF2/AF3/Chai-1 trunk activations | **Survived, now contribution 1** | padding constants are per-shard and run-dependent (\|Δ\|² ≈ 3.9, comparable to real lag-1 displacement); 34–77% of tokens on the original held-out proteins; 50-point inflation of improvement-over-mean |
| Baseline geometry: identity beats the mean below ≈1 ns, loses above; closed-form asymptotes confirmed | **Survived, contribution 2 (credibility anchor)** | sweep, asymptote check (Section 3) |
| Short-lag latent predictability is mean reversion | **Survived** | α = 0.70 at 10 ps; full linear − shrinkage = −0.44 [−1.04, +0.06]; MLP worse (Section 4) |
| Attenuation of the learnable signal under uniform-t CFM | **Survived; Phase 3 adds held-out evidence (see `docs/phase3_report.md`)** | closed-form f(t) predicts the measured f(t) without fitting (Phase 1: 2.76× on matched inputs); both 9M flow models reach the linear-optimal R(t) pointwise on held-out proteins (0.89–1.03 of prediction at milestone 75%); an endpoint-weighting speed advantage is suggestive but not robust across runs. The general principle is prior art (Section 8); the treatment/control flow-loss ratio is NOT an independent confirmation (it follows from the weightings once R(t) = f(t)) |
| No learnability advantage of Chai-1 latents over Cartesian PCA | **Survived only in a narrow form** | indistinguishable on the shrinkage-referenced comparison; one per-channel-linear cell favours Chai-1 on the larger set (Section 5) |
| C3: frozen trunk latents as a privileged dynamics space | **Falsified on the available evidence** | Sections 4–5 |
| C1 midpoint vanishing as a new result | **Dead: prior art** (Midpoint Generative Models Prop 3.1, i.i.d. endpoints); ours is a one-line extension to exchangeable pairs | `docs/prior_art_search.md` |
| OT coupling as part of the trained model's failure | **Withdrawn**: `train.py` never uses OT coupling | log, `theory.md` §6 |

## 1. The correction chain (contributions 1 and 2 in one object)

Percentages are improvements (positive = the first predictor has lower error). "Mean" = predict-the-per-position-mean; "identity" = ẑ₁ = z₀.

**A. MLP improvement over predict-the-mean (the originally reported headline)**

| # | Value | Source | What moved it |
|---|---|---|---|
| 1 | 91.25% | old probe (0.001701 vs 0.01944) | — |
| 2 | 89.58% | S0.0 reproduction (all four old values lie inside the new bootstrap CIs) | nothing: reproduction passes |
| 3 | 89.02% | S0.1 | MLP early stopping moved from the test proteins to training-side entries (−0.6 pts; −4.3 pts vs identity) |
| 4 | **39.19%** | S0.2 | **bucket padding masked: −49.8 pts.** The mean baseline falls 0.016131 → 0.006656; 1bxy_A's padding constant alone was ≈82% of the old pooled mean baseline |
| 5 | 39.41% | S0.3 | leave-one-PDB-entry-out folds (+0.2) |
| 6 | 45.48% [41.15, 50.42] | S0.3, 19 proteins | broader sample |

**B. Identity vs predict-the-mean (the baseline-choice question)**

| # | Value | Source | What moved it |
|---|---|---|---|
| 1 | 87.01% | S0.0 (padding counted) | padding dominates the mean baseline |
| 2 | **21.46%** | S0.2, original 3 proteins | padding masked |
| 3 | **33.28%** [25.27, 42.45] | S0.3, 19 proteins | sample change: 3 → 19 proteins |
| 4 | 33.1% (10 ps) → 17.5% (100 ps) → −3.9% (1 ns) → −41.8% (10 ns) | lag sweep | lag: crossover at ≈1 ns, id/mean = 1.04 |

An earlier reconstruction in the project brief gave "89%" and "21%" for two of these cells; both were wrong, in opposite directions, and are retracted in the log.

**C. Probe improvement over identity (the honest reference)**

| # | Value | Source | What moved it |
|---|---|---|---|
| 1 | linear 22.38% / MLP 19.78% | S0.0 (padding counted) | — |
| 2 | linear 27.9% / MLP 22.6% | S0.2 | padding masked |
| 3 | linear 26.73% [25.59, 27.65] / MLP 16.01% [11.10, 19.99] | S0.3, 19 proteins, (z₀, z_prev) inputs | sample change |
| 4 | **per-channel linear 15.23% [13.01, 17.04]**, shrinkage 15.22%, full linear 14.81%, MLP 13.39% | lag sweep, 10 ps, z₀ only | input change to z₀ alone (the matched comparison for the CFM model, which sees one point) |
| 5 | full linear minus shrinkage −0.44% [−1.04, +0.06] | Phase 1C | the "signal" is one scalar |

**D. Other corrected figures:** the DiT's "held-out" 0.002080 vs 0.002079 was **in-sample** (trained on all 19 proteins); against identity it is −0.02% to +0.01% at every lag.
The cost of uniform-t sampling is **2.76×** [2.73, 2.80] on matched inputs (an earlier draft's larger ratio compared mismatched inputs).

## 2. ATLAS data verification (needed for B1)

(i) residue and atom counts and (ii) frame count, Δt, atoms: **19/19 pass**. Criterion (iii), Mantel alignment with negative controls, side by side:

| | pre-registered (cross-replica + half-shift controls) | revised, POST HOC (within-replica shifts at ¼, ½, ¾ window) |
|---|---|---|
| pass | **8/19**: 1ab1_A, 1ah7_A, 1aol_A, 1b2s_F, 1bq8_A, 1btk_A, 1bxy_A, 1bxy_B | **13/19**: those 8 + 1af7_A, 1b25_D, 1b2s_E, 1b8d_K, 1bkp_A |
| fail | 11 | 6: 16pk_A, 1a62_A, 1ail_A, 1b0n_A, 1bgf_A, 1bx7_A |

The original criterion was badly specified (my error): the three replicas of a protein share a starting structure, so a cross-replica control cannot be near zero (observed 0.15–0.63). The cross-replica numbers are kept as a reported confound. Thresholds (0.20, 0.15) are unchanged.
For 1b0n_A and 1bgf_A the shifted latents correlate with the RMSD matrix as strongly as the aligned ones, so the test cannot discriminate there; 1ail_A has aligned ρ = 0.071. None of the six is called mismatched; they are unverified.

## 3. Lag sweep and asymptote check (19 proteins, 17 PDB-entry folds, masked real tokens; 50 ns greyed, n_eff ≈ 2/protein)

| improvement over identity (%) | 10 ps | 100 ps | 1 ns | 10 ns | 50 ns* |
|---|---|---|---|---|---|
| per-channel linear | +15.23 | +19.10 | +24.84 | +34.69 | +49.62 |
| shrinkage α·z₀ | +15.22 | +19.09 | +24.84 | +34.70 | +49.66 |
| full linear | +14.81 | +18.58 | +24.11 | +33.86 | +48.62 |
| MLP | +13.39 | +17.60 | +23.32 | +33.17 | +48.34 |
| predict-the-mean | −60.28 | −28.76 | −0.49 | +27.79 | +49.62 |
| trained DiT (in-sample), raw/EMA, 1 or 8 steps | −0.02 … +0.00 | | | | |

Asymptote (parameter-free): identity/mean error ratio 0.669, 0.825, 1.039, 1.418, 2.047 [1.89, 2.23] against the structure-function prediction 2(1−ρ) = 0.694, 0.810, 1.019, 1.388, 1.926 and the limit 2; probe-vs-identity → 49.6% (limit 50%); probe-vs-mean → −0.14% (limit 0).
The ACF-based prediction is slightly off at 1 ns (+0.020) and 10 ns (+0.030), CIs excluding 0; reported, not hidden. Figure 2 (`results/phase1/figs/fig2_lag_sweep.*`). **Prescription: report both baselines and state the lag.**

## 4. Phase 1C — the probes' gain is mean reversion (kill condition triggered)

α = 0.704, 0.632, 0.525, 0.334, 0.037 (10 ps … 50 ns). At 10 ps the eigenvalues of the full map have max |λ| = 0.911, median Re = 0.640, fraction complex = 0.000.
Full linear over shrinkage: −0.44 [−1.04, +0.06] (10 ps), −0.58 [−1.19, −0.08] (100 ps), −0.91 [−1.50, −0.45] (1 ns), −1.25 [−1.56, −1.00] (10 ns); MLP −2.04 [−3.38, −0.91] at 10 ps.
Scope: per-residue 384-d rows; cross-residue coupling not tested. Figure 3.

## 5. B1 — Cartesian control (learnability, not information)

Chai-1 latents are a deterministic function of coordinates plus sequence. B1 therefore cannot show they contain less information, and matching Cartesian is the expected null. It tests learnability only.
Both representations PCA-reduced to k = min(3 n_res, 384), leave-one-replica-out, frames at 100 ps stride. The interesting counterfactual — an advantage at 10 ps, where the DiT trained — is unreachable with 100 ps analysis frames.

Paired difference (Chai-1 − Cartesian, % over identity; 95% CI over PDB entries):

| row, set | 100 ps | 1 ns | 10 ns |
|---|---|---|---|
| shrinkage, 8 proteins / 7 entries | −2.79 [−6.62, +1.04] | −4.32 [−8.73, +0.17] | −5.43 [−10.25, −0.05] |
| shrinkage, 13 proteins / 11 entries | +0.56 [−2.85, +3.89] | −0.17 [−4.28, +4.06] | −0.53 [−5.50, +5.01] |
| per-channel linear, 8 | +3.69 [−1.94, +10.89] | −0.41 [−4.52, +4.36] | −3.35 [−7.61, +2.34] |
| per-channel linear, 13 | **+7.10 [+2.10, +12.47]** | +4.35 [−0.32, +9.34] | +4.60 [−2.33, +12.90] |

Supportable statement: *on the shrinkage-referenced comparison — shrinkage is the best probe in both spaces at every lag — frozen Chai-1 trunk latents and matched-dimension aligned-Cα PCA are indistinguishable at lags ≥ 100 ps (differences within about ±0.6 points at 13 proteins; CIs about ±4, so modest differences are not excluded).* The shrinkage row excludes 0 in no cell, under no correction.

**The one cell that favours Chai-1 is not robust to the choice of multiple-comparison correction, and both corrections are reported.** Per-channel linear, 100 ps, 13 proteins: +7.10 [+2.10, +12.47] uncorrected. Bonferroni at m = 12 (the pre-specified trusted cells: {linear_diag, shrinkage} × 3 lags × 2 sets): [+0.08, +14.88], which survives by 0.08 points. At m = 18 (adding the six post hoc ridge cells): [−0.17, +15.28], which does not. Uncorrected, 8 of the 36 interpretable cells exclude 0 (6 favour Chai-1, 2 favour Cartesian), and the cells are strongly correlated. No favourable m is selected.

**The ridge check, and its limits.** A ridge per-channel linear probe (strength chosen on a held-in replica) leaves the cell unchanged (+7.47 [+2.21, +13.10]), so "estimation noise from fitting ≈384 slopes" is not supported and is withdrawn as the explanation. The check was weaker than it looked: ridge shrinks slopes toward 0, but the appropriate null is the *shared shrinkage level* (α ≈ 0.7). The infinite-penalty limit toward that level is exactly the shrinkage row, which shows no difference at any lag or in either set; that is the cleaner argument. The validation replica also chose plain OLS in most folds and the top of the grid in many others (a bimodal choice).

**One descriptive observation.** Per-channel linear loses less relative to shrinkage in Chai-1 space (10.9 → 4.6, −6.4 points) than in Cartesian PCA space (10.4 → −2.5, −12.9 points): Chai-1's per-component slopes transfer across replicas somewhat better. This is the only place the latents show any advantage; both are beaten by one scalar, so it does not rescue the premise that frozen trunk latents are a better dynamics space.

Full-linear and MLP rows overfit in both spaces and are not used. Figure 4. This is **not** "no advantage for linear or MLP probes", which would be too broad.

## 6. Theory and the attenuation measurement

`theory.md`. Under stationarity and reversibility the optimal field is odd about t = ½ and vanishes there; it is **not** zero elsewhere ("optimal field is zero" was wrong and is withdrawn). For a Gaussian reversible pair f(t) = f₀(1−2t)²/(1−2(1−ρ)t(1−t)) and the uniform-t average attains a 1/R ∈ [2.33, 3.0] fraction of the endpoint signal for every ρ.
Measured on matched inputs: direct regression 15.23% [12.98, 17.08] vs f̄ 5.52% [4.67, 6.21] → **2.76×** [2.73, 2.80]; the closed form with ρ_eff fixed from f(0) alone predicts 2.81 and tracks f(t) within 0.08 points for t ≤ ½. f(½) = −0.001%. The OT-coupling remark in §6 of the theory applies to `train_static.py`, not to the trained temporal model.

## 7. Why the trained DiT learned nothing — what is and is not established

Trained DiT output ‖v‖/‖u‖ = 0.0016 (EMA), flat in t, correlation with the target 0.0004: total collapse to identity, not the small structured field the theory allows.
Gradient SNR at the final checkpoint (4096 microbatches of 4): k\* = 25.8 microbatches (jackknife 24.4–26.4), i.e. SNR = 1 needs an effective batch of about 103 vs the 16 used (SNR 0.39 at the batch used). Under the pre-stated thresholds (4 < k\* ≤ 64) this is **ambiguous**: it neither places the signal below the noise floor at feasible batch sizes nor identifies an optimization failure.

## 8. Prior art (`docs/prior_art_search.md`, ~30 calls, tool summaries only)

The general principle that uniform noise-level sampling wastes capacity where the recoverable signal is small is established (EDM; most explicitly arXiv:2506.13763, optimal loss J\*_σ per noise level with training allocated by the gap to it). Our f(t) is that decomposition. Specific here, and not found in the pages read: the equal-marginal observed-pair setting, oddness about t = ½ (the signal vanishes at the midpoint rather than merely varying), the Gaussian reversible closed form with the 1/R bound, and the 2.76× measurement. Midpoint Generative Models' appendices were read: independence assumed, no Gaussian closed form, no quantification of learnable loss.
Unread: appendices of 2506.13763, the last 88k characters of arXiv:2406.12839. "Not found" means not found here.

## 9. Not done / limits

B2 (random-init trunk) not run. B3 rests on the 2026-08 code-review documentation (pre-fix latents purged). 6 proteins unverified. 50 ns not interpretable. All DiT numbers are in-sample. The 10 ps regime is untested by B1.

## 10. Phase 3

See `docs/phase3_report.md` (draft; evaluations at 100% and 50%, the integration-step sweep, arm 3c and the stability diagnostic are pending).
