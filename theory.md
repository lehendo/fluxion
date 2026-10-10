# Conditional flow matching between the two ends of a stationary reversible pair

Status: revised 2026-10-10 after the Phase 1 measurements. The first draft claimed the optimal velocity field is
zero. **That is refuted and has been removed.** What the data confirm is the antisymmetry of the optimal field about
t = 1/2 and a severe, quantifiable attenuation of the learnable signal. Section 5 holds the numerical verification, Section 6 a scoping
remark on OT re-coupling (a general remark that does not apply to the trained temporal model), and Section 7 the prior-art search (still being filled in; see its status line).

Wording rule used throughout: the objective *offers almost no learnable signal*; we never say the model "found
the optimum". The trained model did not reach the (nonzero) optimum either.

## 1. Setup and assumptions

Let (z0, z1) be a pair of latents a fixed lag apart on a molecular-dynamics trajectory, δ = z1 − z0, and
z_t = (1 − t) z0 + t z1 with t ~ Uniform(0, 1). The conditional-flow-matching (CFM) loss of a velocity field v is
L(v) = E‖v(z_t, t) − δ‖². We use only these assumptions:

* **A1 stationarity.** z0 and z1 have the same marginal law p. (Equilibrium MD, after burn-in.)
* **A2 time reversibility.** (z0, z1) has the same joint law as (z1, z0). (Detailed balance; it is inherited by any
  measurable map applied frame by frame, so it holds for the latents if it holds for the coordinates.)
* **A3 the observed coupling.** The pair is the observed lag-ℓ pair from the trajectory. There is no minibatch re-coupling.
* **A4 linear interpolant, uniform t.**

Where real data depart from these: finite replicas (the effective number of independent windows is small, Phase 1
n_eff table), slow non-stationarity inside a 100 ns replica, thermostat-induced departures from exact reversibility,
and frame-level latent noise that is not part of the dynamics.

## 2. The minimizer and what is attainable

**Proposition 1.** The minimizer of L over all measurable v is v*(x, t) = E[δ | z_t = x], and for any v,
L(v) = E‖δ‖² − E‖v*(z_t, t)‖² + E‖v − v*‖². Hence the largest loss reduction any model can obtain relative to the
v = 0 predictor is the fraction

  f̄ = E_t E‖v*(z_t, t)‖² / E‖δ‖² = ∫₀¹ f(t) dt, with f(t) = E‖v*(z_t, t)‖² / E‖δ‖² (fraction of ‖δ‖² explained at time t).

*Proof.* Orthogonal decomposition of the squared error around the conditional mean. ∎

Direct regression of z1 on z0 attains f(0): at t = 0 the input z_t is z0, so E[δ | z_0] is exactly the direct
regression target. This identity is asserted in code (`scripts/phase1_headline.py`, gap ~1e-7).

## 3. Antisymmetry about t = 1/2 (confirmed)

**Proposition 2.** Under A2, v*(x, 1 − t) = −v*(x, t) for all x. Consequently v*(x, 1/2) = 0 for every x, f(1/2) = 0,
and f(t) = f(1 − t).

*Proof.* A2 gives (z0, z1) =ᵈ (z1, z0). Under the swap z_t ↦ z_{1−t} and δ ↦ −δ, so
E[δ | z_t = x] = E[−δ | z_{1−t} = x] = −v*(x, 1 − t). Setting t = 1/2 gives v*(x, 1/2) = −v*(x, 1/2) = 0. ∎

**Corollary (what is dropped).** v* is *not* identically zero: it is odd about t = 1/2, vanishes exactly at the
midpoint, and is nonzero for t ≠ 1/2 whenever δ is predictable from z_t. "The optimal field is zero" was an overreach;
the loss minimizer is a small, structured field, not v = 0.

## 4. Attenuation under uniform t (the quantitative cost)

**Proposition 3 (Gaussian reversible case).** Let (z0, z1) be jointly Gaussian with unit variance and correlation ρ
(per-channel the same holds with ρ_c). Then with f₀ = (1 − ρ)/2,

  f(t) = f₀ (1 − 2t)² / (1 − 2(1 − ρ) t (1 − t)),  f̄ = f₀ R(ρ),  R(ρ) = ∫₀¹ (1 − 2t)² / (1 − 2(1 − ρ) t (1 − t)) dt.

R decreases from ≈ 0.4292 at ρ = 0 to 1/3 as ρ → 1. So **for any stationary reversible Gaussian pair the uniform-t
objective can attain only 33–43% of the endpoint signal: the attenuation 1/R lies in [2.33, 3.0] regardless of ρ.**
With per-channel ρ_c, f̄/f₀ is a weighted average of R(ρ_c) and obeys the same bounds.

*Proof sketch.* Cov(δ, z_t) = −(1 − ρ)(1 − 2t), Var(z_t) = 1 − 2(1 − ρ) t (1 − t), Var(δ) = 2(1 − ρ); the explained
fraction of a Gaussian regression is Cov² / (Var(z_t) Var(δ)). Integrating over t gives R. Implemented and
Monte-Carlo-checked in `analysis/theory.py` and `tests/test_analysis.py::TestClosedForm`. ∎

*Interpretation.* The objective samples t uniformly, the recoverable signal is odd about t = 1/2 and vanishes there, so
most draws land where there is little to learn. Degeneracy here means severe attenuation, not exact nullity.

## 5. Numerical verification on the 19-protein ATLAS pilot (E1)

Held-out entries (17 folds, leave-one-PDB-entry-out), masked real tokens, lag 1 (10 ps), per-channel linear estimators on
identical rows, 95% CIs bootstrapped over the 17 entries. Raw numbers: `results/phase1/headline.json`,
`experiments/EXPERIMENT_LOG.md`.

| quantity (fraction of ‖u‖² explained) | value | 95% CI |
|---|---|---|
| direct regression, z0 only (= f(0), asserted) | 15.23% | [12.98, 17.08] |
| optimal linear E[δ \| z_t], uniform-t average f̄ | 5.52% | [4.67, 6.21] |
| ratio direct(z0) / f̄ | 2.76 | [2.73, 2.80] |
| direct regression on (z0, z_prev) | 26.90% | [25.89, 27.75] |
| ratio direct(z0, z_prev) / f̄ | 4.88 | [4.46, 5.54] |

* **Antisymmetry confirmed.** f(1/2) = −0.001% [−0.003, 0.002], f(0.45) = 0.15%, f(0.55) = 0.21%; f(0) = 15.2%, f(1) = 15.7%.
  The kill condition "the empirical conditional mean does not cross zero near t = 1/2" did **not** trigger.
* **Closed form agrees.** Taking only ρ_eff = 1 − 2 f(0) = 0.695, Proposition 3 predicts f̄/f₀ = 0.356, i.e. attenuation 2.81;
  measured 2.76 [2.73, 2.80]. The predicted curve is within 0.08 percentage points of the measured f(t) for every t ≤ 0.5 and within
  0.35 points for t = 0.6 to 0.9. The right half runs slightly above the prediction (f(1) = 15.7% vs f(0) = 15.2%, CIs overlapping), a small
  departure from exact reversibility. No parameter was fitted to the curve.
* **Which ratio is the cost of random-t?** The first (2.76) is the like-for-like footing: the trained CFM model sees z_t only, and the
  direct comparator sees z0 only. The 4.88 compares against a regression that also sees z_prev; the extra factor is the value
  of velocity information, a different question from the cost of the objective. We report 2.76 as the cost of the objective.
* **Caveat.** These are *linear* estimators of the conditional mean, so they lower-bound the attainable fraction. The MLP probe
  on (z0, z_prev) generalized worse than linear across the 19 proteins (+16.0% vs +26.7% over identity), which argues against large
  nonlinear signal at lag 1, but that is not a proof for E[δ | z_t].

## 6. General remark on minibatch optimal-transport re-coupling (does NOT apply to the trained temporal model)

CORRECTION (2026-10-10). An earlier version of this analysis (and the project's "Correction 2") attributed part of the temporal DiT's
failure to OT-style re-coupling. That leg is withdrawn: `training/train.py` pairs z0 = frame i with z1 = frame i+1 and hands the pair to
`FlowMatcher` unchanged. It never imports `training/ot_coupling.py`, so the trained model used the observed coupling (assumption A3), not an OT coupling.

The remark that survives is general and concerns configurations the trained model did not use. If CFM pairs are re-coupled by minibatch
optimal transport between two batches of equilibrium frames, the assignment minimizes Σ‖z1 − z0‖², and for a continuous p the OT map between
identical measures is the identity as the batch grows, so the regression target δ → 0. That is a degenerate objective, and it is a statement about
OT-CFM *as commonly configured* and about `training/train_static.py` (noise → equilibrium frame; the marginals differ there, so it is not this
degenerate case, and the static model is not analysed here). The Hungarian-vs-adjacent displacement comparison planned in 2B was not run.

## 7. Prior-art position (Phase 2C; full query log in `docs/prior_art_search.md`)

* The vanishing of the optimal velocity at the midpoint for *i.i.d.* endpoints with equal marginals is published (Midpoint Generative Models,
  arXiv:2605.29920, Prop 3.1). Proposition 2 here is the same swap argument applied to exchangeable observed pairs (stationarity +
  reversibility), a one-line generalization; we do not claim it as new beyond that.
* **The general principle is prior art.** Allocating training across noise levels by how much loss reduction is attainable at each is established: EDM (Karras et al. 2022) and, most explicitly,
  arXiv:2506.13763 (optimal loss J*_sigma = E||x0||^2 - E||E[x0|x_t]||^2 per noise level, training allocated by the gap J - J*). Our f(t) is that decomposition. What is specific here (not found in the pages read):
  the equal-marginal observed-pair setting, the oddness about t = 1/2 (signal vanishes at the midpoint rather than merely varying), the Gaussian reversible closed form with the [2.33, 3.0] bound, and the 2.76x measurement.
* Midpoint's appendices were read via the tool (independence assumed throughout, no Gaussian closed form, no quantification of learnable loss under uniform t; it notes the field does not vanish for t != 1/2).
* In the pages read we did not find the attenuation result (Proposition 3), the Gaussian closed form with the [2.33, 3.0] bound, or its use to
  diagnose CFM on temporally coupled MD frames. This is a search negative (19 calls; Midpoint's appendices and part of the survey unread),
  not evidence of novelty. Propositions 1-3 are elementary and may be folklore; a deeper search is required before any novelty wording.
* Kernel-Stabilized Flow Matching (arXiv:2411.08314) was reported by a search tool to state zero velocity under identical marginals; the paper
  text read does not say this, and we do not cite it for that.
* Closest published settings to position against: DeepJump (arXiv:2509.13294, CFM between conformations a fixed step apart) and stochastic-interpolant
  forecasting (arXiv:2403.13724, conditional on the present state).
