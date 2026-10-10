# Prior-art search (Phase 2C) — every query run, what came back, what it does and does not support

Run 2026-10-10 (16:34-16:37 UTC), session tools `WebSearch` (standard mode) and `WebFetch`. The tool returns a model-written
summary of the page, not raw text, so each row records what the summary said and, separately, what was NOT read. Nothing here is a
claim of novelty. Search was shallow (25 calls total, 19 on prior art); a negative result means "not found in these calls".

## A. Queries and outcomes

| # | Tool | Query / URL | Outcome |
|---|---|---|---|
| 1 | search | `Flow Matching in the Low-Noise Regime: Pathologies and a Contrastive Remedy` | Zeng & Yan, arXiv:2509.20952 (ICLR 2026 submission). Low-noise ill-conditioning of the velocity target on CIFAR-10/Tiny-ImageNet. Adjacent (small-t pathology), not about stationary pairs or attenuation. Abstract-level only. |
| 2 | search | `On The Hidden Biases of Flow Matching Samplers` | Lim, arXiv:2512.16768. Empirical FM minimizer is generally not a gradient field; kinetic energy set by the source. Adjacent. Only abstract/intro retrieved; v1 vs v2 wording differs. |
| 3 | search | `conditional flow matching identical source and target distributions optimal transport map is identity velocity field zero` | The search model asserted that Kernel-Stabilized FM (arXiv:2411.08314) states identity coupling gives zero velocity. **Checked in #9: not supported.** Also surfaced Weighted CFM, Bridging Simulators with Conditional OT (not read). |
| 4 | search | `flow matching time reversal symmetry antisymmetric velocity field reversible process detailed balance midpoint interpolant` | Physics-only hits on the first form; the second surfaced **Midpoint Generative Models** (arXiv:2605.29920). No source tying the antisymmetry to detailed balance/reversibility found. |
| 5 | fetch | arxiv.org/abs/2605.29920 | Abstract: with linear FM, when the two endpoint distributions coincide the drift vanishes at t = 1/2; norm of that field used as a "Midpoint Divergence" for one-step generators. Proof/assumptions not on that page. |
| 6 | fetch | arxiv.org/abs/2411.08314 | Kernel-Stabilized FM (Generale et al., TMLR 2026). Abstract page only; equal-marginals statement not found. |
| 7 | fetch | arxiv.org/pdf/2605.29920 | Unreadable (compressed stream). Outline only: Sections 1-5, Appendices A-C, propositions 1 and 3, theorem 3. |
| 8 | fetch | arxiv.org/pdf/2411.08314 | Failed: size limit exceeded. |
| 9 | fetch | arxiv.org/html/2605.29920v2 (chars 0-100k of 125k) | **Proposition 3.1 (Midpoint symmetry)**: v_{1/2}(x) = E[X1 - X0 | X_{1/2} = x] = 0 a.e. when X0, X1 are **independent** samples from the same p. Proof: swap leaves the midpoint unchanged and flips the sign of the difference. Not discussed: paired endpoints from a stationary/reversible process, the fraction of loss attainable under uniform t, any Gaussian closed form. |
| 10 | fetch | arxiv.org/html/2411.08314v5 (chars 0-100k of 242k) | Theorem 3.1 requires q(y0) = q(y1) with the **identity coupling** to generate the marginal path; Prop 3.1 (Sec 3.3): identical marginals give zero limiting transport cost in the conditioning dimension. **Does not state zero velocity**; the zero-velocity reading in #3 is the search model's inference. Remaining 142k chars unread. |
| 11 | fetch | arxiv.org/html/2605.29920v2, offset 100k | Only the NeurIPS checklist; **appendices (A-C proofs, related work) not read**. |
| 12 | search | `flow matching regression target variance intermediate time hardest logit-normal timestep sampling Stable Diffusion 3 rectified flow scaling` | Logit-normal t-sampling (SD3, Movie Gen) concentrates mass near t = 1/2 because the signal is in the middle for 1-rectified flow. Nothing found on regression-target variance vs t; no source says mid-t is hardest. Background only. |
| 13 | search | `DeepJump conditional flow matching protein molecular dynamics jump transition` | Costa et al., arXiv:2509.13294 (NeurIPS 2025 / ICLR 2026): CFM between conformation and the one a fixed step ahead; trained on mdCATH. One search; architecture and benchmarks not read. Closest published setting to ours (observed adjacent-frame coupling). |
| 14 | search | `probabilistic forecasting with stochastic interpolants and Föllmer processes conditional on current state` | Chen et al., ICML 2024, arXiv:2403.13724: forecasting as conditional sampling from the present state; follow-up arXiv:2503.12273. Conditional formulation relevant to C4. |
| 15 | search | `flow matching between consecutive frames stationary Markov process trajectory generation identity baseline persistence learned velocity collapses` | Nothing combines stationary Markov process + persistence baseline + velocity collapse. Adjacent: FlowC2S (current-to-next frame flow), Flowception, Velocity Deficit (MSE underestimates velocity magnitude, images). None read beyond the summary. |
| 16 | fetch | arxiv.org/html/2604.25244 (chars 0-100k of 220k) | Survey of AI for protein dynamics: **no DiT / diffusion transformer / Peebles / "absent" in the text read; the sentence quoted in docs/novelty_assessment.md does not appear.** |
| 17 | fetch | same, offset 100k (to 200k) | Same: no such statement. Last ~20.6k chars (references tail) unread. |
| 18 | search | `SimpleFold: Folding Proteins is Simpler than You Think flow matching general-purpose transformer DiT Peebles Xie ATLAS ensemble` | Apple, ICLR 2026, arXiv:2509.18480. Search summary could not confirm DiT or ATLAS. |
| 19 | fetch | arxiv.org/html/2509.18480v3 (chars 0-100k of 135k) | SimpleFold cites Peebles & Xie and states a "simple DiT architecture" (App. 7.1); evaluates on ATLAS (Sec 4.4.1) with a fine-tuned SimpleFold-MD. Last ~35k chars unread. |

ATLAS access queries (not prior art; recorded in the experiment log): two searches for a Zenodo/Hugging Face/institutional mirror found none; the
ATLAS download page, API docs and OpenAPI spec were fetched; see EXPERIMENT_LOG.md 2026-10-10.

## B. What follows for our claims

1. **The vanishing of the optimal field at the midpoint is published** for i.i.d. endpoints (Midpoint Generative Models, Prop 3.1). Our
   Proposition 2 extends the same swap argument to *exchangeable* observed pairs (stationary + reversible), which is a one-line
   generalization of their proof. We do not claim it as new beyond that scoping.
2. **Not found** in the pages read: the attenuation result (Prop 3), the Gaussian closed form with the [2.33, 3.0] bound, or its use
   for diagnosing training on temporally coupled MD frames. "Not found in ~19 calls, with Midpoint's appendices and ~20k chars of the
   survey unread" is the strongest statement available. A claim of novelty for Prop 3 needs a deeper search (Google Scholar, citing
   papers of Midpoint Generative Models) before submission.
3. **Kernel-Stabilized FM does NOT state zero velocity** under identical marginals in the pages read. An earlier research summary said it did;
   that was the search tool's inference and is retracted.
4. **docs/novelty_assessment.md** contained a quotation attributed to arXiv:2604.25244 ("DiT-style architectures appear absent ...")
   that does not appear in ~200k of its 220k characters, and the "no DiT in protein dynamics" claim is contradicted by SimpleFold (DiT;
   evaluated on ATLAS), MDGen (SiT, DiT-derived, as the assessment itself notes) and others. Both are corrected in that file.
5. The closest prior setting is DeepJump (CFM between conformations a fixed step apart); the conditional reformulation (C4) is close to
   stochastic-interpolant forecasting (Chen et al.). Both must be cited and positioned against, not claimed past.

## C. Not done
Google Scholar forward citations of Midpoint Generative Models; Midpoint Appendices A-C; full DeepJump read; any search for the
persistence-vs-mean baseline issue in MD benchmarks (Phase 4 survey); full-text read of the 15 competitor papers beyond what the old assessment claims.
