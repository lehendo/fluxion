# Practitioner notes (drafts; each states its evidence level)

## N1. Endpoint-concentrated t schedules implemented by clamping: a stability hazard (HYPOTHESIS, one run)

Setting: conditional flow matching with t ~ Beta(0.25, 0.25) clamped to [0.01, 0.99] (the clamp was the specification; a truncated version predicts a smaller gain, 7.9% vs 10.25% attainable loss reduction).
Observation: the clamp places 17.1% of draws at exactly t = 0.01 and 17.1% at t = 0.99, i.e. on inputs within 1% of one endpoint. In the Phase 3a treatment arm the training loss jumped to 2.19x the v = 0 loss at step ~350 (end of warmup, peak LR 5.66e-4,
gate LR 5.66e-3), collapsed to ~zero velocity, and re-learned over ~1,000-3,000 steps. The uniform arm showed a smaller transient on the same ramp (step 250; loss never above 1.0003x v = 0 over 50-step windows).
Hypothesis: boundary point masses on near-clean inputs trigger the low-noise ill-conditioning of velocity regression (arXiv:2509.20952) and blow up at peak LR; truncation avoids it.
EVIDENCE LEVEL: one run, confounded with the LR peak and the large gate LR, which both arms shared; no clamp-vs-truncation comparison exists. Do not cite this as established. The in-run guard (training/train_t3.py) is the practical recommendation that does not depend on the cause.
What would test it: the same configuration to ~600 steps with (a) clamp, (b) truncation, (c) uniform, at the same LR (about 1.5 GPU-hours); not run.
