# The frame-variance gate

**Failure mode.** If coordinate injection goes inert (see [chai1_conditioning.md](chai1_conditioning.md)), every frame of a protein maps to nearly the same latent, and the extraction finishes normally after hours of GPU time with a degenerate dataset. Nothing in the output file looks wrong.

**The gate.** `fluxion.data_pipeline.frame_variance.check_frame_variance` compares the trunk's single representation of frame 0 with that of a later frame (frame 500, or the last frame of a shorter trajectory) and raises `RuntimeError` when the mean absolute difference is below 0.01. `extract_latents.py` runs it in the pre-flight stage, before the main loop, so a broken setup fails within minutes. A future `chai-lab` upgrade that changes which feature generators are registered would trip it.

**What it does not catch.** It tests that latents respond to coordinates, not that they are reproducible. Re-extracting the same frame gives different latents (independent per-frame noise, 7 to 41% of a 10 ps step in our data; paper Section 3.2), and the values at padded positions depend on the GPU type (paper Section 3.1). Neither would trip the gate. Check them separately: `scripts/real_token_reproducibility.py` and `scripts/analyze_reextract_padding.py`.

**Where.** `src/fluxion/data_pipeline/frame_variance.py`, its call in `src/fluxion/data_pipeline/extract_latents.py`, and `tests/`.
