# FUTURE_WORK (ideas recorded UNRUN; no new experiments without a gap in the writing that prose cannot hedge)

1. Phase 3b: conditional p(z1 | z0) reformulation (the brief's C4). Not run; the two-arm Phase 3 consumed that budget.
2. B2: random-initialization trunk control (needs a Chai-1 re-extraction with re-initialized weights).
3. A ridge-toward-the-common-slope (toward alpha ~ 0.7) variant of the B1 per-channel-linear check; a better-targeted test than ridge toward zero.
4. A padding-UNMASKED arm in the new setup (decided against: analytically small SNR effect).
5. Clamp-vs-truncation-vs-uniform stability tests at more seeds / longer horizon, if the queued 600-step diagnostic is dropped or inconclusive.
6. Explaining the 4-6x excess of the observed endpoint-weighting speed-up over the 1.89x signal-amplitude ratio (a training-dynamics question).
7. Cross-residue probes (probes in the paper are per-residue 384-d rows; the DiT has cross-residue attention and pair conditioning).
8. Searching for AF2/AF3 trunk-activation padding behaviour (the padding artifact is demonstrated on Chai-1 only).
9. Surveying Timewarp, ITO, UniSim, LD-FPG, TEMPO and the full text of DynaFold and BioEmu for baseline reporting.
10. Separating the nine differences between the original run and the new setup (3c isolates only batch size with its tied schedule scalings).
