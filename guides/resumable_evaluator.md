# The resumable evaluator

**Failure mode.** Evaluating several flow-model arms on held-out and training proteins takes hours, and the evaluation ran on partitions with a four-hour wall limit. A kill at the limit used to mean starting again from the first protein.

**Behaviour.** `scripts/phase3_eval.py` writes `<out>.partial.json` after every protein it finishes. On start, if that file exists it is loaded and the proteins already scored are skipped, so resubmitting the same command continues where the last job stopped. The final `<out>.json` is written only at the end.

**What makes a resumed run equal an uninterrupted one.** The sampled start frames come from a fixed seed and are identical for every method and arm, so each protein's score depends only on the protein, the checkpoint and the seed, not on the order or on which job computed it.

**Caveat.** The partial file is keyed by protein only. It does not record the checkpoint, the seed or the options it was computed with. Delete `<out>.partial.json` whenever any of those change, or a stale protein will be reused.

**Where.** `scripts/phase3_eval.py` (the `.partial.json` logic), and the end-to-end smoke test in `tests/test_phase3_smoke.py`.
