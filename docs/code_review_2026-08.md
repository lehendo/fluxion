# Fluxion — Senior Review, August 2026

**Scope:** full-repository correctness, engineering, and publishability audit against
ICLR 2027 main-track standards. Read every module, every test, and the relevant
internals of `chai-lab==0.6.1` (source downloaded and inspected directly, not
inferred from docs). Conclusions below are labelled **[verified]** where I executed
code or read decisive source, and **[inferred]** where I reasoned from verified
premises but could not execute (no GPU / no extracted data locally).

---

## Verdict up front

The engineering quality of this repository is genuinely high. The DiT, the flow
matcher, the OT coupling, DDP/SLURM orchestration, checkpoint/EMA/resume logic,
sharded extraction with resumability, and the test suite are all well above the
median ICLR submission. The OT coupling in particular is correct in a way that is
easy to get wrong — I verified the cost-matrix orientation and that the resulting
permutation is applied jointly to `z1`, its pair conditioning, **and** its padding
mask.

**But there is one finding that is blocking, and it invalidates the project's
central scientific claim rather than merely denting it.**

> **The frozen Chai-1 trunk almost certainly never sees the per-frame MD
> coordinates that `extract_latents.py` injects. If so, every extracted latent for
> a given protein is bit-identical, the "MD trajectory dataset" is N copies of one
> sequence-only structure prediction, and the project's headline negative
> result — "there is no exploitable dynamics signal in frozen Chai-1 latents" — is
> an artifact of the extraction pipeline, not a fact about protein dynamics or
> about Chai-1.**

This is not a bug that degrades results. It is a bug that makes the observed
results *exactly what they would be under a broken pipeline*, which means the
pivot from temporal to static modeling was made on evidence that could not have
come out any other way.

The good news: the bug is in the *encoder input path*, not in the model, the
objective, or the infrastructure. Almost all of the code you have written remains
usable. And there is a clean repair route inside frozen Chai-1 (templates — see
§5.1) that would make the original temporal idea testable for the first time.

---

## 1. The blocking finding: coordinate injection is inert

### 1.1 What the pipeline intends

`data_pipeline/extract_latents.py` builds **one** `AllAtomFeatureContext` from the
protein's FASTA sequence (`_build_feature_context`, line 114), then per frame calls
`_patch_atom_coords` (line 133) to overwrite `structure_context.atom_gt_coords` with
that frame's real MD coordinates, and runs the frozen trunk. The intended semantics
is "same sequence, different conformation → different latent."

### 1.2 What actually happens

**[verified — source audit]** Of the 32 generators registered in `chai_lab.chai1`'s
inference `feature_generators` dict, I enumerated programmatically which ones
reference `atom_gt_coords` at all. Only four do, and all four are inert for this
input:

| Generator | Registered? | Why it ignores the injected coordinates |
|---|---|---|
| `TokenDistanceRestraint` | yes, `include_probability=1.0` | Coordinates are read **only** inside the `constraints is not None` branch. Extraction passes no `constraint_path`, so it returns a constant fill of `self.ignore_idx`. The restraint branch is additionally wrapped in `try/except` that logs and falls through. |
| `TokenPairPocketRestraint` | yes, `include_probability=1.0` | Same structure, same constant-fill fallthrough. |
| `DockingConstraintGenerator` | yes, but `include_probability=0.0` | The coordinate-consuming chain-partition branch is unreachable by construction. |
| `MissingChainContact` | yes | Short-circuits to zeros for single-chain input. Every ATLAS protein here is a monomer. |

Two generators genuinely consume coordinates and neither helps:
`TokenPairDistance` computes real distances but is **not in the inference
registry**; `BlockedAtomPairDistogram` / `BlockedAtomPairDistances` read
`atom_ref_pos` — idealized per-residue CCD reference conformers — not
`atom_gt_coords`.

I also closed the remaining paths individually: the template generators read only
`template_*` fields (and extraction passes no templates, so `TemplateContext.empty()`
is used); the `cdist` call inside `AllAtomStructureContext` is a glycan-specific
bond-inference helper that returns early for polymers; `get_centre_positions_and_mask`
in `token_utils.py` is called only from ranking, template loading, and the
unregistered `TokenPairDistance`; `token_bond.py` references no coordinate field;
and the collator reads no coordinate field. The trunk's own input is assembled
purely from the embedded feature dict, the bond feature, and the token input
embedder — there is no coordinate path into it.

### 1.3 Runtime confirmation

**[verified — executed]** I installed `chai-lab`'s dependencies and instantiated the
three reachable coordinate-consuming generators with the *exact* keyword arguments
the registry uses, fed each two unrelated random conformations of the same
topology, and reset RNG state between calls so any stochastic branch would be
identical rather than confounding:

```
TokenDistanceRestraint       chains=1  IDENTICAL_ACROSS_CONFORMATIONS=True  uniq_vals=[-1.0]
TokenDistanceRestraint       chains=2  IDENTICAL_ACROSS_CONFORMATIONS=True  uniq_vals=[-1.0]
TokenPairPocketRestraint     chains=1  IDENTICAL_ACROSS_CONFORMATIONS=True  uniq_vals=[-1.0]
TokenPairPocketRestraint     chains=2  IDENTICAL_ACROSS_CONFORMATIONS=True  uniq_vals=[-1.0]
MissingChainContact          chains=1  IDENTICAL_ACROSS_CONFORMATIONS=True  uniq_vals=[0.0]
MissingChainContact          chains=2  IDENTICAL_ACROSS_CONFORMATIONS=True  uniq_vals=[0.0]
```

Every one returns a single constant fill value, unchanged by conformation. The
`-1.0` is `ignore_idx`; the `0.0` is the monomer short-circuit. This is the
predicted behavior, measured.

### 1.4 Why the existing diagnostics could not have detected this

This is the part that matters most for how you interpret your own results.

**[verified]** `scripts/compute_velocity_baseline.py` raises
`RuntimeError("Dataset produced no non-self-paired samples")` when `n_samples == 0`,
which initially looked like proof that latents must vary (a degenerate dataset
would have crashed). It is not: the self-pair exclusion is **index-based**, not
value-based — it excludes pairs where the frame *indices* coincide, never comparing
values. Bit-identical latents at different frame indices sail through the count
check and reach the zero-variance guards, which force the explained-variance term
to zero at degenerate positions. **Exactly 0.00% predictability with no crash is
precisely the signature of frame-invariant latents.**

So the diagnostic chain — static, linear, velocity, and nonlinear probes, at every
lag from 10 ps to 50 ns — is fully consistent with a constant dataset. Those probes
were carefully built (I checked the mechanical-confound reasoning in their
docstrings; the 25% ceiling argument for difference-regression is correct, and the
undifferenced regression is the right number to read). They were measuring the
right thing. The input just carried no information.

**[verified]** Nothing anywhere in the repository asserts that extracted latents
vary across frames. I grepped for it; there is no such check in `scripts/`,
`data_pipeline/`, or `tests/`. `check_extraction.py` validates frame **counts** and
manifest completeness — not frame **content**. No test exercises
`_patch_atom_coords` at all.

That is the single missing assertion that would have caught this months ago, and
it is the one I would add before anything else.

### 1.5 Corroborating detail

**[verified]** Two facts make "small numerical variation" the *most* optimistic
reading available, and it is still fatal:

- Extraction does no seeding (`grep seed data_pipeline/extract_latents.py` → nothing)
  and casts intermediate features to bf16 (`_to_bf16`, line 192). So any across-frame
  variation that does exist is plausibly nondeterministic-kernel/reduction noise,
  uncorrelated with conformation.
- The pair representation is worse than frame-invariant *by design*:
  `extract_pairs.py` computes it once from **frame 0** and `dataset.py` loads that
  single `pair.pt` for every frame. So the conditioning signal the DiT receives is
  provably constant per protein, independent of the `atom_gt_coords` question. Even
  in the best case the model is conditioned on one static descriptor per protein.

### 1.6 How to settle it in five minutes

I have added **`scripts/verify_latent_frame_sensitivity.py`**. It needs no GPU and
no `chai_lab` — it reads your already-extracted `.h5` shards directly and reports
max across-frame deviation, per-position std, the fraction of exactly-zero-variance
channels, and (with `--atlas-raw`) the Spearman correlation between latent
deviation and true Cα RMSD. That correlation is the part that distinguishes real
conformational signal from bf16 noise.

I tested it against synthetic frame-invariant and frame-varying shards; both
branches report correctly.

```bash
python scripts/verify_latent_frame_sensitivity.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --atlas-raw   /scratch/arjunc4/atlas_raw \
    --protein-id  1bxy_A --n-frames 64
```

**Run this before anything else in this report.** Every other item is downstream of
its answer.

---

## 2. A second confound: all reported diversity numbers are uncalibrated

**[verified]** `scripts/evaluate_ensemble.py` line 312 decodes generated sample `i`
with `seed=args.seed + i` — a **different diffusion seed per sample**. Chai-1's
decoder is a genuine multi-step stochastic diffusion sampler. So the reported
"generated ensemble diversity" is the sum of (a) real variation between generated
latents and (b) decoder sampling noise, with no control separating them.

This cuts both ways, and both directions are damaging:

- Low measured diversity does **not** establish mode collapse — the decoder may be
  contracting distinct latents onto one structure (entirely plausible: the trunk
  single/pair representation is a fold-level descriptor, and the diffusion module
  may be insensitive to small latent perturbations).
- High measured diversity does **not** establish that the model learned the
  ensemble — decoder seed noise alone can produce Å-scale spread.

The entire "diversity deficit → root cause is missing OT coupling" narrative rests
on numbers that contain this confound. The narrative may well be right, but as
currently measured it is not supported, and a reviewer who notices this will treat
it as a fatal methodological gap.

I have added **`scripts/verify_decoder_diversity_control.py`**, which measures the
noise floor directly: decode **one fixed latent** N times varying only the decoder
seed, for both a real and a generated latent, and separately report pairwise L2
spread of the generated latents *in latent space* (decoder-free, so it detects
latent-space mode collapse independently). Any diversity claim must be stated
relative to this floor.

**[verified]** Relatedly, `scripts/verify_chai1_decode.py` — the 0.79 Å result cited
in the README as decoder validation — decodes **one real latent** and compares it
to **that same frame's** coordinates. Under the §1 hypothesis, the single latent it
uses is the sequence-only prediction, and 0.79 Å simply says Chai-1 predicts this
fold well. The test cannot distinguish a working extraction from a frame-invariant
one, because it never decodes a *second* frame and checks the structures differ.
Adding that second decode makes it decisive, and costs one more sampling run.

---

## 3. Correctness issues in the extraction path

**[verified] Atom-ordering correspondence is assumed, never checked.** `_get_sequence`
and the coordinate source both use MDAnalysis `select_atoms("protein")`, which
yields atoms in **PDB-file order** (and includes hydrogens if the ATLAS PDB has
them). Chai-1's tokenizer builds its atom list from **CCD reference conformers in
reference-conformer atom-name order**, typically heavy-atom only. These orderings
do not correspond. So even after the §1 bug is fixed, coordinates would be written
into the wrong atom slots unless an explicit name-based mapping is built.

**[verified] `_patch_atom_coords` silently masks that mismatch.** Lines 140–145
truncate when the trajectory has more atoms than the tokenized topology and
zero-pad when it has fewer. A hydrogen-containing ATLAS PDB against a heavy-atom
tokenization would silently discard the tail of the molecule and no error would
surface. This should be a hard assertion plus a name-based permutation, not a
reshape. Zero-padding coordinates is especially dangerous: it places atoms at the
origin, which is a valid-looking but physically absurd structure.

**Minor:** the `_COORD_FIELD_CANDIDATES` loop uses `hasattr` and takes the first
match; `atom_gt_coords` always exists on `AllAtomStructureContext`, so the three
fallbacks are dead code. Harmless, but it signals the original uncertainty about
which field mattered — the question that turned out to be central.

---

## 4. Evaluation and training observations

These are ordinary review comments; none is fatal, but several would draw fire.

**PCA overlap is optimistically biased.** `fit_pca` is fit on the **real** ensemble
and both sets are projected onto it (lines 350–352). The generated set is scored in
a basis chosen to explain the real data's variance. That is a defensible choice,
but it must be stated, and the reverse direction (fit on generated) or a joint fit
should be reported alongside — otherwise the JS divergence is not symmetric in the
way the metric's framing implies.

**Alignment reference is a single arbitrary frame.** `ref = real_coords[0]`
(line 335). RMSF and PCA both depend on this choice. Standard practice is to align
to the ensemble mean iteratively, or at minimum to report sensitivity to the
reference.

**`filter_majority_shape` silently drops samples.** The docstring's rationale
(preserving expensive decodes) is reasonable and the count *is* recorded in the
metrics dict, which is good practice. But atom-count variation across decodes of
the same sequence is itself a symptom worth investigating rather than filtering —
it suggests the substituted representation is pushing Chai-1's token-existence
masking off-distribution. Report the excluded fraction in the paper.

**Minibatch OT and batch size.** `ot_coupling.py`'s own docstring names a batch-size
range in which exact Hungarian assignment is cheap; `train_static.py`'s default
`--batch-size` is below it. This is not a correctness bug — the coupling is
correct at any size — but minibatch OT approximates the true OT map only as batch
size grows relative to dimension, and here the dimension is `L × 384`, which is
enormous. With a small batch the coupling is nearly arbitrary, so the fix that the
project identifies as its root-cause remedy may be substantially weaker in practice
than intended. Worth a batch-size ablation, and worth reporting the batch size
prominently in any OT-CFM claim.

**`train.py` retained as ablation.** Keeping the temporal path is the right call —
but note that under §1 its published negative result is not an ablation of the
temporal *objective*, it is an ablation of nothing.

---

## 5. Remediation routes, ranked

### 5.1 Preferred: encode each MD frame as a Chai-1 *template* **[verified viable]**

`TemplateMaskGenerator`, `TemplateUnitVectorGenerator`, `TemplateResTypeGenerator`,
and `TemplateDistogramGenerator` are **all registered in the frozen inference
registry** (`chai1.py` lines 186–189). And `templates/load.py` computes
`template_pseudo_beta_distances` and the unit vectors **directly from the template
structure context's `atom_gt_coords`** (lines 133, 149). So the template pathway is
a genuine, high-bandwidth, coordinate-dependent input to the frozen trunk:
`template_distances` is `[n_templ, n_tokens, n_tokens]` real pairwise distances plus
`[..., 3]` unit vectors.

Concretely: for each MD frame, construct a `TemplateContext` whose single template
*is that frame*, and pass it instead of `TemplateContext.empty(...)`. The trunk then
provably conditions on that conformation, through a path Chai-1 was trained to use.
This requires no retraining of Chai-1 and no change to the frozen embedder's input
signature — which is exactly why it is preferable to registering
`TokenPairDistance` (that would change the feature set the exported embedder
expects, and it is TorchScript-frozen).

This route also makes the **original temporal formulation testable for the first
time.** That is worth emphasizing: the temporal idea was never actually falsified.

### 5.2 Also available: real distance restraints

Supply per-frame distance restraints via `constraint_path`, activating the
already-registered `TokenDistanceRestraint` / `TokenPairPocketRestraint` branches
that currently fall through. Lower bandwidth (sparse residue pairs) and requires
choosing which pairs to restrain, but legitimate and cheap to try as a
cross-check on 5.1.

### 5.3 Necessary regardless: fix the pair representation

Extract `pair` **per frame** rather than once from frame 0, or drop the pretense
that it is conformational conditioning. As it stands the DiT's conditioning signal
is constant per protein (§1.5). This is a storage cost question — `[L, L, 256]` per
frame is large — so the honest options are per-frame pair extraction with
aggressive sharding, a low-rank/compressed pair summary, or explicitly reframing
the pair as static per-protein context in the paper.

### 5.4 If Chai-1's trunk turns out to be genuinely fold-level

A real possibility worth naming: even conditioned properly, the trunk single/pair
representation may be too fold-level to resolve thermal conformational
fluctuations — it was trained to predict *one* structure per sequence. If §5.1
lands and predictability is still ~0, *that* is a genuine and publishable negative
result about pretrained structure-prediction representations. But it can only be
claimed after the encoder demonstrably sees conformation. Right now the experiment
that would support it has not been run.

---

## 6. Publishability assessment

**Current state: not submittable, and the gap is not one of polish.** The four-way
novelty combination in `docs/novelty_assessment.md` (OT-CFM + frozen
structure-prediction latents + DiT + protein dynamics) is a real gap in the
literature and I found no reason to doubt the survey work — the claim structure is
sound and the competitor analysis is thorough. Novelty is not the problem.

The problem is that the empirical content currently consists of:

1. a negative result about latent dynamics that is confounded by §1;
2. a positive-then-negative diversity story that is confounded by §2;
3. a decoder validation that cannot discriminate the failure mode (§2).

A competent ICLR reviewer with access to the code will find §1 — it is one grep and
one assertion away — and it invalidates the paper's central empirical claim. Better
to find it yourself now, which is what this review is for.

**The path to a strong submission is clearer than it looks.** Fix §5.1 + §5.3, rerun
the diagnostic chain that already exists (it is well-built and will simply work),
and you get one of two outcomes, both publishable:

- **Signal appears.** The original temporal OT-CFM-in-latent-space formulation
  becomes viable, the novelty claim stands as written, and you have a positive
  result with a strong methods story — plus a genuinely instructive account of an
  encoder-conditioning bug that silently produces a perfect null.
- **Signal still absent.** You have a rigorous, well-controlled negative result on
  whether frozen structure-prediction trunk representations encode
  thermally-accessible conformational variation — with the calibration controls
  (§2) and the coordinate-sensitivity verification (§1.6) to make it airtight. That
  is a real contribution, and the field currently lacks it.

What is *not* available is publishing the current negative result as-is.

**Timeline:** ICLR 2027 is far enough out that this is recoverable without drama.
The infrastructure — extraction, DDP training, checkpointing, evaluation, the
probe suite — is the expensive part and it is done and sound. What remains is a
change to the encoder input path and a rerun.

---

## 7. What I added

- **`scripts/verify_latent_frame_sensitivity.py`** — the missing foundational check.
  No GPU, no `chai_lab`; reads extracted shards. Reports across-frame deviation,
  zero-variance fraction, and latent-vs-coordinate-RMSD rank correlation. Tested on
  synthetic invariant and varying inputs.
- **`scripts/verify_decoder_diversity_control.py`** — decoder noise floor. Decodes one
  fixed latent N times varying only the decoder seed (real and generated), plus
  decoder-free latent-space spread to detect mode collapse independently.

## 8. Recommended order of work

1. Run `verify_latent_frame_sensitivity.py`. Everything below depends on the answer.
2. Add a permanent assertion in `extract_latents.py` (or `check_extraction.py`) that
   consecutive frames produce different latents, and a unit test around
   `_patch_atom_coords`. This bug class must not be able to recur silently.
3. Build the name-based atom mapping and replace the truncate/pad in
   `_patch_atom_coords` with a hard assertion (§3).
4. Implement the template-based conditioning route (§5.1) and per-frame pair
   extraction or an explicit reframing (§5.3).
5. Re-run the existing diagnostic chain unchanged. It is sound; it was starved of
   input.
6. Run `verify_decoder_diversity_control.py` before quoting any diversity number
   again, and add the second-frame decode to `verify_chai1_decode.py`.
7. Then, and only then, interpret model results.
