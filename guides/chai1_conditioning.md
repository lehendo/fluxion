# Per-frame conditioning of the frozen Chai-1 trunk

**Failure mode.** Our first extraction wrote latents that were almost identical for every frame of a protein, because the per-frame coordinates never reached the trunk. Only two of Chai-1's registered feature generators read atom coordinates at all, and both return a constant unless a constraint file is supplied. The generator that would use raw coordinates is not registered. The positional truncate-and-pad that patched the coordinates in was also wrong: ATLAS PDB files include hydrogens (1,006 protein atoms for 1bxy_A), while Chai-1 tokenizes heavy atoms only (476).

**Fix.** `fluxion.data_pipeline.chai1_conditioning` does two things.

1. `build_atom_correspondence` maps Chai-1's heavy atoms to the structure's atoms by (residue rank, atom name), with a documented fallback for GROMACS-versus-PDB naming (isoleucine CD1/CD, C-terminal O versus OXT). All 476 atoms of 1bxy_A matched, three of them through the fallback. An unmatched atom raises an error; nothing is silently reshaped.
2. `build_self_template_context` treats the frame's own structure as its own template, with the identity alignment. The trunk then receives the coordinates through the template features that Chai-1 does compute from atom positions: pseudo-beta distances (binned) and backbone-frame unit vectors.

**What reaches the trunk.** Two frames of 1bxy_A with 0.71 Å mean displacement gave template distances that differ by 0.62 Å on average. The trunk sees side-chain direction (through C-beta) and backbone orientation, not only C-alpha positions; the paper's comparison with C-alpha coordinates depends on this (paper Section 4.2). `scripts/verify_template_features.py` checks an independent reimplementation of these features against the tensors this pipeline produces.

**Imports.** Chai-1 is imported inside the functions, so the module and its tests work without `chai-lab` installed (`pip install -e ".[extract]"` installs it).

**Next.** [frame_variance_gate.md](frame_variance_gate.md) is the regression check for this failure.
