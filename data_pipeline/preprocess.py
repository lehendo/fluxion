"""
data_pipeline/preprocess.py

Never implemented as a standalone module: MDAnalysis-based frame loading
(originally planned here) ended up implemented directly inside
extract_latents.py (see its `mda.Universe(args.pdb, args.traj)` /
`u.trajectory` iteration), since the frame loop and the per-frame Chai-1
trunk call needed to be interleaved rather than staged through a separate
preprocessing pass. Kept as an empty stub rather than deleted only because
nothing currently imports it; safe to remove.
"""
