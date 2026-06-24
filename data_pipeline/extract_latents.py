"""
Extract single [L, d_single] and pair [L, L, d_pair] representations from
frozen Chai-1 and write them to HDF5 shards (~1000 frames each).

Constraint: Chai-1's core model is a ModuleWrapper over a TorchScript
jit_module. register_forward_hook on internal layers is unavailable.
Extraction must go through the official Chai-1 inference API.

Output layout:
  <SCRATCH>/latents/<protein_id>/shard_XXXXXX.h5
    datasets: single [N, L, d_single]
              pair   [N, L, L, d_pair]
              timestamps [N]
"""
# TODO: implement Chai-1 JIT-safe extraction
