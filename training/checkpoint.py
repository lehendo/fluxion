"""
Checkpoint save and restore with full RNG state.

Saves:
  - model state_dict
  - optimizer state_dict
  - lr_scheduler state_dict
  - torch CPU + CUDA rng states
  - numpy rng state
  - global step, epoch, loss history

Restore guarantees bit-identical data order and augmentation on resume.
"""
# TODO: implement save_checkpoint / load_checkpoint
