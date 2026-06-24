"""
PyTorch Dataset that streams (z_t, z_t_plus_dt, dt) pairs from HDF5 shards.
Uses filelock to allow concurrent shard access across DataLoader workers.
"""
# TODO: implement HDF5 shard dataset
