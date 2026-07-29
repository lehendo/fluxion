"""
tests/test_visualize.py

Unit tests for inference/visualize.py.

CPU-only, no Chai-1, no real HDF5/XTC files needed.  IO-heavy functions
(load_all_latents, extract_ca_trajectory, write_*) are tested with temporary
fixtures via pytest's tmp_path.

Run with:
    pytest tests/test_visualize.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from inference.visualize import (
    nn_decode,
    compute_pairwise_rmsds,
    load_all_latents,
    plot_nn_summary,
    write_pdb_trajectory,
    extract_ca_from_cif,
)


# ---------------------------------------------------------------------------
# Shared dims
# ---------------------------------------------------------------------------
L, D_S = 6, 8
N_TRAIN = 20   # synthetic training-set frames
T_ROLLOUT = 10  # rollout steps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_latents():
    """Random latent store: [N_TRAIN, L, D_S]."""
    torch.manual_seed(42)
    return torch.randn(N_TRAIN, L, D_S)


@pytest.fixture
def synthetic_trajectory(synthetic_latents):
    """Rollout trajectory close to training frames (small noise)."""
    torch.manual_seed(0)
    indices = torch.randint(0, N_TRAIN, (T_ROLLOUT,))
    traj = synthetic_latents[indices] + 0.01 * torch.randn(T_ROLLOUT, L, D_S)
    return traj.float(), indices.tolist()


@pytest.fixture
def frame_idxs():
    return list(range(N_TRAIN))


@pytest.fixture
def replica_ids():
    # Alternate between replica 1 and 2
    return [1 + (i % 2) for i in range(N_TRAIN)]


@pytest.fixture
def latents_dir_fixture(tmp_path):
    """
    Writes a minimal latent store on disk so load_all_latents can be tested
    without real cluster data.

    Layout:
        tmp_path/
            prot_A/
                manifest.json
                prot_A/
                    prot_A_R1_shard_000000.h5
                    prot_A_R2_shard_000000.h5
    """
    protein_id  = "prot_A"
    protein_dir = tmp_path / protein_id
    shard_dir   = protein_dir / protein_id   # double-nested (matches extract_latents.py)
    shard_dir.mkdir(parents=True)

    torch.manual_seed(7)
    L_shard, N_shard = 4, 5

    shards = []
    for replica in [1, 2]:
        singles    = torch.randn(N_shard, L_shard, D_S).half().numpy()
        frame_idxs = np.arange(N_shard, dtype=np.int32)
        shard_name = f"{protein_id}_R{replica}_shard_000000.h5"
        shard_path = shard_dir / shard_name
        with h5py.File(shard_path, "w") as hf:
            hf.create_dataset("single",    data=singles)
            hf.create_dataset("frame_idx", data=frame_idxs)

        # path relative to protein_dir (how manifest stores it)
        rel_path = str(Path(protein_id) / shard_name)
        shards.append({"path": rel_path, "n_frames": N_shard,
                        "first_idx": 0, "last_idx": N_shard - 1})

    manifest = {protein_id: {"shards": shards}}
    with open(protein_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return tmp_path, protein_id


# ---------------------------------------------------------------------------
# nn_decode
# ---------------------------------------------------------------------------

class TestNNDecode:
    def test_output_shapes(self, synthetic_latents, synthetic_trajectory,
                            frame_idxs, replica_ids):
        traj, _ = synthetic_trajectory
        nn_frames, nn_reps, nn_dists = nn_decode(
            traj, synthetic_latents, frame_idxs, replica_ids
        )
        assert nn_frames.shape   == (T_ROLLOUT,)
        assert nn_reps.shape     == (T_ROLLOUT,)
        assert nn_dists.shape    == (T_ROLLOUT,)

    def test_finds_close_frame(self, synthetic_latents, synthetic_trajectory,
                                frame_idxs, replica_ids):
        """Each rollout step was constructed near a specific training frame;
        NN should recover it."""
        traj, source_indices = synthetic_trajectory
        nn_frames, _, _ = nn_decode(
            traj, synthetic_latents, frame_idxs, replica_ids
        )
        # Source index for step i is frame_idxs[source_indices[i]] == source_indices[i]
        for i, expected in enumerate(source_indices):
            assert nn_frames[i] == expected, \
                f"Step {i}: expected frame {expected}, got {nn_frames[i]}"

    def test_distances_nonneg(self, synthetic_latents, synthetic_trajectory,
                               frame_idxs, replica_ids):
        traj, _ = synthetic_trajectory
        _, _, nn_dists = nn_decode(traj, synthetic_latents, frame_idxs, replica_ids)
        assert (nn_dists >= 0).all()

    def test_exact_match_zero_distance(self, synthetic_latents,
                                        frame_idxs, replica_ids):
        """When rollout latent exactly equals a training latent, distance is tiny.
        Float32 accumulation error over L*D_S=48 elements can give ~1e-3 residual
        after the ||a||^2 + ||b||^2 - 2<a,b> expansion, so we allow 1e-2."""
        traj = synthetic_latents[:T_ROLLOUT].clone()
        _, _, nn_dists = nn_decode(traj, synthetic_latents, frame_idxs, replica_ids)
        assert (nn_dists < 1e-2).all(), f"Expected near-zero distances, got max={nn_dists.max():.4f}"

    def test_replica_ids_valid(self, synthetic_latents, synthetic_trajectory,
                                frame_idxs, replica_ids):
        traj, _ = synthetic_trajectory
        _, nn_reps, _ = nn_decode(traj, synthetic_latents, frame_idxs, replica_ids)
        assert set(nn_reps.tolist()).issubset({1, 2})

    def test_batch_size_1_same_result(self, synthetic_latents, synthetic_trajectory,
                                       frame_idxs, replica_ids):
        """batch_size should not affect which frame is chosen (NN indices identical)."""
        traj, _ = synthetic_trajectory
        f1, r1, d1 = nn_decode(traj, synthetic_latents, frame_idxs, replica_ids,
                                batch_size=64)
        f2, r2, d2 = nn_decode(traj, synthetic_latents, frame_idxs, replica_ids,
                                batch_size=1)
        np.testing.assert_array_equal(f1, f2)
        # Distances may differ by float32 matmul order; just check frames agree
        np.testing.assert_array_equal(r1, r2)

    def test_single_training_frame(self):
        """With only one training frame, every rollout step maps to it."""
        latents    = torch.randn(1, L, D_S)
        traj       = torch.randn(5, L, D_S)
        nn_frames, nn_reps, nn_dists = nn_decode(traj, latents, [0], [1])
        assert (nn_frames == 0).all()
        assert (nn_reps   == 1).all()


# ---------------------------------------------------------------------------
# compute_pairwise_rmsds
# ---------------------------------------------------------------------------

class TestComputeRMSDs:
    def test_output_shape(self):
        coords = np.random.randn(T_ROLLOUT, 10, 3).astype(np.float32)
        rmsds  = compute_pairwise_rmsds(coords)
        assert rmsds.shape == (T_ROLLOUT,)

    def test_ref_frame_rmsd_is_zero(self):
        coords = np.random.randn(T_ROLLOUT, 10, 3).astype(np.float32)
        rmsds  = compute_pairwise_rmsds(coords, ref_idx=0)
        assert abs(rmsds[0]) < 1e-5, f"RMSD of frame 0 to itself should be 0, got {rmsds[0]}"

    def test_nonneg(self):
        coords = np.random.randn(T_ROLLOUT, 10, 3).astype(np.float32)
        rmsds  = compute_pairwise_rmsds(coords)
        assert (rmsds >= 0).all()

    def test_identical_frames_zero_rmsd(self):
        ref    = np.random.randn(10, 3).astype(np.float32)
        coords = np.stack([ref] * T_ROLLOUT, axis=0)
        rmsds  = compute_pairwise_rmsds(coords)
        assert np.allclose(rmsds, 0.0, atol=1e-5)

    def test_custom_ref_idx(self):
        coords = np.random.randn(T_ROLLOUT, 10, 3).astype(np.float32)
        rmsds  = compute_pairwise_rmsds(coords, ref_idx=3)
        assert abs(rmsds[3]) < 1e-5

    def test_dtype_float32(self):
        coords = np.random.randn(5, 8, 3).astype(np.float32)
        rmsds  = compute_pairwise_rmsds(coords)
        assert rmsds.dtype == np.float32


# ---------------------------------------------------------------------------
# load_all_latents  (uses latents_dir_fixture)
# ---------------------------------------------------------------------------

class TestLoadAllLatents:
    def test_shapes(self, latents_dir_fixture):
        latents_dir, protein_id = latents_dir_fixture
        latents, frame_idxs, replica_ids = load_all_latents(latents_dir, protein_id)
        # 2 replicas × 5 frames = 10 total
        assert latents.shape == (10, 4, D_S)
        assert len(frame_idxs)  == 10
        assert len(replica_ids) == 10

    def test_dtype_float32(self, latents_dir_fixture):
        latents_dir, protein_id = latents_dir_fixture
        latents, _, _ = load_all_latents(latents_dir, protein_id)
        assert latents.dtype == torch.float32

    def test_replica_ids_correct(self, latents_dir_fixture):
        latents_dir, protein_id = latents_dir_fixture
        _, _, replica_ids = load_all_latents(latents_dir, protein_id)
        assert set(replica_ids) == {1, 2}

    def test_dedup_duplicate_shards(self, latents_dir_fixture, tmp_path):
        """Duplicate manifest entries for the same shard path are deduplicated."""
        latents_dir, protein_id = latents_dir_fixture
        protein_dir = latents_dir / protein_id
        mf_path = protein_dir / "manifest.json"
        with open(mf_path) as f:
            manifest = json.load(f)
        # Double every shard entry (simulates old concurrency bug)
        original_shards = manifest[protein_id]["shards"]
        manifest[protein_id]["shards"] = original_shards + original_shards
        with open(mf_path, "w") as f:
            json.dump(manifest, f)

        latents, frame_idxs, replica_ids = load_all_latents(latents_dir, protein_id)
        # Should still be 10 frames, not 20
        assert len(frame_idxs) == 10

    def test_missing_manifest_raises(self, tmp_path):
        (tmp_path / "ghost").mkdir()
        with pytest.raises(FileNotFoundError):
            load_all_latents(tmp_path, "ghost")


# ---------------------------------------------------------------------------
# plot_nn_summary — smoke test (just check it doesn't crash)
# ---------------------------------------------------------------------------

class TestPlotNNSummary:
    def test_produces_png(self, tmp_path):
        nn_frames   = np.arange(T_ROLLOUT, dtype=np.int64)
        nn_replicas = np.array([1] * T_ROLLOUT, dtype=np.int64)
        nn_dists    = np.random.rand(T_ROLLOUT).astype(np.float32)
        rmsds       = np.random.rand(T_ROLLOUT).astype(np.float32)
        plot_nn_summary(nn_frames, nn_replicas, nn_dists, rmsds,
                        tmp_path, "test_protein")
        assert (tmp_path / "test_protein_rollout_summary.png").exists()

    def test_no_rmsds(self, tmp_path):
        """Should not crash when rmsds=None."""
        nn_frames   = np.arange(T_ROLLOUT, dtype=np.int64)
        nn_replicas = np.ones(T_ROLLOUT, dtype=np.int64)
        nn_dists    = np.ones(T_ROLLOUT, dtype=np.float32)
        plot_nn_summary(nn_frames, nn_replicas, nn_dists, None,
                        tmp_path, "test_protein")
        assert (tmp_path / "test_protein_rollout_summary.png").exists()


# ---------------------------------------------------------------------------
# extract_ca_from_cif — parsed against REAL chai_lab CIF output (2026-07),
# fetched directly from a real verify_chai1_decode.py run on the cluster,
# not fabricated — MDAnalysis (installed version 2.9.0) does not support
# CIF as a topology format, confirmed via a real ValueError on that run, so
# this is a from-scratch mmCIF _atom_site parser that needs its own
# ground-truth-backed test.
# ---------------------------------------------------------------------------

# Verbatim excerpt (unrelated preceding loop_ + the real _atom_site loop
# header + first 8 real ATOM rows) from
# results/verify_chai1_decode/cif/pred.model_idx_0.cif, protein 1bxy_A,
# generated by the corrected chai1_decode_to_cif() on 2026-07.
_REAL_CIF_EXCERPT = """\
data_model
_entry.id model
#
loop_
_audit_author.name
_audit_author.pdbx_ordinal
'Chai Discovery team' 1
#
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.label_entity_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET 1 1 ? A -6.406 -18.332 -5.029 1.000 1 A MET 64.875 1
ATOM 2 C CA . MET 1 1 ? A -6.283 -16.896 -5.344 1.000 1 A MET 68.835 1
ATOM 3 C C . MET 1 1 ? A -4.876 -16.399 -5.032 1.000 1 A MET 70.310 1
ATOM 4 O O . MET 1 1 ? A -4.364 -16.619 -3.941 1.000 1 A MET 67.591 1
ATOM 5 C CB . MET 1 1 ? A -7.309 -16.084 -4.512 1.000 1 A MET 63.172 1
ATOM 6 C CG . MET 1 1 ? A -8.743 -16.452 -4.851 1.000 1 A MET 58.872 1
ATOM 7 S SD . MET 1 1 ? A -9.191 -16.030 -6.535 1.000 1 A MET 57.132 1
ATOM 8 C CE . MET 1 1 ? A -9.200 -14.220 -6.480 1.000 1 A MET 50.308 1
#
"""


class TestExtractCaFromCif:
    def test_extracts_only_ca_atoms_with_correct_coordinates(self, tmp_path):
        cif_path = tmp_path / "real_excerpt.cif"
        cif_path.write_text(_REAL_CIF_EXCERPT)

        coords = extract_ca_from_cif(cif_path)

        # Only 1 CA atom in this excerpt (atom id 2, residue MET 1).
        assert coords.shape == (1, 3)
        assert coords.dtype == np.float32
        assert coords[0] == pytest.approx([-6.283, -16.896, -5.344], abs=1e-6)

    def test_skips_unrelated_preceding_loop(self, tmp_path):
        """The real file has an _audit_author loop before _atom_site —
        confirms the parser doesn't misidentify it as the atom loop."""
        cif_path = tmp_path / "real_excerpt.cif"
        cif_path.write_text(_REAL_CIF_EXCERPT)
        # If the wrong loop were parsed, this would raise before ever
        # reaching a shape assertion — the mere absence of an exception
        # here, combined with the correct coordinate in the test above,
        # confirms correct loop selection.
        coords = extract_ca_from_cif(cif_path)
        assert len(coords) > 0

    def test_no_atom_site_loop_raises(self, tmp_path):
        cif_path = tmp_path / "empty.cif"
        cif_path.write_text("data_model\n_entry.id model\n#\n")
        with pytest.raises(ValueError, match="No _atom_site loop"):
            extract_ca_from_cif(cif_path)

    def test_no_ca_atoms_raises(self, tmp_path):
        no_ca_cif = _REAL_CIF_EXCERPT.replace(
            "ATOM 2 C CA . MET 1 1 ? A -6.283 -16.896 -5.344 1.000 1 A MET 68.835 1\n",
            "",
        )
        cif_path = tmp_path / "no_ca.cif"
        cif_path.write_text(no_ca_cif)
        with pytest.raises(ValueError, match="No CA atoms"):
            extract_ca_from_cif(cif_path)

    def test_multiple_residues_multiple_ca(self, tmp_path):
        """Extend the real excerpt with a second residue's CA to confirm
        multi-row extraction, not just a single-match coincidence. Must be
        inserted BEFORE the trailing '#' loop terminator, not after."""
        extra_row = (
            "ATOM 9 C CA . PRO 2 2 ? A -10.500 -13.100 -7.200 "
            "1.000 1 A PRO 55.000 1\n"
        )
        last_atom_line = (
            "ATOM 8 C CE . MET 1 1 ? A -9.200 -14.220 -6.480 1.000 1 A MET 50.308 1\n"
        )
        assert last_atom_line in _REAL_CIF_EXCERPT
        extended = _REAL_CIF_EXCERPT.replace(last_atom_line, last_atom_line + extra_row)

        cif_path = tmp_path / "two_residues.cif"
        cif_path.write_text(extended)

        coords = extract_ca_from_cif(cif_path)
        assert coords.shape == (2, 3)
        assert coords[1] == pytest.approx([-10.500, -13.100, -7.200], abs=1e-6)
