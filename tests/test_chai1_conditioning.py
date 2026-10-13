"""
tests/test_chai1_conditioning.py

Unit tests for data_pipeline/chai1_conditioning.py's atom-correspondence
logic (docs/code_review_2026-08.md Sec 3, 8) — the part most prone to subtle
bugs and cheapest to regression-test without chai_lab installed (a
cluster-only dependency; see that module's docstring for why its functions
use deferred imports).

build_self_template_context's exact tensor math is NOT unit-tested here —
it calls chai_lab helpers directly and needs chai_lab installed to exercise
meaningfully. It was validated empirically against real data on the cluster
(2026-08-13): two frames of 1bxy_A (0.71 A mean / 7.39 A max real Cartesian
displacement) produced template_distances differing by mean=0.62 A,
max=4.66 A, template_unit_vector differing by mean=0.11, max=1.51 — see
docs/code_review_2026-08.md's implementation notes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_pipeline.chai1_conditioning import (
    GROMACS_ATOM_NAME_FALLBACKS,
    build_atom_correspondence,
    decode_atom_name,
    gather_frame_coords,
)


def encode_atom_name(name: str) -> torch.Tensor:
    """Matches chai_lab's _atom_names_to_tensor exactly: ord(c)-32 per char,
    left-justified to 4 chars with spaces. Used only to build test fixtures."""
    return torch.tensor([ord(c) - 32 for c in name.ljust(4, " ")][:4], dtype=torch.int)


class _FakeStructureContext:
    """Minimal duck-typed stand-in for AllAtomStructureContext, exposing only
    the fields build_atom_correspondence actually reads."""

    def __init__(self, residue_ranks: list[int], atom_names: list[str], n_tokens: int | None = None):
        self.atom_token_index = torch.tensor(residue_ranks, dtype=torch.long)
        # token_residue_index is the identity here (token i is residue rank i) --
        # real chai_lab data is the same for standard-residue-only proteins,
        # confirmed empirically for 1bxy_A (see build_self_template_context docstring).
        n_tok = n_tokens if n_tokens is not None else (max(residue_ranks) + 1)
        self.token_residue_index = torch.arange(n_tok, dtype=torch.long)
        self.atom_ref_name_chars = torch.stack([encode_atom_name(n) for n in atom_names])
        self.atom_gt_coords = torch.zeros(len(atom_names), 3)  # only .shape[0] is read


class TestBuildAtomCorrespondence:
    def test_exact_match(self):
        # Two residues, standard heavy-atom backbone names, direct match.
        sc = _FakeStructureContext(
            residue_ranks=[0, 0, 0, 1, 1, 1],
            atom_names=["N", "CA", "C", "N", "CA", "C"],
        )
        resindices = np.array([5, 5, 5, 5, 6, 6, 6, 6])  # residue 5 has an extra H atom
        names = np.array(["N", "CA", "C", "H", "N", "CA", "C", "H"])

        mapping = build_atom_correspondence(resindices, names, sc)

        assert mapping.tolist() == [0, 1, 2, 4, 5, 6]

    def test_gathered_coords_match_expected_atoms(self):
        sc = _FakeStructureContext(
            residue_ranks=[0, 0, 1, 1],
            atom_names=["N", "CA", "N", "CA"],
        )
        resindices = np.array([0, 0, 0, 1, 1])
        names = np.array(["N", "CA", "HA", "N", "CA"])
        mapping = build_atom_correspondence(resindices, names, sc)

        positions = np.array([
            [1.0, 0.0, 0.0],   # N  (res 0)
            [2.0, 0.0, 0.0],   # CA (res 0)
            [9.0, 9.0, 9.0],   # HA (res 0) -- should never be selected
            [3.0, 0.0, 0.0],   # N  (res 1)
            [4.0, 0.0, 0.0],   # CA (res 1)
        ], dtype=np.float32)

        gathered = gather_frame_coords(positions, mapping)
        expected = torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
        assert torch.allclose(gathered, expected)

    def test_ile_cd1_gromacs_fallback(self):
        # Real case found on ATLAS 1bxy_A: Ile's CD1 (Chai-1/PDB) vs CD (GROMACS).
        sc = _FakeStructureContext(residue_ranks=[0], atom_names=["CD1"])
        resindices = np.array([0])
        names = np.array(["CD"])  # GROMACS naming, no "CD1" atom exists at all

        mapping = build_atom_correspondence(resindices, names, sc)
        assert mapping.tolist() == [0]

    def test_cterminal_oxygen_gromacs_fallback(self):
        # Real case found on ATLAS 1bxy_A: C-terminal O/OXT (PDB) vs OT1/OT2 (GROMACS).
        sc = _FakeStructureContext(residue_ranks=[0, 0], atom_names=["O", "OXT"])
        resindices = np.array([0, 0])
        names = np.array(["OT1", "OT2"])

        mapping = build_atom_correspondence(resindices, names, sc)
        assert mapping.tolist() == [0, 1]

    def test_direct_match_preferred_over_fallback(self):
        # If the real structure DOES have the exact CCD name, the fallback
        # table must never be consulted -- direct match always wins.
        sc = _FakeStructureContext(residue_ranks=[0], atom_names=["CD1"])
        resindices = np.array([0, 0])
        names = np.array(["CD1", "CD"])  # both present; must pick CD1 (index 0)

        mapping = build_atom_correspondence(resindices, names, sc)
        assert mapping.tolist() == [0]

    def test_genuinely_missing_atom_raises(self):
        sc = _FakeStructureContext(residue_ranks=[0, 0], atom_names=["N", "SG"])
        resindices = np.array([0])
        names = np.array(["N"])  # "SG" has no match, real or fallback

        with pytest.raises(ValueError, match=r"1/2.*unmatched|no name-matched"):
            build_atom_correspondence(resindices, names, sc)

    def test_missing_atom_error_message_identifies_the_atom(self):
        sc = _FakeStructureContext(residue_ranks=[3], atom_names=["SG"])
        resindices = np.array([3])
        names = np.array(["SD"])  # wrong name, no fallback registered for "SG"

        with pytest.raises(ValueError) as excinfo:
            build_atom_correspondence(resindices, names, sc)
        assert "SG" in str(excinfo.value)

    def test_residue_rank_not_absolute_resid(self):
        # Correspondence must be built on SEQUENCE RANK among residues
        # present in the selection, not absolute/global resindex values --
        # robust to non-contiguous numbering (e.g. a preceding non-protein
        # residue, or a chain break) as long as both sides walk residues in
        # the same order.
        sc = _FakeStructureContext(residue_ranks=[0, 1], atom_names=["CA", "CA"])
        # Global resindices 17 and 42 (non-contiguous, out of naive-index range)
        resindices = np.array([17, 42])
        names = np.array(["CA", "CA"])

        mapping = build_atom_correspondence(resindices, names, sc)
        assert mapping.tolist() == [0, 1]

    def test_gromacs_fallback_table_only_has_documented_entries(self):
        # Guard against silent scope creep of the fallback table -- any new
        # entry should be a deliberate, documented addition (see this
        # module's docstring), not an accidental blanket rule.
        assert set(GROMACS_ATOM_NAME_FALLBACKS.keys()) == {"CD1", "O", "OXT"}


class TestDecodeAtomName:
    def test_round_trip(self):
        for name in ["N", "CA", "CD1", "OXT", "SG"]:
            assert decode_atom_name(encode_atom_name(name)) == name

    def test_strips_padding(self):
        assert decode_atom_name(encode_atom_name("N")) == "N"


class TestFrameVarianceGate:
    """The gate that stops extraction when coordinate injection has gone inert (every frame -> the same latent)."""

    def test_distinct_frames_pass_and_return_the_mean_difference(self):
        import torch
        from data_pipeline.frame_variance import check_frame_variance
        a, b = torch.zeros(1, 4, 8), torch.full((1, 4, 8), 0.5)
        assert abs(check_frame_variance(a, b, 500) - 0.5) < 1e-6

    def test_identical_frames_fail_loudly(self):
        import pytest, torch
        from data_pipeline.frame_variance import check_frame_variance
        a = torch.randn(1, 4, 8)
        with pytest.raises(RuntimeError, match="Frame-variance gate failed"):
            check_frame_variance(a, a.clone(), 500)

    def test_error_names_the_frame_and_the_measured_difference(self):
        import pytest, torch
        from data_pipeline.frame_variance import check_frame_variance
        a = torch.zeros(2, 3)
        with pytest.raises(RuntimeError) as e:
            check_frame_variance(a, a + 1e-4, 37)
        assert "frame 37" in str(e.value) and "mean|diff|=0.000100" in str(e.value)

    def test_threshold_boundary(self):
        import pytest, torch
        from data_pipeline.frame_variance import check_frame_variance, FRAME_VARIANCE_MIN
        a = torch.zeros(10)
        check_frame_variance(a, a + FRAME_VARIANCE_MIN * 1.01, 1)               # just above: passes
        with pytest.raises(RuntimeError):
            check_frame_variance(a, a + FRAME_VARIANCE_MIN * 0.99, 1)           # just below: fails

    def test_a_constant_offset_still_counts_as_variance(self):
        import torch
        from data_pipeline.frame_variance import check_frame_variance
        a = torch.randn(1, 5, 6)
        check_frame_variance(a, a + 1.0, 2)                                     # mean|diff| = 1: the gate measures change, not correlation

    def test_bf16_inputs_are_handled(self):
        import pytest, torch
        from data_pipeline.frame_variance import check_frame_variance
        a = torch.zeros(4, 4, dtype=torch.bfloat16)
        assert check_frame_variance(a, a + 0.25, 9) > 0.2
        with pytest.raises(RuntimeError):
            check_frame_variance(a, a.clone(), 9)

    def test_shape_mismatch_is_an_error_not_a_silent_pass(self):
        import pytest, torch
        from data_pipeline.frame_variance import check_frame_variance
        with pytest.raises(ValueError):
            check_frame_variance(torch.zeros(1, 4, 8), torch.zeros(1, 5, 8), 3)
