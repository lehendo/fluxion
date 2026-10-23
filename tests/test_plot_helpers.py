"""Aggregation helpers behind the flow-model figures: proteins of one PDB entry are averaged first, entries are then averaged."""
import numpy as np

from plot_phase3 import entry_mean_curve, entry_of, entry_values


def test_entry_of_splits_chain_suffix():
    assert entry_of("1bxy_A") == "1bxy" and entry_of("1b2s_F") == "1b2s"


def test_sibling_chains_are_averaged_before_entries_are_averaged():
    flow = {
        "held_out:1bxy_A": {"m": [2.0, 4.0]},
        "held_out:1bxy_B": {"m": [4.0, 8.0]},          # entry 1bxy -> [3, 6]
        "held_out:16pk_A": {"m": [0.0, 0.0]},          # entry 16pk -> [0, 0]
        "in_sample:1a62_A": {"m": [100.0, 100.0]},     # other group must be ignored
    }
    assert np.allclose(entry_mean_curve(flow, "m"), [1.5, 3.0])        # (3+0)/2, (6+0)/2, NOT the protein-weighted mean


def test_entry_values_selects_group_and_method():
    res = {"per_protein": {
        "held_out:1bxy_A": {"pct_vs_identity": {"shrink": 10.0}},
        "held_out:1bxy_B": {"pct_vs_identity": {"shrink": 20.0}},
        "held_out:16pk_A": {"pct_vs_identity": {"shrink": 5.0}},
        "in_sample:1a62_A": {"pct_vs_identity": {"shrink": 99.0}},
    }}
    v = entry_values(res, "shrink")
    assert v == {"1bxy": 15.0, "16pk": 5.0}
    assert entry_values(res, "shrink", group="in_sample") == {"1a62": 99.0}
    assert entry_values(res, "nonexistent") == {}
