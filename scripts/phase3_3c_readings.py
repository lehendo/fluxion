#!/usr/bin/env python3
"""Registered readings for the batch-16 arm 3c against the batch-256 uniform arm (paper Section 6.2)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from phase3_report import per_entry, boot

D = Path("paper_results/phase3")
m = sys.argv[1] if len(sys.argv) > 1 else "100"
U = json.loads((D / "eval_100.json").read_text()); C = json.loads((D / f"eval_3c_{m}.json").read_text())
out = {}
print(f"3c (arm bs16, milestone {m}%, steps {C['arms_steps']}) vs uniform arm (milestone 100%, steps {U['arms_steps']})")
verdicts = []
for w in ("ema", "raw"):
    ru = np.array(U["flow_summary"]["held_out"][f"uniform_{w}"]["per_entry"]); rc = np.array(C["flow_summary"]["held_out"][f"bs16_{w}"]["per_entry"])
    ratio = rc.mean() / ru.mean()
    rng = np.random.default_rng(0); ix = rng.integers(0, len(ru), (20000, len(ru))); bs = rc[ix].mean(1) / np.maximum(ru[ix].mean(1), 1e-12)
    pu, pc = per_entry(U, "held_out"), per_entry(C, "held_out")
    ents = sorted(pu)
    one_u = np.mean([pu[e][f"dit_uniform_{w}_n1"] for e in ents]); one_c = np.mean([pc[e][f"dit_bs16_{w}_n1"] for e in ents])
    shr = np.mean([pu[e]["shrink"] for e in ents])
    B = ratio <= 0.2 and one_c <= 3.0
    P = ratio >= 0.7 and abs(one_c - one_u) <= 0.3 * abs(one_u)
    v = "B" if B else "P" if P else "INTERMEDIATE"
    verdicts.append(v)
    print(f"[{w}] attained flow-loss reduction: uniform {ru.mean():.4f}  3c {rc.mean():.4f}  ratio 3c/uniform {ratio:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]  "
          f"| one-step vs identity: uniform {one_u:+.2f}  3c {one_c:+.2f}  (shrinkage {shr:+.2f})  | 3c minus shrinkage {one_c - shr:+.2f}  | reading {v}")
    out[w] = {"ratio": float(ratio), "one_step_uniform": float(one_u), "one_step_3c": float(one_c), "shrinkage": float(shr), "reading": v}
final = verdicts[0] if len(set(verdicts)) == 1 else "UNCLEAR (EMA and raw disagree: " + "/".join(verdicts) + ")"
print("FINAL READING (EMA and raw must agree):", final)
out["final"] = final
(D / f"3c_readings_{m}.json").write_text(json.dumps(out, indent=1))
