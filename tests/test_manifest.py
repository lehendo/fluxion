"""The manifest covers every script and every saved result."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_current_and_complete():
    before = (ROOT / "manifest.json").read_text()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_manifest.py")], check=True, capture_output=True)
    after = json.loads((ROOT / "manifest.json").read_text())
    assert json.loads(before) == after, "manifest.json is stale: run scripts/make_manifest.py"
    assert after["unmatched_result_files"] == [], after["unmatched_result_files"]
    scripts = {p.name for p in (ROOT / "scripts").glob("*.py")} - {"__init__.py", "make_manifest.py"}
    assert {Path(s["script"]).name for s in after["scripts"]} == scripts
    assert all(s["artifact"] for s in after["scripts"]), "every script needs a one-line docstring naming its paper artifact"
