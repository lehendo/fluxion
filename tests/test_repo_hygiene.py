"""Regression test: no cluster identifiers, private paths or private-record references anywhere in the repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# patterns are built by concatenation so that this file does not match itself
BANNED = ["/scr" + "atch/", "/u/arjun" + "c4", "arjun" + "c4", "jim" + "eng", "cc-" + "login", "campus" + "cluster", "ic-" + "express", "Illinois" + "Computes",
          "EXPERIMENT" + "_LOG", "claims_" + "table", "code_review" + "_2026", "TM" + "LR"]
PRIVATE_DIRS = ["do" + "cs", "experi" + "ments", "res" + "ults", "sl" + "urm", "paper_" + "tex", "supplement" + "_anon"]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}
TEXT = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".cff", ".csv", ".sh", ".cfg", ".ini", ""}


def test_private_directories_absent():
    for d in PRIVATE_DIRS:
        assert not (ROOT / d).exists(), f"private directory {d}/ must not be in the repository"


def test_no_identifying_strings():
    hits = []
    for f in ROOT.rglob("*"):
        if not f.is_file() or SKIP_PARTS & set(f.relative_to(ROOT).parts) or f.suffix not in TEXT or f.name.endswith(".egg-info"):
            continue
        if any(part.endswith(".egg-info") for part in f.parts):
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        for pat in BANNED:
            if pat in text:
                hits.append(f"{f.relative_to(ROOT)}: {pat}")
    assert not hits, "identifying strings found:\n" + "\n".join(hits[:40])
