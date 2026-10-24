"""Where Fluxion reads and writes data, configurable through environment variables.

FLUXION_DATA          root for extracted latents, caches, checkpoints and intermediate results (default ./data)
FLUXION_ATLAS_10PS    ATLAS trajectories saved every 10 ps, one directory per protein (default $FLUXION_DATA/atlas_10ps)
FLUXION_ATLAS_RAW     ATLAS 100 ps trajectories (default $FLUXION_DATA/atlas_raw)
FLUXION_ATLAS_ANALYSIS  ATLAS "analysis" archives (default $FLUXION_DATA/atlas_analysis)
FLUXION_LOGS          job logs read by build_extraction_metadata.py (default ./logs)
FLUXION_PAPER_RESULTS where the analysis scripts write their result summaries (default ./paper_results)
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FLUXION_DATA", "data"))
ATLAS_10PS = Path(os.environ.get("FLUXION_ATLAS_10PS", DATA_ROOT / "atlas_10ps"))
ATLAS_RAW = Path(os.environ.get("FLUXION_ATLAS_RAW", DATA_ROOT / "atlas_raw"))
ATLAS_ANALYSIS = Path(os.environ.get("FLUXION_ATLAS_ANALYSIS", DATA_ROOT / "atlas_analysis"))
LOG_DIR = Path(os.environ.get("FLUXION_LOGS", "logs"))
PAPER_RESULTS = Path(os.environ.get("FLUXION_PAPER_RESULTS", "paper_results"))
