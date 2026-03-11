"""
Production home for CTOExactProfiler.

The class is defined in scoringmetric/approach2/main.py (research script).
This module re-exports it cleanly so the rest of the app can import from
a stable path without sys.path hacks.

Usage:
    from app.modelling.ds.profiler import CTOExactProfiler
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make scoringmetric a discoverable package at the project root level.
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scoringmetric.approach2.main import CTOExactProfiler  # noqa: F401, E402

__all__ = ["CTOExactProfiler"]
