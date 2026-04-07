"""Paths to `ds/` root and sibling `datasets/`."""
from __future__ import annotations

from pathlib import Path


def ds_root() -> Path:
    return Path(__file__).resolve().parent.parent


def datasets_dir() -> Path:
    return ds_root() / "datasets"
