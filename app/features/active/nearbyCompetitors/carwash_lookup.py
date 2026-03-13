"""
Lookup module: match competitor names (from Google Places API) against the internal
car wash dataset (Type_of_carwash_final.xlsx) to enrich with official_website and
primary_carwash_type.

Matching strategy (no external fuzzy library required):
  1. Exact normalised match (lowercase, strip punctuation/common suffixes).
  2. Substring: normalised google name ⊆ normalised client_id or vice-versa.
  3. Token Jaccard: overlap of significant word tokens (ignores "car", "wash", "express", etc.).
  4. difflib SequenceMatcher ratio ≥ FUZZY_THRESHOLD on normalised strings.
Returns None when no match is confident enough.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

XLSX_PATH = Path(__file__).parent / "Type_of_carwash_final.xlsx"

FUZZY_THRESHOLD = 0.72

# Words stripped before comparison (too common to be discriminative)
_NOISE_WORDS = frozenset({
    "car", "wash", "carwash", "express", "auto", "care", "center", "centre",
    "the", "a", "an", "and", "of", "at", "in", "on",
    "inc", "llc", "ltd", "co", "corp",
    "detailing", "detail", "lube", "quick", "fast",
})


def _normalise(name: str) -> str:
    """Lowercase, strip accents, remove non-alphanumeric, collapse spaces."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _tokens(normalised: str) -> frozenset:
    """Return significant tokens (drop noise words and single chars)."""
    return frozenset(w for w in normalised.split() if w not in _NOISE_WORDS and len(w) > 1)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class CarWashLookup:
    """
    Singleton-style lookup: load once, query many times.
    Call CarWashLookup.get() to obtain the shared instance.
    """
    _instance: Optional["CarWashLookup"] = None

    def __init__(self):
        self._index: Dict[str, Dict] = {}   # normalised_name → record
        self._norm_list: list[Tuple[str, frozenset, str]] = []  # (norm, tokens, original_key)
        self._load()

    @classmethod
    def get(cls) -> "CarWashLookup":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if not XLSX_PATH.exists():
            logger.warning("CarWashLookup: xlsx not found at %s — enrichment disabled", XLSX_PATH)
            return
        try:
            import pandas as pd
            df = pd.read_excel(XLSX_PATH, engine="openpyxl")
            required = {"client_id", "official_website", "primary_carwash_type"}
            if not required.issubset(df.columns):
                logger.warning("CarWashLookup: missing columns %s", required - set(df.columns))
                return
            for _, row in df.iterrows():
                raw = str(row["client_id"]).strip() if pd.notna(row["client_id"]) else ""
                if not raw:
                    continue
                norm = _normalise(raw)
                tok = _tokens(norm)
                record = {
                    "client_id": raw,
                    "official_website": str(row["official_website"]).strip() if pd.notna(row["official_website"]) else None,
                    "primary_carwash_type": str(row["primary_carwash_type"]).strip() if pd.notna(row["primary_carwash_type"]) else None,
                }
                # Keep highest-confidence record per normalised key (dedup)
                if norm not in self._index:
                    self._index[norm] = record
                    self._norm_list.append((norm, tok, norm))
            logger.info("CarWashLookup: loaded %d entries from %s", len(self._index), XLSX_PATH.name)
        except Exception as exc:
            logger.warning("CarWashLookup: failed to load xlsx: %s", exc)

    @lru_cache(maxsize=512)
    def match(self, competitor_name: str) -> Optional[Dict]:
        """
        Return {official_website, primary_carwash_type, matched_client_id, match_score}
        or None if no confident match found.
        """
        if not self._index or not competitor_name:
            return None

        q_norm = _normalise(competitor_name)
        q_tok = _tokens(q_norm)

        # 1. Exact normalised match
        if q_norm in self._index:
            rec = self._index[q_norm]
            return {**rec, "match_score": 1.0}

        best_score = 0.0
        best_rec = None

        for norm, tok, key in self._norm_list:
            rec = self._index[key]

            # 2. Substring check (one contained in the other)
            if q_norm and norm:
                if q_norm in norm or norm in q_norm:
                    score = 0.9
                    if score > best_score:
                        best_score = score
                        best_rec = rec
                    continue

            # 3. Token Jaccard on significant words
            j = _jaccard(q_tok, tok)

            # 4. SequenceMatcher on normalised strings
            r = _ratio(q_norm, norm)

            score = max(j, r)
            if score > best_score:
                best_score = score
                best_rec = rec

        if best_score >= FUZZY_THRESHOLD and best_rec is not None:
            return {**best_rec, "match_score": round(best_score, 3)}

        return None
