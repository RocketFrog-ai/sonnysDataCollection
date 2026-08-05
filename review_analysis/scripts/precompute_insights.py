"""Fill data/ai_insights_cache.json for the windows the dashboard opens with.

Every tile's AI button asks a fixed question about a fixed slice of reviews, so
those answers can be generated ahead of time and read from disk instantly.
This script walks the same scopes the app builds — nine tiles x the current
month and the last three months — and writes each answer into the cache under
the same key the app looks up.

Run from the project root, in the env that runs the app:

    python scripts/precompute_insights.py            # both windows, all tiles
    python scripts/precompute_insights.py --window "Current month"
    python scripts/precompute_insights.py --force    # ignore existing entries

Costs roughly one model call per 45 reviews plus one merge per scope, so a full
run is a few dozen calls. Existing cache entries are skipped unless --force.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils import ai as AI                       # noqa: E402
from app.utils import reviews_ui as RU               # noqa: E402
from app.utils.data_loader import SITE_COL           # noqa: E402
from app.utils.metrics import (                      # noqa: E402
    load_scored_data,
    site_table,
    window_frames,
)

WINDOWS = ["Current month", "Last 3 months"]

# (tile key, review lens, "restrict to the best/worst location?")
TILES = [
    ("reviews", "All", None),
    ("rating", "All", None),        # matches DESTINATIONS["rating"]: no review filter
    ("sentiment", "All", None),
    ("positive", "Positive", None),
    ("negative", "Negative", None),
    ("response", "Awaiting owner reply", None),
    ("volume", "All", None),
    ("best", "Positive", "best"),
    ("worst", "Negative", "worst"),
]

MIN_SCORED_FOR_RANK = 20  # mirrors app/Home.py


def leaders(frame):
    """(best, worst) location by net sentiment — same rule as the dashboard."""
    table = site_table(frame)
    if table.empty:
        return None, None
    eligible = table[table["n_scored"] >= MIN_SCORED_FOR_RANK]
    if len(eligible) < 3:
        eligible = table.nlargest(min(3, len(table)), "n_scored")
    ranked = eligible.sort_values("net_sentiment")
    worst = str(ranked.iloc[0]["site"])
    best = str(ranked.iloc[-1]["site"]) if len(ranked) > 1 else None
    return best, worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="append", choices=WINDOWS,
                        help="limit to one window (repeatable)")
    parser.add_argument("--force", action="store_true", help="regenerate cached answers")
    args = parser.parse_args()

    if not AI.is_configured():
        print("No Azure OpenAI key configured — nothing to do.", file=sys.stderr)
        return 1

    df = load_scored_data()
    n_sites = df[SITE_COL].nunique()
    windows = args.window or WINDOWS
    made = skipped = failed = 0

    for window in windows:
        cur, _prior, window_label, _caption = window_frames(df, window)
        if cur.empty:
            print(f"[{window}] no reviews — skipped")
            continue
        best, worst = leaders(cur)

        for tile, lens, leader in TILES:
            scope_df = RU.apply_filter(cur, lens)
            site = best if leader == "best" else (worst if leader == "worst" else None)
            if leader and not site:
                continue
            if site:
                scope_df = scope_df[scope_df[SITE_COL] == site]

            # Must match app/Home.py's tile_ai() exactly, or the app will miss.
            scope_label = (f"{window_label} · {site or f'{n_sites} location(s)'}"
                           + (f" · {lens.lower()} reviews only" if lens != "All" else ""))
            question = AI.DEFAULT_QUESTIONS.get(tile, AI.DEFAULT_QUESTION)
            digest = AI.review_digest(scope_df)

            if not digest:
                print(f"[{window}] {tile:<10} no reviews with text — skipped")
                continue

            key = AI.digest_key(question, scope_label, digest)
            if not args.force and AI.cache_get(key):
                skipped += 1
                print(f"[{window}] {tile:<10} cached")
                continue

            started = time.time()
            answer = AI.build_insight(question, scope_label, digest, use_cache=not args.force)
            if answer.get("error"):
                failed += 1
                print(f"[{window}] {tile:<10} FAILED: {answer['error']}")
                continue
            made += 1
            print(f"[{window}] {tile:<10} {len(digest):>3} reviews, "
                  f"{answer.get('n_calls', 1)} calls, {time.time() - started:.1f}s, "
                  f"{len(answer.get('key_points', []))} key points")

    print(f"\ndone: {made} generated, {skipped} already cached, {failed} failed")
    print(f"cache: {AI.CACHE_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
