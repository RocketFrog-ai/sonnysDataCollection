# `experiments/council` — ── IN PROGRESS ── Council of Agents + Retrospective Backtesting

**Status:** experiment. Not live. Nothing in `app/` imports this, and it does not affect any
forecast the backend or the Streamlit app produces. `proforma` is the live version.

**Reads:** `proforma/data/panel/main-data-v2-stitched.csv` only, from the shared store.
**Writes:** `experiments/council/outputs/`.

**Run:** `python -m experiments.council.harness --limit 8` — from the repo root, conda
`sonnysDataCollection`. The full run is ~2000 LLM calls; start with `--limit`.

It *does* import `app/pnl_analysis/insights/*` (the seats are the existing insight functions,
called unchanged). That is the one direction of coupling: `v1_6` → `app`, never the reverse.

---

An **isolated** experiment (nothing under `app/` is modified) that:

1. Assembles the car-wash insight modes that already ship in `app/pnl_analysis/insights/` into a
   **council** of analysts that reason about one build/no-build decision and can *disagree*, then
   resolves them with a **deterministic adjudication rulebook** — not a free-form LLM fusion.
2. **Retrospectively backtests** that council: freeze the clock at each site's opening month `T`,
   feed the council only the local market as it looked strictly *before* `T`, and grade its go/no-go
   + projected wash level against what the site *actually* did after `T`.

Motivation: a council that just averages conflicting LLMs hallucinates. The fix (Dhruv's idea) is a
*Skill* = exact deterministic steps that know the seats in advance and encode conflict-resolution, so
the verdict is **computed**, not free-associated. And the only way to trust the whole thing is
retrospective testing (the manager's point).

## Design

**Four seats** (each an existing insight function, called **unchanged**):

| Seat | What | Weight | Source |
| --- | --- | --- | --- |
| `internal` | grounded read of the pre-T neighbour panel + a leakage-free mature-neighbour **anchor** (projected washes) | **max** | `insights/graph.market_insights` + `metrics.compute_metrics` |
| `independent` | external LLM market sizing; emits its own `wash_volume` projection | peer | `insights/location_poc.independent_market_research` |
| `competition` | competitive saturation / headroom | peer | `insights/location_poc.competition_scale_analysis` |
| `location` | qualitative world-knowledge read (lean via a small extraction) | peer | `insights/location_poc.location_market_analysis` |

**Adjudication rulebook** (`council.adjudicate`, `problem_type="go_no_go"`):
- internal seat carries a dominant weight (≈ the sum of the external weights);
- all seats agree → that verdict, high confidence;
- **internal contradicts the external majority → the data (internal) wins, and the disagreement is
  surfaced explicitly** (never averaged into mush) — resolved to the internal call if it's confident,
  else `Conditional`, always with a `conflict_note` and the condition that would flip it.

**Leakage control** (in `snapshot.build_snapshot`): panel filtered to `date < T`; neighbours must have
opened before `T` with ≥6 months of pre-T history; `is_entrant` re-derived as-of-T; the competition
seat's nearby set comes from the CSV (never live Google Places); web search disabled. The only
unclosable leak is the external LLMs' **training cutoff** — flagged, not hidden. The internal seat is
pure-Python + a pre-T anchor, so it has **no future leakage**.

## Data

Single source: `proforma/data/panel/main-data-v2-stitched.csv` (2020-01 → 2027-01, 2,103
sites, `imputed`=0). Build date `T` = each site's `operational_start` (first-data-entry month — an
approved proxy). Nothing from `proforma/`, no side-files.

**Backtest sample: N ≈ 420** — sites that (a) opened 2021+ and aren't left-censored, (b) have ≥4 months
in their own 18–30mo maturity window, and (c) have ≥2 pre-T neighbours within 20 km carrying ≥6 months
of pre-T history (the same filter the snapshot applies). ~88% are express-like (membership + volume);
the rest are small/retail-only washes the express-oriented council can't size well, flagged in the report.

## Run

```bash
# from the repo root, so both `app` and `council` import
python -m experiments.council.harness --limit 8           # cheap smoke over 8 sites spanning 2021–2024
python -m experiments.council.harness --limit 40 --workers 5
python -m experiments.council.harness                      # full N≈420 (~2000 LLM calls — slow/$$)
```
Flags: `--radius`, `--min-neighbours`, `--backend {azure|local}`, `--no-location-extract`
(skip the one extra LLM call for the location seat's lean), `--w-internal`.

Outputs to `experiments/council/outputs/`:
- `retro_council_results.csv` — one row per (site × seat) with lean, projection, realized level, APE,
  go/no-go correctness, `express_like`, conflict flags.
- `retro_council_report.md` — aggregate: go/no-go accuracy per seat vs the base rate, the internal-vs-
  external projection MAPE table, projection spread, surfaced conflicts, and the leakage banner.

## Reading the report

- **Base rate** = fraction of sites that were *actually* good builds. "Always Build" scores ≈ the base
  rate — a seat only has skill if it beats that.
- **Projection MAPE table** is the internal-vs-external head-to-head the manager asked for.
- The neighbour-based internal anchor **over-predicts sites that underperform their market** — a real,
  informative finding retrospective testing exposes, not a bug.

## Live use (no backtest)

`council.council_decision(snapshot)` also works on a present-day pin (pass `as_of=None`, real Google
Places `nearby_washes`, `use_web_search=True`) to get a live adjudicated verdict.

## Not in v1 (see the plan's Phase C)

Real multi-vendor peers (Claude / Gemini adapters in `app/pnl_analysis/insights/llm.py`), the cold-start forecast as a
retrained-as-of-T internal seat, an explicit "advise as of {year}" clause in the external prompts, and a
Streamlit "🧭 Council" tab. All deferred to keep v1 isolated and touching zero production files.
