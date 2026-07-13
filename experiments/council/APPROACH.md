# AI Site-Selection Committee — how it works

Drop a pin → **five AI domain experts investigate it with real data, argue with each other** to a
build / no-build call, and emit an in-depth report + an animated "boardroom" you can watch.

---

## The agentic framework — how they argue, and how we built it

**No agent-framework magic.** It's a classic **blackboard** + a **typed message protocol**, run as a
bounded loop. Each "expert" is a *separate* LLM call (Azure GPT-4o) with its own persona and its own
evidence. They **never call each other** — they read from and write to one **shared board**, and a
deterministic coordinator routes everything.

**The 7 things a seat can say** (the whole language):
`PUBLISH` (assert a finding) · `QUESTION` · `CHALLENGE` (dispute, citing a number) · `REQUEST` (ask for
a deeper dig) · `REVISE` (change *my own* mind) · `ENDORSE` (agree) · `VOTE`.

**The loop** (a LangGraph `StateGraph`, with a hand-rolled fallback):
1. **Investigate** — each seat runs its data tools (no LLM) and posts `Evidence` to the board.
2. **Publish round** — each seat states its opening finding + lean.
3. **Discuss ×3** — each seat sees the *whole board* + the recent messages + **what's aimed at it**, and
   reacts (challenge a conflicting number, answer a challenge, revise its own belief, endorse, vote). A
   challenge is tracked until the target answers it.
4. **Converge** — stop when there are no open challenges and beliefs stop moving (or a 3-round cap).
5. **Decide** — a deterministic **weighted vote**.

**The one rule that makes it trustworthy:** the LLM only *proposes* text; **deterministic Python routes
messages, weights the votes, and writes the verdict.** Uncited claims are dropped; **data-grounded seats
out-weigh world-knowledge ones** (1.0 vs 0.4). That's what stops a room of confident LLMs from drifting to
"always build" — the exact failure the previous, un-grounded version was deleted for. ~15 small,
self-contained Python modules; nothing under `app/` or `proforma/` is touched.

## The five experts (each fetches real data)
| Seat | Investigates with |
|---|---|
| 📊 **Historical** | 12-mile cluster track record + each qualified comparable + a forecast |
| 🛰️ **Competition** | live Google rivals @3 mi → filtered to *true express tunnels* → scored on the industry feasibility benchmark |
| 🏙️ **Local-Market** | census demographics/income + a **live web-search** read (down-weighted) |
| 🏗️ **Capacity** | tunnel length sized from the projected peak demand |
| 💰 **Finance** | revenue (washes × ASP), **demand-sized CAPEX**, opex → 5-yr net + breakeven (+ web cost benchmarks) |

## Example dry run — Atlanta (33.75, −84.39)
```
INVESTIGATE  → each seat posts real numbers on the board:
  Historical : 12-mi cluster = 11 sites; 3 comparables (Top Wash Clairmont 12.6k/mo $236k rev, …);
               projected mature 12,588 washes/mo  (vs the 7,473 healthy floor)
  Competition: 19 Google listings → only 1 true express tunnel → score 90 → LOW saturation
  Local-Market: pop 119k, median income $118k, + 8 live web sources
  Capacity   : projected peak 15.4k/mo → tunnel 82 ft
  Finance    : revenue $22.8M, CAPEX $1.9M (sized to the 82 ft tunnel), 5-yr net +$10.3M, breakeven mo 28

ROUND 0  Historical PUBLISH   "cluster matures ~12.6k/mo, above the 7.5k floor"     [cites hist.projected_mature]
         Capacity   PUBLISH   "peak 15.4k justifies an 82 ft tunnel"
ROUND 1  Historical CHALLENGE→Capacity  "your Build ignores competition + the mature anchor"
         Capacity   REVISE    Build → Conditional          ← changed its mind under cross-examination
ROUND 2  beliefs stable, no open challenges → CONVERGE

VERDICT: BUILD   (committee consensus, data-weighted vote)
🔍 cross-check: the data signal reads only P(good build) = 24% — worth a second look.
```
That `REVISE` — Capacity moving Build→Conditional *because Historical argued it down* — is the whole
point: the agents genuinely move each other, not five monologues.

## Why you can trust the answer
- **Data beats vibes** — every claim cites a real number on the board; uncited ones are dropped; the
  data-grounded seats out-weigh the world-knowledge one.
- **An honest yardstick** — a leakage-clean **data signal** (validated with *no peeking at the future*;
  AUC 0.572, the only component with a *measured* edge) rides along as a quiet **P(good-build) cross-check**
  and flags when the committee is more optimistic than the data. It doesn't vote; it keeps everyone honest.
