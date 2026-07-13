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
3. **Discuss ×3** — each seat sees the *whole board* + every peer's current position + **what's aimed at
   it**, and reacts: challenge a conflicting number, answer a challenge, revise its own belief, endorse,
   vote. A challenge stays "open" until the target answers it.
4. **Converge** — stop when there are no open challenges and beliefs stop moving (or a 3-round cap).
5. **Decide** — deterministic: the verdict is the committee's **weighted-majority lean**.

**The rules that make it trustworthy** — the LLM only *proposes* text; **deterministic Python routes
messages, tallies the votes, and writes the verdict**:
- every claim must **cite a real evidence id** on the board — uncited claims are silently dropped;
- **data-grounded seats out-weigh the world-knowledge one** (1.0 vs 0.4);
- the verdict is the **majority lean, not a mean** — one seat can't tip a weighted average over a
  threshold and override a room that deliberated the other way;
- a **Build must be earned**: a mere plurality, or challenges left unresolved, degrade it to Conditional.

That's what stops a room of confident LLMs drifting to "always build" — the exact failure the previous,
un-grounded version was deleted for. ~15 small, self-contained Python modules; nothing under `app/` or
`proforma/` is touched.

## The five experts (each fetches real data)
| Seat | Investigates with |
|---|---|
| 📊 **Historical** | 12-mile cluster track record + each qualified comparable + a forecast |
| 🛰️ **Competition** | live Google rivals @3 mi → filtered to *true express tunnels* → scored on the industry feasibility benchmark |
| 🏙️ **Local-Market** | census demographics/income + a **live web-search** read (down-weighted) |
| 🏗️ **Capacity** | tunnel length sized from the projected peak demand |
| 💰 **Finance** | revenue (washes × ASP), **demand-sized CAPEX**, opex → 5-yr net + breakeven (+ web cost benchmarks) |

## A real run — Atlanta (33.75, −84.39), verbatim
```
INVESTIGATE  → each seat posts real numbers on the board:
  Historical : 12-mi cluster = 11 sites; comparables Clairmont 12.6k/mo, Peachtree 13.3k/mo
               → projects 12,588 mature washes/mo (healthy floor is 7,473)
  Competition: 19 Google listings → only 1 true express tunnel → score 90/100 → LOW saturation
  Local-Market: pop 119k · income $118k · growth +0.4% · 2.33 mass-merchants per capita
  Capacity   : projected peak 15.4k/mo → tunnel 81.5 ft
  Finance    : 5-yr net +$10.3M, CAPEX sized to the tunnel, breakeven month 28

ROUND 0  Historical   PUBLISH               "projected 12,588/mo, backed by Clairmont & Peachtree; only
                                             1 express rival in 3 mi — strong headroom"
         Local-Market CHALLENGE→Historical  "Clairmont/Peachtree had stronger growth than this site's
                                             +0.4%. How does slow growth not temper your 12,588?"
         Local-Market CHALLENGE→Competition "your 90/100 ignores 2.33 mass merchants per capita
                                             competing for the same discretionary spend"
ROUND 1  Historical   REVISE  Build→Conditional  "You raised valid concerns about the slow population
                                                  growth… I now lean Conditional."
         Competition  REVISE  Build→Conditional  "I concede slow growth and retail density temper
                                                  demand… I revise my lean to Conditional."
ROUND 2+ ENDORSE / VOTE — the room settles on Conditional.

VERDICT: CONDITIONAL  (committee consensus, weighted-majority lean)
         condition: resolve the challenge(s) the committee left standing
🔍 quiet cross-check: the data signal reads P(good build) = 24% — worth a second look.
```
Those two `REVISE`s — Historical and Competition abandoning **Build** *because Local-Market's growth and
retail-density numbers argued them down* — are the whole point: the seats genuinely move each other.
And the headline is **faithful to the room**: the seats landed on Conditional, so the verdict is
Conditional (an earlier version let a weighted mean say "Build" over a room that had talked itself to
Conditional; the majority rule fixed that).

## Why you can trust the answer
- **Data beats vibes** — every claim cites a real number on the board; uncited ones are dropped; the
  data-grounded seats out-weigh the world-knowledge one; the verdict is the room's majority, not a
  tippable average.
- **An honest yardstick** — a leakage-clean **data signal** (validated with *no peeking at the future*;
  AUC 0.572, the only component with a *measured* edge) rides along as a quiet **P(good-build)
  cross-check**. It does **not** vote — but when it disagrees with the room, the report says so.
