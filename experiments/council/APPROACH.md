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
1. **Investigate** — each seat runs its data tools (no LLM) and posts `Evidence` to the board. **Finance
   consolidates here too** — revenue, opex, CAPEX, net, breakeven land on the board *before* the debate,
   so the economics are debate subjects, not an epilogue.
2. **Publish round** — every seat (Finance included) states its opening finding + lean.
3. **Discuss ×3, on an AGENDA** — a real meeting covers the whole case, so each round has a topic:
   **round 1 the demand case** (projection vs the cluster's observed numbers, each comparable, the ramp,
   demographics) · **round 2 competition & pricing** (true express rivals, mass merchants, what the cluster's
   ASPs say about pricing power) · **round 3 economics & capacity** (5-yr net, CAPEX, breakeven, does the
   tunnel survive the peak). A lightweight **control unit** wakes the agenda's owner seats plus anyone with a
   live reason to speak (a challenge aimed at them, an unsettled/still-moving lean, a peer engaging their
   numbers); settled, unengaged seats stay quiet. Each active seat sees the *whole board* + every peer's
   position + **everything still aimed at it** (from the full log, so nothing scrolls out of view). Answering
   a challenge means a defended PUBLISH or a REVISE — an ENDORSE or VOTE doesn't close one.
4. **Converge** — stop when there are no open challenges and beliefs stop moving (or the round cap).
5. **Resolution** — any seat with a challenge still aimed at it gets ONE focused reply (defend or concede),
   then challenges between seats that ended up agreeing are retired as moot — nothing dangles.
6. **Decide** — deterministic: the verdict is the committee's **weighted-majority lean**.
7. **Recap** — the Facilitator writes a plain-English per-round summary + insight in one Azure pass, plus a
   **claim-level synthesis** (Council-Mode pattern): what the seats *agreed* on, what stayed *unresolved*,
   and any *unique* point only one seat raised — surfacing the lone-voice insight a majority vote would bury.
8. **Report** — beyond go/no-go, an **operating spec**: recommended tunnel length, ASP anchors from the
   cluster's observed pricing, membership-mix target, the 24–30-month maturation path (with the local
   ramp-to-90% month), capital + payback — plus a deterministic **what-if table** (demand ±10/20%, ASP −$1)
   computed from the committee's own settled numbers.

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
| 💰 **Finance** | revenue (washes × ASP), **demand-sized CAPEX**, opex → 5-yr net + breakeven (+ web cost benchmarks) — consolidated **before** the debate, and the seat argues its P&L in the room like everyone else |

## A real run — Atlanta (33.75, −84.39), verbatim
```
INVESTIGATE  → each seat posts real numbers on the board (Finance's P&L included, up front):
  Historical : 12-mi cluster; comparables Clairmont 12,588/mo · Peachtree 13,310/mo (each with its own
               revenue, mem-buys, ASP, distance, opened date) → projects 12,588 mature washes/mo
  Competition: 19 Google listings → 1 true express tunnel → benchmark score 90/100 → LOW saturation
  Local-Market: pop 119k · income $118k · growth +0.4% trailing (COVID window) / +5.1% PROJECTED 2025-30
  Capacity   : projected peak 15.4k/mo → tunnel 81.5 ft
  Finance    : 5-yr revenue $22.8M → net +$10.3M · CAPEX $1.9M (demand-sized) · breakeven month 13

ROUND 1 (demand)       Historical→Local-Market  CHALLENGE  "cluster median is 7,902/mo, but the qualified
                                                            comparables run 12.6-13.3k — the projection
                                                            stands on them, not the median"
                       Local-Market REVISE   "I concede the comparables justify the 12,588 projection"
ROUND 2 (competition   Competition→Historical  CHALLENGE  "how does the 90/100 score square with cluster
         & pricing)                                        median ASPs of $47.18 mem / $19.45 retail —
                                                           is there pricing power here at all?"
                       Local-Market→Competition CHALLENGE "1 express rival, yes — but 19 total listings;
                                                           does headroom survive the broader field?"
                       Competition REVISE   "conceded — 19 total listings could dilute differentiation"
ROUND 3 (economics     Historical→Finance    CHALLENGE  "your $10.3M net assumes Clairmont's ASPs; at the
         & capacity)                                     cluster median the margin thins"
                       Capacity→Finance      CHALLENGE  "does breakeven month 13 hold if congestion caps
                                                         throughput below the 15.4k peak?"
                       Local-Market→Capacity CHALLENGE  "81.5 ft assumes ideal throughput — Atlanta
                                                         congestion says otherwise"
                       Capacity REVISE   "conceded — bottleneck risk at peak is real"

VERDICT: BUILD (0.79) · 6 mind-changes · 0 disagreements left standing
```
That's a committee covering the *whole* case — demand, pricing power, true-vs-broad competition, the P&L's
assumptions, tunnel throughput — with seats conceding on the record when a peer's number beats theirs
(6 REVISEs), and every challenge answered before the verdict. The agenda is what forces that breadth: an
earlier version let the room spend five rounds on one demographic figure (which turned out to be a
COVID-window artifact — the fix was posting BOTH growth windows, trailing and projected, clearly labeled).

## Why you can trust the answer
- **Data beats vibes** — every claim cites a real number on the board; uncited ones are dropped; the
  data-grounded seats out-weigh the world-knowledge one; the verdict is the room's majority, not a
  tippable average, and a Build must be earned (real majority, no unresolved challenge).
- **The committee IS the product** — no side-model shares the stage. The leakage-clean signal that used
  to ride along lives on only as the **offline backtest** (`harness.py`, N=420, out-of-fold) that grades
  the method honestly; it never appears in a live meeting.

## Why this design (grounded in the research)
This isn't a bespoke guess — it's the pattern the current literature converges on, and it's built to dodge
the one failure that sinks most "N agents arguing" demos.

- **The trap we avoid.** When agents share a model *and* get identical inputs, debate is a *martingale* —
  correctness doesn't improve across rounds, and confident-but-wrong majorities form echo chambers
  ([Can LLM Agents Really Debate?, 2511.07784](https://arxiv.org/pdf/2511.07784)). Our seats each hold
  **different evidence** (cluster panel vs Google Places vs tunnel math vs P&L), so this is the
  *information-asymmetry* regime where deliberation genuinely helps — measured **+15–25%** over independent
  aggregation ([Diverse Evidence, Better Forecasts, 2607.01661](https://arxiv.org/pdf/2607.01661)).
- **The backbone is validated.** A **blackboard** architecture beats RAG and master-slave setups by
  **13–57%** at low token cost ([Blackboard MAS, 2507.01701](https://arxiv.org/html/2507.01701v1)); there's
  even a published *"Council Mode"* for heterogeneous-agent consensus
  ([2604.02923](https://arxiv.org/pdf/2604.02923)).
- **What we took from it.** LbMAS's roles map onto ours (its planner/critic/conflict-resolver/cleaner/decider
  ≈ our seed-plan, cite-or-drop rule, resolution round, dedup + moot-sweep, and `decide_final`). Its one
  extra idea we adopted is the **control unit** — wake only the relevant seats each round (§loop step 3).
  Its LLM decider/cleaner we deliberately kept **deterministic**: for a decision product the stop rule and
  the vote should be auditable Python, not another model's opinion.
- **Open next step.** The papers show correlated errors when every seat shares one model; running seats on
  **different model families** would be the biggest remaining robustness gain — currently gated by the
  Azure-only constraint.
