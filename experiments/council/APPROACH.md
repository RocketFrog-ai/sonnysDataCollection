# AI Site-Selection Committee — how it works

Drop a pin → **five AI domain experts investigate it with real data, argue with each other** to a
build / no-build call, and emit a crisp report + an animated "boardroom" you can watch.

---

## The agentic framework — how they argue, and how we built it

**No agent-framework magic.** It's a classic **blackboard** + a **typed message protocol**, run as a
bounded loop. Each "expert" is a *separate* LLM call (Azure GPT-4o) with its own persona and its own
evidence. They **never call each other** — they read from and write to one **shared board**, and a
deterministic coordinator routes everything. The one orchestration dependency is **LangGraph** (the same
idiom as the repo's insights graph): a `StateGraph` with a conditional self-loop.

**The 7 things a seat can say** (the whole language):
`PUBLISH` (assert a finding) · `QUESTION` · `CHALLENGE` (dispute, citing a number) · `REQUEST` (ask for
a deeper dig) · `REVISE` (change *my own* mind) · `ENDORSE` (agree) · `VOTE`.

**The loop** (LangGraph `StateGraph`, hand-rolled fallback if it's absent):
1. **Investigate** — each seat runs its data tools (no LLM) and posts `Evidence` to the board. **Finance
   consolidates here too** — revenue, opex, CAPEX, net, breakeven land on the board *before* the debate,
   so the economics are debate subjects, not an epilogue.
2. **Publish round** — every seat (Finance included) states its opening finding + lean.
3. **Discuss ×3, on an AGENDA** — a real meeting covers the whole case, so each round has a topic:
   **round 1 the demand case** (projection vs the cluster's observed numbers, each considered site, the
   ramp, demographics) · **round 2 competition & pricing** (true express rivals, mass merchants, what the
   cluster's ASPs say about pricing power) · **round 3 economics & capacity** (5-yr net, CAPEX, breakeven,
   does the tunnel survive the peak). A lightweight **control unit** wakes the agenda's owner seats plus
   anyone with a live reason to speak; settled, unengaged seats stay quiet.
4. **Converge** — stop when there are no open challenges and beliefs stop moving (or the round cap).
5. **Resolution** — any seat with a challenge still aimed at it gets ONE focused reply (defend or concede),
   then challenges between seats that ended up agreeing are retired as moot — nothing dangles.
6. **Decide** — deterministic: the verdict is the committee's **weighted-majority lean**.
7. **Recap** — the Facilitator writes a plain-English per-round summary + insight, plus a **claim-level
   synthesis**: what the seats *agreed* on, what stayed *unresolved*, and any *unique* point only one seat
   raised — surfacing the lone-voice insight a majority vote would bury.
8. **Report** — an **operating spec**, not just go/no-go: recommended tunnel length, ASP anchors, membership
   mix, the 24–30-month maturation path, capital + payback, and a deterministic **what-if table**.

## Behind the scenes — the life of one message (the actual mechanics)

What a seat *sees* when it reacts (one JSON context, assembled by Python — this is the "shared memory"):

```json
{ "my_current_belief":   {"lean": "Build", "confidence": 0.7, "key_number": 12588,
                          "memory": "argued the 12.6k projection; conceded ASP risk last round"},
  "committee_positions": ["competition leans Build (0.75, rivals=1)", "local_market leans Conditional …"],
  "you_must_respond_to": ["CHALLENGE from local_market: your comparably-priced rivals … (cites mkt.traffic)"],
  "my_evidence":         ["hist.projected_mature 🔒 …", "hist.considered_sites 🔒 …"],
  "board":               ["… every seat's evidence, ids first …"],
  "recent_discussion":   ["CHALLENGE local_market→historical: …", "…"] }
```
plus a phase prompt: round 0 = opening statements; rounds 1–3 = "answer what's aimed at you (a defended
PUBLISH or a REVISE — an ENDORSE/VOTE does **not** answer), then debate THIS round's agenda."
`you_must_respond_to` is built from the **full log**, not a recency window — an old unanswered challenge
cannot scroll out of sight. `memory` is the seat's own 1–2 sentence running note, rewritten by the seat
every round — that's how it stays consistent across rounds.

The seat returns strict JSON — an updated belief + up to 3 proposed messages. Then **deterministic Python
decides what actually happens** (`coordinator.route`):

1. **Cite-or-drop** — a PUBLISH/CHALLENGE/ENDORSE whose cited evidence ids don't exist on the board is
   silently dropped (anti-hallucination: you cannot argue from numbers that aren't on the table).
2. **Re-litigation dedup** — same sender → same target arguing from ≥50% the same evidence (Jaccard on the
   cite sets) is a repeat, whether or not the earlier one was answered: a won argument retires.
3. **Broadcast challenges** (no addressee) are recorded as concerns, never as standing disagreements —
   there is no one whose reply could ever close them.
4. **Open-challenge ledger** — a CHALLENGE/QUESTION aimed at seat X stays open until X's next PUBLISH or
   REVISE closes it. The convergence test refuses to stop while any are open (until the round cap).
5. **REVISE applies only to the sender's own belief** — no seat can edit another's position; ENDORSE nudges
   the endorsed evidence's confidence up slightly.
6. Beliefs are **snapshotted every round** → the belief-history trail ("Historical: Build → Conditional")
   the UI shows is the actual recorded state, not a narrative.

Every number the room argues from was computed by tools *before* the debate; every message either cites the
board or dies; the verdict is `argmax` over weighted vote-mass. **The LLM proposes; Python disposes.**

## The five experts (each fetches real data)
| Seat | Investigates with |
|---|---|
| 📊 **Historical** | 12-mile cluster track record + **the consideration set** — the only sites that qualify to be considered (≥30mo history, matured, non-COVID), each with its own numbers — + a forecast anchored on exactly that set |
| 🛰️ **Competition** | live Google rivals @3 mi → filtered to *true express tunnels* → scored on the industry feasibility benchmark (with the zero-competition-fallacy guard: an empty field usually means no demand, not headroom) |
| 🏙️ **Local-Market** | 3-mile trade-area demographics/income/vehicles + a **live web-search** read (down-weighted; abstains if the data is borrowed from a distant reference site) |
| 🏗️ **Capacity** | tunnel length sized from the projected peak demand |
| 💰 **Finance** | revenue (washes × ASP), **demand-sized CAPEX**, opex → 5-yr net + breakeven — consolidated **before** the debate, and the seat argues its P&L in the room like everyone else (abstains if the demand anchor is a fallback) |

## A real run — Atlanta (33.75, −84.39), verbatim
```
INVESTIGATE  → each seat posts real numbers on the board (Finance's P&L included, up front):
  Historical : 12-mi cluster; considered sites Clairmont 12,588/mo · Peachtree 13,310/mo (each with its
               own revenue, mem-buys, ASP, distance, opened date) → projects 12,588 mature washes/mo
  Competition: 19 Google listings → 1 true express tunnel → benchmark score 90/100 → LOW saturation
  Local-Market: pop 119k in the 3-mi trade area · income $118k · growth +0.4% trailing (COVID window)
               / +5.1% PROJECTED 2025-30
  Capacity   : projected peak 15.4k/mo → tunnel 81.5 ft
  Finance    : 5-yr revenue $22.8M → net +$10.3M · CAPEX $1.9M (demand-sized) · breakeven month 13

ROUND 1 (demand)       Historical→Local-Market  CHALLENGE  "cluster median is 7,902/mo, but the considered
                                                            sites run 12.6-13.3k — the projection stands
                                                            on them, not the median"
                       Local-Market REVISE   "I concede the considered sites justify the 12,588 projection"
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
And the same committee at a **remote pin** (Lebanon KS — zero panel sites, demographics 186 km away)
argued its way to a **unanimous Don't-build (0.85, 14 mind-changes)**: *"the absence of car washes within
3 miles likely signals insufficient demand, undermining fallback projections."* Same machinery, opposite
context, opposite — and correct — behavior.

**The emergent-insight case — Houston (29.78, −95.50; 48 sites, 25 considered).** On paper a seductive
Build: projection 9,979/mo *above* the 8,676 cluster median, $8.9M 5-yr net, breakeven month 7. The room
instead found a **self-consistency trap our own pipeline doesn't model**: the tunnel is sized from average
demand (peak 10,988/mo → 64 ft), but the P&L's month-7 breakeven silently assumes *unconstrained* peak
throughput — and Local-Market coupled in seasonality (spikes make the peak peakier). Finance **conceded on
the record** that breakeven could slip. Unanimous Conditional (0.85), with the tunnel-capacity question as
the condition — a capacity × seasonality × P&L interaction no single seat (and no single forecast) carries
alone; it only exists in the cross-examination.

## What the papers contributed — finding → where it lives in the code
We didn't invent this pattern in a vacuum; four papers shaped specific mechanisms:

| Paper / finding | How it's used here |
|---|---|
| **[Can LLM Agents Really Debate? (2511.07784)](https://arxiv.org/pdf/2511.07784)** — identical-input debate is a *martingale*: correctness doesn't improve; same-model agents form echo chambers | The reason each seat holds **different evidence** (cluster panel vs Google Places vs trade-area demographics vs tunnel math vs P&L). We debate under *information asymmetry*, the regime where deliberation provably helps — and the tally is deterministic, so confident repetition can't win by volume |
| **[Diverse Evidence, Better Forecasts (2607.01661)](https://arxiv.org/pdf/2607.01661)** — deliberation beats independent aggregation by 15–25% *when agents hold complementary evidence*; use structured rounds, explicit evidence, expertise weighting, hard stopping rules | The **publish → agenda-driven discuss → recap** round structure; the **cite-or-drop rule** (evidence must be explicit); **expertise weighting** (data seats 1.0, world-knowledge 0.4); the **2–3 round cap + convergence predicate** (their "diminishing returns after 2–3 cycles") |
| **[Blackboard MAS / LbMAS (2507.01701)](https://arxiv.org/html/2507.01701v1)** — agents coordinating only via a shared blackboard beat RAG/master-slave by 13–57%; a control unit activates only relevant agents; roles: planner/critic/conflict-resolver/cleaner/decider | The **Workspace blackboard is the only channel** (seats never call each other). Their control unit → our `_active_experts()` (wake agenda owners + seats with a live reason). Their five roles map to ours: planner→the facilitator's seed plan, critic→cite-or-drop, conflict-resolver→the resolution round, cleaner→re-litigation dedup + moot sweep, decider→`decide_final`. **Deliberately deterministic** where theirs are LLMs — a decision product needs an auditable stop rule and vote |
| **[Council Mode (2604.02923)](https://arxiv.org/pdf/2604.02923)** — synthesize multi-agent output at the *claim level* (consensus / disagreement / unique findings), not by ranking whole answers | The digest's closing **✅ consensus / ⚖️ unresolved / 💡 only-one-seat-flagged** block — the lone-voice insight a majority vote would structurally bury. (Its other engine — parallel *diverse models* — is the open next step, gated by the Azure-only constraint) |

## Why you can trust the answer
- **Data beats vibes** — every claim cites a real number on the board; uncited ones are dropped; the
  data-grounded seats out-weigh the world-knowledge one; the verdict is the room's majority, not a
  tippable average, and a Build must be earned (real majority, no unresolved challenge).
- **It knows what it doesn't know** — fallback projections are labeled and down-weighted; borrowed
  demographics are marked NOT LOCAL; Finance refuses to underwrite a P&L whose demand anchor is a fallback;
  an empty competitive field reads as "probably no demand," not "no competition!"; and if the LLM is
  unreachable, the verdict says **no debate ran** instead of dressing up data-only leans.
- **The committee IS the product** — no side-model shares the stage. The leakage-clean signal that used
  to ride along lives on only as the **offline backtest** (`harness.py`, N=420, out-of-fold) that grades
  the method honestly; it never appears in a live meeting.
