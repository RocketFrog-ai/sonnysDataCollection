# Council Meeting Notes — Rebuild Session

**Topic:** Why does the council of agents have no predictive edge, and can we fix it?
**Method:** three exploration agents run in parallel + offline validation, then a rebuild.

---

## The problem on the table

The LLM council (4 seats → adjudicated build/pass) scored **~30% go/no-go vs a ~32–43% base rate** in the retrospective backtest — i.e. **no better than "always build."** Tuning attempts (5-mile competition radius, threshold changes) moved nothing. We convened three agents to find out *why* and *what would actually help*.

## What the agents found (evidence, not opinion)

1. **The signal wasn't there — you can't tune your way to it.** Every seat's forward projection had ~zero correlation with the realized outcome: the internal neighbour-median "anchor" vs realized level was corr **0.16**; the proj/floor ratio that drove Build/Pass had corr **0.02** with the good-build label.

2. **The LLM seats emit a literal constant "Build"** across all sites (verified `{'Build': 1.0}`). They see only lat/lon and world knowledge — no operating data — so they *cannot* discriminate a good site from a bad one. Averaging their bullishness in is what pinned the council at the base rate.

3. **The repo's trained forecast looked like the answer — but it was leakage.** In-sample corr 0.74; an honest **as-of-T refit collapsed to corr −0.03**. It memorizes existing sites (fits its own pre-T labels at 0.89) but cannot forecast *new* ones. The only real level-signal: when the **operator already runs a matured site nearby** (corr +0.24).

4. **There IS a real, modest, leakage-clean signal — for the binary good/bad, not the level.** A small classifier over **local-market structure + operator scale** hit **out-of-fold AUC 0.57 (operator-grouped) / 0.68 on sites with matured neighbours**, above the permutation noise ceiling (0.53). Demographics / traffic / income were **near-zero predictors** (and leaky 2025 snapshots) — dropped.

## Decision

**Rebuild the council as signal-first:**
- A new **data signal seat** (`council/decider.py` over `council/features.py`) makes the go/no-go call — leakage-clean, honestly out-of-fold evaluated.
- The four LLM seats are **demoted to context/explanation** — they annotate and can flag disagreement, but **can no longer flip the verdict**.
- The trained-forecast level model is **not** wired in for greenfield (it's leakage); kept as a FUTURE task only for the operator-anchored regime with a proper as-of-T refit.

## Result (honest, out-of-fold, N=420)

| Approach | good-build precision |
| --- | ---: |
| Always build (base rate) | 42.6% |
| Old LLM council | ~30% (no edge) |
| **Signal decider — build top 30% by score** | **52.4% (+10 pts)** |

Out-of-fold AUC **0.572** (operator-grouped) > permutation ceiling **0.529** → the edge is **real, not overfit** — but modest. **~Half of build outcomes remain unpredictable** from any pre-build data available here. This is the first thing in the codebase to beat "always build."

## The signal, in plain terms
- **Headroom wins:** if even the *weakest* mature wash nearby is small, there's room → good build. If the weakest is already big → saturated → bad.
- **Mixed markets win:** dispersed neighbour outcomes = opportunity, not a uniformly saturated field.
- **Scale wins:** bigger operators build better.

## Honest caveats / what this is NOT
- Not a mature-*level* forecast — greenfield level is ~unpredictable here.
- Not a demographics/traffic story — those add nothing.
- Modest edge, strongest where neighbours have matured; flagged "weak signal" otherwise.

## Next steps (see FUTURE_TASKS.md)
1. **Market-relative target** ("beat your neighbours", base rate ~55%) — more stable and reveals a mean-reversion edge the absolute floor hides.
2. **Operator-anchored refit-as-of-T** forecast seat — the one regime with real level signal (+0.24).
3. Calibrated probability + P10–P90 interval instead of the seat-spread; grow N beyond 420.
