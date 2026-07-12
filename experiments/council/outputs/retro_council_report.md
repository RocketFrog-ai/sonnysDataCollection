# Council backtest — signal decider (honest, out-of-fold)

The go/no-go anchor is a leakage-clean **data signal** (local-market structure + operator scale). This is the offline, out-of-fold evaluation over **N=420** focal builds (openings 2021+, single source `experiments/council/data/Council--historical-data.csv`, data 2020-01-01 → 2027-01-01). The committee's LLM deliberation never touches this number.

## Does it beat “always build”?

| Approach | good-build rate (precision) |
| --- | ---: |
| Always build (base rate) | 42.6% |
| **Signal — build top 30% by score** | **52.4%  (+10 pts)** |
| Signal — build top 20% | 52.4%  (+10 pts) |

## Out-of-fold AUC (real signal, or luck?)

| Split | AUC |
| --- | ---: |
| GroupKFold by operator (the honest number) | **0.572** |
| StratifiedKFold | 0.597 |
| Sites with ≥2 matured neighbours | 0.683 |
| Permutation noise ceiling (p95) | 0.529 |

AUC 0.572 sits above the permutation ceiling 0.529 → the edge is real but modest; ~half of build outcomes stay unpredictable from any pre-build data.

## What the signal is
- **Is:** local-market STRUCTURE + operator scale — a weak weakest-neighbour = headroom = good; a market where even the worst wash is already big = saturated = bad; bigger operators build better.
- **Isn’t:** demographics / traffic / income (near-zero predictors), nor a mature-LEVEL forecast (greenfield level is ~unpredictable; we predict the binary good/bad).

In the committee, the LLM seats reason and argue around this signal (which sits on the board as evidence); the committee decides, and any divergence from the signal is surfaced in the report.