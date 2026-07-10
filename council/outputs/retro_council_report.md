# Council Backtest — signal decider (honest, out-of-fold)

The council was rebuilt: a leakage-clean **data signal** now makes the build/pass call and the LLM seats are demoted to explanation. This is the offline, out-of-fold evaluation over **N=420** focal builds (openings 2021–2024, single source `earnest-proforma-final-1.6/...stitched.csv`, data 2020-01-01 → 2027-01-01).

## Does it finally beat “always build”?

| Approach | good-build rate (precision) |
| --- | ---: |
| Always build (base rate) | 42.6% |
| Old LLM council | ~30% (no edge — = base rate) |
| **Signal decider — build top 30% by score** | **52.4%  (+10 pts)** |
| Signal decider — build top 20% | 52.4%  (+10 pts) |

## Out-of-fold AUC (is the signal real, or luck?)

| Split | AUC |
| --- | ---: |
| GroupKFold by operator (the honest number) | **0.572** |
| StratifiedKFold | 0.597 |
| Sites with ≥2 matured neighbours (where the signal applies) | 0.683 |
| Permutation noise ceiling (p95) | 0.529 |

AUC 0.572 sits **above** the permutation ceiling 0.529 → the edge is real, not overfit — but modest. Roughly half of build outcomes stay unpredictable from any pre-build data.

## What the signal is (and isn’t)

- **Is:** local-market STRUCTURE + operator scale — weak weakest-neighbour = headroom = good; a market where even the worst wash is already big = saturated = bad; bigger operators build better.
- **Isn’t:** demographics / traffic / income — near-zero predictors (and leaky 2025 snapshots), so dropped.
- **Isn’t:** a mature-LEVEL forecast — greenfield level is ~unpredictable here; the trained model’s apparent skill was operator-identity leakage (honest as-of-T refit → corr ≈0). We predict the binary good/bad instead.

## Caveats
- LLM seats no longer vote — they were structurally bullish (constant “Build”) and added no discrimination; they remain as the live explanation layer.
- Signal is strongest for sites with matured neighbours to learn from; it flags “weak signal” otherwise.
- Base rate is the absolute-floor good-build definition (42.6%); a market-relative target is the next step.