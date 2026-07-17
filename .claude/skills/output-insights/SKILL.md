---
name: output-insights
description: Whenever presenting any table, plot, chart, metric, query result, or analysis output (in chat, a notebook, a report, or an artifact), ALWAYS follow it immediately with insights derived from that specific output. Use when producing analysis results, backtest tables, correlation matrices, model scorecards, EDA figures, or dashboards.
---

# Output insights — every table/plot gets its "so what"

Never show a bare table, plot, or metric block. Every output is immediately
followed by an **Insights** block derived from *that* output's actual numbers.

## Format

- In chat/markdown: a short `**Insights:**` bullet list (2-4 bullets) right after
  the table/figure.
- In a notebook: a markdown cell immediately after each output cell (for
  generated notebooks, e.g. via `make_notebook.py`, emit the markdown cell from
  the same script).
- In an artifact/report: a caption or callout box attached to the figure.

## What a good insight bullet is

Each bullet must be one of these, **quoting the actual numbers shown**:

1. **Reading** — the one-sentence takeaway a decision-maker should walk away
   with ("2 pay stations → 6.1k washes/mo median, 3+ → 9.4k: capacity is a
   ~1.5x lever").
2. **So-what** — what the number changes about the decision or model ("traffic
   alone has oos R² ≈ 0, so a traffic-proportional projection cannot rank
   sites").
3. **Caveat** — the strongest honest reason to distrust it (small n in a cell,
   FDR-uncorrected, confound not controlled, censoring, selection). If a
   headline number would move under a reasonable alternative choice (filter,
   gate, dedup), say by how much.

## Rules

- Derive from the output shown, not from priors; if a prior (memory/doc) is
  cited, mark it as prior and say whether this output confirms or contradicts it.
- State n and the uncertainty/significance treatment (permutation p, FDR q,
  CI) whenever a claim of "real effect" is made; call sub-10 cells anecdotes.
- Check and report monotonicity/sign, not just magnitude, for ordered factors.
- If the output contradicts an earlier one in the same analysis, reconcile
  explicitly — never leave two conflicting numbers unexplained.
- No filler ("interesting", "as expected") — a bullet that could be written
  without looking at the output must be deleted.
