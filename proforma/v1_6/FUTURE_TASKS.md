# Council — future tasks (Phase C + refinements)

Deferred from v1 to keep the first cut isolated and touching zero production files. Ordered roughly by
value. See the approved plan at `~/.claude/plans/temporal-wandering-prism.md`.

## Bigger pieces (Phase C)

- [ ] **Refit-as-of-T internal seat.** Add the cold-start forecast as a numeric internal seat, but trained
      *only on data `< T`* so it's leakage-clean. Concretely: refactor `coldstart_model.fit` to accept an
      optional `panel`/`as_of` (one line: filter the panel to `date < T` before `build_features`), cache one
      `art_asof` per **quarter** (~10 trains across 2021–2024, not one per site), and pass it into
      `compute_trajectory(..., art=art_asof)` to get `plateau_med` as the internal projection. Report the
      shipped model (in-sample, optimistic) and the fit-asof model (honest) side by side. Cost: medium;
      ~400 training sites at a 2023 T vs ~1,292 today; below ~2021-09 there are 0 mature labels so those
      sites can't be scored this way. This is the rigorous version of today's leakage-free neighbour anchor.

- [ ] **Real multi-vendor peers.** Extend `app/pnl_analysis/insights/llm.py` (clean seam at
      `complete(messages, backend=...)`) with **Anthropic (Claude)** and **Gemini** adapters + `*_available()`
      guards, and a `complete_on(seat_model)` that pins a seat to a *named vendor* instead of the azure↔local
      cascade. Then the three external seats become genuine peer models (the manager's "Claude ≈ latest
      Gemini" council; exclude Fable), internal still weighted max. **Consult the `claude-api` skill** for the
      current Messages-API shape, headers, and model IDs — don't hand-wire from memory.

- [ ] **Streamlit tab (proper).** v1 renders a live council section inline (`council/streamlit_view.py`,
      wired near the KPI charts in `earnest-proforma-2.0/streamlits/app.py`). Promote it to a real tab with an
      "as-of year" slider so you can watch the *retrospective* verdict too, not just the present-day one.

- [ ] **`POST /insights/council` route** mirroring `/insights/pollinated` (deferred from the plan).

## Methodology refinements

- [ ] **"Advise as of {year}" clause** in the external prompts (needs owning/forking those prompts — a
      production edit, hence Phase C). Marginal today because LLM training-cutoff leakage already dominates
      and is flagged, but worth it once we care about tighter as-of realism.

- [ ] **Better "good build" definition.** Today = realized mature ≥ *all-site* median (~7,473). That makes
      the base rate sample-sensitive (saw 32–57% across runs). Try a **market-relative** rule: did the site hit
      ~what its own pre-T neighbours do? Reduces the arbitrariness of an absolute floor.

- [ ] **Calibrated prediction interval.** The report's "projected range" is just the seat-spread (min–max of
      point estimates), which brackets reality only ~20% of the time. Replace with a real interval (e.g. the
      internal seat's P10–P90 once the refit-asof model exists) and report true coverage.

- [ ] **More adjudication problem-types.** v1 ships `go_no_go` only. Add `diagnostic` ("why is this site
      under/over-performing?") and `capacity` (the "add $10M revenue next quarter" trigger) rule-sets to
      `council.adjudicate`.

## Follow-ups from the first backtest

- [ ] **Run the full N≈420** for stable numbers (`python -m council.harness`, ~1700 LLM calls), and once with
      `--backend local` to compare vendors.
- [ ] **Investigate the competition seat's edge.** In the 40-site run it was the most discriminating
      (~70% go/no-go accuracy vs ~30% for the bullish LLM seats). Understand why (it says Pass when saturated)
      and consider up-weighting it in the rulebook.
- [ ] **Reconcile live data source.** The live Streamlit view reads the 1.6 panel while the app map uses 2.0.
      Either make the council's data source configurable or feed it the app's already-computed neighbourhood.
- [ ] **Address council over-optimism.** All LLM seats skew Build (high recall, low precision). The neighbour
      anchor over-projects for underperformers. Both are the headline findings to act on.
