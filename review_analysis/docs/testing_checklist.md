# Manual QA Checklist

Run this after all agent deliverables have been merged into `app/`. Check off each item;
note any failures with enough detail (screenshot, error text, steps) to file a fix.

## 0. Setup

- [ ] `pip install -r requirements.txt` completes with no errors.
- [ ] `streamlit run app/Home.py` launches without a stack trace or import error.
- [ ] App loads in browser within a few seconds; no infinite spinner.

## 1. Landing dashboard

- [ ] Landing page renders with summary tiles (e.g. total reviews, avg rating, location
      count, recent activity).
- [ ] Tile numbers match a manual spot-check against `data/final_reviews.csv`
      (e.g. `len(df)` for total review count).
- [ ] Clicking/selecting a tile (or its associated location) navigates to the location
      detail / drilldown view.
- [ ] No tiles show blank, `NaN`, `None`, or placeholder values.

## 2. Trend charts

- [ ] Month/day toggle is present and switches the chart's granularity when clicked.
- [ ] Monthly view aggregates correctly (spot-check one month's count against the CSV).
- [ ] Daily view aggregates correctly (spot-check one day's count against the CSV).
- [ ] Chart visually reflects growth/decline in review volume over time (not flat/static
      regardless of toggle state).
- [ ] Date axis is sorted chronologically and labeled legibly.
- [ ] Chart handles the known data range (2020-11 through 2026-07) without errors or
      dropped points.

## 3. Location drilldown

- [ ] Location filter/selector is present and lists real location/business names from
      the data.
- [ ] Selecting a location filters all relevant views (reviews list, charts, tiles) to
      that location only.
- [ ] Location detail shows address/city/state consistent with the CSV.
- [ ] Behavior when only one location exists in the data (current dataset has a single
      `businessName`) is sane — no broken "all locations" state.

## 4. Review list & search

- [ ] Review list sorts by recency (most recent `reviewDate` first) by default.
- [ ] Search box filters review text as expected (substring match, case-insensitive
      ideally).
- [ ] Search with no matches shows a clear empty state, not a crash.
- [ ] Positive and negative sections both populate with reviews (not one empty).
- [ ] Reviews with null/empty `reviewText` (there are ~1,740 in current data) are handled
      gracefully — skipped or shown as "no comment," not blank crashes.
- [ ] Long review text doesn't break page layout (wraps/truncates properly).

## 5. Sentiment (VADER)

- [ ] Each review displays a sentiment label (e.g. positive/neutral/negative).
- [ ] Sentiment label is directionally consistent with star rating and review text for a
      handful of manually spot-checked reviews (5-star + positive text -> positive label;
      1-star + complaint text -> negative label).
- [ ] Sentiment computation doesn't visibly slow down page load to the point of being
      unusable (if it recomputes on every interaction, check for caching).

## 6. Cross-cutting / robustness

- [ ] Switching between all major views (landing -> trend -> location -> search) and back
      doesn't lose state unexpectedly or throw errors.
- [ ] No Python exceptions/tracebacks appear in the browser or terminal during normal
      navigation.
- [ ] App only reads from files inside `review_analysis/` (no absolute paths pointing
      elsewhere). Check with:
      ```bash
      grep -rnE "open\(|read_csv\(|pd\.read_|Path\(" app/ | grep -vE "review_analysis|\.\./|^[^:]*:\s*#"
      ```
      then manually inspect any hits that reference a path — confirm every file path used
      is relative to the project or explicitly under `review_analysis/`. Also grep for
      likely absolute-path leaks:
      ```bash
      grep -rn "/Users/\|/home/\|C:\\\\" app/
      ```
      Any match outside `review_analysis/` itself is a failure.
- [ ] No secrets, API keys, or credentials are hardcoded anywhere in `app/`.

## 7. Final sign-off

- [ ] All items above pass, or failures are documented and triaged.
- [ ] README.md launch instructions were verified to work as written (fresh venv, follow
      the README exactly, confirm the app comes up).
