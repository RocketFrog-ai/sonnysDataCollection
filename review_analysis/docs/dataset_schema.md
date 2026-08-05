# Dataset Schema & Data Quality — `data/final_reviews.csv`

Owner: Agent 2 (Dataset & Data Pipeline). Companion to `app/utils/data_loader.py`,
which implements every cleaning decision documented here. Read this before
writing any page that touches the raw CSV directly — it shouldn't be
necessary, since `data_loader.load_data()` returns an already-clean
DataFrame, but if something looks surprising downstream, the reasoning is
here.

All numbers below were produced by loading the actual file with
`pandas.read_csv` (not by eyeballing `head`), on 2026-08-04.

## 0. Row count correction

`wc -l data/final_reviews.csv` reports **7,897** lines. The actual row count,
confirmed by both pandas' C and Python CSV parsers with zero bad-line
warnings, is **6,209** (6,087 after dedup — see §4). The discrepancy is
because `reviewText` and `ownerResponseText` contain embedded newlines
inside quoted CSV fields (multi-paragraph reviews), which is valid CSV but
inflates a raw newline count. Never use `wc -l` as a row-count proxy for
this file; always parse with pandas.

## 1. Column-by-column documentation

24 columns. Null % and blank-string % are identical for every column below
(no column has non-null-but-whitespace-only values except where noted), so
one "null %" figure covers both.

| Column | Dtype (raw) | Null % | Cardinality | Notes / example values |
|---|---|---|---|---|
| `site` | str | 0% | 25 | Per-location friendly label, e.g. `"Big Dan's Rome"`. **The location key** — see §3. |
| `businessName` | str | 0% | 1 | Constant: `"Big Dan's Car Wash"`. This is one chain, not multiple businesses — do not use as a location dimension. |
| `address` | str | 0% | 25 | Full street address; maps 1:1 with `site` (see §3). Format is inconsistent for 5 sites — see §3. |
| `city` | str | 0% | 20 | NOT a safe location key alone — 3 cities each host 2–3 distinct `site`s (Bradenton ×3, Woodstock ×2, Orlando ×2). |
| `state` | str | 0% | 6 raw → 4 after normalization | **Inconsistent formatting**: mixes full names (`Georgia` 2276, `Florida` 3421) and abbreviations (`FL` 294, `SC` 66, `AL` 17, plus `South Carolina` 135). See §3 for the fix. |
| `postalCode` | str | 0% | 24 | 5-digit zips, no leading-zero truncation issues (all 5 chars). |
| `placeId` | str | 6.1% (377 rows) | 20 | Google Maps place ID. Null for exactly 5 of 25 sites — a structural gap, not random (see §3). |
| `category` | str | 0% | 2 | `"Car wash"` (6,157) / `"Self service car wash"` (52). Stable per site (verified: no site has >1 category). |
| `businessAvgRating` | str→float | 6.1% (377) | 6 | Google's own aggregate rating for the site (e.g. `4.6`). Null for the same 5 sites as `placeId`. |
| `businessReviewCount` | str→int | 6.1% (377) | 20 | Google's own total review count for the site. Same 5-site null pattern. |
| `reviewId` | str | 2.0% (125) | 6,084 (of 6,084 non-null) | Opaque Google review ID. **No genuine duplicates among non-null values** — see §4 for an important correction to an earlier team note. |
| `reviewerName` | str | 0% | 5,979 | Display name of reviewer. |
| `rating` | str→Int64 | 0% | 5 | Clean integers 1–5, no nulls, no out-of-range values. Distribution: 5★=5,418 (87.3%), 4★=325, 3★=120, 2★=76, 1★=270. |
| `reviewText` | str | 28.0% (1,740) | 4,273 | Free text. Null = legitimate rating-only review (Google allows a star rating with no text), not a data error — text presence correlates with rating (94% of 1★ reviews have text vs 71% of 5★). |
| `reviewDate` | str→datetime64[UTC] | 0% | 5,751 | **The** date column. Mixes two ISO-8601 shapes — see §5. Range after parsing: 2020-11-05 to 2026-07-22. |
| `reviewDateRelative` | str | 0% | 57 | Human string, e.g. `"3 years ago"`, `"Edited a year ago"`. Redundant with `reviewDate` — **not used for aggregation**, but the `"Edited ..."` prefix is diagnostic (see §5 caveat). |
| `language` | str | 6.6% (411) | 3 (+null) | `en` (5,794), `en-US` (3), `es` (1). Not very informative; mostly a constant. |
| `isLocalGuide` | str→boolean | 8.0% (497) | 2 | `"true"`/`"false"` strings; cast to nullable pandas `boolean`, not assumed clean bool. |
| `reviewerReviewCount` | str→Int64 | 8.0% (497) | 348 | Reviewer's total review count on Google. Range 0–1,330. |
| `likesCount` | str→Int64 | 8.0% (497) | 8 | "Helpful" likes on the review. Range 0–9, median 0. |
| `ownerResponseDate` | str→datetime64[UTC] | 11.2% (697) | 5,512 | When the business replied. Same 88.8% non-null rate as `ownerResponseText` (paired). |
| `ownerResponseText` | str | 11.2% (697) | 4,850 | Business's reply text. **88.8% of reviews got a reply** — usable for a future "reply rate" metric per Agent 6's deferred-metrics note. 393 rows have leading/trailing whitespace (stripped by `data_loader`). |
| `reviewUrl` | str | 8.0% (497) | 5,712 | Deep link to the review. Null for the same 497-row own_crawler group (see §3). |
| `dataSource` | str | 0% | 3 | `v1_v2_scrape` (5,712), `own_crawler_crawler` (372), `own_crawler_api` (125). Explains most of the null patterns above — see §3. |

Whitespace check: ran a strip-vs-original comparison on every string column.
Only `ownerResponseText` had whitespace issues (393 rows); all other columns
were already clean. `site` has no case-variant duplicates (25 unique
case-sensitive == 25 unique case-insensitive).

## 2. Rating / text / date column determination

- **Rating column: `rating`.** Clean `Int64`, values strictly 1–5, zero
  nulls. No decision required.
- **Review text column: `reviewText`.** 28.0% null (1,740 rows) —
  confirmed these are legitimate rating-only submissions (Google permits a
  star rating with no written review), not corrupted rows: null-text rate
  varies sensibly by rating (94% of 1★ reviews have text — angry customers
  write more — vs. 57–71% for 4–5★). Any sentiment/keyword logic must treat
  null as "no text supplied," never coerce to `""` and score it as neutral
  text (VADER on `""` also happens to score 0.0/neutral, which would be
  indistinguishable and misleading — Agent 6's `sentiment.py` already
  handles this correctly with a dedicated `"no_text"` label).
- **Date column: `reviewDate`.** ISO-8601 UTC, but **mixed format** — see
  §5 for the parsing gotcha and a caveat about what "the date" actually
  means for edited reviews. `reviewDateRelative` is a derived, redundant
  human string (`"3 years ago"`) and must not be parsed/used for time-series
  aggregation; it does not carry the review's exact date on its own, only
  what `reviewDate` already encodes.

## 3. Location key determination

**Verdict: `site` is the correct, collision-free location key.** Verified
programmatically, not assumed:

- 25 unique `site` values, 25 unique `address` values, and the mapping is
  exactly 1:1 in both directions — zero `site`s span more than one
  `address`, zero `address`es are shared by more than one `site`.
- `businessName` is constant (`"Big Dan's Car Wash"`) across all 6,209 rows
  — it identifies the chain, not a location, despite superficially looking
  like a business-identity column. **Do not use `businessName` for location
  filtering/grouping.**
- `city` is **not** sufficient alone: 3 cities host multiple distinct
  sites — Bradenton (`Big Dan's Bradenton 14th St`, `Big Dan's Bradenton
  301`, and `Big Dan's Lakewood Ranch`, whose `city` is also `Bradenton`),
  Woodstock (`Big Dan's Woodstock`, `Big Dan's Woodstock 2`), and Orlando
  (`Big Dan's John Young Pkwy`, `Big Dan's OBT`). A city-level rollup will
  silently merge distinct physical locations unless it's built as a
  secondary grouping on top of `site`, not a replacement for it.
- `placeId` (Google's own unique location identifier) confirms the `site`
  grouping wherever it's present: every non-null `placeId` maps 1:1 with
  exactly one `site` and vice versa. It's null for exactly 5 of 25 sites.
- `category` is stable per site (checked: no site has more than one
  distinct `category` value), so it's safe as a location attribute, not
  just a review attribute.

**Data quality / structural finding — 5 sites are "thin" records:**
`Big Dan's Academy`, `Big Dan's Fountain Inn`, `Big Dan's Kissimmee OBT`,
`Big Dan's Muscle Shoals`, and `Big Dan's St Pete` have **null** `placeId`,
`businessAvgRating`, `businessReviewCount`, `isLocalGuide`,
`reviewerReviewCount`, `likesCount`, and `reviewUrl` — same 377 rows null
across every one of those columns. This is not random missingness: **100%**
of these 5 sites' rows come from `dataSource` ∈ {`own_crawler_api`,
`own_crawler_crawler`}, a different collection pipeline than the other 20
sites (which are ≈98% `v1_v2_scrape`). Their `address` field is also
formatted differently — just a street name (`"2375 E Irlo Bronson Memorial
Hwy"`) rather than the `"street, city, state zip"` format used by the other
20 sites (`"3150 Blue Springs Rd, Kennesaw, GA 30144"`) — and their `state`
value is an abbreviation rather than a full name (see below). This is a
genuine gap in what the own_crawler pipeline captures, not something to
paper over with a fabricated placeId.

**`state` formatting is inconsistent and must be normalized.** Raw value
counts: `Florida` 3421, `Georgia` 2276, `FL` 294, `South Carolina` 135, `SC`
66, `AL` 17. The abbreviation vs. full-name split correlates exactly with
the 5-site own_crawler group above (those 5 sites use abbreviations; the
other 20 use full names) — same root cause, not a second independent issue.
`data_loader.py` normalizes via an explicit, observed-only map
(`STATE_ABBR_TO_FULL = {"FL": "Florida", "GA": "Georgia", "SC": "South
Carolina", "AL": "Alabama"}`) rather than a general US-state lookup table,
since only these 4 abbreviations actually appear in the data.

**Decision required (flagging, not resolving): should `Big Dan's
Lakewood Ranch` be treated as part of "Bradenton" for city-level rollups?**
Its `address` puts it in Bradenton, FL, but it's a physically distinct car
wash from the other two Bradenton sites. `data_loader.get_locations()`
keeps all three as separate rows (grouped by `site`, the defensible
default); if the app wants a "3 locations in Bradenton" city-level tile,
that's a legitimate aggregation on top of `site`, not evidence that
`city` should replace `site` as the key.

## 4. Duplicate detection & dedup rule

- **Zero fully-duplicate rows** (all 24 columns identical).
- **Zero genuine duplicate `reviewId` values** among the 6,084 non-null
  reviewIds (6,084 unique == 6,084 non-null count).
- **Correction to an earlier team note:** `df['reviewId'].duplicated().sum()`
  run naively (without excluding nulls) reports **124**, and it would be
  easy to read that as "124 duplicate reviews requiring dedup." That number
  is a false positive: pandas' `.duplicated()` treats `NaN == NaN` as a
  match, so with 125 null-`reviewId` rows, 124 of them (all but the first)
  get flagged as "duplicates" of each other purely because they're all
  missing the same field — they are 125 distinct, real reviews (verified:
  their `reviewerName`+`reviewDate`+`reviewText` combinations are almost
  all unique). **Do not run `drop_duplicates(subset='reviewId')` on the raw
  column** — it would silently delete 124 legitimate rows. `data_loader.py`
  restricts reviewId-based dedup to non-null values only.
- **One genuine cross-pipeline duplicate found** via a natural key (`site`
  + `reviewerName` + `reviewDate` + `rating` + `reviewText`): a 5-star
  review by "Debi Madden" at `Big Dan's Fountain Inn` ("Quick and easy,
  always enough cleaning bay available.", 2026-07-08) appears twice —
  once via `own_crawler_api` (null `reviewId`) and once via
  `own_crawler_crawler` (real `reviewId`). Same reviewer, site, date,
  rating, and exact text — the same physical review, captured by two
  overlapping collection runs.
- **Dedup rule implemented in `data_loader.load_data()`:** (1) drop
  reviewId duplicates among non-null reviewId rows only; (2) drop natural-key
  duplicates (site, reviewerName, reviewDate, rating, reviewText),
  preferring to keep the row that has a non-null `reviewId` when a
  collision occurs; (3) **cross-scrape duplicates** (added 2026-08-04 after
  the app rendered visibly repeated reviews): the natural key includes
  `reviewDate`, so it only catches copies sharing a timestamp, and the
  own_crawler_api / own_crawler_crawler runs stamped the same reviews a day
  apart. Pass (3) is date-blind for rows **with text** — identical (site,
  reviewerName, rating, reviewText) is one review, since Google permits one
  review per person per place — and for **rating-only** rows collapses
  (site, reviewerName, rating) only within a 7-day window (38 of the 43
  candidate pairs are 0–3 days apart; the 5 pairs months apart are left
  alone rather than guessed at). Net effect: 6,209 → **6,087** rows
  (1 + 81 + 40 removed).

## 5. Date parsing details

- `reviewDate` mixes two ISO-8601 shapes: full timestamps with milliseconds
  (`"2026-07-17T15:45:58.308Z"`, 5,712 rows, 100% from `v1_v2_scrape`) and
  date-only strings (`"2026-07-17"`, 497 rows, 100% from `own_crawler_*`).
  Calling `pd.to_datetime(col, utc=True, errors="coerce")` **without** an
  explicit format silently turns every date-only row into `NaT` (pandas'
  format inference locks onto the first shape it sees). Fix, used
  throughout `data_loader.py`: pass `format="mixed"` (equivalently
  `"ISO8601"` here), which parses both shapes with **zero** failures —
  confirmed on the full 6,209-row file.
- `ownerResponseDate` parses cleanly with the same `format="mixed"` call,
  full timestamps only, zero failures among its 5,512 non-null values.
- **Caveat for trend/time-series work:** `reviewDate` reflects Google's
  displayed date, which for an edited review is the **edit** time, not the
  original post time. Evidence: 512 rows have `ownerResponseDate` earlier
  than `reviewDate` (which should be structurally impossible if
  `reviewDate` were always the original post date — an owner can't reply
  before the review exists). Of those 512 rows, 502 (98%) have
  `reviewDateRelative` starting with `"Edited ..."` (e.g. `"Edited a year
  ago"`), confirming the reviewer edited their review after posting and
  Google's displayed/scraped date moved to reflect the edit. Gaps run from
  minutes up to **1,584 days**. Recommendation: treat `reviewDate` as the
  best available single "effective date" per review (it's what's
  consistently populated and orderable) for month/quarter-level trend
  charts, but don't present day-level precision on older reviews as
  necessarily the original post day. This is a characteristic of the
  source data (Google's own display behavior), not a scraping bug, and
  isn't something to "fix" — just something chart captions/tooltips should
  not overclaim.

## 6. Owner response availability (for a future reply-rate metric)

`ownerResponseText`/`ownerResponseDate` are null together on the same 697
rows (11.2%) — i.e. **88.8% of all reviews received an owner reply**. This
pairs cleanly (no row has one populated without the other) and is ready to
use for a "reply rate" KPI or per-site reply-rate breakdown; `get_locations()`
in `data_loader.py` already computes `pct_with_owner_response` per site as
a starting point. Per-site reply rates range from the high-80s to high-90s
in a quick spot check — no site appears to systematically skip replies, but
a full per-site breakdown is left to whichever agent builds that page.

## 7. Data loading module

Implemented at `app/utils/data_loader.py`. Public surface:

- `load_data(path=CSV_PATH) -> pd.DataFrame` — the canonical loader
  (`st.cache_data`-decorated when Streamlit is available, plain function
  otherwise). Applies every cleaning step in this document.
- `load_reviews(csv_path=CSV_PATH) -> pd.DataFrame` — alias for
  `load_data`, kept for interface compatibility.
- `get_filter_options(df) -> dict`, `filter_data(df, ...) -> pd.DataFrame`,
  `get_kpis(df) -> dict` — match the exact signatures specified in
  `docs/architecture.md` §3.
- `get_locations(df) -> pd.DataFrame` — one row per `site` with address/
  city/state/postalCode/placeId/category, `review_count`, `avg_rating`,
  `pct_5_star`, `pct_with_owner_response`, `first_review_date`,
  `last_review_date`.
- Column-name constants (`SITE_COL`, `RATING_COL`, `TEXT_COL`, `DATE_COL`,
  `LOCATION_COLS`, etc.) and a `FIELD_MAP` dict, so no downstream page needs
  to hardcode a raw column-name string.
- Self-test: `python3 app/utils/data_loader.py` loads the real CSV and
  prints row counts, dtypes, KPIs, and the top-5 locations table — this is
  how every number in this document was produced/verified.

## 8. Open items / decisions required (explicitly flagged, not resolved here)

1. **Bradenton grouping** (§3): whether `Big Dan's Lakewood Ranch` should
   ever be rolled into a "Bradenton" city-level view alongside the other
   two Bradenton sites, or always shown as its own location. No data
   answers this — it's a product/labeling decision.
2. **5 thin-data sites** (§3): whether the app should visually flag
   Academy/Fountain Inn/Kissimmee OBT/Muscle Shoals/St Pete as missing
   Google-native metrics (avg rating, place link, etc.), since those fields
   are structurally absent for them rather than just sparse.
3. **Edited-review dates** (§5): whether trend charts should carry a
   caption/tooltip caveat about `reviewDate` reflecting edit time for the
   ~8% of reviews that were edited, or whether this level of precision is
   out of scope for the MVP.

## 9. Scrape artefacts that distort any time series (added 2026-08-05)

Three findings from an independent audit of the loaded panel. None is a bug in
the code; all three change how the charts should be read.

- **`own_crawler_crawler` dates are scrape-batch dates, not review dates.** Of
  its 364 rows, 123 carry `2026-06-22`, 80 carry `2026-04-23` and 77 carry
  `2026-05-23` — three collection runs, 77% of that source. 314 loaded rows sit
  on one of those three days. The five own_crawler sites (St Pete 73, Academy
  66, Kissimmee OBT 61, Fountain Inn 53, Muscle Shoals) therefore have no
  meaningful day- or month-level history. This is separate from the edited-review
  caveat in note 8.
- **The monthly volume "growth" is coverage, not demand.** Sites reporting per
  month: 13 in Mar-2026 → 24 in Apr-2026 → 25 in Jun. Volume 193 → 500 (+159%)
  at that seam, because **12 of 25 sites have zero rows before 2026-04** and
  Academy none before 2026-06. Per-site capture ranges from 7.3% (Bradenton 301)
  to ~100% (Rome, Woodstock), so raw counts compare scrape depth as much as
  anything else.
- **Owner response rate is right-censored.** Monthly: Feb-26 100.0%, Mar 99.0%,
  Apr 83.0%, May 82.5%, Jun 69.2%, Jul 53.9%. The last review is 2026-07-22 and
  the last owner reply 2026-07-21 — recent reviews simply have not been answered
  yet, so the current month always looks worst.
