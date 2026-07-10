# Membership Sustainability — Plain-Language Guide

A top-to-bottom walkthrough of `membership_sustainability.ipynb`. Each section says **what the chart shows**, **how to read it**, and **an example / scenario**. No stats background needed.

---

## The 3 questions we're answering
A **new operator** opening a car-wash site wants to know:
1. **How much revenue should I aim for?**
2. **What % of it should come from membership?**
3. **If I can't get many members, can lots of walk-in (retail) washes keep me afloat instead?**

We answer using monthly data for **1,679 sites / 616 operators**, Jan-2024 → Mar-2026.

> **One honest caveat up front:** the data has **no profit/cost numbers** and **no customer IDs**. So "sustainable" means **revenue that is steady and not shrinking** (a stand-in for "healthy"), and "returning customers" is measured as **retail wash volume** (washes, not unique people).

---

## Step 1–2 · Get the data clean
Each row in the file is one *package* (e.g. "Premium Monthly") at a site in a month. We roll those up to **one row per site per month**, using the pre-built `membership_sales_total` and `membership_pct_sales_total` columns (so retail isn't double-counted across packages).

**Key metrics per site:**
- **Membership %** = membership revenue ÷ total revenue
- **Total revenue** = membership + retail
- **Retail washes** = walk-in wash count (our "returning retail" proxy)

*Example:* 360 Car Wash, Mar-2024 = $8,795 membership + $52,360 retail → membership % = **14%**.

---

## Step 3 · Membership % by state — two charts

### Chart 3a — small multiples (25 states)
**What:** one mini-panel per state, showing membership % over time. **Blue = that state**, **grey = national average**, **red dotted = the 40% line**.

**How to read:** in each panel, is blue above or below grey? Above the red line or below?

**Example:** Kentucky climbs from ~35% to ~65% over two years (members growing). Florida dips below 40% in mid-2024, then recovers.

**Takeaway:** big established states cluster in the low-50s% and sit above 40%. Thin states (few sites) are noisier — read them with caution.

### Chart 3b — dumbbell (multi-state operators)
**What:** one row per operator that runs in many states. Each dot = one state's membership %; the bar = its full range. **Red label = weakest state, green = strongest.**

**How to read:** a long bar means the *same brand* performs very differently depending on the state.

**Example:** Magnolia gets **71% membership in Tennessee but only 40% in Ohio**. Mach 1 gets **58% in Ohio but 29% in New Jersey**.

**Takeaway / scenario:** membership is **not just about how well you run the site** — the local market sets a big part of the ceiling. A new operator should benchmark its membership goal **against its state**, not a national number.

---

## Step 4 · What does "sustainable" mean? (the foundation)

We label each site **sustainable** if its revenue is **smooth** *and* **not shrinking** — measured carefully so it's fair:
- **Smooth** = revenue stays close to its own trend line (not jumping around). A site that *grows steadily* counts as smooth — growth is good, not "instability."
- **Not shrinking** = revenue in the last 2 quarters is within −5% of the first 2 quarters (a small dip is forgiven).

**Result: 38 out of every 100 sites are "sustainable."** That **38% is the baseline** — the bar every target has to beat.

### The "Why 38%" scatter plot
**What:** every site is a dot. **X = how bumpy** (left = smooth). **Y = growing or shrinking** (up = growing). **Green = sustainable**, sits in the upper-left.

**Example:** A site doing $80k, $82k, $79k every month → far left (smooth) → green. A site doing $40k, $120k, $30k → far right (bumpy) → red. A site smoothly growing $50k→$150k → **also green** (smooth around its rising line — the new rule no longer punishes it for growing).

> **Why this matters:** every later chart asks *"what % of sites at this level are green?"* and compares it to **38%**. Beating 38% = better-than-average odds.

---

## Step 4 (cont.) · The three answers — one figure, three bars

Each panel groups **all** sites by a lever and shows **what % are sustainable** in each group, vs the 38% baseline line.

### Q1 — Target revenue
**Reads:** below ~$38k/mo, only ~24% of sites are healthy. Once a site clears **~$76k/mo (~$0.9M/yr)**, odds jump to **42–59%**. Sweet spot: **$85k–166k/mo**.
**Plain version:** *you have to be big enough first.* A tiny site rarely stabilizes no matter the mix.
**Scenario:** a new site stuck at $40k/mo → fix scale before worrying about membership %.

### Q2 — Target membership %
**Reads:** under ~30% membership, odds are low (9–27%). The **40–60% band is the sweet spot (~46%)**. Above 60%, odds fade back to ~35%.
**Plain version:** aim for the **40–60% band**. More membership is **not** always better — past 60% it stops helping (you're giving up retail revenue without gaining stability).
**Scenario:** an operator at 25% membership is in the danger zone; one at 50% is in the sweet spot; one at 75% is fine but no better than average.

### Scenario B — The retail fallback
**Reads:** for *low-membership* sites, adding retail washes helps — odds rise from ~12% to a peak of **~32%** around ~2,800 washes/mo — but **never reach the 38% baseline**.
**Plain version:** heavy walk-in volume **cushions** weak membership but **can't replace** it.
**Scenario:** a low-membership site doing 5,000+ steady washes/mo is *better off than a quiet low-membership site*, but still less dependable than a site with balanced membership.

---

## The rule (the headline)

> **A new site is on track to be sustainable if it does ≥ ~$76k/month with membership in the ~40–60% band.**
> **If membership stays low,** ~2,800+ steady retail washes/month lifts the odds toward average — but membership remains the stronger lever.

---

## Step 4b · Same story at the operator level
A quick cross-check: a typical **sustainable operator** runs at a **higher revenue and higher membership** than a non-sustainable one — confirming the per-site rule holds when you zoom out to whole brands. (Most operators are single-site, so operator ≈ site here.)

---

## Step 4c (Scenario B+) · "But my region is just retail-heavy" — two charts

### Heatmap — retail share by state over time
**What:** every state is a row, colour = retail share of revenue. **Red = retail-led, green = membership-led.** Columns = quarters.

**How to read:** a row that **stays the same colour left-to-right** means that market's character **doesn't change** — it's structural.

**Example:** New York is red the whole time (~75–85% retail every quarter); West Virginia is green throughout. (Tiny-n states are noisy — trust the big-n rows.)

**Takeaway:** retail-heavy markets stay retail-heavy. It's a fixed condition to plan around, not a passing phase.

### Bar chart — sustainability by membership tier
**What:** % of sites that are healthy, split into retail-led (<40%), balanced (40–60%), membership-led (>60%). Dotted line = 38% average.

**Reads:** retail-led **30%** | balanced **46%** | membership-led **35%**.

**Scenario / so-what:** if you're in a structurally retail-led market, **don't chase a membership number the market won't give you.** Instead:
1. Get to **scale** (~$76k/mo),
2. Push **steady, high retail volume**,
3. Capture whatever membership you can toward the 40–60% band.

---

## Step 5 · Tableau export
The notebook writes `membership_sustainability_tableau.csv` — a tidy table with **Region / State / Year / Month** and a **Monthly-recurring vs Annual-recurring vs Retail** revenue split, ready to build the side-by-side operator dashboards.

---

## Bottom line in one breath
- **Be big enough** (~$76k/mo+).
- **Aim for 40–60% membership** (not more, not less).
- **Retail volume helps but never fully replaces membership.**
- **Benchmark to your state** — geography sets much of the membership ceiling.

## Two things to keep honest
1. **"Sustainable" = steady, non-declining revenue**, not measured profit (no cost data). Add a P&L feed later and these become true profit thresholds.
2. These are **patterns ("sites at this level tended to be healthier")**, not proof that hitting a number *causes* health. Strong as a benchmark to underwrite against; not a guarantee.
