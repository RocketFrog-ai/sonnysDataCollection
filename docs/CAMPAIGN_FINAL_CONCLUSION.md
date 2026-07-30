# Do car wash campaigns actually work?

**Final conclusion.** Source: `proforma/notebooks/book_v4.ipynb`, Section 7.
Data: 162 sites, 184 campaigns, Dec 2022 – Jun 2026 (43 months).

📹 **[Watch the 1-minute explainer](figures/campaign_causality_explainer.mp4)**

---

## The short version

| | |
|---|---|
| **What the old method said** | +15.9% revenue after a campaign |
| **Was it seasonality?** | **No** — that explains −0.7 points |
| **Was it something else?** | **Yes** — new sites growing on their own |
| **The corrected answer** | **+7.1% revenue** over the three months after a campaign |
| **How sure?** | 95% confident it's between +0.6% and +13.6%. Real, but only just |

---

## 1. The problem with the old number

Every earlier section measured a campaign like this: *revenue after* minus *the same site's revenue before*. That gives **+15.9%**.

Tim's objection: **that's not a fair comparison.** If a campaign runs in May and revenue rises in June, and June is simply a good month for car washes, the campaign gets credit it didn't earn.

He's right that before/after can't tell the difference. So we tested it.

---

## 2. Is it seasonality? No.

**Test A — measure seasonality directly.**

![Seasonality by month](figures/final_1_seasonality.png)

Across every campaign-free month, the best month runs **+15%** above a site's own annual average and the worst **−12%**. Real, but nowhere near enough to manufacture a +16% lift. Campaign start dates are also spread across **all twelve months**, not bunched before the summer peak — so it averages out.

**Test B — delete seasonality and re-measure.**

| | Revenue lift, months +1 to +3 |
|---|---|
| As measured | +15.9% |
| With the calendar effect stripped out | +16.6% |
| **Seasonality's contribution** | **−0.7 points** |

**Test C — the control group.** For every campaign we found sites that ran **no** campaign, in the same region, at a similar age, living through the **exact same calendar months**.

![Event study, all campaigns](figures/final_2_event_study.png)

The orange line is "what would have happened anyway." **It doesn't move.** Those months are not naturally strong months. Seasonality is not the driver — all three tests agree.

---

## 3. But the test found a different problem

Look at the **left half** of that chart, before the campaign starts. The gap should be zero — nothing has happened yet. It isn't: the two groups are already **+5.9% apart**.

And a harder check fails too. Pretend the campaign happened **9 months earlier** than it did, at a time when nothing occurred. A clean method returns zero. This one returned **+18.6%**.

**The cause: site age.** 100 of 184 campaigns are run by sites **less than a year old** — newly opened, still filling up, growing every month regardless of marketing. The before/after baseline was crediting the campaign for the site simply maturing.

> Tim was pointing at the right flaw. He just named the wrong culprit — it isn't the calendar, it's the opening ramp.

---

## 4. The fix, and the real answer

Keep only campaigns at sites **old enough that the ramp is over**, then watch the bias drain away.

![Age threshold sweep](figures/final_3_age_sweep.png)

| Minimum site age | Bias (must be 0) | Effect | Campaigns left |
|---|---|---|---|
| Any age | +5.9% ❌ | +14.2% | 82 |
| 12+ months | +3.8% ❌ | +13.4% | 55 |
| **18+ months** | **+2.3% ✅** | **+7.1%** | **17** |
| 24+ months | +1.2% ✅ | +9.1% | 5 — too few |

The bias falls **smoothly** as the age bar rises. That's the proof the contamination really was the ramp: if it were seasonality or a market trend, filtering on site age wouldn't touch it.

**18 months is the cutoff** — the lowest bar that passes its own placebo while keeping enough campaigns to measure. (24 months keeps only 5; its +9.1% is noise.)

![Clean event study](figures/final_4_mature_event_study.png)

Pre-campaign months are now flat. This is the version that passes every check.

> ### **A campaign raises revenue by about +7% in the three months after it runs.**
> 95% confidence: +0.6% to +13.6%. Effect holds at +9.3% through month +6.

---

## 5. What else survives — and what doesn't

![What survives](figures/final_5_what_survives.png)

- ✅ **Revenue: +14.2%** (all campaigns) / **+7.1%** (clean subset)
- ✅ **Member washes: +6.1%**
- ✅ **Retail washes: +8.6%** — note this *flips sign*. The old method said −8.5%, because retail volume was falling market-wide anyway.
- ❌ **New member sign-ups: −0.8%, no effect.** The old method said +14.9%. **The entire "promo converts retail customers into members" story was the opening ramp.** The revenue gain is real; this explanation for it is not.

---

## 6. Are they stealing from neighbours?

![Neighbours vs far sites](figures/final_6_neighbours.png)

Neighbours within 20 km lose **−24.8%** of retail washes after a nearby campaign. But sites **100+ km away**, which cannot possibly be cannibalised, lose **−13.8%** over the same months.

**Only about half the decline is real stealing** (−12.7% attributable to proximity), and that on 76 noisy pairs. Neighbours' *total revenue* is unaffected (−0.5%).

---

## 7. Every objection we checked

| Objection | Answer |
|---|---|
| "It's just seasonality" | No — three independent tests, contributes −0.7 pts |
| "It's the site's own growth" | **Yes** — this was the real problem; fixed by the 18-month filter |
| "Different brands aren't comparable" | Tested: comparing only **within the same brand** gives +14.9% vs +14.2%. Barely moves |
| "Maybe the matching is doing the work" | With no matching at all: +15.8%. Same answer |
| "Is the campaign trigger even right?" | It fires on `cogs + expenses`, and COGS rises with volume (corr 0.87). Re-running on **fixed expenses only** gives +18.1%. Holds |
| "Only 17 campaigns?" | Yes — small. But the result still clears statistical significance, and every robustness cut agrees on direction |

---

## 8. Honest limitations

1. **17 campaigns** in the clean estimate. The direction is solid; the exact number isn't precise.
2. **The confidence interval nearly touches zero** (+0.6%). Don't plan as if +7% is guaranteed.
3. **Campaigns are inferred, not recorded.** We detect them from OPEX spikes, so some may be renovations or one-off costs rather than marketing.
4. **Persistence beyond 6 months is unproven** — too few mature sites have a full year of follow-up.
5. **Operators choose when to promote.** No amount of matching removes that choice. It's the one confound left.

### What would settle it

A **staggered rollout**: one operator runs the campaign at half their sites in a market and not the other half, same month. Identical in every way except the campaign. That would turn "about +7%" into a number worth planning against.

---

## Bottom line

**Campaigns work. Just less than we thought.**

The seasonality challenge was answered and the effect survived it. A second, bigger flaw was found and corrected. What's left is a **modest but real ~+7% revenue lift** — with the caveat that the sign-up mechanism everyone assumed was behind it doesn't hold up.
