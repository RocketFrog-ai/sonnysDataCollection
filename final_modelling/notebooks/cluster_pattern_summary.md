# Do Nearby Car Washes Behave Alike? — Simple Summary

**Question:** Do car-wash sites that are close to each other show the same month-to-month
ups and downs in their KPIs? *(Monthly Recurring Plan only.)*

**Answer:** **Yes — location clearly matters, but it's not the whole story.**

---

## Methodology (how I did it, in plain terms)

1. **Found the real sites.** The `site_id` column is just a number-within-a-brand, so I
   used `client_id + site_id` as the true site key → **1,674 sites**, 27 months of data.

2. **Grouped sites by location.** Bundled together sites that sit within ~**20 km** of each
   other — small local neighborhoods, not whole cities. (Plain DBSCAN/HDBSCAN merged entire
   metros into 70–100-site blobs, so I used a method with a hard 20 km cap instead.)

3. **Measured if grouped sites move together.** For each KPI I lined up each site's monthly
   curve and computed a **correlation** (1.0 = perfectly in sync, 0 = unrelated).

4. **Compared against random — the key test.** High correlation alone proves nothing (if
   business booms everywhere, everyone looks similar). So I also measured correlation for
   **random sites from anywhere in the country**. If location matters, *neighbors must beat
   random strangers*.

5. **Pulled out the actual groups (Step 7).** Finally I extracted the real neighborhoods of
   look-alike sites: groups where **every member is within 20 km AND correlates ≥ 0.70**.
   Saved to `kpi_buffer_clusters.csv`.

---

## Results: neighbors vs random strangers

| KPI | Nearby sites (in sync) | Random sites | Extra from being neighbors | Real effect? |
|---|---|---|---|---|
| Retail washes | **0.62** | 0.24 | **+0.39** | ✅ strongest |
| Total washes | 0.58 | 0.24 | +0.34 | ✅ |
| Retail sales ($) | 0.52 | 0.19 | +0.33 | ✅ |
| Membership washes | 0.49 | 0.26 | +0.23 | ✅ |
| Memberships sold | 0.20 | 0.11 | +0.09 | ✅ weak |
| Membership sales ($) | 0.23 | 0.16 | +0.06 | ✅ weakest |

*Read row 1 as: "Nearby retail-wash numbers move together at 0.62, random sites only 0.24,
so being neighbors adds +0.39." All 6 KPIs beat random with near-zero chance of luck (p≈0.005).*

---

## Interpretation (what it means)

- **Location matters most for wash volume** (retail/total washes) — people in the same area
  react to the same weather, traffic, and local habits, so their wash counts rise and fall
  together.
- **Location barely matters for membership dollars** — price and plans are set by head
  office, not the local area, so they don't follow geography.
- **It's necessary but not sufficient.** Of the 1,321 site-pairs that sit within 20 km of
  each other, only about **half actually move together** (51% for total washes, 54% for
  retail washes at correlation ≥ 0.70) — the other half are close but behave differently.
  If location alone explained behavior this would be ~100%, not ~50%. To explain the rest
  you'd also need **weather, demographics, pricing, and store age**.

### Where are the most similar nearby sites?
**Florida leads by far** (56 clusters, 184 sites), then **California, Texas, Georgia** — the
Sun Belt, where car washes cluster thickest.

### Examples
- **Look-alikes (almost identical curves, r ≈ 0.95):** a 6-site **St. Louis, MO** group
  (r=0.96) and a 6-site **Orange County, CA** group (r=0.94) — the CA one even mixes **4
  different brands** within 10 km, all moving together. That's the cleanest proof it's the
  *location* doing it, not the operator.
- **Exceptions that break the rule:** a few nearby pairs moved *opposite* each other
  (correlation −0.18, −0.12) — reminders that being close doesn't guarantee similarity.

---

## Outputs
- `kpi_buffer_clusters.csv` — every group of similar nearby sites (members, correlation, radius, state)
- `kpi_buffer_clusters_map.html` — map of those groups
- `cluster_map.html` — the 20 km geographic neighborhoods

*Tune `BUFFER_KM` and `CORR_MIN` at the top of Step 7 to trade tightness for coverage.*
