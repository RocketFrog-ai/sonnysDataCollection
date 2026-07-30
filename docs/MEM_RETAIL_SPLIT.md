# How the forecast splits Membership vs Retail washes

A plain-language explanation of how a dropped pin gets a **membership** wash forecast and a
**retail** wash forecast — not just a total.

Everything here lives in [`proforma/models/coldstart.py`](../proforma/models/coldstart.py).
Model 5 ("super ensemble") does **not** do the split — see the last section.

---

## The one-line summary

We predict **one total wash number**, then we predict **what fraction of it is membership**, and
then we let membership and retail **grow on their own separate curves** over the 5 years.

```
total washes  ──split by "share"──►  membership washes  + retail washes
                                     (each rides its own ramp over time)
```

---

## Step 1 — Predict the total washes

For the pin, the model first predicts the site's **mature total wash volume** (the plateau it
settles at). This is the number Model 5 sharpens using pay stations / vacuums / site type / traffic.

Call this `total`.

## Step 2 — Predict the membership *share*

Separately, we have a small trained model whose only job is to answer:

> *"At a mature site in this location, what fraction of washes are from unlimited members
> (vs. one-off retail customers)?"*

- **What it learned from:** every real mature site in our panel, using its actual
  `membership washes ÷ total washes`.
  ([coldstart.py:414](../proforma/models/coldstart.py#L414))
- **What it looks at to decide:** purely **location signals** — where the pin is, how many
  neighbours are nearby (5/10/20 km), how big the local cluster is, the brand, the region/state.
  It is a LightGBM model.
  ([coldstart.py:427](../proforma/models/coldstart.py#L427))
- **At pin-drop** it outputs a single number, e.g. `share = 0.60` → "60% of washes will be
  membership." We clamp it to a sane 5%–95% range, and if the location is too unusual to score we
  fall back to the **region's median** (or the global median).
  ([coldstart.py:556](../proforma/models/coldstart.py#L556))

So the split is **not** a fixed 50/50, and it's **not** based on price/ASP. It's a location-driven
prediction: dense, membership-heavy markets get a high share; thin retail markets get a low one.

## Step 3 — Split the total

```
membership washes (at maturity) = total ×  share
retail     washes (at maturity) = total × (1 − share)
```

## Step 4 — Let each side follow its own time curve

A car wash doesn't hit its mature numbers on day one. Memberships take time to build; retail is
closer to full speed immediately. So we don't just multiply by `share` and stop — each stream is
walked forward month-by-month on its **own empirical ramp curve**:

- **Membership ramp (`rm`)** — starts low and climbs over roughly the first ~7 months as people
  sign up for unlimited plans.
- **Retail ramp (`rr`)** — close to flat from month 1; retail customers show up right away.

Both curves are built from real site histories, pooled from the **local market first**
(cluster → region → global), so a pin inherits how sites *near it* actually ramped.
([coldstart.py:581-582](../proforma/models/coldstart.py#L581-L582))

There's also a small optional long-run drift after maturity: membership can keep drifting up a bit,
retail down a bit (saturating after ~2 years), if those knobs are turned on.

Putting it together, for each month `m` the forecast is:

```python
membership[m] = total × share       × rm[m] × (1 + mem_growth)^years_past_maturity
retail[m]     = total × (1 − share) × rr[m] × (1 + ret_change)^years_past_maturity
```

([coldstart.py:594-595](../proforma/models/coldstart.py#L594-L595))

That's the whole split. The forecast then reports `mem`, `ret`, and `total = mem + ret` for every
month, plus a low/median/high band.

---

## Where does Model 5 fit in?

Model 5 (the "super ensemble") **only improves Step 1 — the total level.** It takes the total the
base model produced and rescales it up or down using the site's capacity inputs (pay stations,
vacuum slots, lot type) and traffic.

Crucially, it rescales **membership, retail, and total by the exact same factor**, so the
membership/retail *proportion never changes*. Model 5 moves the whole trajectory up or down; the
split from Steps 2–4 rides through untouched.
([super_ensemble.py:140-146](../proforma/models/super_ensemble.py#L140-L146))

```
Model 5:  total ─── ×adj ──► bigger/smaller total   (mem, ret both ×adj → same share)
```

---

## What this does and doesn't use

| Question | Answer |
|---|---|
| Is the split a fixed ratio? | **No** — it's predicted per location. |
| Is it based on price / ASP? | **No** — ASP only affects *revenue* later in the P&L, not the wash-count split. |
| Is it a model output? | **Yes** — a LightGBM "share" model, plus two ramp curves for the timing. |
| Does it use the cluster/neighbours? | **Yes** — the share model's inputs and both ramp curves are pulled from the local market first. |
| Does Model 5 change the split? | **No** — Model 5 only rescales the total; the proportion is preserved. |

---

## The exact code, if you want to read it

- Membership **share** trained: [coldstart.py:414](../proforma/models/coldstart.py#L414) (target) &
  [:427](../proforma/models/coldstart.py#L427) (fit)
- Share predicted at pin-drop: [coldstart.py:556](../proforma/models/coldstart.py#L556)
- Region/global fallback shares: [coldstart.py:442](../proforma/models/coldstart.py#L442),
  [:451](../proforma/models/coldstart.py#L451)
- Ramp curves selected (mem vs ret): [coldstart.py:581-582](../proforma/models/coldstart.py#L581-L582)
- The month-by-month split: [coldstart.py:594-595](../proforma/models/coldstart.py#L594-L595)
- Model 5's proportion-preserving rescale:
  [super_ensemble.py:140-146](../proforma/models/super_ensemble.py#L140-L146)
</content>
</invoke>
