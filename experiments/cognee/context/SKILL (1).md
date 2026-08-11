# SKILL.md — Car Wash POS Schema

Quick reference for writing SQL against the multi-tenant car wash POS DB (MySQL 5.7). Load this before writing any query touching `trans`, `trans_item`, `washbook*`, `sales_device`, `item`, `vehicle`, or `customer`.

Use this to skip the boilerplate. Explore the schema for anything not covered here.

---

## 1. Composite keys - always join on both columns

Almost every table uses `(id, sales_device_id)` as PK. FKs reference other tables via `(<other>_id, <other>_sales_device_id)`.

**Rule:** join on both. Never join on `id` alone.

```sql
-- ✅ correct
JOIN trans_item ti
    ON (ti.trans_id, ti.trans_sales_device_id) = (t.id, t.sales_device_id)

-- ❌ wrong — silently returns bad data
JOIN trans_item ti ON ti.trans_id = t.id
```

**Naming quirk:**

- Own device column: `sales_device_id`
- FK to another table's device: `<target>_sales_device_id`

So `customer.sales_device_id` vs `washbook_account.customer_sales_device_id`.

---

## 2. Magic numbers cheat sheet

Apply these unless you have a reason not to.

| Column | Value | Meaning |
| --- | --- | --- |
| `trans.trans_state_id` | `3` | Completed (only valid state) |
| `trans.trans_type_id` | `1, 5, 7, 8` | Valid types (5 = refund) |
| `trans.trans_type_id` | `5` | Refund — apply `IF(...=5, -1, 1)` sign flip |
| `trans_item.trans_item_state_id` | `1, 5, 8, 6` | Valid line items |
| `trans_item.item_department_category_id` | `1` | Wash line item |
| `item.item_department_id` | `1` | Car wash item |
| `item.item_department_id` | `13` | Subscription payment item (signup + renewal) |
| `item.item_department_id` | `5` | System / excluded from discount join |
| `washbook_item.washbook_item_type_id` | `1` | Prepaid book (NOT a real membership) |
| `washbook_item.washbook_item_type_id` | `!= 1` | Actual membership |
| `washbook_balance_history.washbook_balance_history_type_id` | `2, 3` | Wash redemption event |
| `trans_tender.tender_type_id` | `12` | Prepaid wash tender |
| `trans_tender.sub_tender_type_id` | `6, 7, 8, 9, 20` | Membership / washbook tenders |
| `sales_device.id` | `999` | System device — exclude from site resolution |
| `site.site_type_id` | `1` | Real site (not test) |
| `site.excluded` | `NULL or 0` | Site is active |
| `item.is_system` | `FALSE` | Real item (not system-generated) |
| `item_site.active` | `1` | Current active price row |

**Standard date bound:** `t.complete_date >= '2020-01-01'`. Use `complete_date` (finished), not `created_date` (started).

**Standard transaction filter (use this at minimum):**

```sql
WHERE t.trans_state_id = 3
  AND t.trans_type_id IN (1, 5, 7, 8)
  AND ti.trans_item_state_id IN (1, 5, 8, 6)
  AND t.complete_date >= '2020-01-01'
```

---

## 3. Refund handling

Refunds are `trans_type_id = 5`. Standard sign-flip pattern for counts and amounts:

```sql
SUM(IF(t.trans_type_id = 5, -1, 1) * ABS(ti.quantity)) AS net_count
SUM(IF(t.trans_type_id = 5, -1, 1) * ABS(ti.amount))   AS net_amount
```

Always `ABS()` the value before flipping — refund rows may already have negative amounts stored.

---

## 4. Grain conventions (for pipeline metric queries)

Metric queries return `(site_id, <grain>, metric)` — 3 columns, nothing else. Metadata joined downstream.

**Monthly:**

```sql
DATE_FORMAT(t.complete_date, '%Y-%m-01') AS month
GROUP BY sd.site_id, DATE_FORMAT(t.complete_date, '%Y-%m-01')
```

**Hourly:**

```sql
DATE_FORMAT(t.complete_date, '%Y-%m-%d %H:00:00') AS hour
GROUP BY sd.site_id, DATE_FORMAT(t.complete_date, '%Y-%m-%d %H:00:00')
```

Same logic, only the format string and column name change.

---

## 5. Common flag logic (member vs retail)

For classifying whether a wash is a member wash or retail wash:

```sql
-- Wash?
ti.item_department_category_id = 1 AS isWash

-- Member wash via prepaid tender?
IFNULL(pwtt.trans_id, 0) > 0 AS isPrepaidWashTender
-- (pwtt = trans_tender filtered to tender_type_id = 12)

-- Member wash via recurring subscription redemption?
COUNT(wbr.id) > 0 AS isRecurringRedemption
-- (wbr = washbook_recurring joined through washbook_balance_history type 2/3)
```

**Member wash** = `isWash AND (isRecurringRedemption OR isPrepaidWashTender)` — use **OR**, not AND.

**Retail wash** = `isWash AND NOT isRecurringRedemption AND NOT isPrepaidWashTender`.

Filter mismatches between OR and AND across queries are a known source of 10x discrepancies. Be deliberate.

---

## 6. Payment history: signups vs renewals

**Critical:** `washbook_billing_history` contains ONLY renewals, not signups.

- **Signup payment** = `trans_item` with `item.item_department_id = 13`, on the same day as `washbook_account.created_date`, matched via `(customer_id, customer_sales_device_id)`
- **Renewal payment** = via `washbook_billing_history` → `trans` → `trans_item`

For complete payment history, `UNION ALL` both. Tag rows with `payment_type = 'signup' | 'renewal'`.

Signup match is fuzzy (customer + same day) — may duplicate in rare cases where a customer has multiple accounts created same day.

---

## 7. Net-of-discount amounts

For revenue queries that need discount-adjusted amounts, use the stored proc:

```sql
ROUND(reports_getTransItemNetAmount(ti.quantity, ti.amount, discounts.discount_total_amount), 2)
```

Where `discounts` is a subquery aggregating `trans_promo_distribution.amount` per trans_item:

```sql
LEFT JOIN (
    SELECT ti.id, ti.sales_device_id,
           SUM(ROUND(tpd.amount, 2)) AS discount_total_amount
    FROM trans_item ti
    JOIN trans t ON (ti.trans_id, ti.trans_sales_device_id) = (t.id, t.sales_device_id)
    JOIN item i ON (i.id, i.sales_device_id) = (ti.item_id, ti.item_sales_device_id)
    LEFT JOIN trans_promo_distribution tpd
        ON i.item_department_id != 5
        AND (tpd.trans_item_id, tpd.trans_item_sales_device_id) = (ti.id, ti.sales_device_id)
    WHERE ti.trans_item_state_id IN (1, 5, 8, 6)
      AND t.trans_state_id = 3
      AND t.trans_type_id IN (1, 5, 7, 8)
      AND t.complete_date >= '<date>'
    GROUP BY ti.id, ti.sales_device_id
) discounts ON (discounts.id, discounts.sales_device_id) = (ti.id, ti.sales_device_id)
```

Skip this for simple counts. Only needed when cent-accuracy on revenue matters.

---

## 8. Reusable join snippets

**Site of transaction:**

```sql
JOIN sales_device sd ON sd.id = t.sales_device_id
-- sd.site_id
```

**Vehicle at wash time (from plate reader):**

```sql
LEFT JOIN vehicle v
    ON (v.id, v.sales_device_id) = (t.vehicle_id, t.sales_device_id)
```

May be NULL if plate not captured.

**Customer's registered vehicles (all of them):**

```sql
LEFT JOIN vehicle v
    ON (v.customer_id, v.customer_sales_device_id) = (c.id, c.sales_device_id)
```

Row-multiplies by number of vehicles per customer.

**Current package list price at a site:**

```sql
LEFT JOIN item_site ist
    ON (ist.item_id, ist.item_sales_device_id) = (i.id, i.sales_device_id)
    AND ist.site_id = <billing_site_id>
    AND ist.active = 1
-- ist.amount = current list price
```

**Full chain: transaction → customer → membership package → current price:**

```sql
JOIN trans t ...
JOIN trans_item ti ON (ti.trans_id, ti.trans_sales_device_id) = (t.id, t.sales_device_id)
JOIN item i ON (i.id, i.sales_device_id) = (ti.item_id, ti.item_sales_device_id)
JOIN washbook_item wbi ON (wbi.id, wbi.sales_device_id) = (i.id, i.sales_device_id)
JOIN washbook_account wba ON (wba.washbook_item_id, wba.washbook_item_sales_device_id) = (wbi.id, wbi.sales_device_id)
JOIN customer c ON (c.id, c.sales_device_id) = (wba.customer_id, wba.customer_sales_device_id)
JOIN washbook_recurring wr ON (wr.washbook_account_id, wr.washbook_account_sales_device_id) = (wba.id, wba.sales_device_id)
```

---

## 9. Query template (starting point)

Adapt this scaffold for most reporting queries:

```sql
SELECT
    sd.site_id AS site_id,
    DATE_FORMAT(t.complete_date, '%Y-%m-01') AS month,
    <your aggregation> AS <metric_name>
FROM trans t
JOIN sales_device sd ON sd.id = t.sales_device_id
JOIN trans_item ti ON (ti.trans_id, ti.trans_sales_device_id) = (t.id, t.sales_device_id)
-- add other joins here
WHERE t.trans_state_id = 3
  AND t.trans_type_id IN (1, 5, 7, 8)
  AND ti.trans_item_state_id IN (1, 5, 8, 6)
  AND t.complete_date >= '2020-01-01'
  -- add metric-specific filters here
GROUP BY sd.site_id, DATE_FORMAT(t.complete_date, '%Y-%m-01')
ORDER BY sd.site_id, month;
```

---

## 10. When exploring the schema

For anything not in this file:

- **Composite keys apply everywhere.** Assume `(id, sales_device_id)` unless proven otherwise.
- **Test filter values on real data** before trusting them. `SELECT DISTINCT <col> FROM <table>` is your friend.
- **Compare cross-tenant** if a pattern seems off — some quirks are tenant-specific.
- **Ambiguous metric names lie.** Read the actual query logic, not the column name.
- **When a metric returns 0 or looks off**, first check whether `ti.amount` on those rows is literally 0 (subscription model) before debugging joins.

---