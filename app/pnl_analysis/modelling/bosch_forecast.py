"""
Bosch prediction engine — a straight port of Rafal's proforma Excel formula (the "Input Form"
layout used by 111 of 119 Express-Exterior workbooks in
experiments/old-proforma-analysis/old-proforma-data/), NOT a learned model. Given the 10
site-selection factors + 4 demographic components + a traffic-count input, it reproduces the
sheet's Year 1-5 car-wash-volume estimate.

Formula chain (see experiments/bosch-prediction-api/agent.md §2 for the cell-by-cell derivation):
  1. cumulative_site_score       = sum of the 10 chosen factor weights (SITE_FACTOR_WEIGHTS)
  2. cumulative_demographic_score = mean of the 4 demographic scores (each vs. a fixed target)
  3. first_year / second_year / mature target scores, from (1) and (2)
  4. yearly wash volume = target-score-for-that-year * effective traffic that year * operating_days
     (operating_days is NOT a fixed constant -- a sweep of all 119 workbooks found 300 (88x), 280
     (24x), 330 (6x) and 310 (1x), so it's a per-site editable assumption, not baked-in math.
     Defaults to 300, the modal value; see agent.md §6.4)
  5. monthly = yearly / 12

The Year 1/3 divisor (85) and the 92/82/76 divisors for years 2/4/5 are reproduced exactly as
found in the sheet; do not "simplify" this ramp, it is unexplained but deliberate (agent.md §2e).

Validated against every "Input Form"-layout workbook with complete inputs (111 files) using each
one's own cached (Excel-computed) Year 1-5 outputs: 106/111 exact matches; the 5 misses were traced
to individual workbooks with hand-edited formula cells (e.g. a changed base-price target), not a
port bug -- see agent.md §8. That check also caught two wrong assumptions in the first draft, since
fixed:
  * year{3,4,5}_growth_pct are ADDITIVE rates (0.02 = +2%, -0.01 = -1%, 0 = flat), matching the
    sheet's raw `(1 + rate)` formula -- NOT multipliers (1.02 = +2%) as first assumed.
  * each year's effective traffic is base_traffic * (1 + that year's OWN rate) -- years do NOT
    compound off each other, despite the (unused-here) legacy Sheet1 layout doing exactly that.
"""
from __future__ import annotations

from typing import Dict

# ─────────────────────── site-selection factors (§2a) ───────────────────────
# option key -> site-score weight, one dict per factor. Weights verified against the Excel formulas
# (agent.md §2a); factor/option KEY STRINGS match the front end's FACTORS config verbatim (the
# `name`/`value` fields, snake_cased) -- these are the literal wire values, not just labels, so they
# must match exactly or every real request 422s.
SITE_FACTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "area_profile": {
        "shopping": 0.15, "business": 0.1, "residential": 0.05, "industrial": -0.25,
    },
    "nearest_competition": {
        "one_in_4_miles": 0.15, "multiple_in_4_miles": 0.125, "one_in_2_miles": 0.075, "multiple_in_2_miles": -0.025,
    },
    "weekly_hours_category": {
        "more_than_70": 0.15, "65_to_70": 0.1, "60_to_64": 0.025, "less_than_60": 0.0,
    },
    "type_of_site": {
        "corner_lot_with_light": 0.15, "corner_lot_without_light": 0.125, "inside_lot_near_light": 0.075, "inside_lot_no_light": 0.05,
    },
    "site_accessibility": {
        "easy_in_easy_out": 0.15, "easy_in_out_divided_highway": 0.1, "easy_in_or_out_one_way": 0.05, "difficult_in_and_out": 0.0,
    },
    "entrance_stack_up_area": {
        "more_than_20_vehicles": 0.15, "20_to_15_vehicles": 0.125, "14_to_10_vehicles": 0.075, "less_than_10_vehicles": 0.05,
    },
    "number_of_free_vacuum_slots": {
        "more_than_20": 0.15, "12_to_20": 0.1, "less_than_12": 0.05, "coin_or_none": -0.25,
    },
    "number_of_pay_stations": {
        "3_or_more": 0.15, "2": 0.1, "1": 0.05, "live_person": 0.0,
    },
    "visibility": {
        "more_than_500_ft": 0.15, "400_to_500_ft": 0.1, "300_to_400_ft": 0.05, "less_than_300_ft": 0.0,
    },
    "traffic_speed": {
        "less_than_30_mph": 0.15, "30_to_40_mph": 0.1, "40_to_50_mph": 0.05, "more_than_50_mph": 0.0,
    },
}

SITE_FACTOR_KEYS = tuple(SITE_FACTOR_WEIGHTS.keys())


def cumulative_site_score(selections: Dict[str, str]) -> float:
    """Sum of the 10 chosen factor weights. `selections` maps each of SITE_FACTOR_KEYS to one of
    its option keys (validated by the request schema, not re-validated here)."""
    return sum(SITE_FACTOR_WEIGHTS[factor][selections[factor]] for factor in SITE_FACTOR_KEYS)


# ─────────────────────── demographic components (§2b) ───────────────────────

def _score_avg_household_size(value: float) -> float:
    if value < 2.101:
        return -0.1 * (2.1 - value)
    if value > 3:
        return -0.5 * (value - 3)
    return 0.0


def _score_pct_pop_25_65(value: float) -> float:
    return -0.3 * (0.55 - value)


def _score_pct_hh_income_over_35k(value: float) -> float:
    return -0.75 * (0.5 - value)


def _score_base_price_carwash(value: float) -> float:
    if value < 5.01:
        return 0.5 * (5 - value)
    return -0.075 * (value - 5)


def cumulative_demographic_score(avg_household_size: float, pct_pop_25_65: float,
                                  pct_hh_income_over_35k: float, base_price_carwash: float) -> float:
    """Mean of the 4 demographic scores, each vs. a fixed target (§2b). `pct_*` are fractions
    (0.55 = 55%), matching the Input Form sheet's own scale -- not percentages."""
    scores = (
        _score_avg_household_size(avg_household_size),
        _score_pct_pop_25_65(pct_pop_25_65),
        _score_pct_hh_income_over_35k(pct_hh_income_over_35k),
        _score_base_price_carwash(base_price_carwash),
    )
    return sum(scores) / 4


# ─────────────────────── target scores (§2c) ───────────────────────

def target_scores(site_score: float, demog_score: float) -> Dict[str, float]:
    growth = 1 + demog_score
    return {
        "first_year": (site_score * 0.7) * growth / 85,
        "second_year": site_score * growth / 92,
        "mature": site_score * growth / 76,
    }


# ─────────────────────── wash-volume forecast (§2d-2e) ───────────────────────

def bosch_forecast(
    site_factors: Dict[str, str],
    avg_household_size: float, pct_pop_25_65: float, pct_hh_income_over_35k: float, base_price_carwash: float,
    base_traffic: float, year3_growth_pct: float = 0.0, year4_growth_pct: float = 0.0, year5_growth_pct: float = 0.0,
    operating_days_per_year: float = 300.0,
) -> Dict[str, object]:
    """The full Bosch prediction: site factors + demographics + traffic -> Year 1-5 wash-volume
    estimate, yearly and monthly. `year{3,4,5}_growth_pct` are ADDITIVE rates (0.0 = flat,
    0.02 = +2% that year), each applied independently to `base_traffic` (years do NOT compound
    off each other -- verified against the sheet's own cached outputs, see module docstring).
    `operating_days_per_year` defaults to 300 (the modal value across 119 source workbooks; 280,
    330 and 310 all appear too -- it's a per-site assumption, not a universal constant)."""
    site_score = cumulative_site_score(site_factors)
    demog_score = cumulative_demographic_score(
        avg_household_size, pct_pop_25_65, pct_hh_income_over_35k, base_price_carwash,
    )
    targets = target_scores(site_score, demog_score)
    growth = 1 + demog_score

    # Each year's effective traffic is base_traffic * (1 + that year's OWN rate) -- verified against
    # the sheet's cached outputs: it is NOT compounded/chained across years (year4 does not build on
    # year3's effective traffic; each multiplies the raw base_traffic independently).
    effective_traffic_y3 = base_traffic * (1 + year3_growth_pct)
    effective_traffic_y4 = base_traffic * (1 + year4_growth_pct)
    effective_traffic_y5 = base_traffic * (1 + year5_growth_pct)

    yearly = {
        "year1": targets["first_year"] * base_traffic * operating_days_per_year,
        "year2": targets["second_year"] * base_traffic * operating_days_per_year,
        "year3": (site_score * growth) / 85 * effective_traffic_y3 * operating_days_per_year,
        "year4": (site_score * growth) / 82 * effective_traffic_y4 * operating_days_per_year,
        "year5": targets["mature"] * effective_traffic_y5 * operating_days_per_year,
    }
    monthly = {year: value / 12 for year, value in yearly.items()}

    return {
        "cumulative_site_score": site_score,
        "cumulative_demographic_score": demog_score,
        "target_scores": targets,
        "yearly": yearly,
        "monthly": monthly,
    }
