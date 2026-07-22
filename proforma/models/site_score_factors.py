"""
The client's proforma site-score sheet as a LEVEL ADJUSTMENT layer for Model 5 (SUPER).

The sheet scores 10 site factors (option → additive score, best ≈ +0.15, worst down to −0.25).
The old-proforma backtest (121 proformas vs actuals) showed only the capacity factors survive FDR —
pay stations, free vacuums, type of site — and those (plus traffic count) are ALREADY inputs of the
SUPER ridge (level_A), so they are NEVER double counted here. The seven remaining factors
(area profile, nearest competition, weekly hours, site accessibility, entrance stack-up, visibility,
traffic speed) have weak/no measured correlation with actual mature washes (r ≈ −0.08…0.21, none
survive FDR): this layer exists because the CLIENT'S sheet uses them, and it keeps them honest —
RELATIVE BUCKETING (each chosen option's score centered on its factor's option mean, so a
mid-bucket pick is ≈ neutral), summed and applied as a level multiplier clipped to
[MULT_LO, MULT_HI]. Every contribution is itemised in the output so the adjustment is auditable,
and the bounded multiplier keeps a stack of cosmetic factors from ever overwhelming the
calibrated level.

Shared by the Streamlit Model-5 sidebar and the /pinpoint-forecast API. Pure math, no I/O.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MULT_LO, MULT_HI = 0.75, 1.25   # the factor layer may move the level at most ±25%

# The client's sheet verbatim (name/value/label/score). `ridge_input=True` marks factors whose
# information already enters the SUPER ridge — they are accepted but SKIPPED (reported, not scored).
FACTORS: List[Dict[str, Any]] = [
    {"name": "areaProfile", "label": "Area Profile", "options": [
        {"value": "shopping", "label": "Shopping", "score": 0.15},
        {"value": "business", "label": "Business", "score": 0.10},
        {"value": "residential", "label": "Residential", "score": 0.05},
        {"value": "industrial", "label": "Industrial", "score": -0.25},
    ]},
    {"name": "nearestCompetition", "label": "Nearest Competition", "options": [
        {"value": "one_in_4_miles", "label": "One in 4 Miles", "score": 0.15},
        {"value": "multiple_in_4_miles", "label": "Multiple in 4 Miles", "score": 0.125},
        {"value": "one_in_2_miles", "label": "One in 2 Miles", "score": 0.075},
        {"value": "multiple_in_2_miles", "label": "Multiple in 2 Miles", "score": -0.025},
    ]},
    {"name": "weeklyHoursCategory", "label": "Weekly Hours of Operation", "options": [
        {"value": "more_than_70", "label": "More Than 70 Hours", "score": 0.15},
        {"value": "65_to_70", "label": "65 - 70 Hours", "score": 0.10},
        {"value": "60_to_64", "label": "60 - 64 Hours", "score": 0.025},
        {"value": "less_than_60", "label": "Less Than 60 Hours", "score": 0.0},
    ]},
    {"name": "typeOfSite", "label": "Type of Site", "ridge_input": True, "options": [
        {"value": "corner_lot_with_light", "label": "Corner Lot With Light", "score": 0.15},
        {"value": "corner_lot_without_light", "label": "Corner Lot Without Light", "score": 0.125},
        {"value": "inside_lot_near_light", "label": "Inside Lot Near Light", "score": 0.075},
        {"value": "inside_lot_no_light", "label": "Inside Lot No Light", "score": 0.05},
    ]},
    {"name": "siteAccessibility", "label": "Site Accessibility", "options": [
        {"value": "easy_in_easy_out", "label": "Easy In & Easy Out", "score": 0.15},
        {"value": "easy_in_out_divided_highway", "label": "Easy In/Out With Divided Highway", "score": 0.10},
        {"value": "easy_in_or_out_one_way", "label": "Easy In Or Easy Out One Way", "score": 0.05},
        {"value": "difficult_in_and_out", "label": "Difficult In and Out", "score": 0.0},
    ]},
    {"name": "entranceStackUpArea", "label": "Entrance Stack Up Area", "options": [
        {"value": "more_than_20_vehicles", "label": "More Than 20 Vehicles", "score": 0.15},
        {"value": "20_to_15_vehicles", "label": "20 - 15 Vehicles", "score": 0.125},
        {"value": "14_to_10_vehicles", "label": "14 - 10 Vehicles", "score": 0.075},
        {"value": "less_than_10_vehicles", "label": "Less Than 10 Vehicles", "score": 0.05},
    ]},
    {"name": "numberOfFreeVacuumSlots", "label": "Number of Free Vacuum Slots", "ridge_input": True, "options": [
        {"value": "more_than_20", "label": "More Than 20 Vehicles", "score": 0.15},
        {"value": "12_to_20", "label": "12 - 20 Vehicles", "score": 0.10},
        {"value": "less_than_12", "label": "Less Than 12 Vehicles", "score": 0.05},
        {"value": "coin_or_none", "label": "Coin or None", "score": -0.25},
    ]},
    {"name": "numberOfPayStations", "label": "Number of Pay Stations", "ridge_input": True, "options": [
        {"value": "3_or_more", "label": "3 or More", "score": 0.15},
        {"value": "2", "label": "2", "score": 0.10},
        {"value": "1", "label": "1", "score": 0.05},
        {"value": "live_person", "label": "Live Person", "score": 0.0},
    ]},
    {"name": "visibility", "label": "Visibility", "options": [
        {"value": "more_than_500_ft", "label": "More Than 500 Feet Both Directions", "score": 0.15},
        {"value": "400_to_500_ft", "label": "400 - 500 Feet Both Directions", "score": 0.10},
        {"value": "300_to_400_ft", "label": "300 - 400 Feet Both Directions", "score": 0.05},
        {"value": "less_than_300_ft", "label": "Less Than 300 Feet Both Directions", "score": 0.0},
    ]},
    {"name": "trafficSpeed", "label": "Traffic Speed", "options": [
        {"value": "less_than_30_mph", "label": "Less Than 30 MPH", "score": 0.15},
        {"value": "30_to_40_mph", "label": "30 - 40 MPH", "score": 0.10},
        {"value": "40_to_50_mph", "label": "40 - 50 MPH", "score": 0.05},
        {"value": "more_than_50_mph", "label": "More Than 50 MPH", "score": 0.0},
    ]},
]

_BY_NAME = {f["name"]: f for f in FACTORS}
# case-insensitive alias → canonical name (accept snake_case posts like "area_profile" too)
_NAME_ALIASES = {f["name"].lower(): f["name"] for f in FACTORS}
_NAME_ALIASES.update({
    "area_profile": "areaProfile", "nearest_competition": "nearestCompetition",
    "weekly_hours_category": "weeklyHoursCategory", "weekly_hours": "weeklyHoursCategory",
    "type_of_site": "typeOfSite", "site_accessibility": "siteAccessibility",
    "entrance_stack_up_area": "entranceStackUpArea", "entrance_stack_up": "entranceStackUpArea",
    "number_of_free_vacuum_slots": "numberOfFreeVacuumSlots",
    "number_of_pay_stations": "numberOfPayStations", "traffic_speed": "trafficSpeed",
})


def adjustment_factor_names() -> List[str]:
    """The factor names this layer actually scores (the sheet minus the ridge inputs)."""
    return [f["name"] for f in FACTORS if not f.get("ridge_input")]


def _match_option(f: Dict[str, Any], choice: str) -> Optional[Dict[str, Any]]:
    c = str(choice).strip().lower()
    for o in f["options"]:
        if c == o["value"].lower() or c == o["label"].lower():
            return o
    return None


def factor_adjustment(selections: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """The bounded level multiplier for a set of sheet selections ({factor: option value-or-label}).

    Relative bucketing: contribution = score(option) − mean(that factor's option scores), so picking
    a factor's middle bucket is ≈ neutral and only deviations from typical move the level. The
    summed delta becomes multiplier = clip(1 + Σdelta, 0.75, 1.25). Ridge-input factors (pay
    stations / vacuums / type of site) are accepted but skipped — their effect already lives in the
    SUPER ridge. Unknown factors/options are reported in `ignored`, never an error.
    Returns {multiplier, total_delta, n_used, contributions, skipped_ridge_inputs, ignored}."""
    contributions: List[Dict[str, Any]] = []
    skipped: List[str] = []
    ignored: List[str] = []
    total = 0.0
    for raw_name, choice in (selections or {}).items():
        name = _NAME_ALIASES.get(str(raw_name).strip().lower())
        f = _BY_NAME.get(name) if name else None
        if f is None:
            ignored.append(f"{raw_name} (unknown factor)")
            continue
        o = _match_option(f, choice)
        if o is None:
            ignored.append(f"{raw_name}={choice} (unknown option)")
            continue
        if f.get("ridge_input"):
            skipped.append(f"{f['name']} (already a SUPER ridge input — not double counted)")
            continue
        scores = [x["score"] for x in f["options"]]
        delta = o["score"] - sum(scores) / len(scores)
        ranked = sorted(f["options"], key=lambda x: -x["score"])
        contributions.append({
            "factor": f["name"], "label": f["label"],
            "option": o["value"], "option_label": o["label"],
            "score": o["score"], "rank": ranked.index(o) + 1, "of": len(ranked),
            "delta": round(delta, 4),
        })
        total += delta
    mult = min(max(1.0 + total, MULT_LO), MULT_HI)
    return {"multiplier": round(mult, 4), "total_delta": round(total, 4),
            "clipped": bool(1.0 + total != mult),
            "n_used": len(contributions), "contributions": contributions,
            "skipped_ridge_inputs": skipped, "ignored": ignored,
            "bounds": [MULT_LO, MULT_HI]}
