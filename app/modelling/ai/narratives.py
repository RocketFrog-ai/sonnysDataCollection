"""Top-level narrative orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modelling.ai.common import load_input_payload
# Feature narrative modules
from app.modelling.ai.weather import (
    get_feature_narratives as get_weather_feature_narratives,
    get_insight as get_weather_insight,
    get_overall_narrative as get_weather_overall_narrative,
)
from app.modelling.ai.competition import (
    get_feature_narratives as get_competition_feature_narratives,
    get_insight as get_competition_insight,
    get_overall_narrative as get_competition_overall_narrative,
)
from app.modelling.ai.retail import (
    get_feature_narratives as get_retail_feature_narratives,
    get_insight as get_retail_insight,
    get_overall_narrative as get_retail_overall_narrative,
)
from app.modelling.ai.gas import (
    get_feature_narratives as get_gas_feature_narratives,
    get_insight as get_gas_insight,
    get_overall_narrative as get_gas_overall_narrative,
)

# V3 feature keys per dimension (for filtering narratives)
COMPETITION_V3_KEYS = frozenset({
    "competitors_count_4miles",
    "competitor_1_distance_miles",
    "competitor_1_google_rating",
    "competitor_1_rating_count",
    "competition_quality",
})

RETAIL_V3_KEYS = frozenset({
    "costco_enc",
    "distance_nearest_walmart(5 mile)",
    "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile",
    "count_food_joints_0_5miles (0.5 mile)",
})

GAS_V3_KEYS = frozenset({
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
})

# Weather feature keys used by the weather narrative module
WEATHER_V3_KEYS = frozenset({
    "weather_rainy_days",
    "weather_total_snowfall_cm",
    "weather_days_pleasant_temp",
    "weather_days_below_freezing",
})


def _strip_markdown(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.replace("**", "")
    elif isinstance(obj, dict):
        return {k: _strip_markdown(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_strip_markdown(i) for i in obj]
    return obj


def _compose_observation(overall: Optional[Dict[str, Any]]) -> Optional[str]:
    """Normalize overall payloads that may return observation or pro/con."""
    if not overall or not isinstance(overall, dict):
        return None
    observation = overall.get("observation")
    if observation:
        return observation
    pro = overall.get("pro")
    con = overall.get("con")
    if pro and con:
        return f"Pro: {pro} Con: {con}"
    return pro or con


def _extract_pro_con(overall: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Return explicit pro/con keys from either dedicated keys or observation text."""
    out: Dict[str, Optional[str]] = {"pro": None, "con": None}
    if not overall or not isinstance(overall, dict):
        return out
    pro = overall.get("pro")
    con = overall.get("con")
    if pro:
        out["pro"] = pro
    if con:
        out["con"] = con
    if out["pro"] or out["con"]:
        return out

    observation = overall.get("observation")
    if not isinstance(observation, str) or not observation.strip():
        return out
    text = observation.strip()
    lower = text.lower()
    pro_idx = lower.find("pro:")
    con_idx = lower.find("con:")
    if pro_idx != -1 and con_idx != -1:
        if pro_idx < con_idx:
            out["pro"] = text[pro_idx + 4:con_idx].strip()
            out["con"] = text[con_idx + 4:].strip()
        else:
            out["con"] = text[con_idx + 4:pro_idx].strip()
            out["pro"] = text[pro_idx + 4:].strip()
    return out


def get_feature_narratives(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build per-feature narrative entries from all enabled feature modules (weather, competition).
    """
    out: List[Dict[str, Any]] = []
    out.extend(get_weather_feature_narratives(quantile_result, feature_values))
    out.extend(get_competition_feature_narratives(quantile_result, feature_values))
    out.extend(get_retail_feature_narratives(quantile_result, feature_values))
    out.extend(get_gas_feature_narratives(quantile_result, feature_values))
    return _strip_markdown(out)


def get_overall_narrative(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
    feature_narratives: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build overall narrative (insight, observation, conclusion) for weather.
    Competition has its own insight/observation in narratives["competition"].
    """
    if feature_narratives is None:
        feature_narratives = get_feature_narratives(quantile_result, feature_values)
    # Weather overall narrative must only use the 4 weather metric narratives.
    weather_narratives = [n for n in feature_narratives if n.get("feature_key") in WEATHER_V3_KEYS]
    insight = get_weather_insight(quantile_result, weather_narratives)
    overall = get_weather_overall_narrative(
        quantile_result, weather_narratives
    )
    pro_con = _extract_pro_con(overall)
    return _strip_markdown({
        "insight": insight,
        "observation": _compose_observation(overall),
        "pro": pro_con.get("pro"),
        "con": pro_con.get("con"),
        "conclusion": overall.get("conclusion"),
    })


def get_competition_narrative(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    feature_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Competition-only insight and observation (used by competition route)."""
    comp_narratives = [n for n in feature_narratives if n.get("feature_key") in COMPETITION_V3_KEYS]
    insight = get_competition_insight(quantile_result, comp_narratives, feature_values=feature_values)
    overall = get_competition_overall_narrative(quantile_result, comp_narratives, feature_values=feature_values)
    pro_con = _extract_pro_con(overall)
    return _strip_markdown({
        "insight": insight,
        "observation": _compose_observation(overall),
        "pro": pro_con.get("pro"),
        "con": pro_con.get("con"),
    })


def get_retail_narrative(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    feature_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retail-only insight, observation, and conclusion (used by retail route)."""
    retail_narratives = [n for n in feature_narratives if n.get("feature_key") in RETAIL_V3_KEYS]
    insight = get_retail_insight(quantile_result, retail_narratives)
    overall = get_retail_overall_narrative(
        quantile_result,
        retail_narratives,
        feature_values=feature_values,
    )
    pro_con = _extract_pro_con(overall)
    return _strip_markdown({
        "insight": insight,
        "observation": _compose_observation(overall),
        "pro": pro_con.get("pro"),
        "con": pro_con.get("con"),
        "conclusion": overall.get("conclusion"),
    })


def get_gas_narrative(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Gas-only insight, observation, and conclusion (used by gas route)."""
    gas_narratives = [n for n in feature_narratives if n.get("feature_key") in GAS_V3_KEYS]
    insight = get_gas_insight(quantile_result, gas_narratives)
    overall = get_gas_overall_narrative(quantile_result, gas_narratives)
    pro_con = _extract_pro_con(overall)
    return _strip_markdown({
        "insight": insight,
        "observation": _compose_observation(overall),
        "pro": pro_con.get("pro"),
        "con": pro_con.get("con"),
        "conclusion": overall.get("conclusion"),
    })


if __name__ == "__main__":
    payload = load_input_payload(
        {
            "quantile_result": {
                "predicted_wash_quantile": 3,
                "predicted_wash_quantile_label": "Q3",
                "predicted_wash_range": {"label": "170-220 washes/day"},
                "feature_analysis": {},
            },
            "feature_values": {},
        }
    )
    qr = payload.get("quantile_result", {})
    fv = payload.get("feature_values", {})
    feature_narratives = get_feature_narratives(qr, fv)
    print("Feature narratives:", feature_narratives)
    print("Weather overall:", get_overall_narrative(qr, fv, feature_narratives))
    print("Competition overall:", get_competition_narrative(qr, feature_narratives, feature_values=fv))
    print("Retail overall:", get_retail_narrative(qr, feature_narratives, fv))
    print("Gas overall:", get_gas_narrative(qr, feature_narratives))
