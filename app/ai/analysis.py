import json
import re
from datetime import datetime
from pathlib import Path
from app.utils.llm import local_llm as llm
from .prompts import build_normalized_strength_prompt, build_rationale_prompt
from .signals import CUTOFF, CORR_FLOOR, POSITIVE_SIGNALS, NEGATIVE_SIGNALS
from app.server.app import get_climate, get_competitors, get_traffic_lights
from app.features.nearbyStores.nearby_stores import get_nearby_stores_data
from app.utils import common as calib

FEATURE_VALUES_LOG = Path(__file__).resolve().parent.parent / "feature_values.log"

def get_llm_normalized_and_strength(feature_values):
    prompt = build_normalized_strength_prompt({k: v for k, v in feature_values.items() if v is not None})
    try:
        response = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.2)
        text = (response or {}).get("generated_text", "").strip()
        if not text:
            return {}
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            parsed = json.loads(json_match.group(0))
            out = {}
            for k, v in parsed.items():
                if isinstance(v, dict) and "normalized" in v and "strength" in v:
                    n = float(v["normalized"])
                    out[k] = {"normalized": max(0.0, min(1.0, n)), "strength": str(v["strength"]).lower()}
            return out
    except Exception as e:
        print(f"[ERROR] LLM normalized/strength: {e}")
    return {}

def calculate_feature_impact(feature_name, value, signal_info, normalized_value, strength):
    effective_corr = max(abs(signal_info['corr']), CORR_FLOOR)
    impact = normalized_value * signal_info['score'] * effective_corr
    return impact, strength, normalized_value

def generate_llm_rationale(feature_values, pros, cons):
    pros_text = "\n".join([f"- {p['description']}: {p['value']:.2f} (correlation: {p['correlation']:.4f})" for p in pros[:5]])
    cons_text = "\n".join([f"- {c['description']}: {c['value']:.2f} (correlation: {c['correlation']:.4f})" for c in cons[:5]])
    prompt = build_rationale_prompt(pros_text, cons_text, feature_values)
    try:
        response = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3)
        text = (response or {}).get("generated_text", "")
        return text if text else "Analysis completed based on feature correlations."
    except Exception as e:
        print(f"[ERROR] LLM rationale: {e}")
        return "Analysis completed based on feature correlations."

def generate_llm_pros_cons(feature_values, pros, cons):
    return (
        ["Strong climate and traffic visibility.", "Favorable nearby amenities and competition profile."],
        ["Consider weather and distance factors where relevant."],
    )

def _create_feature_dict(feature_name, value, signal_info, impact, strength):
    return {
        'feature': feature_name,
        'value': value,
        'description': signal_info['description'],
        'correlation': signal_info['corr'],
        'signal_score': signal_info['score'],
        'impact': impact,
        'strength': strength,
    }

def _process_positive_signal(feature_name, value, signal_info, pros, cons, ns):
    if feature_name not in ns:
        return
    n, s = ns[feature_name]["normalized"], ns[feature_name]["strength"]
    impact, strength, normalized = calculate_feature_impact(feature_name, value, signal_info, n, s)
    magnitude = abs(impact)
    if normalized >= CUTOFF:
        pros.append(_create_feature_dict(feature_name, value, signal_info, magnitude, strength))
    else:
        cons.append(_create_feature_dict(feature_name, value, signal_info, magnitude, strength))

def _process_negative_signal(feature_name, value, signal_info, pros, cons, ns):
    if feature_name not in ns:
        return
    n, s = ns[feature_name]["normalized"], ns[feature_name]["strength"]
    impact, strength, normalized = calculate_feature_impact(feature_name, value, signal_info, n, s)
    magnitude = abs(impact)
    is_distance = 'distance' in feature_name.lower()
    if is_distance:
        if normalized >= CUTOFF:
            pros.append(_create_feature_dict(feature_name, value, signal_info, magnitude, strength))
        else:
            cons.append(_create_feature_dict(feature_name, value, signal_info, magnitude, strength))
    else:
        if normalized < CUTOFF:
            pros.append(_create_feature_dict(feature_name, value, signal_info, magnitude, strength))
        else:
            cons.append(_create_feature_dict(feature_name, value, signal_info, magnitude, strength))

def _calculate_impact_percentages(items):
    total = sum(item['impact'] for item in items) or 1
    for item in items:
        item['impact_pct'] = round(100 * item['impact'] / total, 1)
    return total

def analyze_site_features(feature_values):
    ns = get_llm_normalized_and_strength(feature_values)
    pros = []
    cons = []
    for feature_name, signal_info in POSITIVE_SIGNALS.items():
        if feature_name in feature_values and feature_values[feature_name] is not None:
            _process_positive_signal(feature_name, feature_values[feature_name], signal_info, pros, cons, ns)
    for feature_name, signal_info in NEGATIVE_SIGNALS.items():
        if feature_name in feature_values and feature_values[feature_name] is not None:
            _process_negative_signal(feature_name, feature_values[feature_name], signal_info, pros, cons, ns)
    pros.sort(key=lambda x: x['impact'], reverse=True)
    cons.sort(key=lambda x: x['impact'], reverse=True)
    total_pro_impact = _calculate_impact_percentages(pros)
    total_con_impact = _calculate_impact_percentages(cons)
    llm_rationale = generate_llm_rationale(feature_values, pros, cons)
    llm_pros, llm_cons = generate_llm_pros_cons(feature_values, pros, cons)
    net_score = round(total_pro_impact - total_con_impact, 6)
    all_signals = {**POSITIVE_SIGNALS, **NEGATIVE_SIGNALS}
    features_analyzed = len([f for f in feature_values if f in all_signals])
    return {
        'rationale': llm_rationale,
        'pros': pros,
        'cons': cons,
        'llm_pros': llm_pros,
        'llm_cons': llm_cons,
        'total_pro_impact': round(total_pro_impact, 6),
        'total_con_impact': round(total_con_impact, 6),
        'net_score': net_score,
        'features_analyzed': features_analyzed,
    }

def analyze_site_from_dict(address):
    geo_cord = calib.get_lat_long(address)
    lat = geo_cord["lat"]
    lon = geo_cord["lon"]
    weather_details = get_climate(lat, lon)
    nearby_stores_data = get_nearby_stores_data(lat, lon)
    competitors_data = get_competitors(lat, lon)
    competitor_1_google_user_rating_count = competitors_data["competitor_1_google_user_rating_count"]
    competitors_count = competitors_data["competitors_count"]
    traffic_data = get_traffic_lights(lat, lon)
    nearby_traffic_lights_count = traffic_data["nearby_traffic_lights_count"]
    distance_nearest_traffic_light_2 = traffic_data["distance_nearest_traffic_light_2"]
    distance_nearest_traffic_light_3 = traffic_data["distance_nearest_traffic_light_3"]
    distance_nearest_traffic_light_4 = traffic_data["distance_nearest_traffic_light_4"]
    distance_nearest_traffic_light_7 = traffic_data["distance_nearest_traffic_light_7"]
    distance_nearest_traffic_light_9 = traffic_data["distance_nearest_traffic_light_9"]
    sunny_days_per_year = (weather_details["total_sunshine_hours"] / 12.0) if weather_details.get("total_sunshine_hours") is not None else None
    feature_values = {
        "sunny_days_per_year": sunny_days_per_year,
        "total_precipitation_mm": weather_details["total_precipitation_mm"],
        "days_pleasant_temp": weather_details["days_pleasant_temp"],
        "rainy_days": weather_details["rainy_days"],
        "avg_daily_max_windspeed_ms": weather_details["avg_daily_max_windspeed_ms"],
        "days_below_freezing": weather_details["days_below_freezing"],
        "total_snowfall_cm": weather_details["total_snowfall_cm"],
        "count_of_costco_5miles": nearby_stores_data.get("count_of_costco_5miles", 0) or 0,
        "count_of_walmart_5miles": nearby_stores_data.get("count_of_walmart_5miles", 0) or 0,
    }
    if nearby_stores_data.get("distance_from_nearest_costco") is not None:
        feature_values["distance_from_nearest_costco"] = nearby_stores_data["distance_from_nearest_costco"]
    if nearby_stores_data.get("distance_from_nearest_walmart") is not None:
        feature_values["distance_from_nearest_walmart"] = nearby_stores_data["distance_from_nearest_walmart"]
    feature_values["count_of_gas_stations_5miles"] = nearby_stores_data.get("count_of_gas_stations_5miles", 0) or 0
    if nearby_stores_data.get("distance_from_nearest_gas_station") is not None:
        feature_values["distance_from_nearest_gas_station"] = nearby_stores_data["distance_from_nearest_gas_station"]
    feature_values["nearby_traffic_lights_count"] = nearby_traffic_lights_count
    feature_values["distance_nearest_traffic_light_2"] = distance_nearest_traffic_light_2
    feature_values["distance_nearest_traffic_light_3"] = distance_nearest_traffic_light_3
    feature_values["distance_nearest_traffic_light_4"] = distance_nearest_traffic_light_4
    feature_values["distance_nearest_traffic_light_7"] = distance_nearest_traffic_light_7
    feature_values["distance_nearest_traffic_light_9"] = distance_nearest_traffic_light_9
    feature_values["competitor_1_google_user_rating_count"] = competitor_1_google_user_rating_count
    feature_values["competitors_count"] = competitors_count
    try:
        with open(FEATURE_VALUES_LOG, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"run_at={datetime.utcnow().isoformat()}Z address={address}\n")
            f.write(f"lat={lat} lon={lon}\n")
            f.write(json.dumps(feature_values, indent=2) + "\n")
    except Exception as e:
        print(f"Error log: {e}")
    result = analyze_site_features(feature_values)
    result["feature_values"] = feature_values
    return result

if __name__ == "__main__":
    analyze_site_from_dict("1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA")
