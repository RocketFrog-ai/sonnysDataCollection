from typing import Dict, List, Tuple
import json
import re
import requests
from datetime import datetime
from pathlib import Path
from agentic_pipeline.utils import get_llm_response
from .prompts import build_rationale_prompt, build_pros_cons_prompt
from app import get_climate, get_competitors, get_traffic_lights
from nearbyStores.nearby_stores import get_nearby_stores_data
from agentic_pipeline.utils import commons as calib

FEATURE_VALUES_LOG = Path(__file__).resolve().parent.parent / "feature_values.log"

POSITIVE_SIGNALS = {
    'total_sunshine_hours': {'score': 0.8523, 'corr': 0.1442, 'description': 'Total sunshine hours'},
    'total_precipitation_mm': {'score': 0.7033, 'corr': 0.0441, 'description': 'Total precipitation'},
    'days_pleasant_temp': {'score': 0.5601, 'corr': 0.1307, 'description': 'Days with pleasant temperature'},
    'nearby_traffic_lights_count': {'score': 0.4658, 'corr': 0.1372, 'description': 'Count of nearby traffic lights'},
    'rainy_days': {'score': 0.3574, 'corr': 0.0399, 'description': 'Number of rainy days'},
    'count_of_costco_5miles': {'score': 0.2188, 'corr': 0.1097, 'description': 'Costco stores within 5 miles'},
    'count_of_walmart_5miles': {'score': 0.2188, 'corr': 0.1097, 'description': 'Walmart stores within 5 miles'},
    'competitor_1_google_user_rating_count': {'score': 0.2182, 'corr': 0.0354, 'description': 'Top competitor rating count'},
    'competitors_count': {'score': 0.0432, 'corr': 0.0257, 'description': 'Total competitors count'}
}

NEGATIVE_SIGNALS = {
    'days_below_freezing': {'score': 0.5977, 'corr': -0.0900, 'description': 'Days below freezing'},
    'distance_nearest_traffic_light_3': {'score': 0.5014, 'corr': -0.0991, 'description': 'Distance to 3rd nearest traffic light'},
    'distance_nearest_traffic_light_9': {'score': 0.4776, 'corr': -0.1540, 'description': 'Distance to 9th nearest traffic light'},
    'distance_from_nearest_costco': {'score': 0.4476, 'corr': -0.1663, 'description': 'Distance from nearest Costco'},
    'distance_nearest_traffic_light_2': {'score': 0.4282, 'corr': -0.1041, 'description': 'Distance to 2nd nearest traffic light'},
    'total_snowfall_cm': {'score': 0.4131, 'corr': -0.0918, 'description': 'Total snowfall'},
    'distance_nearest_traffic_light_7': {'score': 0.4023, 'corr': -0.1268, 'description': 'Distance to 7th nearest traffic light'},
    'distance_from_nearest_walmart': {'score': 0.3863, 'corr': -0.0710, 'description': 'Distance from nearest Walmart'},
    'distance_nearest_traffic_light_4': {'score': 0.3714, 'corr': -0.1149, 'description': 'Distance to 4th nearest traffic light'},
    'avg_daily_max_windspeed_ms': {'score': 0.3616, 'corr': -0.0057, 'description': 'Average daily max wind speed'}
}

def normalize_feature_value(feature_name: str, value: float, feature_type: str = 'positive') -> float:
    if 'distance' in feature_name.lower():
        normalized = max(0, min(1, 1 - (value / 10.0)))
    elif 'count' in feature_name.lower() or 'hours' in feature_name.lower() or 'days' in feature_name.lower():
        normalized = max(0, min(1, value / 365.0))
    elif 'precipitation' in feature_name.lower() or 'snowfall' in feature_name.lower():
        normalized = max(0, min(1, value / 2000.0))
    elif 'windspeed' in feature_name.lower():
        normalized = max(0, min(1, value / 50.0))
    elif 'rating' in feature_name.lower():
        normalized = max(0, min(1, value / 10000.0))
    else:
        normalized = max(0, min(1, value / 100.0))
    return normalized

def calculate_feature_impact(feature_name: str, value: float, signal_info: Dict) -> Tuple[float, str]:
    normalized_value = normalize_feature_value(feature_name, value, 'positive' if signal_info['corr'] > 0 else 'negative')
    impact = normalized_value * abs(signal_info['corr']) * signal_info['score']
    
    if signal_info['corr'] > 0:
        if normalized_value > 0.7:
            strength = "strong"
        elif normalized_value > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
    else:
        if 'distance' in feature_name.lower():
            if normalized_value > 0.7:
                strength = "strong"
            elif normalized_value > 0.4:
                strength = "moderate"
            else:
                strength = "weak"
        else:
            if normalized_value < 0.3:
                strength = "strong"
            elif normalized_value < 0.6:
                strength = "moderate"
            else:
                strength = "weak"
    
    return impact, strength

def generate_llm_rationale(feature_values: Dict[str, float], pros: List[Dict], cons: List[Dict]) -> str:
    pros_text = "\n".join([f"- {p['description']}: {p['value']:.2f} (correlation: {p['correlation']:.4f})" for p in pros[:5]])
    cons_text = "\n".join([f"- {c['description']}: {c['value']:.2f} (correlation: {c['correlation']:.4f})" for c in cons[:5]])
    
    prompt = build_rationale_prompt(pros_text, cons_text)
    
    try:
        response = get_llm_response(prompt, reasoning_effort="low", temperature=0.3)
        text = response.get("generated_text", "")
        print(f"[DEBUG] Raw LLM response for rationale:\n{text}\n{'='*80}")
        return text if text else "Analysis completed based on feature correlations."
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] LLM rationale request timed out: {e}")
        return "Analysis completed based on feature correlations."
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM rationale request failed: {e}")
        return "Analysis completed based on feature correlations."
    except Exception as e:
        print(f"[ERROR] Unexpected error in generate_llm_rationale: {type(e).__name__}: {e}")
        return "Analysis completed based on feature correlations."

def generate_llm_pros_cons(feature_values: Dict[str, float], pros: List[Dict], cons: List[Dict]) -> Tuple[List[str], List[str]]:
    pros_data = "\n".join([f"- {p['description']}: {p['value']:.2f}" for p in pros])
    cons_data = "\n".join([f"- {c['description']}: {c['value']:.2f}" for c in cons])
    
    prompt = build_pros_cons_prompt(pros_data, cons_data)
    
    try:
        response = get_llm_response(prompt, reasoning_effort="medium", temperature=0.4)
        text = response.get("generated_text", "")
        
        print(f"[DEBUG] Raw LLM response for pros/cons:\n{text}\n{'='*80}")
        
        if not text:
            print("[WARNING] Empty LLM response")
            return [], []
        
        llm_pros = []
        llm_cons = []
        
        try:
            text_clean = text.strip()
            
            json_match = re.search(r'\{[^{}]*"pros"[^{}]*\[[^\]]*\][^{}]*"cons"[^{}]*\[[^\]]*\][^{}]*\}', text_clean, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                llm_pros = parsed.get("pros", [])
                llm_cons = parsed.get("cons", [])
                print(f"[DEBUG] Successfully parsed JSON: {len(llm_pros)} pros, {len(llm_cons)} cons")
                return llm_pros[:5], llm_cons[:5]
            
            try:
                parsed = json.loads(text_clean)
                llm_pros = parsed.get("pros", [])
                llm_cons = parsed.get("cons", [])
                print(f"[DEBUG] Successfully parsed JSON (direct): {len(llm_pros)} pros, {len(llm_cons)} cons")
                return llm_pros[:5], llm_cons[:5]
            except json.JSONDecodeError:
                pass
            
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parsing failed: {e}, falling back to text parsing")
        
        in_pros = False
        in_cons = False
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if 'PROS:' in line.upper() or 'PRO:' in line.upper():
                in_pros = True
                in_cons = False
                continue
            if 'CONS:' in line.upper() or 'CON:' in line.upper():
                in_cons = True
                in_pros = False
                continue
                
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                content = line.lstrip('- •*').strip()
                if in_pros and content:
                    llm_pros.append(content)
                elif in_cons and content:
                    llm_cons.append(content)
            elif in_pros and line and not line.startswith('{') and not line.startswith('['):
                llm_pros.append(line)
            elif in_cons and line and not line.startswith('{') and not line.startswith('['):
                llm_cons.append(line)
        
        print(f"[DEBUG] Text parsing result: {len(llm_pros)} pros, {len(llm_cons)} cons")
        return llm_pros[:5], llm_cons[:5]
        
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] LLM request timed out: {e}")
        return [], []
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM request failed: {e}")
        return [], []
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error: {e}")
        return [], []
    except Exception as e:
        print(f"[ERROR] Unexpected error in generate_llm_pros_cons: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return [], []

def analyze_site_features(feature_values: Dict[str, float]) -> Dict:
    pros = []
    cons = []
    
    for feature_name, signal_info in POSITIVE_SIGNALS.items():
        if feature_name in feature_values:
            value = feature_values[feature_name]
            impact, strength = calculate_feature_impact(feature_name, value, signal_info)
            normalized = normalize_feature_value(feature_name, value, 'positive')
            
            if normalized > 0.5:
                pros.append({
                    'feature': feature_name,
                    'value': value,
                    'description': signal_info['description'],
                    'correlation': signal_info['corr'],
                    'signal_score': signal_info['score'],
                    'impact': impact,
                    'strength': strength,
                })
            elif normalized < 0.3:
                cons.append({
                    'feature': feature_name,
                    'value': value,
                    'description': signal_info['description'],
                    'correlation': signal_info['corr'],
                    'signal_score': signal_info['score'],
                    'impact': -impact,
                    'strength': strength,
                })
    
    for feature_name, signal_info in NEGATIVE_SIGNALS.items():
        if feature_name in feature_values:
            value = feature_values[feature_name]
            impact, strength = calculate_feature_impact(feature_name, value, signal_info)
            normalized = normalize_feature_value(feature_name, value, 'negative')
            
            if 'distance' in feature_name.lower():
                if normalized > 0.7:
                    pros.append({
                        'feature': feature_name,
                        'value': value,
                        'description': signal_info['description'],
                        'correlation': signal_info['corr'],
                        'signal_score': signal_info['score'],
                        'impact': -impact,
                        'strength': strength,
                    })
                elif normalized < 0.3:
                    cons.append({
                        'feature': feature_name,
                        'value': value,
                        'description': signal_info['description'],
                        'correlation': signal_info['corr'],
                        'signal_score': signal_info['score'],
                        'impact': impact,
                        'strength': strength,
                    })
            else:
                if normalized < 0.3:
                    pros.append({
                        'feature': feature_name,
                        'value': value,
                        'description': signal_info['description'],
                        'correlation': signal_info['corr'],
                        'signal_score': signal_info['score'],
                        'impact': -impact,
                        'strength': strength,
                    })
                elif normalized > 0.7:
                    cons.append({
                        'feature': feature_name,
                        'value': value,
                        'description': signal_info['description'],
                        'correlation': signal_info['corr'],
                        'signal_score': signal_info['score'],
                        'impact': impact,
                        'strength': strength,
                    })
    
    pros.sort(key=lambda x: x['impact'], reverse=True)
    cons.sort(key=lambda x: x['impact'], reverse=True)
    
    llm_rationale = generate_llm_rationale(feature_values, pros, cons)
    llm_pros, llm_cons = generate_llm_pros_cons(feature_values, pros, cons)
    
    return {
        'rationale': llm_rationale,
        'pros': pros,
        'cons': cons,
        'llm_pros': llm_pros,
        'llm_cons': llm_cons,
        'features_analyzed': len([f for f in feature_values.keys() if f in POSITIVE_SIGNALS or f in NEGATIVE_SIGNALS])
    }

def analyze_site_from_dict(address:str) -> Dict:
    geo_cord = calib.get_lat_long(address)
    lat = geo_cord["lat"]
    lon = geo_cord["lon"]
    # print("---------------------------")
    # print("Geocode Details")
    # print(geo_cord)
    # print("---------------------------")
    weather_details = get_climate(lat, lon)

    total_sunshine_hours = weather_details["total_sunshine_hours"]
    total_precipitation_mm = weather_details["total_precipitation_mm"]
    days_pleasant_temp = weather_details["days_pleasant_temp"]
    rainy_days = weather_details["rainy_days"]
    avg_daily_max_windspeed_ms = weather_details["avg_daily_max_windspeed_ms"]
    days_below_freezing = weather_details["days_below_freezing"]
    total_snowfall_cm = weather_details["total_snowfall_cm"]

    nearby_stores_data = get_nearby_stores_data(lat, lon)
    distance_from_nearest_costco = nearby_stores_data.get("distance_from_nearest_costco")
    count_of_costco_5miles = nearby_stores_data.get("count_of_costco_5miles", 0) or 0
    count_of_walmart_5miles = nearby_stores_data.get("count_of_walmart_5miles", 0) or 0
    distance_from_nearest_walmart = nearby_stores_data.get("distance_from_nearest_walmart")

    competitors_data = get_competitors(lat, lon)
    print("---------------------------")
    print("Competitors Data")
    print(competitors_data)
    print("---------------------------")
    # competitor_1_google_user_rating_count = competitors_data["competitor_1_google_user_rating_count"]
    # competitors_count = competitors_data["competitors_count"]
    
    traffic_data = get_traffic_lights(lat, lon)
    nearby_traffic_lights_count = traffic_data["nearby_traffic_lights_count"]
    distance_nearest_traffic_light_2 = traffic_data["distance_nearest_traffic_light_2"]
    distance_nearest_traffic_light_3 = traffic_data["distance_nearest_traffic_light_3"]
    distance_nearest_traffic_light_4 = traffic_data["distance_nearest_traffic_light_4"]
    distance_nearest_traffic_light_7 = traffic_data["distance_nearest_traffic_light_7"]
    distance_nearest_traffic_light_9 = traffic_data["distance_nearest_traffic_light_9"]    
    
    feature_values = {}
    feature_values["total_sunshine_hours"] = total_sunshine_hours
    feature_values["total_precipitation_mm"] = total_precipitation_mm
    feature_values["days_pleasant_temp"] = days_pleasant_temp
    feature_values["rainy_days"] = rainy_days
    feature_values["avg_daily_max_windspeed_ms"] = avg_daily_max_windspeed_ms
    feature_values["days_below_freezing"] = days_below_freezing
    feature_values["total_snowfall_cm"] = total_snowfall_cm
    if distance_from_nearest_costco is not None:
        feature_values["distance_from_nearest_costco"] = distance_from_nearest_costco
    feature_values["count_of_costco_5miles"] = count_of_costco_5miles
    feature_values["count_of_walmart_5miles"] = count_of_walmart_5miles
    if distance_from_nearest_walmart is not None:
        feature_values["distance_from_nearest_walmart"] = distance_from_nearest_walmart
    # feature_values["competitor_1_google_user_rating_count"] = competitor_1_google_user_rating_count
    # feature_values["competitors_count"] = competitors_count
    feature_values["nearby_traffic_lights_count"] = nearby_traffic_lights_count
    feature_values["distance_nearest_traffic_light_2"] = distance_nearest_traffic_light_2
    feature_values["distance_nearest_traffic_light_3"] = distance_nearest_traffic_light_3
    feature_values["distance_nearest_traffic_light_4"] = distance_nearest_traffic_light_4
    feature_values["distance_nearest_traffic_light_7"] = distance_nearest_traffic_light_7
    feature_values["distance_nearest_traffic_light_9"] = distance_nearest_traffic_light_9

    try:
        with open(FEATURE_VALUES_LOG, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"run_at={datetime.utcnow().isoformat()}Z address={address}\n")
            f.write(f"lat={lat} lon={lon}\n")
            f.write(json.dumps(feature_values, indent=2) + "\n")
    except Exception as e:
        print(f"Error log: {e}")

    return analyze_site_features(feature_values)
    # return feature_values
    # return None

if __name__ == "__main__":
    address = "1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
    feature_values = analyze_site_from_dict(address)
    # print(feature_values)
