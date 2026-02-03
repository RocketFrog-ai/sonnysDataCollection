
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from core import analyze_site_from_dict

app = FastAPI(title="Site Analysis Agentic Pipeline API ")

# class SiteFeatures(BaseModel):
#     total_sunshine_hours: Optional[float] = None
#     total_precipitation_mm: Optional[float] = None
#     days_pleasant_temp: Optional[float] = None
#     nearby_traffic_lights_count: Optional[float] = None
#     rainy_days: Optional[float] = None
#     count_of_walmart_5miles: Optional[float] = None
#     competitor_1_google_user_rating_count: Optional[float] = None
#     competitors_count: Optional[float] = None
#     days_below_freezing: Optional[float] = None
#     distance_nearest_traffic_light_2: Optional[float] = None
#     distance_nearest_traffic_light_3: Optional[float] = None
#     distance_nearest_traffic_light_4: Optional[float] = None
#     distance_nearest_traffic_light_7: Optional[float] = None
#     distance_nearest_traffic_light_9: Optional[float] = None
#     distance_from_nearest_costco: Optional[float] = None
#     total_snowfall_cm: Optional[float] = None
#     distance_from_nearest_walmart: Optional[float] = None
#     avg_daily_max_windspeed_ms: Optional[float] = None

class AnalyseRequest(BaseModel):
    address: str = None

@app.post("/analyze-site")
def analyze_site(features: AnalyseRequest):
    # feature_dict = {k: v for k, v in features.dict().items() if v is not None}
    
    if "address" not in features:
        raise HTTPException(status_code=400, detail="No site adddress provided")
    
    try:
        result = analyze_site_from_dict(features["address"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}
