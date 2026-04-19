from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum


class AnalyseRequest(BaseModel):
    address: str = None
    tunnel_count: Optional[int] = Field(None, ge=1, le=4, description="Number of wash tunnels at the site (1-4). Improves quantile prediction accuracy.")
    carwash_type_encoded: Optional[int] = Field(None, ge=1, le=3, description="Car wash type: 1=Express Tunnel, 2=Mobile, 3=Hand Wash/Detail. Improves quantile prediction.")
    tier_strategy: Optional[str] = Field("4-class-wide-middle", description="Tiering strategy to use (e.g. '4-class-wide-middle', '3-class-bottom-heavy').")


class ClusterRangeCheckRequest(BaseModel):
    """Request for cluster assignment + historical range check."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="Latitude of site.")
    longitude: Optional[float] = Field(None, description="Longitude of site.")
    radius: str = Field("12km", description="Cluster radius to use: 12km or 18km.")


class ClusterProjectionRequest(BaseModel):
    """Request for multi-horizon cluster projection."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="Latitude of site.")
    longitude: Optional[float] = Field(None, description="Longitude of site.")
    radius: str = Field("12km", description="Cluster radius to use: 12km or 18km.")
    method: str = Field(
        "blend",
        description="Projection method: 'holt_winters', 'arima', or 'blend'.",
    )


class WeatherRequest(BaseModel):
    address: str = None
    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)


class WeatherDataMetricRequest(BaseModel):
    """Request for a single weather metric at api/v1/weather/data/{metric_key}."""
    address: str = Field(..., description="Site address to geocode and fetch weather for")
    start_date: Optional[str] = Field(None, description="Start date (ISO) for climate range")
    end_date: Optional[str] = Field(None, description="End date (ISO) for climate range")


class WeatherDataMetricResponse(BaseModel):
    """Response for api/v1/weather/data/{metric_key}: value + unit, and scale/quantile block (summary filled later)."""
    value: float = Field(..., description="Metric value (e.g. 120 days)")
    unit: str = Field(..., description="Unit label (e.g. 'days/year')")
    quantile_score: Optional[float] = Field(None, description="Score 0–100 vs portfolio (to be wired)")
    quantile: Optional[str] = Field(None, description="Quantile band (to be wired)")
    category: Optional[str] = Field(None, description="Category e.g. Poor / Fair / Good / Strong (to be wired)")
    min: Optional[float] = Field(None, description="Scale min for UI (to be wired)")
    max: Optional[float] = Field(None, description="Scale max for UI (to be wired)")
    summary: Optional[str] = Field(None, description="Short narrative (to be wired)")


class GasStationRequest(BaseModel):
    """Request for nearby gas stations (fuel stations within radius)."""
    address: str
    radius_miles: Optional[float] = Field(2.0, ge=0.1, le=25.0, description="Search radius in miles (default 2)")
    max_results: Optional[int] = Field(10, ge=1, le=20, description="Max number of stations to return")
    fetch_place_details: Optional[bool] = Field(True, description="Whether to fetch Place Details (fuelOptions, types) per station")


class DataFetchNearestGasRequest(BaseModel):
    """Request for data-fetch endpoint: single nearest gas station only (no radius/limit)."""
    address: str
    fetch_place_details: Optional[bool] = Field(True, description="Whether to fetch Place Details (fuelOptions, types)")


class RetailersRequest(BaseModel):
    """Request for nearby retailers (complementary businesses within radius)."""
    address: str
    radius_miles: Optional[float] = Field(0.5, ge=0.1, le=5.0, description="Driving distance radius in miles (default 0.5)")
    fetch_place_details: Optional[bool] = Field(True, description="Whether to fetch Place Details (website, googleMapsUri)")


class CompetitorsDynamicsRequest(BaseModel):
    """Request for competition dynamics (nearby car washes within radius by route distance)."""
    address: str
    radius_miles: Optional[float] = Field(4.0, ge=0.5, le=15.0, description="Driving distance radius in miles (default 4)")
    fetch_place_details: Optional[bool] = Field(True, description="Whether to fetch Place Details (website, googleMapsUri, primaryTypeDisplayName)")


class CompetitorItem(BaseModel):
    name: Optional[str] = None
    distance_miles: Optional[float] = None
    rating: Optional[float] = None
    user_rating_count: Optional[int] = None
    address: Optional[str] = None
    place_id: Optional[str] = None
    website: Optional[str] = None
    google_maps_uri: Optional[str] = None
    primary_type_display_name: Optional[str] = None


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskResponse(BaseModel):
    """Response model for task submission"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Task status")
    message: str = Field(..., description="Status message")


class TaskStatusResponse(BaseModel):
    """Response model for task status check"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    result: Optional[Any] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: Optional[datetime] = Field(None, description="Task creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    redis_connected: bool = Field(..., description="Redis connection status")
    celery_active: bool = Field(..., description="Celery worker status")


# ── Quantile Profiling Summary Models ────────────────────────────

class DimensionSummaryResponse(BaseModel):
    """Response model for a single-dimension summary (Approach 2 categories or quantile)."""
    task_id: str = Field(..., description="Task identifier from /v1/analyze-site")
    dimension: str = Field(..., description="Dimension name (Weather, Traffic, etc.)")
    predicted_tier: str = Field(..., description="Predicted tier: Approach 2 (Excellent/Very Good/Good/Fair/Poor/Very Poor) or quantile (High/Avg/Low)")
    fit_score: float = Field(0, description="Fit score (%) — average of Approach 2 final scores when using Approach 2")
    features_scored: int = Field(..., description="Number of features scored in this dimension")
    feature_breakdown: Dict = Field(default_factory=dict, description="Per-feature details: Approach 2 (value, percentile, category) or IQR ranges")
    discriminatory_power: Dict = Field(default_factory=dict, description="Historical discriminatory power (quantile) or empty for Approach 2")
    summary: str = Field(..., description="Executive summary — Approach 2 rationale (percentile, category) or LLM-generated")
    feature_values_slice: Dict = Field(default_factory=dict, description="Input feature values for this dimension")
    feature_performance: Dict[str, str] = Field(default_factory=dict, description="Per-feature category (Excellent, Very Good, Good, Fair, Poor, Very Poor)")


class QuantileSummaryResponse(BaseModel):
    """Response model for the full quantile profiling summary across all dimensions."""
    task_id: str = Field(..., description="Task identifier")
    overall_tier: str = Field(..., description="Overall predicted tier (majority vote)")
    overall_fit_score: float = Field(..., description="Overall fit score from quantile scoring")
    expected_volume: Dict = Field(default_factory=dict, description="Conservative / likely / optimistic volume range")
    dimensions: Dict[str, Dict] = Field(default_factory=dict, description="Per-dimension results")
    vote: Dict = Field(default_factory=dict, description="Majority vote breakdown")
    strengths: List[str] = Field(default_factory=list, description="Dimensions predicted as High Performing")
    weaknesses: List[str] = Field(default_factory=list, description="Dimensions predicted as Low Performing")
    rationale: str = Field("", description="LLM-generated executive rationale")