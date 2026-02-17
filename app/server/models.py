from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum


class AnalyseRequest(BaseModel):
    address: str = None


class WeatherRequest(BaseModel):
    address: str = None
    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)


class WeatherPlotRequest(BaseModel):
    """Request for annual weather plot data (monthly distribution chart)."""
    address: str
    year: int = Field(..., ge=1900, le=2100, description="Calendar year (e.g. 2025)")


class GasStationRequest(BaseModel):
    """Request for nearby gas stations (fuel stations within radius)."""
    address: str
    radius_miles: Optional[float] = Field(2.0, ge=0.1, le=25.0, description="Search radius in miles (default 2)")
    max_results: Optional[int] = Field(10, ge=1, le=20, description="Max number of stations to return")
    fetch_place_details: Optional[bool] = Field(True, description="Whether to fetch Place Details (fuelOptions, types) per station")


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
    """Response model for a single-dimension quantile summary (e.g. weather, competition)."""
    task_id: str = Field(..., description="Task identifier from /v1/analyze-site")
    dimension: str = Field(..., description="Dimension name (Weather, Traffic, etc.)")
    predicted_tier: str = Field(..., description="Predicted performance tier for this dimension")
    fit_score: float = Field(..., description="Fit score (%) — how well the site matches this tier")
    features_scored: int = Field(..., description="Number of features scored in this dimension")
    feature_breakdown: Dict = Field(default_factory=dict, description="Per-feature scoring details with IQR ranges")
    discriminatory_power: Dict = Field(default_factory=dict, description="How well this dimension separates Low from High tiers historically")
    summary: str = Field(..., description="LLM-generated executive summary for this dimension")
    feature_values_slice: Dict = Field(default_factory=dict, description="Input feature values for this dimension")
    feature_performance: Dict[str, str] = Field(default_factory=dict, description="Per-feature performance label (High/Avg/Low)")


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