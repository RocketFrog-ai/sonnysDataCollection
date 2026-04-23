from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Celery task id")
    status: TaskStatus = Field(..., description="Task state")
    message: str = Field(..., description="Human-readable status message")


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ClusteringV2ProjectionRequest(BaseModel):
    address: Optional[str] = Field(None, description="Address (optional if latitude/longitude provided).")
    latitude: Optional[float] = Field(None, description="Latitude.")
    longitude: Optional[float] = Field(None, description="Longitude.")
    method: str = Field("arima", description="Time series method: 'arima', 'holt_winters', or 'blend'.")
    level_model: str = Field("ridge", description="Level model: 'ridge' or 'rf' (if RF checkpoints exist).")
    use_opening_prefix_for_mature_forecast: bool = Field(
        True,
        description="If true, append <2y monthly forecast as context before mature extrapolation.",
    )
    bridge_opening_to_mature_when_prefix: bool = Field(
        True,
        description="If true, bridge month 24→25 when prefix is used.",
    )
    allow_nearest_cluster_beyond_distance_cap: bool = Field(
        False,
        description="If true, allow nearest centroid assignment beyond the 20 km cap.",
    )

    # Optional PnL inputs (to compute revenue/opex/profit summary on top of wash projections).
    wash_prices: Optional[List[float]] = Field(None, description="List of wash tier prices in dollars.")
    wash_pcts: Optional[List[float]] = Field(None, description="List of wash tier user share percentages (0–100).")
    opex_years: Optional[List[float]] = Field(None, description="Operating expense by year: [y1,y2,y3,y4].")


class CentralInputFormRequest(BaseModel):
    """Central input-form request (no model configuration knobs)."""
    address: Optional[str] = Field(None, description="Address (optional if latitude/longitude provided).")
    latitude: Optional[float] = Field(None, description="Latitude.")
    longitude: Optional[float] = Field(None, description="Longitude.")

    # Optional PnL inputs.
    wash_prices: Optional[List[float]] = Field(None, description="List of wash tier prices in dollars.")
    wash_pcts: Optional[List[float]] = Field(None, description="List of wash tier user share percentages (0–100).")
    opex_years: Optional[List[float]] = Field(None, description="Operating expense by year: [y1,y2,y3,y4].")
    capex_initial: Optional[float] = Field(None, description="Initial capex / investment ($).")
