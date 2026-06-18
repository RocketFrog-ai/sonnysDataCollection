from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
from enum import Enum


class AnalyseRequest(BaseModel):
    """Kickoff for the external-data fetch pipeline (also used by /traffic-lights, /nearby-stores)."""
    address: str = Field(..., description="Site address to geocode and fetch nearby data for.")


class SiteContextRequest(BaseModel):
    """Synchronous lat/lon site-analysis (the shared map pin). Provide latitude+longitude OR an address.

    Returns weather / competitors / retail anchors / gas stations + map markers + rule-based insights in ONE
    response (no task polling) — the lat/lon counterpart to the async /analyze-site pipeline."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="Pin latitude.")
    longitude: Optional[float] = Field(None, description="Pin longitude.")
    include_ai: bool = Field(False, description="Also rewrite each dimension's insight with the internal LLM (skipped if unreachable).")
    demo: bool = Field(False, description="Anonymized demo: hide the origin address on the markers.")


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
