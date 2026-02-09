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