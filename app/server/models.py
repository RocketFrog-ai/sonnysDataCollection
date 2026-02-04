from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum


class AnalyseRequest(BaseModel):
    address: str = None

class QueryRequest(BaseModel):
    """Request model for data query endpoint"""
    tenant_id: str = Field(..., description="Tenant identifier")
    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Password for authentication")
    query: str = Field(..., description="Natural language query")
    operator: str = Field(..., description="Subdomain of the Operator")
    site_id: str = Field(..., description="Site Id")


class QueryResponse(BaseModel):
    """Response model for data query endpoint"""
    success: bool = Field(..., description="Whether the query was successful")
    query : Optional[str] = Field(default=None, description="Natural language query")
    answer: str = Field(..., description="Natural language answer to the query")
    generated_sql: Optional[str] = Field(default=None, description="Generated SQL query")
    raw_data: Optional[List[List[Any]]] = Field(default=None, description="Raw query results")
    anomaly_found: Optional[bool] = Field(default=None, description="Whether anomaly was detected")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    tda_response_code: Optional[str] = Field(default=None, description="TDA pipeline response code")
    tda_response_description: Optional[str] = Field(default=None, description="TDA pipeline response description")
    node_timings: Optional[Dict[str, float]] = Field(default=None, description="Node execution timings in milliseconds")
    question_id: Optional[str] = Field(default=None, description="Unique identifier for this question")
    session_id: Optional[str] = Field(default=None, description="Session ID for chat history (created on first request)")


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


class AnomalyAnalyzeRequest(BaseModel):
    """Request model for anomaly analysis endpoint"""
    site_address: str = Field(..., description="Full address of the business site")
    anomaly: str = Field(..., description="Description of the business anomaly")
    # days_back: int = Field(default=7, ge=1, le=90, description="Number of days to look back for data (1-90)")
    start_date : date = Field(..., description="Start Date")
    end_date : date = Field(..., description="End Date")
    operator: str = Field(..., description="Subdomain of the Operator")
    site_id: str = Field(..., description="Site Id")
    anomaly_id: str = Field(..., description="Anomaly Id")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    redis_connected: bool = Field(..., description="Redis connection status")
    celery_active: bool = Field(..., description="Celery worker status")