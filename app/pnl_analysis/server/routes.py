from __future__ import annotations

import logging
from typing import Any, Dict

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query

from app.celery.celery_app import celery_app
from app.pnl_analysis.modelling.clustering_v2 import (
    run_clustering_v2_projection_task,
    run_pnl_central_input_form_task,
)
from app.pnl_analysis.server.models import (
    ClusteringV2ProjectionRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl_analysis")


@router.post("/clustering-v2/project", response_model=TaskResponse)
def start_clustering_v2_projection(req: ClusteringV2ProjectionRequest):
    """
    Enqueue clustering_v2 projection (greenfield wash-count forecast).
    Returns `task_id`; poll `/pnl_analysis/task/{task_id}` or fetch data via `/pnl_analysis/wash-count-plot?task_id=...`.
    """
    if not (req.address and req.address.strip()) and (req.latitude is None or req.longitude is None):
        raise HTTPException(status_code=400, detail="Provide either address or latitude+longitude.")
    try:
        result = run_clustering_v2_projection_task.delay(req.model_dump())
    except Exception as e:
        logger.exception("Failed to enqueue clustering_v2 projection")
        raise HTTPException(status_code=500, detail=str(e))
    return TaskResponse(task_id=result.id, status=TaskStatus.PENDING, message="Clustering v2 projection submitted")


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_pnl_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    status_str = task_result.state
    try:
        status = TaskStatus(status_str)
    except ValueError:
        status = TaskStatus.PENDING

    response = TaskStatusResponse(task_id=task_id, status=status, result=None, error=None)
    if status == TaskStatus.SUCCESS:
        response.result = task_result.result
    elif status == TaskStatus.FAILURE:
        response.error = str(task_result.result) if task_result.result else "Task failed"
    return response


@router.get("/wash-count-plot")
def get_wash_count_plot_data(task_id: str = Query(..., description="Celery task id from /clustering-v2/project")):
    """
    Return wash-count projection *data* (no plots) for an existing clustering_v2 task_id.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res: Dict[str, Any] = task_result.result or {}
        # Prefer flattened timeline if present (task output), else fall back to raw response.
        timeline = res.get("monthly_projection_48mo")
        if timeline is None and isinstance(res.get("raw"), dict):
            from app.pnl_analysis.clustering_v2.runtime import build_monthly_wash_projection_48mo

            timeline = build_monthly_wash_projection_48mo(res["raw"])
        return {
            "task_id": task_id,
            "status": "success",
            "input": res.get("input"),
            "method": res.get("method"),
            "level_model": res.get("level_model"),
            "calendar_year_washes": res.get("calendar_year_washes"),
            "monthly_projection_48mo": timeline,
            "dollars_per_wash": res.get("dollars_per_wash"),
            "revenue_by_year": res.get("revenue_by_year"),
            "opex_by_year": res.get("opex_by_year"),
            "profit_by_year": res.get("profit_by_year"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


def _get_task_payload(task_id: str) -> Dict[str, Any] | None:
    task_result = AsyncResult(task_id, app=celery_app)
    if (task_result.state or "").upper() != TaskStatus.SUCCESS.value:
        return None
    res = task_result.result or {}
    return res if isinstance(res, dict) else None


@router.get("/wash_volume_range_minmax")
def get_wash_volume_range_minmax(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "wash_volume_range_minmax": res.get("wash_volume_range_minmax"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.get("/wash_volume_projection")
def get_wash_volume_projection(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "wash_volume_projection": res.get("wash_volume_projection"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}
