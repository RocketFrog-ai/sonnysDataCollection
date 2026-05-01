from __future__ import annotations

import logging
from typing import Any, Dict

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query

from app.celery.celery_app import celery_app
from app.pnl_analysis.modelling.zeta_pnl import run_clustering_v2_projection_task
from app.pnl_analysis.server.models import (
    BreakevenRequest,
    BreakevenResponse,
    ClusteringV2ProjectionRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl_analysis")


@router.post("/breakeven", response_model=BreakevenResponse)
def calculate_breakeven(req: BreakevenRequest):
    """
    Compute breakeven from project cost vs cumulative yearly net cashflows.
    """
    cumulative_values = req.net_cashflows_cumulative
    project_cost = float(req.project_cost)

    for idx, cumulative in enumerate(cumulative_values, start=1):
        if cumulative >= project_cost:
            prev_cumulative = cumulative_values[idx - 2] if idx > 1 else 0.0
            delta = cumulative - prev_cumulative

            # Linear interpolation within the current year interval.
            if delta > 0:
                fraction = (project_cost - prev_cumulative) / delta
                fraction = max(0.0, min(1.0, fraction))
                breakeven_year_fractional = (idx - 1) + fraction
            else:
                breakeven_year_fractional = float(idx)

            return BreakevenResponse(
                project_cost=project_cost,
                net_cashflows_cumulative=cumulative_values,
                breakeven_achieved=True,
                breakeven_year=idx,
                breakeven_year_fractional=round(breakeven_year_fractional, 4),
                cumulative_at_breakeven_year=cumulative,
            )

    return BreakevenResponse(
        project_cost=project_cost,
        net_cashflows_cumulative=cumulative_values,
        breakeven_achieved=False,
        breakeven_year=None,
        breakeven_year_fractional=None,
        cumulative_at_breakeven_year=None,
    )


@router.post("/clustering-v2/project", response_model=TaskResponse)
def start_clustering_v2_projection(req: ClusteringV2ProjectionRequest):
    """
    Enqueue zeta_modelling wash projection (legacy path name kept for API compatibility).
    Returns `task_id`; poll `/pnl_analysis/task/{task_id}` or fetch data via `/pnl_analysis/wash-count-plot?task_id=...`.
    """
    if not (req.address and req.address.strip()) and (req.latitude is None or req.longitude is None):
        raise HTTPException(status_code=400, detail="Provide either address or latitude+longitude.")
    try:
        result = run_clustering_v2_projection_task.delay(req.model_dump())
    except Exception as e:
        logger.exception("Failed to enqueue zeta_modelling projection")
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
    Return wash-count projection *data* (no plots) for an existing projection task_id (zeta_modelling).
    """
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res: Dict[str, Any] = task_result.result or {}
        # Prefer 60-month timeline if present, retain 48-month for compatibility.
        timeline_60 = res.get("monthly_projection_60mo")
        timeline_48 = res.get("monthly_projection_48mo")
        return {
            "task_id": task_id,
            "status": "success",
            "input": res.get("input"),
            "method": res.get("method"),
            "level_model": res.get("level_model"),
            "calendar_year_washes": res.get("calendar_year_washes"),
            "monthly_projection_48mo": timeline_48,
            "monthly_projection_60mo": timeline_60,
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


@router.get("/wash-count-minmax")
def get_wash_count_minmax(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "wash_count_minmax": res.get("wash_volume_range_minmax"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.api_route("/wash_volume_projection", methods=["GET", "POST"])
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


@router.get("/cash_flow_projections")
def get_cash_flow_projections(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "cash_flow_projections": res.get("cash_flow_projections"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.get("/cash-on-cash-return")
def get_cash_on_cash_return(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "project_cost": res.get("project_cost"),
            "cash_on_cash_return": res.get("cash_on_cash_return"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.get("/expense-breakdown")
def get_expense_breakdown(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "expense_breakdown": res.get("expense_breakdown"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.get("/headlines-washcast")
def get_headlines_washcast(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "headlines_washcast": res.get("headlines_washcast"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.get("/membership_retail_count")
def get_membership_retail_count(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "membership_retail_count": res.get("membership_retail_count"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}


@router.get("/membership_retail_revenue")
def get_membership_retail_revenue(task_id: str = Query(..., description="Task id from /v1/input-form")):
    task_result = AsyncResult(task_id, app=celery_app)
    state = (task_result.state or "PENDING").upper()
    if state == TaskStatus.SUCCESS.value:
        res = _get_task_payload(task_id) or {}
        return {
            "task_id": task_id,
            "status": "success",
            "membership_retail_revenue": res.get("membership_retail_revenue"),
        }
    if state == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
        }
    return {"task_id": task_id, "status": state.lower()}
