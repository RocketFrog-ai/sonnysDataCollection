from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.pnl_analysis.modelling.clustering_v2 import run_pnl_central_input_form_task
from app.pnl_analysis.server.models import CentralInputFormRequest, TaskResponse, TaskStatus


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/input-form", response_model=TaskResponse)
def submit_central_input_form(req: CentralInputFormRequest):
    """
    Shared input-form endpoint.
    Returns one central task id that downstream modules can read from.
    """
    if not (req.address and req.address.strip()) and (req.latitude is None or req.longitude is None):
        raise HTTPException(status_code=400, detail="Provide either address or latitude+longitude.")
    try:
        result = run_pnl_central_input_form_task.delay(req.model_dump())
    except Exception as exc:
        logger.exception("Failed to enqueue central input form task")
        raise HTTPException(status_code=500, detail=str(exc))
    return TaskResponse(task_id=result.id, status=TaskStatus.PENDING, message="Submitted")
