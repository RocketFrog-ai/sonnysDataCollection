from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.pnl_analysis.modelling.zeta_pnl import run_pnl_central_input_form_task
from app.pnl_analysis.server.central_input_db import save_central_input_form_submission
from app.pnl_analysis.server.models import CentralInputFormRequest, TaskResponse, TaskStatus


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/input-form", response_model=TaskResponse)
def submit_central_input_form(req: CentralInputFormRequest):
    """
    Shared input-form endpoint.
    Accepts the full nested input form, persists it to PostgreSQL (CAR_WASH_DB_URL), and enqueues processing.
    Returns one central task id that downstream modules can read from.

    Wash volumes and P10/P90-style year ranges come from ``zeta_modelling`` (``model_1`` / ``data_1``).
    Optional body field ``zeta_forecast`` overrides margin/costs/scenario (see ``ZetaForecastParams``).
    """
    payload = req.to_task_payload()
    try:
        result = run_pnl_central_input_form_task.delay(payload)
    except Exception as exc:
        logger.exception("Failed to enqueue central input form task")
        raise HTTPException(status_code=500, detail=str(exc))
    if not save_central_input_form_submission(task_id=result.id, payload=payload):
        logger.warning("Central input form not saved to DB (task_id=%s); check CAR_WASH_DB_URL and connectivity.", result.id)
    return TaskResponse(task_id=result.id, status=TaskStatus.PENDING, message="Submitted")
