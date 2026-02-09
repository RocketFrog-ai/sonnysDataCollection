import redis
import logging
from app.utils import common as calib
from datetime import datetime, timezone
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from app.server.models import *
from app.ai.analysis import analyze_site_from_dict
from app.celery.tasks import analyse_site
from app.celery.celery_app import celery_app


REDIS_HOST = calib.REDIS_HOST
REDIS_PORT = calib.REDIS_PORT
REDIS_DB = calib.REDIS_DB
REDIS_PASSWORD = calib.REDIS_PASSWORD
CELERY_BROKER_URL = calib.CELERY_BROKER_URL
CELERY_RESULT_BACKEND = calib.CELERY_RESULT_BACKEND


logger = logging.getLogger(__name__)
router = APIRouter()


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True
    )


@router.post("/analyze-site")
def analyze_site(features: AnalyseRequest):
    # feature_dict = {k: v for k, v in features.dict().items() if v is not None}
    
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    
    try:
        result = analyse_site.delay(features.address)
        return TaskResponse(
            task_id=result.id,
            status=TaskStatus.PENDING,
            message="Site succesfully submitted for Analysis"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Get the status and result of an analyze-site task by its task_id."""
    task_result = AsyncResult(task_id, app=celery_app)
    status_str = task_result.state
    # Map Celery state to our TaskStatus enum (Celery uses same names)
    try:
        status = TaskStatus(status_str)
    except ValueError:
        status = TaskStatus.PENDING

    response = TaskStatusResponse(
        task_id=task_id,
        status=status,
        result=None,
        error=None,
        created_at=None,
        completed_at=None,
    )

    if status == TaskStatus.SUCCESS:
        response.result = task_result.result
    elif status == TaskStatus.FAILURE:
        response.error = str(task_result.result) if task_result.result else "Task failed"

    return response


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}