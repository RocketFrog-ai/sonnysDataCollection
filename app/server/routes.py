import redis
import logging
from app.utils import common as calib
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from app.server.models import *
from app.ai.analysis import analyze_site_from_dict
from app.celery.tasks import analyse_site


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
    
    if "address" not in features:
        raise HTTPException(status_code=400, detail="No site adddress provided")
    
    try:
        result = analyse_site.delay(features["address"])
        return TaskResponse(
            task_id=result.id,
            status=TaskStatus.PENDING,
            message="Site succesfully submitted for Analysis"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}