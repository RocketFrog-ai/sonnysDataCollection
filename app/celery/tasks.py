import json
import logging
import redis

from app.utils import common as calib
from app.celery.celery_app import celery_app
from app.modelling.site_analysis import run_site_analysis
from app.server.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD

TASK_RETRY_DELAY = calib.TASK_RETRY_DELAY
TASK_MAX_RETRIES = calib.TASK_MAX_RETRIES
EXTERNAL_SERVICE_URL = calib.EXTERNAL_SERVICE_URL
EXTERNAL_SERVICE_TIMEOUT = calib.EXTERNAL_SERVICE_TIMEOUT

# Partial result cache TTL (1 day)
RESULT_CACHE_TTL = 86400

logger = logging.getLogger(__name__)


def _redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT or 6379),
        db=int(REDIS_DB or 0),
        password=REDIS_PASSWORD or None,
        decode_responses=True,
    )


@celery_app.task(bind=True)
def analyse_site(self, address: str = "1"):
    """
    Run site analysis: fetch (data available quickly) → quantile → narratives.
    Writes partial result to Redis after each stage so GET /result/{task_id} can
    return data as soon as fetch is done, then quantile, then full result.
    """
    task_id = self.request.id
    cache_key = f"site_analysis:{task_id}"

    def set_progress(partial: dict):
        try:
            client = _redis_client()
            # Serialize for Redis; numpy etc. via default=str
            payload = json.dumps(partial, default=str)
            client.setex(cache_key, RESULT_CACHE_TTL, payload)
        except Exception as e:
            logger.warning("Redis set_progress failed: %s", e)

    res = run_site_analysis(
        address,
        run_quantile=True,
        run_narratives=True,
        set_progress=set_progress,
    )
    return res