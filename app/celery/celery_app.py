from celery import Celery
from app.utils import common as calib

CELERY_BROKER_URL = calib.CELERY_BROKER_URL
CELERY_RESULT_BACKEND = calib.CELERY_RESULT_BACKEND
TASK_RETRY_DELAY = calib.TASK_RETRY_DELAY
TASK_MAX_RETRIES = calib.TASK_MAX_RETRIES

celery_app = Celery(
    "proforma-backend",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.celery.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_default_retry_delay=TASK_RETRY_DELAY,
    task_max_retries=TASK_MAX_RETRIES,
)