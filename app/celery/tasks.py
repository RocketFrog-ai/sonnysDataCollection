import logging
from app.utils import common as calib
from app.celery.celery_app import celery_app
from app.ai.analysis import analyze_site_from_dict

TASK_RETRY_DELAY = calib.TASK_RETRY_DELAY
TASK_MAX_RETRIES = calib.TASK_MAX_RETRIES
EXTERNAL_SERVICE_URL = calib.EXTERNAL_SERVICE_URL
EXTERNAL_SERVICE_TIMEOUT = calib.EXTERNAL_SERVICE_TIMEOUT

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def analyse_site(
    self, 
    address : str = "1"
    ):
    res = analyze_site_from_dict(address)
    return res