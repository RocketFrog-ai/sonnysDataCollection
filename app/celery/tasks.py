import logging
from app.utils import common as calib
from app.celery.celery_app import celery_app


TASK_RETRY_DELAY = calib.TASK_RETRY_DELAY
TASK_MAX_RETRIES = calib.TASK_MAX_RETRIES
EXTERNAL_SERVICE_URL = calib.EXTERNAL_SERVICE_URL
EXTERNAL_SERVICE_TIMEOUT = calib.EXTERNAL_SERVICE_TIMEOUT


logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def test_task(
    self, 
    param_1 : str = "1"
    ):
    #res = process_func(param_1, self.request.id)
    return None