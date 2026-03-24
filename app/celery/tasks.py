import logging

from app.utils import common as calib
from app.celery.celery_app import celery_app
from app.modelling.site_analysis import run_site_analysis

TASK_RETRY_DELAY = calib.TASK_RETRY_DELAY
TASK_MAX_RETRIES = calib.TASK_MAX_RETRIES
EXTERNAL_SERVICE_URL = calib.EXTERNAL_SERVICE_URL
EXTERNAL_SERVICE_TIMEOUT = calib.EXTERNAL_SERVICE_TIMEOUT

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def analyse_site(self, address: str = "1", tunnel_count: int = None, carwash_type_encoded: int = None, tier_strategy: str = "4-class-90pct-custom"):
    """
    Run full site analysis: fetch → quantile → narratives. Returns only when all stages complete.
    tunnel_count (1–4): actual tunnel count for the site; improves quantile prediction.
    carwash_type_encoded (1–3): 1=Express, 2=Mobile, 3=Hand Wash; improves quantile prediction.
    tier_strategy: e.g. '4-class-wide-middle', '3-class-bottom-heavy'.
    """
    task_id = self.request.id
    logger.info(
        "analyse_site task started: task_id=%s address=%s tunnel_count=%s carwash_type=%s strategy=%s",
        task_id, address, tunnel_count, carwash_type_encoded, tier_strategy,
    )

    res = run_site_analysis(
        address,
        tunnel_count=tunnel_count,
        carwash_type_encoded=carwash_type_encoded,
        tier_strategy=tier_strategy,
        run_quantile=True,
        run_narratives=True,
    )
    logger.info("analyse_site task finished: task_id=%s", task_id)
    return res
