from __future__ import annotations

import json
import logging
from typing import Any, Dict

from sqlalchemy import text

from app.site_analysis.server.db_cache import get_car_wash_engine

logger = logging.getLogger(__name__)

_central_input_table_ready = False

# --- PostgreSQL (current; CAR_WASH_DB_URL is postgresql+psycopg2://...) ---
_CREATE_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS pnl_central_input_form (
        id BIGSERIAL PRIMARY KEY,
        task_id VARCHAR(128) NOT NULL,
        input_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_pnl_central_input_form_task_id UNIQUE (task_id)
    )
    """
)

_CREATE_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS idx_pnl_central_input_form_created_at
    ON pnl_central_input_form (created_at)
    """
)

_INSERT_SQL = text(
    """
    INSERT INTO pnl_central_input_form (task_id, input_json)
    VALUES (:task_id, CAST(:input_json AS jsonb))
    ON CONFLICT (task_id) DO UPDATE SET
        input_json = EXCLUDED.input_json
    """
)

# --- MySQL (re-enable when CAR_WASH_DB_URL points at MySQL again) ---
# _CREATE_TABLE_SQL = text(
#     """
#     CREATE TABLE IF NOT EXISTS pnl_central_input_form (
#         id BIGINT AUTO_INCREMENT PRIMARY KEY,
#         task_id VARCHAR(128) NOT NULL,
#         input_json LONGTEXT NOT NULL,
#         created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
#         UNIQUE KEY uq_pnl_central_input_form_task_id (task_id),
#         KEY idx_pnl_central_input_form_created_at (created_at)
#     )
#     """
# )
#
# _INSERT_SQL = text(
#     """
#     INSERT INTO pnl_central_input_form (task_id, input_json)
#     VALUES (:task_id, :input_json)
#     ON DUPLICATE KEY UPDATE
#         input_json = VALUES(input_json),
#         created_at = created_at
#     """
# )


def save_central_input_form_submission(*, task_id: str, payload: Dict[str, Any]) -> bool:
    """Persist the full central input-form JSON keyed by Celery task id."""
    if not task_id or not isinstance(payload, dict):
        return False
    global _central_input_table_ready
    eng = get_car_wash_engine()
    if eng is None:
        logger.warning("CAR_WASH_DB_URL not set or DB init failed; central input form not persisted.")
        return False
    try:
        with eng.connect() as conn:
            if not _central_input_table_ready:
                conn.execute(_CREATE_TABLE_SQL)
                conn.execute(_CREATE_INDEX_SQL)
                conn.commit()
                _central_input_table_ready = True
            conn.execute(
                _INSERT_SQL,
                {"task_id": task_id, "input_json": json.dumps(payload, default=str)},
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error("Central input form DB save failed task_id=%s: %s", task_id, e)
        return False
