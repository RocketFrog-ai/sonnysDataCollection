"""Lazy, process-wide SQLAlchemy engine for the Azure MySQL `proforma_schema` database.

Built ONCE on first use and reused (the engine owns a small connection pool). Everything is
best-effort: if CAR_WASH_DB_URL is empty or the engine cannot be constructed, `get_engine()` returns
None and every caller falls back to computing live — the API must never fail because the cache is
down. Azure enforces TLS, so we hand pymysql an SSL context by default (see `_ssl_connect_args`).
"""
from __future__ import annotations

import logging
import ssl
import threading
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core import common as calib

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_lock = threading.Lock()
_init_failed = False  # remember a hard failure so we don't retry-storm engine creation on every call


def _ssl_connect_args() -> dict:
    """SSL args for pymysql. Azure MySQL requires TLS; without a pinned CA we encrypt but skip
    identity verification (set CAR_WASH_DB_SSL_CA to a CA .pem to verify). CAR_WASH_DB_SSL=0 disables."""
    if not calib.CAR_WASH_DB_SSL:
        return {}
    ca = calib.CAR_WASH_DB_SSL_CA or None
    ctx = ssl.create_default_context(cafile=ca)
    if ca is None:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return {"ssl": ctx}


def get_engine() -> Optional[Engine]:
    """The shared engine, or None if the DB is not configured / could not be built."""
    global _engine, _init_failed
    if _engine is not None:
        return _engine
    if _init_failed or not calib.CAR_WASH_DB_URL:
        return None
    with _lock:
        if _engine is not None:
            return _engine
        if _init_failed:
            return None
        try:
            _engine = create_engine(
                calib.CAR_WASH_DB_URL,
                pool_pre_ping=True,   # Azure silently drops idle connections; revalidate before use
                pool_recycle=1800,
                pool_size=5,
                max_overflow=5,
                pool_timeout=10,
                connect_args={"connect_timeout": 10, **_ssl_connect_args()},
                future=True,
            )
            logger.info("pin-cache: DB engine ready (%s)", _engine.url.render_as_string(hide_password=True))
            return _engine
        except Exception:
            logger.warning("pin-cache: could not build DB engine; caching disabled", exc_info=True)
            _init_failed = True
            return None


def reset_engine() -> None:
    """Dispose the pool and clear the failure latch (tests / after a config change)."""
    global _engine, _init_failed
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _init_failed = False
