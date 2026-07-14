"""The pin read-through cache table + its accessors.

One table, `pin_api_cache` (name overridable via PIN_CACHE_TABLE), in the Azure MySQL
`proforma_schema` DB. Each row is one API search:

    cache_key       sha256(endpoint + canonical(request params incl. resolved lat/lon))  -- UNIQUE
    endpoint        which /v1/pnl_analysis/* route produced it (e.g. "pinpoint-forecast")
    latitude,       the RESOLVED pin, rounded to 6 dp and broken out into indexed columns so you can
    longitude       query "everything ever searched near here" independent of the JSON blob
    request_params  the full request, as-is                                    (MySQL JSON)
    response        the full response JSON, as-is                              (MySQL JSON)
    hit_count       how many times this row was served from cache
    created_at / updated_at / last_read_at / expires_at   bookkeeping + TTL

Read path: `lookup()` returns the stored response for an identical, unexpired search, else None.
Write path: `store()` upserts the freshly-computed response and (re)stamps expires_at = now + TTL.

MySQL has no JSONB (that is PostgreSQL); its `JSON` type is the binary-stored equivalent and is what
we use. Every function here is best-effort — any DB error is logged and swallowed so the API keeps
serving live results (see `app.db.engine`)."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Double,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.core import common as calib
from app.db.engine import get_engine

logger = logging.getLogger(__name__)

_TABLE = calib.PIN_CACHE_TABLE
metadata = MetaData()

pin_api_cache = Table(
    _TABLE,
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("cache_key", String(64), nullable=False, unique=True),
    Column("endpoint", String(64), nullable=False, index=True),
    Column("latitude", Double, nullable=False),
    Column("longitude", Double, nullable=False),
    Column("request_params", JSON, nullable=False),
    Column("response", JSON, nullable=False),
    Column("hit_count", Integer, nullable=False, server_default=text("0")),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("last_read_at", DateTime, nullable=True),
    Column("expires_at", DateTime, nullable=False, index=True),
    Index(f"ix_{_TABLE}_lat_lon", "latitude", "longitude"),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


# ─────────────────────────── keys ───────────────────────────
def _canonical(params: dict) -> str:
    """Stable JSON for the key: sorted keys, no whitespace, non-JSON types coerced via str."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def make_key(endpoint: str, params: dict) -> str:
    """sha256 over the endpoint + canonicalised params. `params` already carries the resolved,
    6-dp-rounded lat/lon (added by the caller), so an identical search maps to an identical key."""
    return hashlib.sha256(f"{endpoint}\n{_canonical(params)}".encode("utf-8")).hexdigest()


def _expires_expr():
    """SQL for the expiry stamp, computed on the DB clock: NOW() + INTERVAL <ttl> DAY.
    ttl is int-coerced, so the interpolation is injection-safe."""
    ttl = max(0, int(calib.PIN_CACHE_TTL_DAYS))
    return text(f"NOW() + INTERVAL {ttl} DAY")


def is_enabled() -> bool:
    return bool(calib.PIN_CACHE_ENABLED and get_engine() is not None)


# ─────────────────────────── read / write ───────────────────────────
def lookup(endpoint: str, params: dict) -> Optional[dict]:
    """Return the cached response for this exact, unexpired search, or None (miss/disabled/error).
    Bumps hit_count + last_read_at on a hit. Never raises."""
    if not calib.PIN_CACHE_ENABLED:
        return None
    eng = get_engine()
    if eng is None:
        return None
    key = make_key(endpoint, params)
    try:
        with eng.begin() as conn:
            row = conn.execute(
                select(pin_api_cache.c.id, pin_api_cache.c.response)
                .where(pin_api_cache.c.cache_key == key)
                .where(pin_api_cache.c.expires_at > func.now())  # TTL enforced on the DB clock
            ).first()
            if row is None:
                return None
            conn.execute(
                update(pin_api_cache)
                .where(pin_api_cache.c.id == row.id)
                .values(hit_count=pin_api_cache.c.hit_count + 1, last_read_at=func.now())
            )
            return row.response  # MySQL JSON → already a decoded dict/list
    except Exception:
        logger.warning("pin-cache lookup failed for %s; serving live", endpoint, exc_info=True)
        return None


def store(endpoint: str, latitude: float, longitude: float, params: dict, response) -> None:
    """Upsert the freshly-computed response and (re)stamp expiry. Keyed on cache_key, so a repeat
    search overwrites in place. Best-effort; never raises."""
    if not calib.PIN_CACHE_ENABLED:
        return
    eng = get_engine()
    if eng is None:
        return
    key = make_key(endpoint, params)
    expires = _expires_expr()
    try:
        with eng.begin() as conn:
            stmt = mysql_insert(pin_api_cache).values(
                cache_key=key,
                endpoint=endpoint,
                latitude=float(latitude),
                longitude=float(longitude),
                request_params=params,
                response=response,
                expires_at=expires,
            )
            stmt = stmt.on_duplicate_key_update(
                endpoint=stmt.inserted.endpoint,
                latitude=stmt.inserted.latitude,
                longitude=stmt.inserted.longitude,
                request_params=stmt.inserted.request_params,
                response=stmt.inserted.response,
                updated_at=func.now(),
                expires_at=expires,  # a recompute refreshes the TTL window; hit_count is preserved
            )
            conn.execute(stmt)
    except Exception:
        logger.warning("pin-cache store failed for %s; continuing", endpoint, exc_info=True)


# ─────────────────────────── maintenance ───────────────────────────
def init(create: bool = True, prune: bool = False) -> bool:
    """Create the table if absent (idempotent CREATE TABLE IF NOT EXISTS) and optionally prune expired
    rows. Returns True if the DB is reachable and ready. Never raises."""
    eng = get_engine()
    if eng is None:
        return False
    try:
        if create:
            metadata.create_all(eng, checkfirst=True)
        if prune:
            prune_expired()
        return True
    except Exception:
        logger.warning("pin-cache init failed; caching disabled for now", exc_info=True)
        return False


def prune_expired() -> int:
    """Delete rows past their TTL. Returns the number removed (0 on any error)."""
    eng = get_engine()
    if eng is None:
        return 0
    try:
        with eng.begin() as conn:
            res = conn.execute(delete(pin_api_cache).where(pin_api_cache.c.expires_at <= func.now()))
            return res.rowcount or 0
    except Exception:
        logger.warning("pin-cache prune failed", exc_info=True)
        return 0


def clear(endpoint: Optional[str] = None) -> int:
    """Delete all cached rows (or just one endpoint's). Use after a model/data retrain. Returns count."""
    eng = get_engine()
    if eng is None:
        return 0
    try:
        stmt = delete(pin_api_cache)
        if endpoint:
            stmt = stmt.where(pin_api_cache.c.endpoint == endpoint)
        with eng.begin() as conn:
            return conn.execute(stmt).rowcount or 0
    except Exception:
        logger.warning("pin-cache clear failed", exc_info=True)
        return 0


def stats() -> dict:
    """Row counts (total / live / expired), rows + hits per endpoint, total cache hits served."""
    eng = get_engine()
    if eng is None:
        return {"enabled": False, "reason": "no DB engine (CAR_WASH_DB_URL unset or unreachable)"}
    try:
        with eng.begin() as conn:
            total = conn.execute(select(func.count()).select_from(pin_api_cache)).scalar() or 0
            live = conn.execute(
                select(func.count()).select_from(pin_api_cache).where(pin_api_cache.c.expires_at > func.now())
            ).scalar() or 0
            hits = conn.execute(select(func.coalesce(func.sum(pin_api_cache.c.hit_count), 0))).scalar() or 0
            per_ep = conn.execute(
                select(
                    pin_api_cache.c.endpoint,
                    func.count().label("rows"),
                    func.coalesce(func.sum(pin_api_cache.c.hit_count), 0).label("hits"),
                ).group_by(pin_api_cache.c.endpoint).order_by(func.count().desc())
            ).all()
        return {
            "enabled": bool(calib.PIN_CACHE_ENABLED),
            "table": _TABLE,
            "ttl_days": int(calib.PIN_CACHE_TTL_DAYS),
            "rows_total": int(total),
            "rows_live": int(live),
            "rows_expired": int(total) - int(live),
            "hits_served": int(hits),
            "by_endpoint": [{"endpoint": e, "rows": int(r), "hits": int(h)} for e, r, h in per_ep],
        }
    except Exception:
        logger.warning("pin-cache stats failed", exc_info=True)
        return {"enabled": bool(calib.PIN_CACHE_ENABLED), "error": "stats query failed (see logs)"}
