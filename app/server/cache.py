"""`@cached(endpoint)` — the read-through wrapper the pin endpoints in router.py wear.

On every call it:
  1. resolves lat/lon ONCE (lat/lon as given, else geocode the address) and writes them back onto the
     request, so the wrapped handler's own resolve_lat_lon is a no-op — no double geocode;
  2. builds the cache key from the resolved, 6-dp-rounded lat/lon + the remaining request params
     (address dropped: two inputs that resolve to the same pin + params share one entry);
  3. returns the stored response on a hit; on a miss runs the handler, stores the JSON, returns it.

The cache is best-effort (app.db.pin_cache swallows DB errors), so wrapping a handler cannot break it.
A handler that raises (e.g. HTTPException 404/503) never reaches `store()`, so only real successes are
cached. FastAPI introspects the wrapped handler's signature via functools.wraps' __wrapped__, so the
request model + docs are unchanged."""
from __future__ import annotations

import functools
import logging

from fastapi.encoders import jsonable_encoder

from app.server import service

logger = logging.getLogger(__name__)

# The DB layer (sqlalchemy + a MySQL driver) is an OPTIONAL dependency. If it can't be imported — the
# packages aren't installed, or CAR_WASH_DB_URL points at a broken build — the cache must simply not
# exist: `cached` becomes an identity decorator and every endpoint behaves exactly as it did before
# caching. This keeps a missing driver from taking down the whole API on import.
try:
    from app.db import pin_cache
    _CACHE_IMPORTABLE = True
except Exception:  # noqa: BLE001 -- ImportError today, but never let anything here break app import
    pin_cache = None
    _CACHE_IMPORTABLE = False
    logger.warning("pin-cache: db layer unavailable (import failed); endpoints will run uncached",
                   exc_info=True)

_LATLON_DP = 6  # ~0.11 m; the granularity at which two pins count as "the same" for the cache


def _params_for_key(req, lat: float, lon: float) -> dict:
    """The request as a plain dict, address dropped, with the resolved lat/lon (rounded) substituted in
    — this is exactly what is hashed into the key and stored in request_params."""
    params = req.model_dump()
    params.pop("address", None)
    params["latitude"] = round(float(lat), _LATLON_DP)
    params["longitude"] = round(float(lon), _LATLON_DP)
    return params


def cached(endpoint: str):
    """Decorate a pin-driven route handler `fn(req)` with a DB read-through cache under `endpoint`.
    If the db layer isn't importable, returns `fn` untouched (endpoints stay exactly as before)."""
    def decorator(fn):
        if not _CACHE_IMPORTABLE:
            return fn  # identity: no lat/lon pre-resolve, no re-encode — behavior-identical to pre-cache

        @functools.wraps(fn)  # keeps fn's signature so FastAPI still parses the request model
        def wrapper(req):
            # resolve_lat_lon raises HTTPException(400) if neither lat/lon nor a geocodable address is
            # given — let that propagate unchanged (same 400 the handler would have produced).
            lat, lon = service.resolve_lat_lon(
                getattr(req, "latitude", None), getattr(req, "longitude", None), getattr(req, "address", None)
            )
            req.latitude, req.longitude = lat, lon  # request-local; handler now skips re-geocoding

            params = _params_for_key(req, lat, lon)
            hit = pin_cache.lookup(endpoint, params)
            if hit is not None:
                return hit

            result = fn(req)  # raises on error → nothing cached
            encoded = jsonable_encoder(result)  # same shape FastAPI would send; safe to re-serialize
            pin_cache.store(endpoint, round(float(lat), _LATLON_DP), round(float(lon), _LATLON_DP), params, encoded)
            return encoded

        return wrapper

    return decorator
