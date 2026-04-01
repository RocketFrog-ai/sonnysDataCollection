import os
import json
import ssl
import certifi
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# SQLAlchemy connection pool
engine = None
SessionLocal = None
_site_fetch_table_ready = False
_site_response_table_ready = False

def init_db():
    global engine, SessionLocal
    db_url = os.getenv("CAR_WASH_DB_URL")
    if not db_url:
        logger.warning("CAR_WASH_DB_URL not set. Caching will be disabled.")
        return False
        
    try:
        # Create a connection pool optimized for threading
        # Azure requires secure transport. Match DBeaver: Require SSL = True, Verify Cert = False
        connect_args = {
            "ssl": {
                "check_hostname": False
            }
        }

        engine = create_engine(
            db_url,
            pool_size=15, 
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=1800,
            connect_args=connect_args
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Initialized Azure MySQL caching connection pool.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database cache: {e}")
        return False


def _ensure_site_fetch_cache_table():
    global _site_fetch_table_ready
    if _site_fetch_table_ready:
        return True
    if not SessionLocal and not init_db():
        return False
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS site_analysis_fetch_cache (
                        address_key VARCHAR(64) PRIMARY KEY,
                        address_input TEXT NULL,
                        normalized_address TEXT NULL,
                        lat DOUBLE NOT NULL,
                        lon DOUBLE NOT NULL,
                        fetched_json LONGTEXT NOT NULL,
                        cache_version VARCHAR(32) NOT NULL DEFAULT 'v1',
                        expires_at DATETIME NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_site_analysis_fetch_cache_expires_at (expires_at)
                    )
                    """
                )
            )
            conn.commit()
        _site_fetch_table_ready = True
        return True
    except Exception as e:
        logger.error("Failed creating site_analysis_fetch_cache table: %s", e)
        return False


def _ensure_site_response_cache_table():
    global _site_response_table_ready
    if _site_response_table_ready:
        return True
    if not SessionLocal and not init_db():
        return False
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS site_analysis_response_cache (
                        address_key VARCHAR(64) PRIMARY KEY,
                        address_input TEXT NULL,
                        normalized_address TEXT NULL,
                        lat DOUBLE NOT NULL,
                        lon DOUBLE NOT NULL,
                        response_json LONGTEXT NOT NULL,
                        cache_version VARCHAR(32) NOT NULL DEFAULT 'v1',
                        expires_at DATETIME NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_site_analysis_response_cache_lat_lon (lat, lon),
                        INDEX idx_site_analysis_response_cache_expires_at (expires_at)
                    )
                    """
                )
            )
            conn.commit()
        _site_response_table_ready = True
        return True
    except Exception as e:
        logger.error("Failed creating site_analysis_response_cache table: %s", e)
        return False


def get_cached_site_fetch(address_key: str):
    if not address_key:
        return None
    if not _ensure_site_fetch_cache_table():
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT lat, lon, fetched_json, cache_version, expires_at, updated_at
                    FROM site_analysis_fetch_cache
                    WHERE address_key = :address_key
                    """
                ),
                {"address_key": address_key},
            ).mappings().first()
            if not row:
                return None
            expires_at = row.get("expires_at")
            if expires_at:
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                if expires_at < now_utc:
                    return None
            fetched_json = row.get("fetched_json")
            if not fetched_json:
                return None
            return {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "fetched": json.loads(fetched_json),
                "cache_version": row.get("cache_version"),
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
    except Exception as e:
        logger.error("Site cache read error for key=%s: %s", address_key, e)
        return None


def save_site_fetch_cache(
    *,
    address_key: str,
    address_input: Optional[str],
    normalized_address: Optional[str],
    lat: float,
    lon: float,
    fetched: dict,
    cache_version: str = "v1",
    ttl_days: Optional[int] = 30,
):
    if not address_key or fetched is None:
        return False
    if not _ensure_site_fetch_cache_table():
        return False
    expires_at = None
    if ttl_days is not None and ttl_days > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(days=int(ttl_days))
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO site_analysis_fetch_cache
                        (address_key, address_input, normalized_address, lat, lon, fetched_json, cache_version, expires_at)
                    VALUES
                        (:address_key, :address_input, :normalized_address, :lat, :lon, :fetched_json, :cache_version, :expires_at)
                    ON DUPLICATE KEY UPDATE
                        address_input = VALUES(address_input),
                        normalized_address = VALUES(normalized_address),
                        lat = VALUES(lat),
                        lon = VALUES(lon),
                        fetched_json = VALUES(fetched_json),
                        cache_version = VALUES(cache_version),
                        expires_at = VALUES(expires_at),
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "address_key": address_key,
                    "address_input": address_input,
                    "normalized_address": normalized_address,
                    "lat": float(lat),
                    "lon": float(lon),
                    "fetched_json": json.dumps(fetched),
                    "cache_version": cache_version,
                    "expires_at": expires_at,
                },
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error("Site cache save error for key=%s: %s", address_key, e)
        return False


def get_cached_site_analysis_by_latlon(lat: float, lon: float, tolerance: float = 0.5):
    if not _ensure_site_response_cache_table():
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        lat, lon, response_json, cache_version, expires_at, updated_at,
                        ((lat - :lat) * (lat - :lat) + (lon - :lon) * (lon - :lon)) AS distance_score
                    FROM site_analysis_response_cache
                    WHERE ABS(lat - :lat) <= :tolerance
                      AND ABS(lon - :lon) <= :tolerance
                    ORDER BY distance_score ASC, updated_at DESC
                    LIMIT 1
                    """
                ),
                {"lat": float(lat), "lon": float(lon), "tolerance": float(tolerance)},
            ).mappings().first()
            if not row:
                return None
            expires_at = row.get("expires_at")
            if expires_at:
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                if expires_at < now_utc:
                    return None
            response_json = row.get("response_json")
            if not response_json:
                return None
            return {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "response": json.loads(response_json),
                "cache_version": row.get("cache_version"),
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
    except Exception as e:
        logger.error("Site response cache read error at (%.6f, %.6f): %s", lat, lon, e)
        return None


def save_site_analysis_response(
    *,
    address_key: str,
    address_input: Optional[str],
    normalized_address: Optional[str],
    lat: float,
    lon: float,
    response: dict,
    cache_version: str = "v1",
    ttl_days: Optional[int] = 30,
):
    if not address_key or response is None:
        return False
    if not _ensure_site_response_cache_table():
        return False
    expires_at = None
    if ttl_days is not None and ttl_days > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(days=int(ttl_days))
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO site_analysis_response_cache
                        (address_key, address_input, normalized_address, lat, lon, response_json, cache_version, expires_at)
                    VALUES
                        (:address_key, :address_input, :normalized_address, :lat, :lon, :response_json, :cache_version, :expires_at)
                    ON DUPLICATE KEY UPDATE
                        address_input = VALUES(address_input),
                        normalized_address = VALUES(normalized_address),
                        lat = VALUES(lat),
                        lon = VALUES(lon),
                        response_json = VALUES(response_json),
                        cache_version = VALUES(cache_version),
                        expires_at = VALUES(expires_at),
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "address_key": address_key,
                    "address_input": address_input,
                    "normalized_address": normalized_address,
                    "lat": float(lat),
                    "lon": float(lon),
                    "response_json": json.dumps(response),
                    "cache_version": cache_version,
                    "expires_at": expires_at,
                },
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error("Site response cache save error for key=%s: %s", address_key, e)
        return False


def get_all_site_analysis_cache(page: int = 1, page_size: int = 50, include_response: bool = True):
    if not _ensure_site_response_cache_table():
        return {"total": 0, "page": page, "page_size": page_size, "rows": []}
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 1
    if page_size > 200:
        page_size = 200
    offset = (page - 1) * page_size
    try:
        with engine.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM site_analysis_response_cache")
            ).scalar()
            rows = conn.execute(
                text(
                    """
                    SELECT address_key, normalized_address, lat, lon, cache_version, updated_at, response_json
                    FROM site_analysis_response_cache
                    ORDER BY updated_at DESC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                {"limit": int(page_size), "offset": int(offset)},
            ).mappings().all()
            out = []
            for r in rows:
                response = None
                if include_response and r.get("response_json"):
                    try:
                        response = json.loads(r.get("response_json"))
                    except Exception:
                        response = None
                out.append(
                    {
                        "address_key": r.get("address_key"),
                        "normalized_address": r.get("normalized_address"),
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "cache_version": r.get("cache_version"),
                        "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None,
                        "response": response,
                    }
                )
            return {
                "total": int(total or 0),
                "page": int(page),
                "page_size": int(page_size),
                "rows": out,
            }
    except Exception as e:
        logger.error("Site response cache list error: %s", e)
        return {"total": 0, "page": page, "page_size": page_size, "rows": []}

def get_cached_classification(place_id: str) -> dict:
    if not SessionLocal:
        if not init_db():
            return None
            
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM car_wash_classifications WHERE place_id = :place_id")
            result = conn.execute(query, {"place_id": place_id}).mappings().first()
            
            if result and result.get("ai_primary_type"):
                logger.info(f"CACHE HIT: Classification found for {place_id}")
                return {
                    "primary_type": result["ai_primary_type"],
                    "secondary_types": json.loads(result["ai_secondary_types"]) if result["ai_secondary_types"] else [],
                    "confidence_score": result["ai_confidence_score"],
                    "found_packages": json.loads(result["ai_found_packages"]) if result["ai_found_packages"] else [],
                    "detected_markers": json.loads(result["ai_detected_markers"]) if result["ai_detected_markers"] else [],
                    "reasoning": result["ai_reasoning"]
                }
    except Exception as e:
        logger.error(f"Cache read error for {place_id}: {e}")
        
    return None

def save_classification(comp_dict: dict, classification: dict):
    if not SessionLocal:
        if not init_db():
            return
            
    try:
        place_id = comp_dict.get("place_id")
        if not place_id:
            logger.warning("Attempted to cache classification without a place_id. Skipping.")
            return

        with engine.connect() as conn:
            query = text("""
                INSERT INTO car_wash_classifications (
                    place_id, name, address, website, google_maps_uri, rating, user_rating_count, primary_type_display_name,
                    ai_primary_type, ai_secondary_types, ai_confidence_score, ai_found_packages, ai_detected_markers, ai_reasoning
                ) VALUES (
                    :place_id, :name, :address, :website, :google_maps_uri, :rating, :user_rating_count, :primary_type_display_name,
                    :ai_primary_type, :ai_secondary_types, :ai_confidence_score, :ai_found_packages, :ai_detected_markers, :ai_reasoning
                ) ON DUPLICATE KEY UPDATE
                    name=VALUES(name),
                    address=VALUES(address),
                    website=VALUES(website),
                    rating=VALUES(rating),
                    user_rating_count=VALUES(user_rating_count),
                    ai_primary_type=VALUES(ai_primary_type),
                    ai_secondary_types=VALUES(ai_secondary_types),
                    ai_confidence_score=VALUES(ai_confidence_score),
                    ai_found_packages=VALUES(ai_found_packages),
                    ai_detected_markers=VALUES(ai_detected_markers),
                    ai_reasoning=VALUES(ai_reasoning),
                    updated_at=CURRENT_TIMESTAMP
            """)
            
            params = {
                "place_id": place_id,
                "name": comp_dict.get("name"),
                "address": comp_dict.get("address"),
                "website": comp_dict.get("website"),
                "google_maps_uri": comp_dict.get("google_maps_uri"),
                "rating": comp_dict.get("rating"),
                "user_rating_count": comp_dict.get("user_rating_count"),
                "primary_type_display_name": comp_dict.get("primary_type_display_name"),
                
                "ai_primary_type": classification.get("primary_type"),
                "ai_secondary_types": json.dumps(classification.get("secondary_types", [])),
                "ai_confidence_score": classification.get("confidence_score"),
                "ai_found_packages": json.dumps(classification.get("found_packages", [])),
                "ai_detected_markers": json.dumps(classification.get("detected_markers", [])),
                "ai_reasoning": classification.get("reasoning")
            }
            
            conn.execute(query, params)
            conn.commit()
            logger.info(f"CACHE SAVE: Classification cached for {place_id}")
            
    except Exception as e:
        logger.error(f"Cache save error for {comp_dict.get('name')}: {e}")