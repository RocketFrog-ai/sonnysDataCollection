import os
import json
import certifi
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

# SQLAlchemy connection pool
engine = None
SessionLocal = None


def init_db():
    global engine, SessionLocal
    db_url = os.getenv("CAR_WASH_DB_URL")

    if not db_url:
        logger.warning("CAR_WASH_DB_URL not set. Caching will be disabled.")
        return False

    try:
        # PostgreSQL SSL config (Azure)
        connect_args = {
            "sslmode": "require",
            "sslrootcert": certifi.where()
        }

        engine = create_engine(
            db_url,
            pool_size=15,
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=1800,
            connect_args=connect_args
        )

        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

        logger.info("Initialized PostgreSQL caching connection pool.")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database cache: {e}")
        return False


def get_cached_classification(place_id: str) -> dict:
    if not SessionLocal:
        if not init_db():
            return None

    try:
        with engine.connect() as conn:
            query = text(
                "SELECT * FROM car_wash_classifications WHERE place_id = :place_id"
            )

            result = conn.execute(
                query, {"place_id": place_id}
            ).mappings().first()

            if result and result.get("ai_primary_type"):
                logger.info(f"CACHE HIT: Classification found for {place_id}")

                return {
                    "primary_type": result["ai_primary_type"],
                    "secondary_types": result["ai_secondary_types"] or [],
                    "confidence_score": result["ai_confidence_score"],
                    "found_packages": result["ai_found_packages"] or [],
                    "detected_markers": result["ai_detected_markers"] or [],
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
            logger.warning(
                "Attempted to cache classification without a place_id. Skipping."
            )
            return

        with engine.connect() as conn:
            query = text("""
                INSERT INTO car_wash_classifications (
                    place_id, name, address, website, google_maps_uri, rating, user_rating_count, primary_type_display_name,
                    ai_primary_type, ai_secondary_types, ai_confidence_score, ai_found_packages, ai_detected_markers, ai_reasoning
                ) VALUES (
                    :place_id, :name, :address, :website, :google_maps_uri, :rating, :user_rating_count, :primary_type_display_name,
                    :ai_primary_type, :ai_secondary_types, :ai_confidence_score, :ai_found_packages, :ai_detected_markers, :ai_reasoning
                )
                ON CONFLICT (place_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    address = EXCLUDED.address,
                    website = EXCLUDED.website,
                    rating = EXCLUDED.rating,
                    user_rating_count = EXCLUDED.user_rating_count,
                    ai_primary_type = EXCLUDED.ai_primary_type,
                    ai_secondary_types = EXCLUDED.ai_secondary_types,
                    ai_confidence_score = EXCLUDED.ai_confidence_score,
                    ai_found_packages = EXCLUDED.ai_found_packages,
                    ai_detected_markers = EXCLUDED.ai_detected_markers,
                    ai_reasoning = EXCLUDED.ai_reasoning,
                    updated_at = CURRENT_TIMESTAMP
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

                # ✅ Direct JSON (PostgreSQL JSONB handles it)
                "ai_primary_type": classification.get("primary_type"),
                "ai_secondary_types": classification.get("secondary_types", []),
                "ai_confidence_score": classification.get("confidence_score"),
                "ai_found_packages": classification.get("found_packages", []),
                "ai_detected_markers": classification.get("detected_markers", []),
                "ai_reasoning": classification.get("reasoning")
            }

            conn.execute(query, params)
            conn.commit()

            logger.info(f"CACHE SAVE: Classification cached for {place_id}")

    except Exception as e:
        logger.error(f"Cache save error for {comp_dict.get('name')}: {e}")