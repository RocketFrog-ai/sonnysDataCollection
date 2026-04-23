import logging
import concurrent.futures
from typing import List, Dict, Any
from type_car_wash.scraper import scrape_site
from type_car_wash.analyzer import classify_car_wash_with_ai
from type_car_wash.finder import find_official_website
from app.site_analysis.server.db_cache import get_cached_classification, save_classification

logger = logging.getLogger(__name__)

def _process_single_competitor(comp_dict: Dict[str, Any]) -> Dict[str, Any]:
    comp_dict["classification"] = None
    
    name = comp_dict.get("name")
    if not name:
        return comp_dict
        
    # 1. Quick Cache Check
    place_id = comp_dict.get("place_id")
    if place_id:
        cached_data = get_cached_classification(place_id)
        if cached_data:
            comp_dict["classification"] = cached_data
            return comp_dict

    url = comp_dict.get("website")
    if not url:
        address = comp_dict.get("address")
        found_url, confidence = find_official_website(name, address=address)
        if found_url:
            url = found_url
            # Write the discovered URL back into the dict so routes.py
            # can surface it as official_website (even when no Place Details URL existed)
            comp_dict["website"] = found_url
            logger.info(f"Found fallback URL for {name} ({address}): {url}")
        else:
            logger.info(f"Could not find URL for {name} ({address})")
            comp_dict["classification_error"] = "No fallback URL found."
            
    if url:
        # Ensure URL formatting
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            text = scrape_site(url)
            if text:
                classification = classify_car_wash_with_ai(text)
                if classification:
                    comp_dict["classification"] = classification.model_dump() if hasattr(classification, "model_dump") else classification.dict()
                    # 2. Save new classification to cache
                    save_classification(comp_dict, comp_dict["classification"])
                else:
                    comp_dict["classification_error"] = "AI classification returned None (API Key may be invalid or rate limited)."
            else:
                comp_dict["classification_error"] = "Failed to scrape any text from website."
        except Exception as e:
            logger.error(f"Error classifying {name} at {url}: {e}")
            comp_dict["classification_error"] = str(e)
            
    return comp_dict

def classify_competitors(competitors: List[Dict[str, Any]], max_classifications: int = 20) -> List[Dict[str, Any]]:
    # Elements to process via AI
    to_process = competitors[:max_classifications]
    # Elements to skip immediately
    to_skip = competitors[max_classifications:]
    
    results = []
    
    # Process up to max_classifications concurrently
    if to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(to_process), max_classifications)) as executor:
            # Map returns the sequence in exactly the same order as to_process
            processed_results = list(executor.map(_process_single_competitor, to_process))
            results.extend(processed_results)
            
    # Process the skipped elements directly
    for comp_dict in to_skip:
        comp_dict["classification"] = None
        comp_dict["classification_error"] = "Skipped to optimize API latency."
        results.append(comp_dict)
        
    return results
