"""
Council-local web-search grounding — a self-contained copy of the insights layer's `websearch.py`, so the
council imports nothing from `app/`. **Best-effort: NEVER raises**, returns `[]` when no provider is
configured (the analysis silently falls back to un-grounded world knowledge).

Providers (auto: azure → tavily → duckduckgo; override with `WEB_SEARCH_BACKEND = azure|tavily|duckduckgo|off`):
  • azure      — the SAME Azure OpenAI resource (Responses API `web_search` tool, `POST {endpoint}/openai/v1/responses`).
                 Reuses `AZURE_OPENAI_*` (via `council.llm`); grounds on Bing, returns real url_citations.
  • tavily     — REST, requests-only. Set `TAVILY_API_KEY`.
  • duckduckgo — the `ddgs` package if installed (no key).

Feeds fresh sources (title / url / snippet) into the Local-Market + research prompts so the read cites real,
current facts, not static world knowledge. Web results are present-day → flagged 🌐, never leakage-safe.
Web search incurs Grounding-with-Bing tool costs and sends the query outside the Azure compliance boundary.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import List, Optional, Tuple

import requests

from experiments.council import llm
from experiments.council import places

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
AZURE_RESPONSES_PATH = "/openai/v1/responses"


# ─────────────────────────── provider selection ───────────────────────────
def _azure_available() -> bool:
    return bool(llm.ENDPOINT and llm.API_KEY and llm.DEPLOYMENT)


def _tavily_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def _ddgs_available() -> bool:
    return (importlib.util.find_spec("ddgs") is not None
            or importlib.util.find_spec("duckduckgo_search") is not None)


def active_provider() -> Optional[str]:
    be = os.getenv("WEB_SEARCH_BACKEND", "auto").strip().lower()
    if be == "azure":
        return "azure" if _azure_available() else None
    if be == "tavily":
        return "tavily" if _tavily_available() else None
    if be in ("duckduckgo", "ddg", "ddgs"):
        return "duckduckgo" if _ddgs_available() else None
    if be in ("off", "none", "disabled", "false", "0"):
        return None
    if _azure_available():
        return "azure"
    if _tavily_available():
        return "tavily"
    if _ddgs_available():
        return "duckduckgo"
    return None


def search_available() -> bool:
    return active_provider() is not None


# ─────────────────────────── the searches ───────────────────────────
def _azure_web_search(query: str, max_results: int) -> List[dict]:
    ep, key, dep = (llm.ENDPOINT or "").rstrip("/"), llm.API_KEY or "", llm.DEPLOYMENT or "gpt-4o"
    if not (ep and key):
        return []
    body = {"model": dep, "tools": [{"type": "web_search"}], "tool_choice": "auto",
            "include": ["web_search_call.action.sources"], "max_output_tokens": 700, "input": query}
    headers = {"api-key": key, "Content-Type": "application/json"}
    out = None
    for attempt in (1, 2):
        try:
            r = requests.post(f"{ep}{AZURE_RESPONSES_PATH}", headers=headers, json=body, timeout=120)
            r.raise_for_status()
            out = (r.json().get("output") or [])
            break
        except requests.exceptions.Timeout:
            logger.warning("Azure web_search timed out (attempt %d) for %r", attempt, query[:80])
            continue
        except Exception as e:
            logger.warning("Azure web_search failed for %r: %s", query[:80], e)
            return []
    if out is None:
        return []
    sources, seen = [], set()
    for it in out:
        if it.get("type") == "message":
            for c in (it.get("content") or []):
                for a in (c.get("annotations") or []):
                    u = a.get("url")
                    if a.get("type") == "url_citation" and u and u not in seen:
                        seen.add(u)
                        sources.append({"title": a.get("title") or u, "url": u, "content": ""})
    for it in out:
        if it.get("type") == "web_search_call":
            for s in ((it.get("action") or {}).get("sources") or []):
                u = s.get("url")
                if u and u not in seen:
                    seen.add(u)
                    sources.append({"title": u, "url": u, "content": ""})
    return sources[:max_results]


def _tavily_search(query: str, max_results: int) -> List[dict]:
    body = {"api_key": os.getenv("TAVILY_API_KEY", ""), "query": query, "max_results": max_results,
            "search_depth": "basic", "include_answer": False}
    try:
        r = requests.post(TAVILY_URL, json=body, timeout=15)
        r.raise_for_status()
        return [{"title": x.get("title") or "", "url": x.get("url") or "", "content": (x.get("content") or "").strip()}
                for x in (r.json().get("results") or []) if x.get("url")]
    except Exception as e:
        logger.warning("Tavily search failed for %r: %s", query, e)
        return []


def _ddg_search(query: str, max_results: int) -> List[dict]:
    try:
        try:
            from ddgs import DDGS
        except Exception:
            from duckduckgo_search import DDGS
        out: List[dict] = []
        with DDGS() as d:
            for x in d.text(query, max_results=max_results):
                u = x.get("href") or x.get("url")
                if u:
                    out.append({"title": x.get("title") or "", "url": u, "content": (x.get("body") or "").strip()})
        return out
    except Exception as e:
        logger.warning("DuckDuckGo search failed for %r: %s", query, e)
        return []


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """One search via the active provider → [{title, url, content}]. [] when unavailable / on error."""
    prov = active_provider()
    if prov == "azure":
        return _azure_web_search(query, max_results)
    if prov == "tavily":
        return _tavily_search(query, max_results)
    if prov == "duckduckgo":
        return _ddg_search(query, max_results)
    return []


def reverse_geocode(lat: float, lon: float) -> str:
    key = places.GOOGLE_MAPS_API_KEY or ""
    if not key:
        return ""
    try:
        js = requests.get(GEOCODE_URL, params={"latlng": f"{lat},{lon}", "key": key}, timeout=8).json()
        if js.get("status") == "OK" and js.get("results"):
            for res in js["results"]:
                if set(res.get("types") or []) & {"locality", "postal_town", "administrative_area_level_2"}:
                    return res.get("formatted_address") or ""
            return js["results"][0].get("formatted_address") or ""
    except Exception as e:
        logger.warning("Reverse geocode failed for (%s, %s): %s", lat, lon, e)
    return ""


def gather_location_sources(lat: float, lon: float, *, radius_km: float = 20.0, per_query: int = 4,
                            max_sources: int = 8) -> Tuple[List[dict], str]:
    """Run targeted market queries for this pin → (deduped sources, resolved place string). ([], place) if
    web search is unavailable. Azure Responses does its own multi-step search in one (Bing-billed) call."""
    place = reverse_geocode(lat, lon) or f"{lat:.4f}, {lon:.4f}"
    prov = active_provider()
    if prov is None:
        return [], place
    miles = f"{radius_km * 0.621:.0f}"
    if prov == "azure":
        research = (
            f"Search the web for current facts about {place} relevant to building a new EXPRESS-TUNNEL car "
            f"wash (trade area ~{miles} miles): population and household income, growth, traffic and major "
            f"roads, retail anchors, and existing express/tunnel car-wash competitors. Do NOT write a report — "
            f"run the searches and give a few bullet points, each with its source citation.")
        return web_search(research, max_results=max_sources), place
    queries = [
        f"{place} express tunnel car wash competitors",
        f"{place} demographics median household income population growth",
        f"{place} traffic counts major roads new retail development",
    ]
    seen, sources = set(), []
    for q in queries:
        for res in web_search(q, max_results=per_query):
            u = res.get("url")
            if u and u not in seen:
                seen.add(u)
                sources.append(res)
                if len(sources) >= max_sources:
                    return sources, place
    return sources, place
