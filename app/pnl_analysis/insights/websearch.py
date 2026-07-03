"""
Optional web-search grounding for the location-only market analysis.

Gives the market read a set of freshly-retrieved sources (title / url / snippet) so the commentary
can cite real links instead of leaning only on the model's static world knowledge — a source of truth
the reader can click through to. Deliberately best-effort and dependency-light: it NEVER raises and
returns [] whenever no provider is configured, so the market analysis silently falls back to its old
un-grounded behaviour.

Providers (first available wins; override with WEB_SEARCH_BACKEND = azure | tavily | duckduckgo | off | auto):
  • azure       — the SAME Azure OpenAI resource used for the summaries, via the Responses API `web_search`
                  tool (POST {endpoint}/openai/v1/responses). No extra key or package — reuses AZURE_OPENAI_*.
                  Grounds on Bing and returns real url_citations. The default when Azure is configured.
  • tavily      — REST, requests-only (no extra dependency). Set TAVILY_API_KEY.
  • duckduckgo  — the `ddgs` package if installed (no key needed).  pip install ddgs
  • none        — returns []; the market analysis runs on world knowledge alone.

Note: Azure web search is on the RESPONSES API, not chat completions — chat completions (what llm.py uses for
the actual write-up) has no web-search tool, so we retrieve sources here and feed them into that prompt.
Web search incurs Grounding-with-Bing tool-call costs and sends the query outside the Azure compliance boundary.

Secrets are read from env only.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import List, Optional, Tuple

import requests

from app.utils import common as calib

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
AZURE_RESPONSES_PATH = "/openai/v1/responses"     # the Azure "v1" surface — no api-version query needed


# ─────────────────────────── provider selection ───────────────────────────
def _azure_available() -> bool:
    return bool(calib.AZURE_OPENAI_ENDPOINT and calib.AZURE_OPENAI_API_KEY
                and calib.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME)


def _tavily_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def _ddgs_available() -> bool:
    return (importlib.util.find_spec("ddgs") is not None
            or importlib.util.find_spec("duckduckgo_search") is not None)


def active_provider() -> Optional[str]:
    """Which web-search provider will actually be used (or None). Honours WEB_SEARCH_BACKEND."""
    be = os.getenv("WEB_SEARCH_BACKEND", "auto").strip().lower()
    if be == "azure":
        return "azure" if _azure_available() else None
    if be == "tavily":
        return "tavily" if _tavily_available() else None
    if be in ("duckduckgo", "ddg", "ddgs"):
        return "duckduckgo" if _ddgs_available() else None
    if be in ("off", "none", "disabled", "false", "0"):
        return None
    # auto: prefer Azure (native, no extra key/package), then Tavily, then DuckDuckGo
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
    """Run the query through the Azure OpenAI Responses API `web_search` tool → [{title, url, content}].
    Uses the same AZURE_OPENAI_* creds as the summaries; extracts url_citation annotations (which carry
    titles) and any action.sources. [] on error — never raises."""
    ep = (calib.AZURE_OPENAI_ENDPOINT or "").rstrip("/")
    key = calib.AZURE_OPENAI_API_KEY or ""
    dep = calib.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME or "gpt-4o"
    if not (ep and key):
        return []
    body = {
        "model": dep,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "max_output_tokens": 700,          # we only need citations, not an essay → keep it fast
        "input": query,
    }
    url = f"{ep}{AZURE_RESPONSES_PATH}"
    headers = {"api-key": key, "Content-Type": "application/json"}
    out = None
    for attempt in (1, 2):                 # web_search latency is variable; retry once on timeout
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
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
    # primary: url_citation annotations on the assistant message — these carry page titles
    for it in out:
        if it.get("type") == "message":
            for c in (it.get("content") or []):
                for a in (c.get("annotations") or []):
                    url = a.get("url")
                    if a.get("type") == "url_citation" and url and url not in seen:
                        seen.add(url)
                        sources.append({"title": a.get("title") or url, "url": url, "content": ""})
    # supplement: the raw sources the tool consulted (urls, no titles)
    for it in out:
        if it.get("type") == "web_search_call":
            for s in ((it.get("action") or {}).get("sources") or []):
                url = s.get("url")
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": url, "url": url, "content": ""})
    return sources[:max_results]


def _tavily_search(query: str, max_results: int) -> List[dict]:
    body = {
        "api_key": os.getenv("TAVILY_API_KEY", ""),
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }
    try:
        r = requests.post(TAVILY_URL, json=body, timeout=15)
        r.raise_for_status()
        return [
            {"title": x.get("title") or "", "url": x.get("url") or "", "content": (x.get("content") or "").strip()}
            for x in (r.json().get("results") or []) if x.get("url")
        ]
    except Exception as e:
        logger.warning("Tavily search failed for %r: %s", query, e)
        return []


def _ddg_search(query: str, max_results: int) -> List[dict]:
    try:
        try:
            from ddgs import DDGS                       # newer package name
        except Exception:
            from duckduckgo_search import DDGS           # legacy package name
        out: List[dict] = []
        with DDGS() as d:
            for x in d.text(query, max_results=max_results):
                url = x.get("href") or x.get("url")
                if not url:
                    continue
                out.append({"title": x.get("title") or "", "url": url, "content": (x.get("body") or "").strip()})
        return out
    except Exception as e:
        logger.warning("DuckDuckGo search failed for %r: %s", query, e)
        return []


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Run one search via the active provider → [{title, url, content}]. [] when unavailable/on error."""
    prov = active_provider()
    if prov == "azure":
        return _azure_web_search(query, max_results)
    if prov == "tavily":
        return _tavily_search(query, max_results)
    if prov == "duckduckgo":
        return _ddg_search(query, max_results)
    return []


# ─────────────────────────── location helpers ───────────────────────────
def reverse_geocode(lat: float, lon: float) -> str:
    """Google reverse geocode (lat,lon) → a human place string for building search queries. '' on failure."""
    key = calib.GOOGLE_MAPS_API_KEY or ""
    if not key:
        return ""
    try:
        r = requests.get(GEOCODE_URL, params={"latlng": f"{lat},{lon}", "key": key}, timeout=8)
        js = r.json()
        if js.get("status") == "OK" and js.get("results"):
            # prefer a locality-level result if present, else the first (usually a street address)
            for res in js["results"]:
                types = res.get("types") or []
                if "locality" in types or "postal_town" in types or "administrative_area_level_2" in types:
                    return res.get("formatted_address") or ""
            return js["results"][0].get("formatted_address") or ""
    except Exception as e:
        logger.warning("Reverse geocode failed for (%s, %s): %s", lat, lon, e)
    return ""


def gather_location_sources(lat: float, lon: float, *, address: Optional[str] = None,
                            radius_km: float = 20, per_query: int = 4,
                            max_sources: int = 10) -> Tuple[List[dict], str]:
    """Run a few targeted market queries for this pin and return (deduped sources, resolved place string).
    Best-effort — returns ([], place) when web search is unavailable."""
    place = (address or "").strip() or reverse_geocode(lat, lon) or f"{lat:.4f}, {lon:.4f}"
    prov = active_provider()
    if prov is None:
        return [], place
    miles = f"{radius_km * 0.621:.0f}"
    # Azure Responses does its OWN multi-step searching in one call — a single research query is cheaper
    # (one Bing-billed call) and returns several citations. Keyword providers get several targeted queries.
    if prov == "azure":
        research = (
            f"Search the web for facts about {place} relevant to building a new express-tunnel car wash "
            f"(trade area ~{miles} miles): population and household income, growth, traffic and major roads, "
            f"retail anchors, and existing car-wash competitors. Do NOT write a report — just run the searches "
            f"and give a few bullet points, each with its source citation."
        )
        return web_search(research, max_results=max_sources), place
    queries = [
        f"{place} car wash competitors express tunnel",
        f"{place} demographics median household income population growth",
        f"{place} traffic counts major roads retail development",
        f"new car wash development {place}",
    ]
    seen, sources = set(), []
    for q in queries:
        for res in web_search(q, max_results=per_query):
            url = res.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(res)
            if len(sources) >= max_sources:
                return sources, place
    return sources, place
