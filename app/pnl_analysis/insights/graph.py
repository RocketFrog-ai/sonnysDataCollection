"""
The 2-node Key-Insights pipeline.

    compute_metrics  ──►  generate_insights
    (pure python)         (LLM, Azure | local)

Built as a LangGraph `StateGraph` when langgraph is installed; otherwise the SAME two node
functions run in sequence via a tiny fallback runner — so the feature ships regardless of whether
langgraph is present in the interpreter that launches Streamlit.

Public entry point: `market_insights(panel, sites_meta, focal_key) -> {"metrics": ..., "insights": ...}`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TypedDict

import pandas as pd

from app.pnl_analysis.insights.metrics import compute_metrics
from app.pnl_analysis.insights.prompts import build_combined_messages, parse_group_sections
from app.pnl_analysis.insights import llm as llm_client

logger = logging.getLogger(__name__)

_GROUPS = ("Washes", "Revenue", "ASPs")


class InsightsState(TypedDict, total=False):
    # inputs
    panel: pd.DataFrame
    sites_meta: pd.DataFrame
    focal_key: str
    last_n_months: int
    backend: Optional[str]
    # node outputs
    metrics: Dict[str, Any]
    insights: Dict[str, str]


# ─────────────────────────── nodes ───────────────────────────
def compute_metrics_node(state: InsightsState) -> Dict[str, Any]:
    metrics = compute_metrics(
        state["panel"], state["sites_meta"], state["focal_key"],
        last_n_months=state.get("last_n_months", 12),
    )
    return {"metrics": metrics}


def generate_insights_node(state: InsightsState) -> Dict[str, Any]:
    """LLM-only: cascade Azure↔local. No fabricated fixed-string narrative — if neither backend answers,
    surface an honest error notice so the operator fixes connectivity instead of trusting canned text."""
    metrics = state["metrics"]
    try:

        print(build_combined_messages(metrics))
        print("*"*50)
        raw, used = llm_client.complete_cascade(build_combined_messages(metrics),
                                                backend=state.get("backend"), json_mode=True)
        logger.info("Insights generated via %s backend.", used)
        parsed = parse_group_sections(raw)
        insights = {g: (parsed.get(g) or f"_The model did not return a {g} section — try regenerating._")
                    for g in _GROUPS}
    except llm_client.LLMUnavailable as exc:
        notice = f"⚠️ Could not generate insights — both Azure and the local LLM are unreachable. ({exc})"
        insights = {g: notice for g in _GROUPS}
    except Exception as exc:  # surface, don't fabricate
        logger.warning("Insights generation failed: %s", exc)
        insights = {g: f"⚠️ Insights generation failed: {exc}" for g in _GROUPS}
    return {"insights": insights}


# ─────────────────────────── graph wiring ───────────────────────────
def _build_graph():
    """Compiled LangGraph, or None if langgraph isn't importable / fails to build."""
    try:
        from langgraph.graph import StateGraph
    except Exception:
        return None
    try:
        g = StateGraph(InsightsState)
        g.add_node("compute_metrics", compute_metrics_node)
        g.add_node("generate_insights", generate_insights_node)
        g.add_edge("compute_metrics", "generate_insights")
        g.set_entry_point("compute_metrics")
        g.set_finish_point("generate_insights")
        return g.compile()
    except Exception as exc:
        logger.warning("Failed to build LangGraph (%s); using sequential fallback.", exc)
        return None


_GRAPH = _build_graph()
USING_LANGGRAPH = _GRAPH is not None


def market_insights(panel: pd.DataFrame, sites_meta: pd.DataFrame, focal_key: str, *,
                    backend: Optional[str] = None, last_n_months: int = 12) -> Dict[str, Any]:
    """Run the pipeline and return {"metrics": dict, "insights": {"Washes","Revenue","ASPs"}}."""
    state: InsightsState = {
        "panel": panel, "sites_meta": sites_meta, "focal_key": focal_key,
        "last_n_months": last_n_months, "backend": backend,
    }
    if _GRAPH is not None:
        out = _GRAPH.invoke(state)
    else:
        out = dict(state)
        out.update(compute_metrics_node(out))
        out.update(generate_insights_node(out))
    return {"metrics": out["metrics"], "insights": out["insights"]}
