"""Key-Insights pipeline for the Explore-markets dashboard (2-node: compute_metrics -> generate_insights).

`market_insights` is imported lazily so that importing the pure `metrics` submodule does not pull in
the LLM client / langgraph chain.
"""

__all__ = ["market_insights"]


def __getattr__(name):
    if name == "market_insights":
        from app.pnl_analysis.insights.graph import market_insights
        return market_insights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
