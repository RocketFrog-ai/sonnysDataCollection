"""Committee experts — Historical · Competition · Local-Market · Capacity · Finance.

`build_experts(order)` returns the seated experts in investigation order, Finance forced last (it
consolidates the others' numbers once they're on the board). `Expert` is the base contract in `base.py`.
"""
from __future__ import annotations

from typing import List, Optional

from experiments.council.experts.base import Expert  # noqa: F401


def build_experts(order: Optional[List[str]] = None) -> List[Expert]:
    """Instantiate the 5 committee seats in `order` (default `config.EXPERT_ORDER`). Finance is always
    sorted last regardless of `order`, since it consolidates the other seats' evidence."""
    from experiments.council import config as C
    from experiments.council.experts.historical import HistoricalExpert
    from experiments.council.experts.competition import CompetitionExpert
    from experiments.council.experts.local_market import LocalMarketExpert
    from experiments.council.experts.capacity import CapacityExpert
    from experiments.council.experts.finance import FinanceExpert

    reg = {"historical": HistoricalExpert, "competition": CompetitionExpert,
           "local_market": LocalMarketExpert, "capacity": CapacityExpert, "finance": FinanceExpert}
    order = order or C.EXPERT_ORDER
    experts = [reg[k]() for k in order if k in reg]
    experts.sort(key=lambda e: e.name == "finance")  # Finance last
    return experts
