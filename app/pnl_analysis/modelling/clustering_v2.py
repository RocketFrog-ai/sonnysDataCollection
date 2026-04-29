"""
Legacy import path for PnL Celery tasks.

The clustering_v2 daily_data pipeline is no longer used here. Wash volume,
ranges (P10/P90 style via calibrated quantiles), and downstream PnL blocks
are produced by :mod:`app.pnl_analysis.modelling.zeta_pnl` using
``zeta_modelling/model_1`` and ``zeta_modelling/data_1``.
"""

from app.pnl_analysis.modelling.zeta_pnl import (
    run_clustering_v2_projection_task,
    run_pnl_central_input_form_task,
)

__all__ = ["run_clustering_v2_projection_task", "run_pnl_central_input_form_task"]
