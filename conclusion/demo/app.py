"""
Conclusions — the evidence pack, built for a CIO review.

Thin entry point only: page config, shared CSS, and the section switch. Each section is a module
that owns its own sidebar controls and tabs, so adding the next one is a two-line change here.

    conda activate sonnys
    streamlit run conclusion/demo/app.py          # from the repo root

Layering, mirroring the repo's own convention (proforma/ui/app.py + panels/):

    tunnel_data.py / proforma_data.py   the maths. Streamlit-free — the notebook imports these too,
                                        so the app and the notebook cannot report different numbers.
    ui.py                               palette, Plotly layout, CSS, table + callout helpers.
    section_*.py                        one Streamlit section each.
    app.py                              this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pandas 3.0.2 stores strings in a pyarrow-backed array by default, and in this environment
# (pyarrow 25.0.0) constructing one segfaults the process — not raises, *segfaults* — after a
# handful of Streamlit script runs. The faulthandler trace lands in
# `pandas/core/arrays/string_arrow.py::_from_sequence`, reached from ordinary calls like
# `DataFrame.dropna(subset=[...])` (which builds a string Index from the column names). Switching
# to the pure-Python string backend takes that code path out of the picture entirely. Set before
# any DataFrame is built. The proper fix is an environment pin — see each section's method tab.
pd.set_option("mode.string_storage", "python")

st.set_page_config(page_title="Conclusions — proforma evidence pack", page_icon="🚗",
                   layout="wide")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402
import section_campaign  # noqa: E402
import section_proforma  # noqa: E402
import section_tunnel  # noqa: E402

ui.inject_css()

# Adding a section = one more entry here.
SECTIONS = [
    ("① Tunnel length", "Is the tunnel we build bigger than the volume needs?",
     section_tunnel.render),
    ("② Proforma backtest", "When the proforma projected a wash count, how close did it land?",
     section_proforma.render),
    ("③ Campaign causality", "Promotions look powerful against a site's own past. Do they survive "
     "a counterfactual?", section_campaign.render),
]

with st.sidebar:
    st.markdown("## Conclusions")
    picked = st.radio("Section", [s[0] for s in SECTIONS], label_visibility="collapsed")
    st.caption({s[0]: s[1] for s in SECTIONS}[picked])
    st.markdown("---")

{s[0]: s[2] for s in SECTIONS}[picked]()
