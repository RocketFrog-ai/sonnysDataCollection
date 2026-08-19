"""
Chart system for the customer-profiling experiment — one palette, one Plotly layout.

Same validated palette the conclusions app uses (`conclusion/demo/ui.py`), restated here because
`experiments/` is standalone and off the import path. Both modes are *selected*: the dark column is
the same hues re-stepped for the dark surface, not an automatic flip of the light one.

The notebook renders on white and calls `theme(dark=False)`; the Streamlit demo pins
`base = "dark"` and calls `theme(dark=True)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import plotly.graph_objects as go


@dataclass(frozen=True)
class Theme:
    dark: bool
    ink: str
    ink2: str
    muted: str
    axis: str
    grid: str
    surface: str
    plane: str
    border: str
    header_bg: str
    series: tuple[str, ...]
    # Status steps are mode-invariant by design: all four clear 3:1 on both surfaces, and always
    # ship with a glyph and a label so hue never carries the meaning alone.
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"
    seq: tuple[str, ...] = field(default_factory=lambda: (
        "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"))

    @property
    def s1(self) -> str: return self.series[0]

    @property
    def s2(self) -> str: return self.series[1]

    @property
    def s3(self) -> str: return self.series[2]

    @property
    def s4(self) -> str: return self.series[3]


LIGHT = Theme(
    dark=False, ink="#0b0b0b", ink2="#52514e", muted="#898781", axis="#c3c2b7",
    grid="#e1e0d9", surface="#fcfcfb", plane="#f9f9f7",
    border="rgba(11,11,11,0.10)", header_bg="#f0efec",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)
DARK = Theme(
    dark=True, ink="#ffffff", ink2="#c3c2b7", muted="#898781", axis="#383835",
    grid="#2c2c2a", surface="#1a1a19", plane="#0d0d0d",
    border="rgba(255,255,255,0.10)", header_bg="#242422",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)


def theme(dark: bool = False) -> Theme:
    return DARK if dark else LIGHT


# Personas keep a fixed colour each, so a filter that drops one never repaints the survivors.
# Order is the validated slot order: blue, orange, aqua, yellow.
SEGMENT_ORDER = ["Power household", "Core regular", "Never activated", "Promo flipper"]


def segment_colors(t: Theme) -> dict[str, str]:
    return dict(zip(SEGMENT_ORDER, t.series[:4]))


def layout(t: Theme, **kw) -> dict:
    base = dict(
        paper_bgcolor=t.surface, plot_bgcolor=t.surface,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=t.ink2, size=12),
        # text="" matters: with only a font set, Plotly leaves the title undefined and Streamlit
        # renders the literal string "undefined" above the chart.
        title=dict(text="", font=dict(color=t.ink, size=15)),
        xaxis=dict(gridcolor=t.grid, zeroline=False, linecolor=t.axis, tickfont=dict(color=t.muted)),
        yaxis=dict(gridcolor=t.grid, zeroline=False, linecolor=t.axis, tickfont=dict(color=t.muted)),
        # y=1.06 (not 1.02) + a taller top margin -- 1.02 sits close enough to the plot top that a
        # long title (or a title on a chart with little headroom, e.g. a 3D scene) can collide with
        # the legend row above it. This gives a guaranteed gap between them on every chart.
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t.ink2), orientation="h",
                    yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=64, r=28, t=64, b=48), hovermode="closest",
    )
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def style(fig: go.Figure, t: Theme, **kw) -> go.Figure:
    """Apply the house layout, letting the caller override any slot (one level deep)."""
    fig.update_layout(**layout(t, **kw))
    return fig
