"""
Historical Analyst — the WHOLE 12-mile neighbour cluster's observed track record + a forward projection +
the leakage-clean signal exhibit.

`investigate` reads every Sonny's site within 12 miles from the council panel and posts the cluster's
OBSERVED performance — how many sites, and their median monthly washcount, revenue, membership purchases and
ASP (mem/retail) — so it is transparent that the read spans the full cluster, not one site. It then runs the
council-local forecast (`forecast.project_site`, whose mature-level anchor + ramp come from the reliable
comparables: ≥30 months of history, matured, non-COVID) to project what a NEW build here would mature into,
and computes the leakage-clean signal decider's independent Build/Pass call (`anchor.compute_anchor`, the one
component with measured out-of-fold edge) as `signal.decider`. `initial_belief` forms its OWN lean from
the cluster forecast vs the healthy-site floor; the signal is a quiet cross-check on the board, not a driver.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from experiments.council import anchor as A
from experiments.council import config as C
from experiments.council import data_1_6 as D
from experiments.council import forecast
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence


def _cluster_observed(lat: float, lon: float) -> Dict[str, Any]:
    """Observed track record across ALL sites within 12 miles: counts + the median of each site's recent
    (last-12-month) monthly washcount, revenue, membership purchases, and ASP."""
    df, site = D.load_panel_1_6()
    nb = D.neighbours_within(site, lat, lon, C.HISTORICAL_CLUSTER_KM)
    stats: Dict[str, Any] = {"n_sites": int(len(nb)), "n_matured": int(nb.mature_level.notna().sum())}
    keys = nb.site_key.tolist()
    if keys:
        recent = df[df.site_key.isin(keys)].sort_values("date").groupby("site_key").tail(12)
        g = recent.groupby("site_key").agg(wash=("tot_wash_count", "mean"), rev=("tot_revenue", "mean"),
                                           mem_purch=("mem_purchase_count", "mean"),
                                           asp_m=("ASP_mem", "median"), asp_r=("ASP_ret", "median"))

        def _med(s) -> Optional[float]:
            s = s.dropna()
            return float(s.median()) if len(s) else None

        stats.update(wash=_med(g.wash), rev=_med(g.rev), mem_purch=_med(g.mem_purch),
                     asp_mem=_med(g.asp_m), asp_ret=_med(g.asp_r))
    return stats


def _qualified_comparables(lat: float, lon: float, max_show: int = 6) -> List[Dict[str, Any]]:
    """The reliable comparables (≥30mo, matured, non-COVID) INDIVIDUALLY — each site's OWN mature washcount,
    recent revenue, membership purchases and ASP — so all data points (not just their median) inform the
    analysis and are citable in the debate. Exactly the donor set the forecast anchor is built from."""
    df, site = D.load_panel_1_6()
    nb = D.neighbours_within(site, lat, lon, C.HISTORICAL_CLUSTER_KM)
    donors = forecast._qualified_donors(nb)
    rows: List[Dict[str, Any]] = []
    for _, d in donors.sort_values("dist_km").iterrows():
        s = df[df.site_key == d.site_key]
        mat = s[(s.rel >= D.MAT_LO) & (s.rel <= D.MAT_HI)]        # the site's own matured window
        base = mat if len(mat) else s.sort_values("date").tail(12)

        def _mean(col):
            v = base[col].dropna()
            return float(v.mean()) if len(v) else None

        def _med(col):
            v = base[col].dropna()
            return float(v.median()) if len(v) else None

        rows.append({"name": str(d.client_name)[:22], "dist_mi": round(float(d.dist_km) * 0.621, 1),
                     "mature_wash": round(float(d.mature_level)), "revenue_mo": _mean("tot_revenue"),
                     "mem_purch_mo": _mean("mem_purchase_count"), "asp_mem": _med("ASP_mem"),
                     "asp_ret": _med("ASP_ret"), "months": int(d.n_obs)})
    return rows[:max_show]


def _comparables_text(comps: List[Dict[str, Any]]) -> str:
    if not comps:
        return "no sites in the 12-mile cluster meet the ≥30mo / matured / non-COVID bar"
    parts = []
    for c in comps:
        seg = f"{c['name']} ({c['dist_mi']}mi): {c['mature_wash']:,}/mo"
        if c.get("revenue_mo"):
            seg += f", ${round(c['revenue_mo']):,} rev"
        if c.get("mem_purch_mo"):
            seg += f", {round(c['mem_purch_mo']):,} mem-buys"
        if c.get("asp_mem"):
            seg += f", ASP ${c['asp_mem']:.0f}/{c.get('asp_ret') or 0:.0f}"
        parts.append(seg)
    return " · ".join(parts)


class HistoricalExpert(Expert):
    name = "historical"
    role = "Historical Analyst"
    persona = ("You read the WHOLE 12-mile neighbour cluster's track record — how many Sonny's sites are "
               "there and their observed washcount, revenue, membership purchases and ASP — plus how new "
               "sites ramped, to project what a new build here would mature into.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        obs = _cluster_observed(ws.lat, ws.lon)
        proj = forecast.project_site(ws.lat, ws.lon)
        comps = _qualified_comparables(ws.lat, ws.lon)
        out = [
            self.ev("hist.cluster_size", "sites in the 12-mile cluster",
                    f"{obs['n_sites']} Sonny's sites within 12 mi · {obs['n_matured']} matured · "
                    f"{proj['n_donors']} qualify as reliable comparables (≥30mo, matured)",
                    kind="text", source="Council--historical-data.csv (12mi cluster)", confidence=0.8),
            self.ev("hist.comparables", f"each of the {len(comps)} qualified comparable(s) — its OWN numbers",
                    _comparables_text(comps), kind="text", source="12mi cluster (qualified comparables)",
                    confidence=0.75),
            self.ev("hist.cluster_wash", "cluster median monthly washcount (observed, all sites)",
                    obs.get("wash"), unit="washes/mo", source="12mi cluster observed", confidence=0.75),
            self.ev("hist.cluster_rev", "cluster median monthly revenue (observed)",
                    obs.get("rev"), unit="$/mo", source="12mi cluster observed", confidence=0.7),
            self.ev("hist.membership", "cluster median monthly membership purchases (observed) + share",
                    {"mem_purchases_per_mo": obs.get("mem_purch"), "membership_share": proj["mem_share"]},
                    kind="table", source="12mi cluster observed", confidence=0.7),
            self.ev("hist.cluster_asp", "cluster observed ASP (membership / retail)",
                    {"asp_mem": obs.get("asp_mem"), "asp_ret": obs.get("asp_ret")},
                    kind="table", unit="$/wash", source="12mi cluster observed", confidence=0.7),
            self.ev("hist.projected_mature", "projected mature washes/mo for a NEW build here (from the comparables)",
                    proj["mature_anchor"], unit="washes/mo", source="forecast.project_site", confidence=0.65),
            self.ev("hist.ramp_pattern", "how new sites in this cluster ramped",
                    f"ramps to ~90% of mature in {proj['ramp_to_90pct_months']} mo, learned from "
                    f"{proj['n_donors']} qualified comparables ({proj['ramp_source']})",
                    kind="text", source="forecast.project_site", confidence=0.6),
        ]
        try:
            out.append(A.compute_anchor(ws.snap).as_evidence())     # eid "signal.decider"
        except Exception:
            pass
        return out

    def initial_belief(self, ws) -> BeliefState:
        proj_ev = ws.evidence.get("hist.projected_mature")
        projected = proj_ev.value if proj_ev is not None else None
        # Historical forms its OWN lean from the cluster forecast vs the healthy-site floor. The data signal
        # is a quiet cross-check on the board (P good-build), not a driver of this seat's vote.
        lean = self.lean_from_level(projected, D.mature_floor())
        return BeliefState(expert=self.name, lean=lean, confidence=0.6, key_number=projected,
                           key_number_label="projected mature washes/mo",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
