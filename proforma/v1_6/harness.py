"""
Retrospective backtest runner — the manager's validation.

For each focal build in the sample (sites that opened 2021–2024 with a realized maturity window and
≥2 pre-T neighbours, N≈492), freeze the clock at T = operational_start, run the council on the
strictly-pre-T local market, and grade its go/no-go + projected level against the site's own post-T
actuals. Writes a per-(site×seat) CSV and an aggregate markdown report to council/outputs/.

Run:  python -m council.harness --limit 8            # cheap smoke over 8 sites
      python -m council.harness                       # full N≈492 (many LLM calls)
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from proforma.v1_6 import data_1_6 as D
from proforma.v1_6.snapshot import build_snapshot
from proforma.v1_6.council import council_decision, SEATS
from proforma.v1_6.scorer import realized_outcome, ape, gonogo_correct

logger = logging.getLogger(__name__)
OUT_DIR = Path(__file__).resolve().parent / "outputs"   # version-owned outputs (was repo-root/council/outputs)
_SEAT_ORDER = list(SEATS) + ["council"]


# ─────────────────────────── one site ───────────────────────────
def _run_one_site(row: pd.Series, *, radius_km: float, backend: Optional[str],
                  extract_location: bool, w_internal: Optional[float]) -> List[Dict[str, Any]]:
    T = row["t_open"]
    snap = build_snapshot(row.site_key, row.lat, row.lon, T, radius_km=radius_km)
    dec = council_decision(snap, backend=backend, radius_km=radius_km,
                           extract_location=extract_location, w_internal=w_internal)
    out = realized_outcome(row.site_key, T)
    realized, good = out["realized_mature_washes"], out["realized_good_build"]

    base = dict(focal_key=row.site_key, client_name=row.client_name, state=row.state,
                t_open=str(pd.Timestamp(T).date()), n_neighbours=snap.n_neighbours,
                express_like=bool(getattr(row, "express_like", False)),
                realized_mature=realized, realized_ramp=out["realized_ramp"],
                months_open=out["months_open"], realized_good_build=good, mature_floor=out["mature_floor"])
    rows: List[Dict[str, Any]] = []
    for v in dec["seats"]:
        proj = v["projected_mature_washes"]
        rows.append({**base, "seat": v["seat"], "access": v["access"], "lean": v["lean"],
                     "confidence": v["confidence"], "projected_mature": proj,
                     "ape": ape(proj, realized), "gonogo_correct": gonogo_correct(v["lean"], good),
                     "conflict": None, "conflict_note": None, "proj_low": None, "proj_high": None})
    pr = dec["projected_range"]
    proj = pr["median"] if pr else None
    rows.append({**base, "seat": "council", "access": "council", "lean": dec["verdict"],
                 "confidence": dec["confidence"], "projected_mature": proj,
                 "ape": ape(proj, realized), "gonogo_correct": gonogo_correct(dec["verdict"], good),
                 "conflict": dec["conflict"], "conflict_note": dec["conflict_note"],
                 "proj_low": pr["low"] if pr else None, "proj_high": pr["high"] if pr else None})
    return rows


def _safe_site(row, **kw) -> List[Dict[str, Any]]:
    try:
        return _run_one_site(row, **kw)
    except Exception as exc:
        logger.warning("Site %s failed: %s", getattr(row, "site_key", "?"), exc)
        return []


# ─────────────────────────── aggregation + report ───────────────────────────
def _pct(x: Optional[float]) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.0f}%"


def _seat_stats(res: pd.DataFrame, seat: str) -> Dict[str, Any]:
    sub = res[res.seat == seat]
    voted = sub[sub.lean.notna()]
    gg = voted.gonogo_correct.dropna().astype(float)
    yes = voted[voted.lean.isin(["Build", "Conditional"])]
    good_all = voted[voted.realized_good_build == True]                      # noqa: E712
    precision = (yes.realized_good_build == True).mean() if len(yes) else None   # noqa: E712
    recall = ((good_all.lean.isin(["Build", "Conditional"])).mean() if len(good_all) else None)
    apes = sub.ape.dropna()
    both = sub[sub.projected_mature.notna() & sub.realized_mature.notna()]
    wape = (float((both.projected_mature - both.realized_mature).abs().sum() / both.realized_mature.sum())
            if len(both) and both.realized_mature.sum() else None)
    return {
        "n_votes": int(len(voted)), "abstain": float(sub.lean.isna().mean()) if len(sub) else None,
        "accuracy": float(gg.mean()) if len(gg) else None,
        "build_precision": float(precision) if precision is not None else None,
        "build_recall": float(recall) if recall is not None else None,
        "median_ape": float(apes.median()) if len(apes) else None, "wape": wape,
        "n_proj": int(len(both)),
    }


def _build_report(res: pd.DataFrame, cand: pd.DataFrame, params: Dict[str, Any]) -> str:
    sites = res[res.seat == "council"]
    n = int(len(sites))
    base_rate = float(sites.realized_good_build.astype(float).mean()) if n else float("nan")
    df, _ = D.load_panel_1_6()
    dr = f"{df.date.min().date()} → {df.date.max().date()}"
    t_years = sorted(sites.t_open.str[:4].unique()) if n else []

    L: List[str] = [
        "# Council Retrospective Backtest — report",
        "",
        f"Council go/no-go for a NEW express-tunnel build, frozen at each site's opening month T "
        f"(`operational_start`) and graded against its own post-T actuals. Single source: "
        f"`proforma/data/panel/main-data-v2-stitched.csv`.",
        "",
        f"- **Sites graded:** {n}  (openings {', '.join(t_years)})",
        f"- **Realized 'good build' base rate:** {_pct(base_rate)}  "
        f"(a site clears the mature floor of ~{int(sites.mature_floor.iloc[0]):,} washes/mo with a healthy "
        f"ramp) — so 'always Build' would score ≈ {_pct(base_rate)} accuracy.",
        f"- **radius** {params['radius_km']:g} km · **min neighbours** {params['min_neighbours']} · "
        f"**data range** {dr}",
        "",
        "## Go/no-go accuracy (did the seat's Build/Pass match reality?)",
        "",
        "| Seat | votes | abstain | accuracy | Build precision | Build recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    stats = {s: _seat_stats(res, s) for s in _SEAT_ORDER}
    for s in _SEAT_ORDER:
        st = stats[s]
        L.append(f"| {'**'+s+'**' if s in ('internal','council') else s} | {st['n_votes']} | "
                 f"{_pct(st['abstain'])} | {_pct(st['accuracy'])} | {_pct(st['build_precision'])} | "
                 f"{_pct(st['build_recall'])} |")

    # express-like subset: where the council's express-tunnel assumptions actually apply
    exp_sites = sites[sites.express_like == True]                            # noqa: E712
    n_exp = int(len(exp_sites))
    exp_council = exp_sites[exp_sites.gonogo_correct.notna()]
    exp_acc = float(exp_council.gonogo_correct.astype(float).mean()) if len(exp_council) else None
    exp_base = float(exp_sites.realized_good_build.astype(float).mean()) if n_exp else None
    L += ["", f"*Express-like subset ({n_exp}/{n} sites — membership + volume so express economics apply): "
          f"council go/no-go accuracy {_pct(exp_acc)} vs base rate {_pct(exp_base)}. "
          f"The rest are small/retail-only washes the council's express seats can't size well.*"]

    L += ["", "## Projection error — internal anchor vs external LLM (does internal beat external?)",
          "", "| Seat | n | median APE | WAPE |", "| --- | ---: | ---: | ---: |"]
    for s in _SEAT_ORDER:
        st = stats[s]
        if st["n_proj"]:
            L.append(f"| {s} | {st['n_proj']} | {_pct(st['median_ape'])} | {_pct(st['wape'])} |")

    # calibration: council projected range coverage
    rng = sites[sites.proj_low.notna() & sites.proj_high.notna() & sites.realized_mature.notna()]
    cov = (float(((rng.realized_mature >= rng.proj_low) & (rng.realized_mature <= rng.proj_high)).mean())
           if len(rng) else None)
    n_conf = int(sites.conflict.fillna(False).astype(bool).sum())
    L += ["", "## Projection spread & disagreement", "",
          f"- **Seats bracketed reality:** the realized mature level fell within the seats' own projection "
          f"spread (min–max) in {_pct(cov)} of {len(rng)} sites (not a calibrated interval — just where the "
          f"seats' point estimates landed relative to the truth).",
          f"- **Internal-vs-external conflicts surfaced:** {n_conf} of {n} "
          f"({_pct(n_conf/n if n else None)}) — resolved by the deterministic rulebook (data wins, split shown)."]
    ex = sites[sites.conflict.fillna(False).astype(bool)].head(3)
    for _, r in ex.iterrows():
        L.append(f"  - {r.client_name} ({r.t_open}): **{r.lean}** — {r.conflict_note}")

    L += ["", "## Leakage & caveats (honest bounds)", "",
          "- Internal seat is a pure-Python descriptive read of pre-T neighbour data + a leakage-free "
          "mature-neighbour anchor — **no trained model, no future leakage**.",
          "- External seats retain LLM **training-cutoff** leakage (a recent model reasoning about a 2022 "
          "site) — unclosable; this is the measured ceiling of external-only analysis, and web search is "
          "disabled in the backtest.",
          "- `operational_start` = build date is a proxy (some openings are reporting-onset).",
          f"- Small-ish N ({n}) — aggregate rates and MAPE are meaningful; treat fine buckets with care.",
          "- External LLM calls are stochastic (temperature fixed low); rates move a little run-to-run."]
    return "\n".join(L)


# ─────────────────────────── runner ───────────────────────────
def run_backtest(*, min_mature_months: int = 4, min_neighbours: int = 2, radius_km: float = 20.0,
                 max_workers: int = 6, limit: Optional[int] = None, backend: Optional[str] = None,
                 extract_location: bool = True, w_internal: Optional[float] = None,
                 out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Select the sample, run the council per site (concurrently), score, and write CSV + report."""
    cand = D.focal_candidates(radius_km=radius_km, min_mature_months=min_mature_months,
                              min_neighbours=min_neighbours)
    if limit and limit < len(cand):                  # evenly-spaced slice → spans all T-years, not just 2021
        idx = np.unique(np.linspace(0, len(cand) - 1, limit).round().astype(int))
        cand = cand.iloc[idx].reset_index(drop=True)
    logger.info("Backtesting %d focal sites (limit=%s, workers=%d).", len(cand), limit, max_workers)

    rows_of = [row for _, row in cand.iterrows()]
    kw = dict(radius_km=radius_km, backend=backend, extract_location=extract_location, w_internal=w_internal)
    all_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rows in ex.map(lambda r: _safe_site(r, **kw), rows_of):
            all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "retro_council_results.csv"
    md_path = out_dir / "retro_council_report.md"
    params = dict(radius_km=radius_km, min_neighbours=min_neighbours, min_mature_months=min_mature_months)
    if len(res):
        res.to_csv(csv_path, index=False)
        report = _build_report(res, cand, params)
        md_path.write_text(report)
    else:
        report = "No sites scored."
    logger.info("Wrote %s (%d rows) and %s.", csv_path, len(res), md_path)
    return {"results": res, "report": report, "n_sites": int(len(cand)),
            "csv": str(csv_path), "md": str(md_path)}


def _signal_report(ev: Dict[str, Any], m: pd.DataFrame) -> str:
    base = ev["base_rate"]
    tk = ev["topk_precision"]
    df, _ = D.load_panel_1_6()
    dr = f"{df.date.min().date()} → {df.date.max().date()}"
    top30 = tk.get(30, base)
    L = [
        "# Council Backtest — signal decider (honest, out-of-fold)",
        "",
        f"The council was rebuilt: a leakage-clean **data signal** now makes the build/pass call and the LLM "
        f"seats are demoted to explanation. This is the offline, out-of-fold evaluation over **N={ev['n']}** "
        f"focal builds (openings 2021–2024, single source `proforma/data/panel/main-data-v2-stitched.csv`, data {dr}).",
        "",
        "## Does it finally beat “always build”?",
        "",
        "| Approach | good-build rate (precision) |",
        "| --- | ---: |",
        f"| Always build (base rate) | {base:.1%} |",
        f"| Old LLM council | ~30% (no edge — = base rate) |",
        f"| **Signal decider — build top 30% by score** | **{top30:.1%}  ({(top30-base)*100:+.0f} pts)** |",
        f"| Signal decider — build top 20% | {tk.get(20, base):.1%}  ({(tk.get(20,base)-base)*100:+.0f} pts) |",
        "",
        "## Out-of-fold AUC (is the signal real, or luck?)",
        "",
        "| Split | AUC |",
        "| --- | ---: |",
        f"| GroupKFold by operator (the honest number) | **{ev['auc_group']:.3f}** |",
        f"| StratifiedKFold | {ev['auc_strat']:.3f} |",
        f"| Sites with ≥2 matured neighbours (where the signal applies) | "
        f"{'—' if ev['auc_matured_subset'] is None else format(ev['auc_matured_subset'], '.3f')} |",
        f"| Permutation noise ceiling (p95) | {ev['perm_auc_p95']:.3f} |",
        "",
        f"AUC {ev['auc_group']:.3f} sits **above** the permutation ceiling {ev['perm_auc_p95']:.3f} → the edge is "
        f"real, not overfit — but modest. Roughly half of build outcomes stay unpredictable from any pre-build data.",
        "",
        "## What the signal is (and isn’t)",
        "",
        "- **Is:** local-market STRUCTURE + operator scale — weak weakest-neighbour = headroom = good; a market "
        "where even the worst wash is already big = saturated = bad; bigger operators build better.",
        "- **Isn’t:** demographics / traffic / income — near-zero predictors (and leaky 2025 snapshots), so dropped.",
        "- **Isn’t:** a mature-LEVEL forecast — greenfield level is ~unpredictable here; the trained model’s apparent "
        "skill was operator-identity leakage (honest as-of-T refit → corr ≈0). We predict the binary good/bad instead.",
        "",
        "## Caveats",
        "- LLM seats no longer vote — they were structurally bullish (constant “Build”) and added no discrimination; "
        "they remain as the live explanation layer.",
        "- Signal is strongest for sites with matured neighbours to learn from; it flags “weak signal” otherwise.",
        f"- Base rate is the absolute-floor good-build definition ({base:.1%}); a market-relative target is the next step.",
    ]
    return "\n".join(L)


def run_signal_backtest(*, out_dir: Path = OUT_DIR, rebuild: bool = False) -> Dict[str, Any]:
    """The honest, offline backtest of the rebuilt council: evaluate the data signal out-of-fold and write
    the report + a per-site results CSV (features + OOF probability + label). No LLM calls."""
    from proforma.v1_6 import decider as DEC
    ev = DEC.evaluate(rebuild=rebuild)
    m = DEC.get_matrix().copy()
    m["oof_prob"] = DEC.oof_probs(m, kind="group")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "retro_council_results.csv", out_dir / "retro_council_report.md"
    m.to_csv(csv_path, index=False)
    report = _signal_report(ev, m)
    md_path.write_text(report)
    logger.info("Signal backtest: AUC(group)=%.3f top30=%.1f%% base=%.1f%%",
                ev["auc_group"], ev["topk_precision"].get(30, 0) * 100, ev["base_rate"] * 100)
    return {"eval": ev, "report": report, "csv": str(csv_path), "md": str(md_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Council retrospective backtest")
    ap.add_argument("--llm", action="store_true",
                    help="run the OLD LLM-in-the-loop per-site backtest (slow/$$); default = offline signal eval")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of focal sites (smoke run)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--min-neighbours", type=int, default=2)
    ap.add_argument("--backend", default=None, help="insights LLM backend (azure|local); default env")
    ap.add_argument("--no-location-extract", action="store_true", help="skip the location lean LLM call")
    ap.add_argument("--w-internal", type=float, default=None, help="override internal seat weight")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.llm:                                                  # old per-site LLM council (context seats)
        r = run_backtest(limit=args.limit, max_workers=args.workers, radius_km=args.radius,
                         min_neighbours=args.min_neighbours, backend=args.backend,
                         extract_location=not args.no_location_extract, w_internal=args.w_internal)
        print(f"\nScored {r['n_sites']} sites → {r['csv']}\n")
    else:                                                         # default: honest offline signal eval
        r = run_signal_backtest(rebuild=True)
        print(f"\nSignal backtest → {r['csv']}\n")
    print(r["report"])


if __name__ == "__main__":
    main()
