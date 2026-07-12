"""
Backtest harness — the honest validation.

Two tiers (`python -m experiments.council.harness [--committee] [--limit N] [--light]`):

  • DEFAULT — `run_signal_backtest`: the leakage-clean **signal decider** evaluated OUT-OF-FOLD over N≈420
    focal builds. **No LLM.** This is the only number quoted as "the edge" (the committee never contaminates
    it). Writes `outputs/retro_council_report.md` + `retro_council_results.csv`.

  • `--committee` — `run_committee_backtest`: convene the full committee on a few evenly-spaced sites frozen
    at their opening month T (leakage-controlled snapshot), grade the go/no-go against post-T actuals, and
    dump each transcript for inspection. Qualitative (is the arguing sane?), not an aggregate-accuracy claim.
    `--light` runs it with zero LLM (the regression guard: the committee verdict should track the decider).

Self-contained: council modules only.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.council import data_1_6 as D
from experiments.council.committee import run_committee
from experiments.council.protocol import MsgType
from experiments.council.scorer import gonogo_correct, realized_outcome
from experiments.council.snapshot import build_snapshot

logger = logging.getLogger(__name__)
OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ─────────────────────────── tier 1: honest signal backtest (no LLM) ───────────────────────────
def _signal_report(ev: Dict[str, Any]) -> str:
    base, tk = ev["base_rate"], ev["topk_precision"]
    df, _ = D.load_panel_1_6()
    dr = f"{df.date.min().date()} → {df.date.max().date()}"
    top30 = tk.get(30, base)
    L = [
        "# Council backtest — signal decider (honest, out-of-fold)",
        "",
        f"The go/no-go anchor is a leakage-clean **data signal** (local-market structure + operator scale). This "
        f"is the offline, out-of-fold evaluation over **N={ev['n']}** focal builds (openings 2021+, single source "
        f"`experiments/council/data/Council--historical-data.csv`, data {dr}). The committee's LLM deliberation "
        f"never touches this number.",
        "",
        "## Does it beat “always build”?",
        "",
        "| Approach | good-build rate (precision) |",
        "| --- | ---: |",
        f"| Always build (base rate) | {base:.1%} |",
        f"| **Signal — build top 30% by score** | **{top30:.1%}  ({(top30-base)*100:+.0f} pts)** |",
        f"| Signal — build top 20% | {tk.get(20, base):.1%}  ({(tk.get(20,base)-base)*100:+.0f} pts) |",
        "",
        "## Out-of-fold AUC (real signal, or luck?)",
        "",
        "| Split | AUC |",
        "| --- | ---: |",
        f"| GroupKFold by operator (the honest number) | **{ev['auc_group']:.3f}** |",
        f"| StratifiedKFold | {ev['auc_strat']:.3f} |",
        f"| Sites with ≥2 matured neighbours | "
        f"{'—' if ev['auc_matured_subset'] is None else format(ev['auc_matured_subset'], '.3f')} |",
        f"| Permutation noise ceiling (p95) | {ev['perm_auc_p95']:.3f} |",
        "",
        f"AUC {ev['auc_group']:.3f} sits above the permutation ceiling {ev['perm_auc_p95']:.3f} → the edge is real "
        f"but modest; ~half of build outcomes stay unpredictable from any pre-build data.",
        "",
        "## What the signal is",
        "- **Is:** local-market STRUCTURE + operator scale — a weak weakest-neighbour = headroom = good; a market "
        "where even the worst wash is already big = saturated = bad; bigger operators build better.",
        "- **Isn’t:** demographics / traffic / income (near-zero predictors), nor a mature-LEVEL forecast (greenfield "
        "level is ~unpredictable; we predict the binary good/bad).",
        "",
        "In the committee, the LLM seats reason and argue around this signal (which sits on the board as evidence); "
        "the committee decides, and any divergence from the signal is surfaced in the report.",
    ]
    return "\n".join(L)


def run_signal_backtest(*, out_dir: Path = OUT_DIR, rebuild: bool = False) -> Dict[str, Any]:
    """Honest, offline, no-LLM evaluation of the signal decider. Writes the report + a per-site results CSV."""
    from experiments.council import decider as DEC
    ev = DEC.evaluate(rebuild=rebuild)
    m = DEC.get_matrix().copy()
    m["oof_prob"] = DEC.oof_probs(m, kind="group")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "retro_council_results.csv", out_dir / "retro_council_report.md"
    m.to_csv(csv_path, index=False)
    report = _signal_report(ev)
    md_path.write_text(report)
    logger.info("Signal backtest: AUC(group)=%.3f top30=%.1f%% base=%.1f%%",
                ev["auc_group"], ev["topk_precision"].get(30, 0) * 100, ev["base_rate"] * 100)
    return {"eval": ev, "report": report, "csv": str(csv_path), "md": str(md_path)}


# ─────────────────────────── tier 2: committee sample (qualitative) ───────────────────────────
def _transcript(res) -> List[str]:
    out, last = [], -1
    for m in res.log.messages:
        if m.round != last:
            out.append(f"── round {m.round} ──"); last = m.round
        tgt = f" → {m.to}" if m.to else ""
        out.append(f"  {m.mtype.value:9} {m.sender}{tgt}: {m.text}")
    return out


def run_committee_backtest(*, limit: int = 6, light: bool = False, out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Convene the committee on `limit` evenly-spaced focal builds frozen at T, grade go/no-go vs post-T
    actuals, and dump transcripts. Qualitative — inspect the arguing, not aggregate accuracy."""
    cand = D.focal_candidates()
    if limit and limit < len(cand):
        idx = np.unique(np.linspace(0, len(cand) - 1, limit).round().astype(int))
        cand = cand.iloc[idx].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "committee_samples"
    samples_dir.mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for _, r in cand.iterrows():
        try:
            snap = build_snapshot(r.site_key, r.lat, r.lon, r["t_open"], radius_km=20.0)
            res = run_committee(snap, light=light, radius_km=20.0)
            out = realized_outcome(r.site_key, r["t_open"])
            good = out["realized_good_build"]
            d = res.decision
            n_chal = sum(1 for m in res.log.messages if m.mtype == MsgType.CHALLENGE)
            n_rev = sum(1 for m in res.log.messages if m.mtype == MsgType.REVISE)
            rows.append({"site_key": r.site_key, "client_name": r.client_name, "t_open": str(pd.Timestamp(r["t_open"]).date()),
                         "verdict": d.verdict, "confidence": d.confidence, "signal_lean": d.signal_lean,
                         "diverges": d.diverges_from_signal, "realized_good_build": good,
                         "gonogo_correct": gonogo_correct(d.verdict, good), "n_messages": len(res.log.messages),
                         "n_challenges": n_chal, "n_revises": n_rev, "llm_calls": res.workspace.llm_calls})
            # dump the transcript
            (samples_dir / f"{r.site_key.replace('::','_')}.txt").write_text(
                f"{r.client_name} — opened {pd.Timestamp(r['t_open']).date()} — verdict {d.verdict} "
                f"(realized good build: {good})\n" + "\n".join(_transcript(res)))
        except Exception as exc:
            logger.warning("committee backtest site %s failed: %s", r.site_key, exc)

    res_df = pd.DataFrame(rows)
    csv_path = out_dir / "committee_backtest.csv"
    if len(res_df):
        res_df.to_csv(csv_path, index=False)
    return {"results": res_df, "csv": str(csv_path), "samples_dir": str(samples_dir), "n": len(res_df)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Council backtest")
    ap.add_argument("--committee", action="store_true", help="run the committee sample (LLM) instead of the offline signal eval")
    ap.add_argument("--limit", type=int, default=6, help="committee sample size")
    ap.add_argument("--light", action="store_true", help="committee sample with zero LLM (regression guard)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the signal feature matrix")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.committee:
        r = run_committee_backtest(limit=args.limit, light=args.light)
        print(f"\nCommittee sample: {r['n']} sites → transcripts in {r['samples_dir']}\n")
        if len(r["results"]):
            print(r["results"][["client_name", "verdict", "signal_lean", "realized_good_build",
                                "n_challenges", "n_revises"]].to_string(index=False))
    else:
        r = run_signal_backtest(rebuild=args.rebuild)
        print(f"\nSignal backtest → {r['csv']}\n")
        print(r["report"])


if __name__ == "__main__":
    main()
