"""print_report / plots / optional LLM text (mixin for `QuantilePredictorV4`)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from app.site_analysis.modelling.ds.prediction.constants import (
    FEATURE_LABELS,
    QUANTILE_TIER_NAMES,
    SIGNAL_THRESHOLD,
)


class QuantileReportingMixin:
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from app.utils.llm import local_llm as llm_module
            response = llm_module.get_llm_response(
                prompt, reasoning_effort="medium", temperature=temperature
            )
            text = (response or {}).get("generated_text", "").strip()
            return text if text else None
        except Exception:
            return None

    def _generate_narrative(self, result: Dict) -> str:
        pred_q    = result["predicted_wash_quantile"]
        tier      = result["predicted_wash_tier"]
        wash_range = result["predicted_wash_range"]["label"]
        proba     = result["quantile_probabilities"]
        conf      = round(proba.get(pred_q, 0) * 100, 1)

        sig_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items()
             if not fa.get("imputed") and fa.get("signal", 0) >= SIGNAL_THRESHOLD
             and fa.get("ml_feature", True)],
            key=lambda x: -x[1].get("importance", 0),
        )[:6]

        feat_lines = []
        for feat, fa in sig_feats:
            label  = fa["label"]
            val    = fa["value"]
            wq     = fa.get("wash_correlated_q")
            q4_med = fa.get("wash_q_q4_median")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            q4_note = f"Q4 typical: {q4_med:.1f}" if q4_med is not None else ""
            wq_note = "exceeds Q4 level" if exceeds else (f"matches Q{wq} sites" if wq else "")
            feat_lines.append(f"  - {label}: {val:.1f}  [{wq_note}, {q4_note}]")

        prompt = f"""You are a car wash site analyst. Write a SHORT narrative summary — strictly 2–3 sentences total.

PREDICTION: Q{pred_q} ({tier}), expected annual count {wash_range}, model confidence {conf}%.

KEY FEATURES (signal-bearing only):
{chr(10).join(feat_lines) if feat_lines else "  (none with significant signal)"}

Rules:
- Exactly 2–3 sentences. No bullets. No headers.
- Sentence 1: state the quartile and expected volume.
- Sentence 2-3: mention 2–3 key features and whether they match high-performing (Q4) site levels or not.
- Do NOT suggest improvements. Do NOT list all features."""

        text = self._call_llm(prompt, temperature=0.25)
        return (text or "").strip()

    def _generate_strengths_weaknesses_llm(self, result: Dict) -> str:
        strengths = result.get("strengths") or []
        weaknesses = result.get("weaknesses") or []
        pred_q     = result["predicted_wash_quantile"]
        tier       = result["predicted_wash_tier"]
        wash_range = result["predicted_wash_range"]["label"]

        def fmt(items):
            return [
                f"  - {s['label']}: {s['value']:.1f}" +
                (f" (Q4 med {s['q4_median']:.1f})" if s.get("q4_median") is not None else "") +
                f"  [{s['note']}]"
                for s in items[:8]
            ]

        s_block = "\n".join(fmt(strengths)) if strengths else "  (none identified)"
        w_block = "\n".join(fmt(weaknesses)) if weaknesses else "  (none identified)"

        prompt = f"""You are a car wash site investment analyst. Write a concise investment-report-style assessment.

SITE PREDICTION: Q{pred_q} ({tier}), expected {wash_range} cars/yr.

STRENGTHS (features matching Q4 high-performer levels):
{s_block}

WEAKNESSES (features at Q1/Q2 low-performer levels):
{w_block}

Output format — exactly 2 short paragraphs:
Paragraph 1 (3–5 sentences): Key strengths. Mention each strength once with its value vs Q4 benchmark. Professional analytical tone.
Paragraph 2 (3–5 sentences): Key weaknesses. Mention each weakness once with its value vs Q4 benchmark.
No bullet points in output. No recommendations. No subheadings."""

        text = self._call_llm(prompt, temperature=0.3)
        return (text or "").strip()

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, result: Dict):
        q = result["predicted_wash_quantile"]
        W = 90

        def hdr(title: str):
            print(f"\n{'─' * W}")
            print(f"  {title}")
            print(f"{'─' * W}")

        print("\n" + "═" * W)
        print("  QUANTILE PREDICTION REPORT  —  v4  (ExtraTrees, Optuna-tuned)")
        print("═" * W)
        print(f"\n  Predicted Quantile  : {result['predicted_wash_quantile_label']}  — {result['predicted_wash_tier']}")
        print(f"  Expected Annual Vol : {result['predicted_wash_range']['label']}")

        hdr("PREDICTION CONFIDENCE")
        for qi in range(1, self.n_quantiles + 1):
            p     = result["quantile_probabilities"].get(qi, 0)
            bar   = "█" * int(p * 36)
            mark  = "  ◄ PREDICTED" if qi == q else ""
            r     = result["wash_count_distribution"][qi]["range"]
            lbl   = f"Q{qi} {QUANTILE_TIER_NAMES[qi][:14]}"
            print(f"  {lbl:<22s} [{bar:<36s}] {p*100:5.1f}%  {r}{mark}")

        hdr(
            "FEATURE ANALYSIS  (★ exceeds Q4  ▲ matches Q4  ▼ matches Q1  ~ low signal  [D]=display-only)\n"
            "  WashQ = car wash tier this value matches  |  Signal = |Spearman r| with count\n"
            "  [D] = shown for context, NOT in ML model (already captured by effective_capacity)"
        )
        print(
            f"  {'Feature':<38s} {'Value':>8s}  {'WashQ':>5s}  {'Pctile':>6s}"
            f"  {'Q4 med':>8s}  {'Signal':>6s}  {'Imprt':>5s}  {'ML':>3s}"
        )
        print("  " + "─" * (W - 2))

        ml_feats = [(f, fa) for f, fa in result["feature_analysis"].items() if fa.get("ml_feature", True)]
        disp_feats = [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("ml_feature", True)]
        sorted_feats = sorted(ml_feats, key=lambda x: -x[1].get("importance", 0)) + disp_feats

        for feat, fa in sorted_feats:
            is_ml   = fa.get("ml_feature", True)
            label   = (fa["label"][:34] + " (imp)" if fa.get("imputed") else fa["label"][:36])
            val     = fa["value"]
            wq      = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            low_sig = fa.get("wash_q_low_signal", False)
            pct     = fa["adjusted_percentile"]
            imp     = fa["importance"]
            q4_med  = fa.get("wash_q_q4_median")
            signal  = fa.get("signal", 0.0)
            ml_tag  = "✓" if is_ml else "D"

            if low_sig or wq is None:
                wq_str = "  ~"
                marker = ""
            else:
                wq_str = f"Q{wq}"
                marker = "★" if exceeds else ("▲" if wq == 4 else ("▼" if wq == 1 else " "))

            q4_str = f"{q4_med:>8.1f}" if q4_med is not None else f"{'n/a':>8s}"
            print(
                f"  {label:<38s} {val:>8.1f}  {wq_str:>4s}{marker}  {pct:>5.1f}%"
                f"  {q4_str}  {signal:>5.3f}  {imp:>5.1%}  {ml_tag:>3s}"
            )

        hdr("WHY CARWASH TYPE IS CAPTURED VIA effective_capacity (not a direct ML feature)")
        cw_fa = result["feature_analysis"].get("carwash_type_encoded")
        if cw_fa:
            print(f"  carwash_type_encoded = {cw_fa.get('value', 'N/A'):.0f}  "
                  f"({FEATURE_LABELS['carwash_type_encoded']})")
            print(f"  WashQ match: Q{cw_fa.get('wash_correlated_q', '?')}  |  "
                  f"Spearman |r| = {cw_fa.get('signal', 0):.3f}  |  "
                  f"Adjusted percentile: {cw_fa.get('adjusted_percentile', 0):.0f}th")
            print()
            print("  Express Tunnel sites average 143K washes/yr vs 79-119K for other types.")
            print("  However, effective_capacity = tunnel_count × is_express already encodes this:")
            print("    Express 1 tunnel → ec=1 (avg 68K), Express 2 → ec=2 (avg 168K), ...")
            print("    Mobile/Flex/Hand Wash → ec=0 (avg 50-83K, uses age+location features)")
            print("  Adding carwash_type_encoded directly creates multicollinearity → −0.9% accuracy.")
        else:
            print("  carwash_type_encoded not provided in input.")

        hdr("STRENGTHS & WEAKNESSES  (LLM investment summary)")
        sw_llm = result.get("strengths_weaknesses_llm")
        if sw_llm:
            for para in sw_llm.split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()

        if result["shift_opportunities"]:
            hard = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) > 0]
            soft = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) == 0]
            if hard:
                hdr("QUANTILE SHIFT OPPORTUNITIES")
                print(f"  {'Feature':<34s} {'Current':>9s} {'Target':>9s} {'Change':>8s}  Wash Q Shift")
                print("  " + "─" * (W - 2))
                for o in hard:
                    d = "+" if o["change_direction"] == "increase" else "−"
                    print(
                        f"  {o['label'][:32]:<34s} {o['current_value']:>9.1f} {o['target_value']:>9.1f} "
                        f"{d}{o['change_needed']:>7.1f}  Q{o['current_wash_q']} → Q{o['simulated_wash_q']}"
                    )
            if soft:
                hdr("PROBABILITY-LIFT OPPORTUNITIES  (no single feature flips the quartile)")
                print(f"  {'Feature':<34s} {'Current':>9s} {'Target':>9s}  Prob Lift")
                print("  " + "─" * (W - 2))
                for o in soft:
                    d = "+" if o["change_direction"] == "increase" else "−"
                    print(
                        f"  {o['label'][:32]:<34s} {o['current_value']:>9.1f} {o['target_value']:>9.1f}"
                        f"  +{o.get('prob_lift',0):.1f}%"
                    )

        hdr("PROFILE COMPARISON  (your value vs each car wash tier's median)")
        print(
            f"  {'Feature':<38s} {'Yours':>8s}  "
            + "  ".join(f"{'Q'+str(qi)+' med':>8s}" for qi in range(1, self.n_quantiles + 1))
        )
        print("  " + "─" * (W - 2))
        for feat, fa in sorted_feats[:12]:
            val  = fa["value"]
            comp = result["profile_comparison"].get(feat, {})
            meds = "  ".join(
                f"{comp.get(f'Q{qi}', {}).get('profile_median', 0.0):>8.1f}"
                for qi in range(1, self.n_quantiles + 1)
            )
            print(f"  {fa['label'][:36]:<38s} {val:>8.1f}  {meds}")

        if result.get("narrative"):
            hdr("NARRATIVE SUMMARY")
            for para in result["narrative"].split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()

        print(
            f"\n  Model v4 (ExtraTrees) — 5-fold CV  exact: {result['model_cv_accuracy']:.1%}  "
            f"within-1-quartile: {result.get('model_adj_accuracy',0):.1%}  "
            f"ML features provided: {result['features_available']}/{len(self.feature_cols)}\n"
            f"  Error analysis: 96% of wrong predictions are adjacent (Q1↔Q2 or Q3↔Q4).\n"
            f"  Theoretical ceiling ~65% — volume also driven by traffic, ops, pricing (not in features)."
        )
        print("═" * W)

    def report_to_string(self, result: Dict) -> str:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.print_report(result)
        return buf.getvalue()

    def save_report(self, result: Dict, path: Optional[Path] = None) -> Path:
        if path is None:
            path = Path(__file__).parent / "quantile_report_v4.txt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report_to_string(result), encoding="utf-8")
        print(f"✓ Report saved: {path}")
        return path

    def plot_feature_quantiles(
        self,
        result: Dict,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (24, 20),
    ) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            print("[Skip] matplotlib not installed.")
            return None

        if output_path is None:
            output_path = Path(__file__).parent / "quantile_analysis_v4.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Show only ML features in plot (skip display-only)
        sorted_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items()
             if not fa.get("imputed") and fa.get("ml_feature", True)],
            key=lambda x: -x[1].get("importance", 0),
        )
        n_feat = len(sorted_feats)
        cols   = 4
        rows   = (n_feat + cols - 1) // cols + 1

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor("#f8f9fa")
        axes_flat = axes.flatten()
        q_colours = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        predicted_q = result["predicted_wash_quantile"]

        ax_prob = axes_flat[0]
        qs    = list(range(1, self.n_quantiles + 1))
        probs = [result["quantile_probabilities"].get(q, 0) * 100 for q in qs]
        bars  = ax_prob.bar([f"Q{q}" for q in qs], probs,
                            color=[q_colours[q-1] for q in qs], edgecolor="white", linewidth=0.8)
        for bar, p in zip(bars, probs):
            ax_prob.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{p:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax_prob.set_title(
            f"Predicted: Q{predicted_q} — {QUANTILE_TIER_NAMES[predicted_q]}\n"
            f"({result['predicted_wash_range']['label']})",
            fontsize=10, fontweight="bold")
        ax_prob.set_ylabel("Probability (%)", fontsize=8)
        ax_prob.set_ylim(0, 105)
        ax_prob.set_facecolor("#f0f0f0")

        ax_bench = axes_flat[1]
        ax_bench.axis("off")
        bench = "CAR WASH COUNT RANGES\n\n"
        for qi in range(1, self.n_quantiles + 1):
            r = result["wash_count_distribution"][qi]["range"]
            bench += f"Q{qi}: {r}{' ◄ YOU' if qi == predicted_q else ''}\n"
        bench += f"\nCV Exact: {result['model_cv_accuracy']:.1%} (v4)"
        bench += f"\nCV Within-1: {result.get('model_adj_accuracy',0):.1%}"
        bench += "\nModel: ExtraTrees (Optuna-tuned)"
        ax_bench.text(0.05, 0.95, bench, transform=ax_bench.transAxes, va="top", ha="left",
                      fontsize=8.5, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="#d4edda", edgecolor="#28a745"))

        ax_sw = axes_flat[2]
        ax_sw.axis("off")
        sw_text = "STRENGTHS & WEAKNESSES\n\n"
        for s in (result.get("strengths") or [])[:4]:
            sw_text += f"✓ {s['label'][:28]}: {s['value']:.1f}\n"
        sw_text += "\n"
        for w in (result.get("weaknesses") or [])[:4]:
            sw_text += f"✗ {w['label'][:28]}: {w['value']:.1f}\n"
        ax_sw.text(0.05, 0.95, sw_text, transform=ax_sw.transAxes, va="top", ha="left",
                   fontsize=8, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107"))

        # Carwash type note in panel 4
        ax_cw = axes_flat[3]
        ax_cw.axis("off")
        cw_fa = result["feature_analysis"].get("carwash_type_encoded", {})
        cw_text = "CARWASH TYPE (display-only)\n\n"
        if cw_fa.get("value") is not None:
            enc_map = {1: "Express Tunnel", 2: "Mobile/Flex", 3: "Hand Wash"}
            enc_val = int(round(cw_fa["value"]))
            cw_text += f"Type: {enc_map.get(enc_val, str(enc_val))}\n"
            cw_text += f"WashQ: Q{cw_fa.get('wash_correlated_q','?')}\n"
            cw_text += f"|r| = {cw_fa.get('signal',0):.3f}\n\n"
        cw_text += "Already captured via:\neffective_capacity = tc × is_express\nDirect use → −0.9% accuracy"
        ax_cw.text(0.05, 0.95, cw_text, transform=ax_cw.transAxes, va="top", ha="left",
                   fontsize=8, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8d5f5", edgecolor="#9b59b6"))

        for idx, (feat, fa) in enumerate(sorted_feats):
            ax            = axes_flat[cols + idx]
            dist          = self.feature_dists.get(feat)
            if dist is None:
                ax.axis("off")
                continue
            data          = dist["data"]
            val           = fa["value"]
            wq            = fa.get("wash_correlated_q", 2)
            exceeds       = fa.get("wash_correlated_exceeds_q4", False)
            group_medians = fa.get("wash_q_group_medians", {})
            low_sig       = fa.get("wash_q_low_signal", False)
            signal        = fa.get("signal", 0.0)

            ax.hist(data, bins=min(30, max(10, len(np.unique(data)))),
                    color="#aec7e8", edgecolor="white", linewidth=0.4, alpha=0.85, zorder=1)

            for qi in range(1, self.n_quantiles + 1):
                gm = group_medians.get(qi)
                if gm is not None:
                    ls = "--" if not low_sig else ":"
                    ax.axvline(gm, color=q_colours[qi-1], linestyle=ls, linewidth=1.3,
                               alpha=0.85, zorder=2, label=f"Q{qi}:{gm:.0f}")

            eff_wq = 4 if exceeds else (wq if wq else 2)
            mc = q_colours[eff_wq - 1] if not low_sig else "#555555"
            ax.axvline(val, color=mc, linestyle="-", linewidth=2.8, zorder=4, label=f"You:{val:.1f}")

            sig_tag  = f"r={signal:.3f}" if signal >= SIGNAL_THRESHOLD else f"r={signal:.3f}~"
            wq_tag   = f"WashQ{wq}" if not low_sig and wq else "~"
            ax.set_title(
                f"{fa['label'][:30]}\n{wq_tag} | {fa['adjusted_percentile']:.0f}th pct | {sig_tag} | imp {fa['importance']:.1%}",
                fontsize=7, fontweight="bold" if fa["importance"] > 0.06 else "normal")
            ax.set_ylabel(f"n={len(data)}", fontsize=6)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc="upper right", ncol=1)
            ax.set_facecolor("#f8f9fa")

        for j in range(cols + n_feat, len(axes_flat)):
            axes_flat[j].axis("off")

        legend_handles = [
            Line2D([0], [0], color=q_colours[q-1], linestyle="--", linewidth=1.5,
                   label=f"Q{q} median ({QUANTILE_TIER_NAMES[q]})")
            for q in range(1, self.n_quantiles + 1)
        ] + [Line2D([0], [0], color="black", linestyle="-", linewidth=2.5, label="Your value")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=5,
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

        fig.suptitle(
            f"Car Wash Quantile Analysis v4  —  Predicted Q{predicted_q} "
            f"({result['predicted_wash_range']['label']})\n"
            f"Model: ExtraTrees (Optuna-tuned, exact CV {result['model_cv_accuracy']:.1%})  |  "
            f"carwash_type shown in summary panel (display-only, captured by effective_capacity)",
            fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"✓ Plot saved: {output_path}")
        return output_path
