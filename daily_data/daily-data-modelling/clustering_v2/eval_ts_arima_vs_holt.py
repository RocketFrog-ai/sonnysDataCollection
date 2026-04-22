"""One-step-ahead on train cluster median monthly series: ARIMA vs Holt-Winters → results/ts_arima_vs_holt_pick.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parent
REPO_ROOT = V2.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2))

from project_site import _forecast, _series_to_df  # noqa: E402


def _eval_cohort(models_subdir: str) -> dict:
    p = V2 / "models" / models_subdir / "cluster_monthly_series_12km.json"
    blob = json.loads(p.read_text())
    hw_mae = ar_mae = 0.0
    hw_absy = ar_absy = 0.0
    n = 0
    for rows in blob["series"].values():
        s = _series_to_df(rows)
        if len(s) < 7:
            continue
        for t in range(6, len(s)):
            tr = s.iloc[:t]
            y1 = float(s.iloc[t])
            n += 1
            p_hw = float(_forecast(tr, 1, "holt_winters").iloc[0])
            p_ar = float(_forecast(tr, 1, "arima").iloc[0])
            hw_mae += abs(p_hw - y1)
            ar_mae += abs(p_ar - y1)
            hw_absy += abs(y1)
            ar_absy += abs(y1)
    return {
        "n_one_step_forecasts": n,
        "mae_holt_winters": hw_mae / max(n, 1),
        "mae_arima": ar_mae / max(n, 1),
        "wape_holt_winters": hw_mae / max(hw_absy, 1e-9),
        "wape_arima": ar_mae / max(ar_absy, 1e-9),
    }


def main() -> None:
    lt = _eval_cohort("less_than")
    mt = _eval_cohort("more_than")
    sum_hw = lt["mae_holt_winters"] + mt["mae_holt_winters"]
    sum_ar = lt["mae_arima"] + mt["mae_arima"]
    pick = "arima" if sum_ar < sum_hw else "holt_winters"
    out = {
        "less_than_2yrs": lt,
        "more_than_2yrs": mt,
        "recommendation": pick,
        "rule": "Pick method with lower (mae_lt + mae_mt) on one-step cluster-median monthly tracks.",
    }
    outp = V2 / "results" / "ts_arima_vs_holt_pick.json"
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {outp.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
