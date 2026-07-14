"""
Expand the one-row-per-proforma summary into MONTHLY long format:
one row per (proforma file x matched-site month), with the proforma input/
factor columns repeated on every month and that month's panel actuals attached.

Unmatched proformas are kept as a single row with blank monthly fields so no
source file is dropped.  Reads the summary CSV (which already carries the
match_client_id / match_site_id chosen by match_panel.py).
"""
import os
import pandas as pd

HERE = os.path.dirname(__file__)
SUMMARY = os.path.join(HERE, "..", "old-proforma-combined.csv")
PANEL = os.path.join(HERE, "..", "..", "..", "proforma", "data", "panel", "main-data-v2-stitched.csv")
OUT = os.path.join(HERE, "..", "old-proforma-combined-monthly.csv")

MONTH_COLS = ["year", "month", "mem_purchase_count", "mem_wash_count", "mem_revenue",
              "ret_wash_count", "ret_revenue", "ASP_mem", "ASP_ret",
              "mem_revenue_pct", "mem_count_pct", "imputed"]


def main():
    pf = pd.read_csv(SUMMARY)
    panel = pd.read_csv(PANEL)
    panel["wash_count"] = panel["mem_wash_count"].fillna(0) + panel["ret_wash_count"].fillna(0)
    panel["revenue"] = panel["mem_revenue"].fillna(0) + panel["ret_revenue"].fillna(0)

    # proforma-side columns to repeat = everything EXCEPT the rolled-up actual_*
    # aggregates (those belong to the summary file, not the monthly grain).
    keep = [c for c in pf.columns if not c.startswith("actual_")]

    pmonth = MONTH_COLS + ["wash_count", "revenue"]
    rows = []
    for _, r in pf.iterrows():
        base = {c: r[c] for c in keep}
        cid, sid = r.get("match_client_id"), r.get("match_site_id")
        if pd.notna(cid) and pd.notna(sid):
            sub = panel[(panel["client_id"] == cid) & (panel["site_id"] == int(sid))]
            sub = sub.sort_values(["year", "month"])
            if len(sub):
                for _, m in sub.iterrows():
                    row = dict(base)
                    for c in pmonth:
                        row[c] = m[c]
                    rows.append(row)
                continue
        # unmatched (or matched site with no panel rows): single blank-month row
        row = dict(base)
        for c in pmonth:
            row[c] = None
        rows.append(row)

    out = pd.DataFrame(rows)
    # column order: proforma/match cols first, then the monthly block
    ordered = keep + pmonth
    out = out[ordered]
    out.to_csv(OUT, index=False)

    n_files = out["source_file"].nunique()
    n_month_rows = out["year"].notna().sum()
    print(f"Wrote {len(out)} rows ({n_files} source files) -> {OUT}")
    print(f"  monthly rows: {n_month_rows} | blank-month (unmatched) rows: {len(out) - n_month_rows}")
    print("  months per matched file (describe):")
    per = out[out["year"].notna()].groupby("source_file").size()
    print("   ", per.describe()[["count", "mean", "min", "max"]].to_dict())


if __name__ == "__main__":
    main()
